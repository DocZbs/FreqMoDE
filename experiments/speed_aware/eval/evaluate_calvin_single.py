#!/usr/bin/env python
"""
Evaluate Speed-Aware Flow Matching model on CALVIN - Single GPU evaluation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).absolute().parents[3]))

import hydra
from omegaconf import DictConfig
import torch
import json
from datetime import datetime
import numpy as np

import experiments.speed_aware.models.speed_aware_agent as speed_aware_models
from mode.evaluation.utils import LangEmbeddings
from mode.evaluation.multistep_sequences import get_sequences


@hydra.main(config_path="../configs", config_name="config_speed_aware_calvin", version_base=None)
def evaluate(cfg: DictConfig):
    """
    Evaluate trained Speed-Aware Flow Matching model on CALVIN using single GPU
    """

    checkpoint_path = cfg.get('checkpoint_path', None)
    if checkpoint_path is None:
        checkpoint_path = "/mnt/nvme-fast/zbs/fp/MoDE_Diffusion_Policy/experiments/speed_aware/logs/checkpoints/last.ckpt"

    print(f"Loading checkpoint from: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    # Load Speed-Aware model
    print("Loading model...")
    model = speed_aware_models.SpeedAwareFlowAgent.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    # Backward compatibility: old rectified flow checkpoints
    if hasattr(model.model, 'flow_type') and model.model.flow_type == 'rectified':
        if not hasattr(model.model, 'rectified_use_linear_time'):
            model.model.rectified_use_linear_time = False
            print("Old checkpoint detected: set rectified_use_linear_time=False")
        else:
            if '20251216' in checkpoint_path or 'rectified_euler' in checkpoint_path:
                model.model.rectified_use_linear_time = False
                print("Pre-fix checkpoint: set rectified_use_linear_time=False for compatibility")

    model.eval()
    model.to(device)
    print("Model loaded successfully")

    # Ensure language encoder is on the correct device
    if hasattr(model, 'language_goal'):
        model.language_goal.device = device
        model.language_goal.to(device)
        if hasattr(model.language_goal, 'clip_rn50'):
            model.language_goal.clip_rn50.to(device)

    # Setup datamodule
    print("Setting up datamodule...")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup("fit")

    # Get validation dataset
    val_dataloaders = datamodule.val_dataloader()
    if isinstance(val_dataloaders, dict):
        dataloader = val_dataloaders.get('lang', next(iter(val_dataloaders.values())))
    else:
        dataloader = val_dataloaders[0] if isinstance(val_dataloaders, list) else val_dataloaders
    dataset = dataloader.dataset
    print("Datamodule ready")

    # Setup rollout callback
    print("Setting up rollout callback...")
    rollout_callback = hydra.utils.instantiate(cfg.rollout_callback)
    rollout_callback.device = device

    # Initialize environment
    print("Initializing CALVIN environment...")
    from mode.wrappers.hulc_wrapper import HulcWrapper
    rollout_callback.env = HulcWrapper(
        dataset,
        device,
        show_gui=False
    )
    print("Environment initialized successfully")

    # Initialize language embeddings
    print("Loading language embeddings...")
    rollout_callback.lang_embeddings = LangEmbeddings(
        dataset.abs_datasets_dir,
        rollout_callback.lang_folder,
        device=device
    )
    print("Language embeddings loaded successfully")

    # Get evaluation sequences
    num_sequences = cfg.get('num_sequences', 1000)
    all_sequences = get_sequences(num_sequences)
    print(f"Generated {len(all_sequences)} evaluation sequences")

    # Set evaluation sequences
    rollout_callback.eval_sequences = all_sequences

    # Evaluate
    print(f"\n{'='*80}")
    print("Starting evaluation...")
    print(f"{'='*80}\n")

    results = rollout_callback.evaluate_policy(model)
    print(f"\nEvaluation completed. Results length: {len(results)}")

    if len(results) == 0:
        print("Error: No results collected")
        return None

    # Calculate statistics
    results_array = np.array(results)
    avg_seq_len = np.mean(results_array)

    # Calculate success rates for each task length (SR_1 to SR_5)
    success_rates = {}
    for i in range(1, 6):
        success_rates[f'SR_{i}'] = np.mean(results_array >= i) * 100

    results_dict = {
        'avg_seq_len': avg_seq_len,
        **success_rates
    }

    # Print results
    print(f"\n{'='*80}")
    print("CALVIN Long-Horizon Task Success Rates")
    print(f"{'='*80}")
    print(f"Total sequences evaluated: {len(results)}")
    for key, value in results_dict.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"{'='*80}\n")

    # Save results to JSON file
    output_dir = Path(checkpoint_path).parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"calvin_eval_results_{timestamp}.json"

    save_data = {
        'timestamp': timestamp,
        'checkpoint_path': str(checkpoint_path),
        'num_sequences': len(results),
        'results': results_dict,
        'raw_results': results_array.tolist()
    }

    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"Results saved to: {results_file}\n")

    return results_dict


if __name__ == "__main__":
    evaluate()
