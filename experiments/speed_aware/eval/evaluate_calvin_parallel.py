#!/usr/bin/env python
"""
Evaluate Speed-Aware Flow Matching model on CALVIN - Multi-GPU parallel evaluation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).absolute().parents[3]))

import hydra
from omegaconf import DictConfig
import torch
import torch.multiprocessing as mp
import json
from datetime import datetime
import numpy as np

import experiments.speed_aware.models.speed_aware_agent as speed_aware_models
from mode.evaluation.utils import LangEmbeddings
from mode.evaluation.multistep_sequences import get_sequences


def evaluate_on_gpu(gpu_id, checkpoint_path, cfg, sequences, return_dict):
    """
    Evaluate on a single GPU

    Args:
        gpu_id: GPU device ID
        checkpoint_path: Path to model checkpoint
        cfg: Hydra config
        sequences: List of evaluation sequences for this GPU
        return_dict: Shared dictionary to store results
    """
    try:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        device = torch.device("cuda:0")
        print(f"[GPU {gpu_id}] Starting evaluation on {len(sequences)} sequences")
        print(f"[GPU {gpu_id}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"[GPU {gpu_id}] PyTorch device: {device}")

        torch.manual_seed(42 + gpu_id)
        np.random.seed(42 + gpu_id)

        # Load Speed-Aware model
        model = speed_aware_models.SpeedAwareFlowAgent.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )

        # Backward compatibility: old rectified flow checkpoints
        if hasattr(model.model, 'flow_type') and model.model.flow_type == 'rectified':
            if not hasattr(model.model, 'rectified_use_linear_time'):
                model.model.rectified_use_linear_time = False
                print(f"[GPU {gpu_id}] Old checkpoint detected: set rectified_use_linear_time=False")
            else:
                if '20251216' in checkpoint_path or 'rectified_euler' in checkpoint_path:
                    model.model.rectified_use_linear_time = False
                    print(f"[GPU {gpu_id}] Pre-fix checkpoint: set rectified_use_linear_time=False for compatibility")

        model.eval()
        model.to(device)

        # Ensure language encoder is on the correct device
        if hasattr(model, 'language_goal'):
            model.language_goal.device = device
            model.language_goal.to(device)
            if hasattr(model.language_goal, 'clip_rn50'):
                model.language_goal.clip_rn50.to(device)

        # Setup datamodule
        datamodule = hydra.utils.instantiate(cfg.datamodule)
        datamodule.setup("fit")

        # Get validation dataset
        val_dataloaders = datamodule.val_dataloader()
        if isinstance(val_dataloaders, dict):
            dataloader = val_dataloaders.get('lang', next(iter(val_dataloaders.values())))
        else:
            dataloader = val_dataloaders[0] if isinstance(val_dataloaders, list) else val_dataloaders
        dataset = dataloader.dataset

        # Setup rollout callback
        rollout_callback = hydra.utils.instantiate(cfg.rollout_callback)
        rollout_callback.device = device

        # Initialize environment
        print(f"[GPU {gpu_id}] Initializing CALVIN environment...")
        from mode.wrappers.hulc_wrapper import HulcWrapper
        rollout_callback.env = HulcWrapper(
            dataset,
            device,
            show_gui=False
        )
        print(f"[GPU {gpu_id}] Environment initialized successfully")

        # Initialize language embeddings
        print(f"[GPU {gpu_id}] Loading language embeddings...")
        rollout_callback.lang_embeddings = LangEmbeddings(
            dataset.abs_datasets_dir,
            rollout_callback.lang_folder,
            device=device
        )
        print(f"[GPU {gpu_id}] Language embeddings loaded successfully")

        # Set evaluation sequences for this GPU
        rollout_callback.eval_sequences = sequences
        print(f"[GPU {gpu_id}] Assigned {len(sequences)} sequences to evaluate")

        # Evaluate
        print(f"[GPU {gpu_id}] Running evaluation...")
        results = rollout_callback.evaluate_policy(model)
        print(f"[GPU {gpu_id}] Evaluation completed. Results length: {len(results)}")

        # Store results
        return_dict[gpu_id] = results
        print(f"[GPU {gpu_id}] Completed evaluation")

    except Exception as e:
        print(f"[GPU {gpu_id}] Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return_dict[gpu_id] = []


@hydra.main(config_path="../configs", config_name="config_speed_aware_calvin", version_base=None)
def evaluate(cfg: DictConfig):
    """
    Evaluate trained Speed-Aware Flow Matching model on CALVIN using multi-GPU parallelism
    """

    checkpoint_path = cfg.get('checkpoint_path', None)
    if checkpoint_path is None:
        checkpoint_path = "/mnt/nvme-fast/zbs/fp/MoDE_Diffusion_Policy/experiments/speed_aware/logs/checkpoints/last.ckpt"

    print(f"Loading checkpoint from: {checkpoint_path}")

    # Get number of GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"Warning: Only {num_gpus} GPU(s) available.")

    # Use specified number of GPUs or all available
    num_gpus_to_use = cfg.get('num_gpus', num_gpus)
    num_gpus_to_use = min(num_gpus_to_use, num_gpus)
    print(f"Using {num_gpus_to_use} GPU(s) for evaluation")

    # Get evaluation sequences
    num_sequences = cfg.get('num_sequences', 1000)
    all_sequences = get_sequences(num_sequences)
    print(f"Generated {len(all_sequences)} evaluation sequences")

    # Split sequences across GPUs
    sequences_per_gpu = len(all_sequences) // num_gpus_to_use
    gpu_sequences = []
    for i in range(num_gpus_to_use):
        start_idx = i * sequences_per_gpu
        if i == num_gpus_to_use - 1:
            end_idx = len(all_sequences)
        else:
            end_idx = (i + 1) * sequences_per_gpu
        gpu_sequences.append(all_sequences[start_idx:end_idx])

    print(f"\nSequence distribution:")
    for i, seqs in enumerate(gpu_sequences):
        print(f"  GPU {i}: {len(seqs)} sequences")

    # Setup multiprocessing
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_dict = manager.dict()

    # Launch evaluation on each GPU
    print(f"\n{'='*80}")
    print(f"Starting parallel evaluation on {num_gpus_to_use} GPUs...")
    print(f"{'='*80}\n")

    processes = []
    for gpu_id in range(num_gpus_to_use):
        p = mp.Process(
            target=evaluate_on_gpu,
            args=(gpu_id, checkpoint_path, cfg, gpu_sequences[gpu_id], return_dict)
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    print("\nWaiting for all GPU processes to complete...")
    for i, p in enumerate(processes):
        p.join()
        exit_code = p.exitcode
        if exit_code != 0:
            print(f"Warning: GPU {i} process exited with code {exit_code}")
        else:
            print(f"GPU {i} process completed successfully")

    # Collect results from all GPUs
    print("\nCollecting results from all GPUs...")
    all_results = []
    for gpu_id in range(num_gpus_to_use):
        if gpu_id in return_dict:
            gpu_results = return_dict[gpu_id]
            print(f"GPU {gpu_id}: collected {len(gpu_results)} results")
            all_results.extend(gpu_results)
        else:
            print(f"Warning: No results from GPU {gpu_id}")

    if len(all_results) == 0:
        print("Error: No results collected from any GPU")
        return None

    # Calculate statistics
    results_array = np.array(all_results)
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
    print(f"Total sequences evaluated: {len(all_results)}")
    for key, value in results_dict.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"{'='*80}\n")

    # Save results to JSON file
    output_dir = Path(checkpoint_path).parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"calvin_eval_results_parallel_{timestamp}.json"

    save_data = {
        'timestamp': timestamp,
        'checkpoint_path': str(checkpoint_path),
        'num_sequences': len(all_results),
        'num_gpus': num_gpus_to_use,
        'results': results_dict,
        'raw_results': results_array.tolist()
    }

    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"Results saved to: {results_file}\n")

    return results_dict


if __name__ == "__main__":
    evaluate()
