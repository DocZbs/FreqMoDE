#!/usr/bin/env python
"""
Vision-Aware Flow Matching Training Script for CALVIN

This script trains a Vision-Aware Flow Matching policy on the CALVIN dataset.
Key innovation: During denoising, concat a vision cls token alongside action tokens,
enabling the model to implicitly learn action information through visual features.
"""

import logging
from pathlib import Path
import sys
import os
import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.base import Container, ContainerMetadata
from omegaconf.nodes import ValueNode
import torch
from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only

torch.serialization.add_safe_globals([
    DictConfig,
    ListConfig,
    Container,
    ContainerMetadata,
    ValueNode
])

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
sys.path.insert(0, str(Path(__file__).absolute().parents[3]))

import experiments.vis_aware.models.vis_aware_agent as vis_aware_models
from mode.utils.utils import (
    get_git_commit_hash,
    get_last_checkpoint,
    initialize_pretrained_weights,
    print_system_env_info
)
from mode.callbacks.loss_logger import LossLogger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


def clear_cuda_cache():
    """Clear CUDA cache and garbage collect"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        for i in range(torch.cuda.device_count()):
            memory_stats = torch.cuda.memory_stats(i)
            allocated = memory_stats.get('allocated_bytes.all.current', 0) / (1024**3)
            reserved = memory_stats.get('reserved_bytes.all.current', 0) / (1024**3)
            logger.info(f"GPU {i} Memory: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


@rank_zero_only
def log_rank_0(*args, **kwargs):
    logger.info(*args, **kwargs)


def setup_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    return [hydra.utils.instantiate(cb) for cb in callbacks_cfg.values()]


def setup_logger(cfg: DictConfig, model: LightningModule):
    pathlib_cwd = Path.cwd()
    if "group" in cfg.logger:
        cfg.logger.group = pathlib_cwd.parent.name
        cfg.logger.name = f"{pathlib_cwd.parent.name}/{pathlib_cwd.name}"
        cfg.logger.id = cfg.logger.name.replace("/", "_")
    return hydra.utils.instantiate(cfg.logger)


@hydra.main(config_path="../configs", config_name="config_vis_aware_calvin", version_base=None)
def train(cfg: DictConfig) -> None:
    try:
        os.environ['HYDRA_FULL_ERROR'] = '1'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'gloo'

        seed_everything(cfg.seed, workers=True)
        torch.set_float32_matmul_precision('medium')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        clear_cuda_cache()

        log_rank_0(f"\nInitializing Vision-Aware Flow Matching training for seed {cfg.seed}")
        log_rank_0(f"Flow type: {cfg.flow_type}")
        log_rank_0(f"Sampling method: {cfg.sampling_method}")
        log_rank_0(f"Vision token dim: {cfg.cls_token_dim} (used as conditioning)")

        datamodule = hydra.utils.instantiate(cfg.datamodule)

        checkpoint = get_last_checkpoint(Path.cwd())
        if checkpoint is None:
            model = hydra.utils.instantiate(cfg.model)
            model.eval()
            log_rank_0("Initializing model with dummy forward pass...")
        else:
            log_rank_0(f"Resuming from checkpoint: {checkpoint}")
            model = getattr(vis_aware_models, cfg.model["_target_"].split(".")[-1]).load_from_checkpoint(
                checkpoint.as_posix()
            )

        if "pretrain_chk" in cfg:
            initialize_pretrained_weights(model, cfg)

        hydra_cfg = HydraConfig.get()
        hydra_output_dir = Path(hydra_cfg.runtime.output_dir)
        log_rank_0(f"Hydra output directory: {hydra_output_dir}")

        work_dir = hydra_output_dir / f"seed_{cfg.seed}"
        work_dir.mkdir(exist_ok=True)
        os.chdir(work_dir)

        if 'checkpoint' in cfg.callbacks:
            cfg.callbacks.checkpoint.dirpath = str(work_dir / "checkpoints")
            log_rank_0(f"Checkpoints will be saved to: {cfg.callbacks.checkpoint.dirpath}")

        train_logger = setup_logger(cfg, model)
        callbacks = setup_callbacks(cfg.callbacks) + [
            LearningRateMonitor(logging_interval="step"),
            LossLogger(log_every_n_steps=50)
        ]

        file_handler = logging.FileHandler(work_dir / 'training.log', mode='a')
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(file_handler)

        trainer_args = {
            **cfg.trainer,
            "logger": train_logger,
            "callbacks": callbacks,
            "benchmark": False,
            "strategy": "ddp_find_unused_parameters_true",
            "accelerator": "gpu",
            "devices": cfg.trainer.devices,
            "use_distributed_sampler": True,
            "default_root_dir": work_dir,
            "sync_batchnorm": True,
        }

        log_rank_0(f"Training config for seed {cfg.seed}:\n{OmegaConf.to_yaml(cfg)}")
        log_rank_0(f"Git commit: {get_git_commit_hash(Path(hydra.utils.to_absolute_path(__file__)))}")
        log_rank_0(print_system_env_info())

        clear_cuda_cache()

        trainer = Trainer(**trainer_args)

        ckpt_path = cfg.get('resume_from_checkpoint', None)
        if ckpt_path and Path(ckpt_path).exists():
            log_rank_0(f"Resuming from checkpoint: {ckpt_path}")
        else:
            ckpt_path = None

        try:
            trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
        except Exception as e:
            log_rank_0("\nDetailed Error Information:")
            log_rank_0("=" * 80)
            log_rank_0(f"Error Type: {type(e).__name__}")
            log_rank_0(f"Error Message: {str(e)}")
            log_rank_0("\nFull Traceback:")
            import traceback
            log_rank_0(''.join(traceback.format_tb(e.__traceback__)))
            log_rank_0("=" * 80)
            raise e

    except Exception as e:
        logger.error(f"\nTraining failed for seed {cfg.seed}:")
        logger.error(f"{'='*80}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"{'='*80}")
        raise


if __name__ == "__main__":
    train()
