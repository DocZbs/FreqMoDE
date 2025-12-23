"""
Extended DiskDataset that supports loading precomputed CLIP features.
"""

import logging
from pathlib import Path
from typing import Any, Dict
import numpy as np

from mode.datasets.disk_dataset import DiskDataset

logger = logging.getLogger(__name__)


class DiskDatasetWithCLIP(DiskDataset):
    """
    Extended DiskDataset that loads precomputed CLIP cls_token features.

    This significantly speeds up training by avoiding redundant CLIP forward passes.

    Args:
        clip_features_dir: Directory containing precomputed CLIP features
        use_precomputed_clip: Whether to load precomputed features
    """

    def __init__(
        self,
        *args: Any,
        clip_features_dir: str = None,
        use_precomputed_clip: bool = True,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.use_precomputed_clip = use_precomputed_clip
        self.clip_features_dir = clip_features_dir

        if self.use_precomputed_clip and self.clip_features_dir is not None:
            self.clip_features_path = Path(self.clip_features_dir)
            if not self.clip_features_path.exists():
                logger.warning(
                    f"CLIP features directory not found: {self.clip_features_dir}. "
                    "Falling back to on-the-fly computation."
                )
                self.use_precomputed_clip = False
            else:
                logger.info(f"Using precomputed CLIP features from: {self.clip_features_dir}")
        else:
            self.use_precomputed_clip = False
            logger.info("Using on-the-fly CLIP feature computation")

    def _get_clip_feature_path(self, file_idx: int, split: str) -> Path:
        """
        Get path to precomputed CLIP feature file.

        Args:
            file_idx: Episode file index
            split: 'training' or 'validation'

        Returns:
            Path to CLIP feature file
        """
        episode_name = f"episode_{file_idx:0{self.n_digits}d}"
        return self.clip_features_path / split / f"{episode_name}_clip.npz"

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load episode with optional precomputed CLIP features.

        Args:
            idx: Index of first frame
            window_size: Length of sampled episode

        Returns:
            episode: Dict containing episode data and optionally CLIP features
        """
        # Load standard episode data
        episode = super()._load_episode(idx, window_size)

        # Load precomputed CLIP features if available
        if self.use_precomputed_clip:
            start_idx = self.episode_lookup[idx]
            end_idx = start_idx + window_size

            # Determine split (training or validation)
            split = self._get_split_for_idx(idx)

            clip_static_list = []
            clip_gripper_list = []
            all_files_exist = True

            for file_idx in range(start_idx, end_idx):
                clip_path = self._get_clip_feature_path(file_idx, split)

                if clip_path.exists():
                    try:
                        clip_data = np.load(clip_path)
                        clip_static_list.append(clip_data['clip_static'])  # (512,)
                        clip_gripper_list.append(clip_data['clip_gripper'])  # (512,)
                    except Exception as e:
                        logger.warning(f"Failed to load CLIP features from {clip_path}: {e}")
                        all_files_exist = False
                        break
                else:
                    logger.debug(f"CLIP feature file not found for episode {file_idx}: {clip_path}")
                    all_files_exist = False
                    break

            # Only add CLIP features if all files were loaded successfully
            if all_files_exist and len(clip_static_list) == window_size:
                episode['clip_static'] = np.stack(clip_static_list, axis=0)  # (T, 512)
                episode['clip_gripper'] = np.stack(clip_gripper_list, axis=0)  # (T, 512)
            else:
                # For this episode only, CLIP features will be computed on-the-fly
                # Don't disable globally to allow other episodes to use precomputed features
                pass

        return episode

    def _get_split_for_idx(self, idx: int) -> str:
        """
        Determine which split (training/validation) an index belongs to.

        This is a simple heuristic - you may need to adjust based on
        how your dataset is structured.
        """
        # This assumes abs_datasets_dir contains either 'training' or 'validation'
        if 'training' in str(self.abs_datasets_dir):
            return 'training'
        elif 'validation' in str(self.abs_datasets_dir):
            return 'validation'
        else:
            # Default fallback - check which directory exists for first episode
            start_idx = self.episode_lookup[idx]
            training_path = self.clip_features_path / 'training' / f"episode_{start_idx:0{self.n_digits}d}_clip.npz"
            if training_path.exists():
                return 'training'
            return 'validation'
