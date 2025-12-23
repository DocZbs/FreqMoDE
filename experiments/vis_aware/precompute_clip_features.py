#!/usr/bin/env python
"""
预计算CLIP cls_token特征并保存到磁盘。

这样训练时就不需要每次都通过CLIP encoder，大幅提速！
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).absolute().parents[2]))

import torch
import numpy as np
from tqdm import tqdm

from experiments.vis_aware.models.clip_vision_encoder import ClipVisionEncoder


def precompute_clip_features(
    data_root: str,
    output_dir: str,
    clip_model_name: str = "ViT-B/32",
    batch_size: int = 64,
    device: str = "cuda",
):
    """
    Precompute CLIP cls_token features for all images in CALVIN dataset.

    Args:
        data_root: CALVIN dataset root directory (contains training/ and validation/)
        output_dir: Output directory for precomputed features
        clip_model_name: CLIP model name
        batch_size: Batch size for processing
        device: Device to use
    """
    print("=" * 60)
    print("Precomputing CLIP cls_token features for CALVIN")
    print("=" * 60)

    # Initialize CLIP encoder
    print(f"Loading CLIP model: {clip_model_name}")
    clip_encoder = ClipVisionEncoder(
        model_name=clip_model_name,
        freeze_backbone=True,
        device=device
    )
    clip_encoder.eval()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process training and validation splits
    for split in ['training', 'validation']:
        print(f"\nProcessing {split} split...")

        data_dir = Path(data_root) / split
        if not data_dir.exists():
            print(f"Skip {split}: directory not found")
            continue

        # Find all episode files
        episode_files = sorted(data_dir.glob("episode_*.npz"))
        print(f"Found {len(episode_files)} episode files")

        # Create output directory for this split
        output_split_dir = output_path / split
        output_split_dir.mkdir(parents=True, exist_ok=True)

        # Process in batches for efficiency
        batch_rgb_static = []
        batch_rgb_gripper = []
        batch_filenames = []

        for episode_file in tqdm(episode_files, desc=f"{split}"):
            # Load episode data (single timestep per file in CALVIN)
            episode_data = np.load(episode_file, allow_pickle=True)

            # Extract RGB images (H, W, C)
            rgb_static = episode_data['rgb_static']  # (200, 200, 3)
            rgb_gripper = episode_data['rgb_gripper']  # (84, 84, 3)

            # Add to batch
            batch_rgb_static.append(rgb_static)
            batch_rgb_gripper.append(rgb_gripper)
            batch_filenames.append(episode_file.stem)

            # Process when batch is full
            if len(batch_rgb_static) >= batch_size:
                process_and_save_batch(
                    batch_rgb_static,
                    batch_rgb_gripper,
                    batch_filenames,
                    clip_encoder,
                    output_split_dir,
                    device
                )
                batch_rgb_static = []
                batch_rgb_gripper = []
                batch_filenames = []

        # Process remaining items
        if len(batch_rgb_static) > 0:
            process_and_save_batch(
                batch_rgb_static,
                batch_rgb_gripper,
                batch_filenames,
                clip_encoder,
                output_split_dir,
                device
            )

        print(f"Completed {split} split!")

    print("\n" + "=" * 60)
    print("Precomputation completed!")
    print(f"Features saved to: {output_dir}")
    print("=" * 60)


def process_and_save_batch(
    rgb_static_list,
    rgb_gripper_list,
    filenames,
    clip_encoder,
    output_dir,
    device
):
    """Process a batch of images and save individual CLIP features."""
    # Stack to batch (B, H, W, C)
    rgb_static_batch = np.stack(rgb_static_list, axis=0)
    rgb_gripper_batch = np.stack(rgb_gripper_list, axis=0)

    # Convert to tensor and preprocess
    static_tensor = torch.from_numpy(rgb_static_batch).float()
    gripper_tensor = torch.from_numpy(rgb_gripper_batch).float()

    # (B, H, W, C) -> (B, C, H, W)
    static_tensor = static_tensor.permute(0, 3, 1, 2)
    gripper_tensor = gripper_tensor.permute(0, 3, 1, 2)

    # Normalize to [0, 1]
    static_tensor = static_tensor / 255.0
    gripper_tensor = gripper_tensor / 255.0

    # Move to device
    static_tensor = static_tensor.to(device)
    gripper_tensor = gripper_tensor.to(device)

    # Extract CLIP features
    with torch.no_grad():
        static_features = clip_encoder(static_tensor)  # (B, 512)
        gripper_features = clip_encoder(gripper_tensor)  # (B, 512)

    # Convert to numpy
    static_features = static_features.cpu().numpy()
    gripper_features = gripper_features.cpu().numpy()

    # Save individual files
    for i, filename in enumerate(filenames):
        output_file = output_dir / f"{filename}_clip.npz"
        np.savez_compressed(
            output_file,
            clip_static=static_features[i],  # (512,)
            clip_gripper=gripper_features[i],  # (512,)
        )


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Precompute CLIP features for CALVIN dataset")
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.environ.get("CALVIN_DATA_ROOT", "./dataset/task_D_D"),
        help="CALVIN dataset root directory (can be set via CALVIN_DATA_ROOT env var)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for precomputed features (default: {data_root}/clip_features)"
    )
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="ViT-B/32",
        help="CLIP model name (e.g., ViT-B/32, ViT-L/14)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )

    args = parser.parse_args()

    # Set default output_dir if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_root, "clip_features")
        print(f"Output directory not specified, using default: {args.output_dir}")

    precompute_clip_features(
        data_root=args.data_root,
        output_dir=args.output_dir,
        clip_model_name=args.clip_model_name,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
