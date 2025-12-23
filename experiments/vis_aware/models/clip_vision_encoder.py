import torch
import torch.nn as nn
from mode.models.networks.clip import build_model, load_clip


class ClipVisionEncoder(nn.Module):
    """
    CLIP Vision Encoder that extracts cls_token.

    This is the PROPER way to get cls_token from a vision foundation model.
    CLIP ViT has a learnable class_embedding token that aggregates visual information.
    """
    def __init__(self, model_name: str = "ViT-B/32", freeze_backbone: bool = True, device=None):
        super(ClipVisionEncoder, self).__init__()

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading CLIP vision model with backbone: {model_name}")
        self._load_clip(model_name)

        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # Set output dimension based on model
        if "ViT-B" in model_name:
            self.output_dim = 512
        elif "ViT-L" in model_name:
            self.output_dim = 768
        elif "RN50" in model_name:
            self.output_dim = 1024
        elif "RN101" in model_name:
            self.output_dim = 512
        else:
            self.output_dim = 512

    def _load_clip(self, model_name: str) -> None:
        model, self.preprocess = load_clip(model_name, device=self.device)
        self.clip_model = build_model(model.state_dict()).to(self.device)
        # Force to float32 to avoid dtype issues
        self.clip_model = self.clip_model.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CLIP vision cls_token.

        Args:
            x: Images (B, C, H, W)

        Returns:
            cls_token: CLIP vision features (B, output_dim)
                      This is the TRUE cls_token from vision transformer!
        """
        with torch.no_grad():
            # Resize to CLIP's expected input size (224x224)
            if x.shape[-2:] != (224, 224):
                x = torch.nn.functional.interpolate(
                    x, size=(224, 224), mode='bilinear', align_corners=False
                )

            # Convert to CLIP's dtype
            x = x.type(self.clip_model.dtype)
            # CLIP's encode_image returns the cls_token directly
            cls_token = self.clip_model.encode_image(x)
            # Convert back to float32
            cls_token = cls_token.float()

        return cls_token


class ClipVisionEncoderWithResNet(nn.Module):
    """
    Hybrid encoder: Use ResNet for state representation + CLIP for cls_token.

    This combines:
    - ResNet (FiLM conditioned): For task-specific visual features
    - CLIP ViT: For semantic cls_token from vision foundation model
    """
    def __init__(
        self,
        resnet_encoder,  # FiLMResNet50Policy or similar
        clip_model_name: str = "ViT-B/32",
        freeze_clip: bool = True,
        device=None
    ):
        super(ClipVisionEncoderWithResNet, self).__init__()

        self.resnet_encoder = resnet_encoder
        self.clip_encoder = ClipVisionEncoder(
            model_name=clip_model_name,
            freeze_backbone=freeze_clip,
            device=device
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor, return_cls_token: bool = False):
        """
        Args:
            x: Images (B, C, H, W)
            condition: Language condition (B, D)
            return_cls_token: Whether to return CLIP cls_token

        Returns:
            resnet_features: Task-specific features from ResNet (B, feature_dim)
            clip_cls_token (optional): Semantic cls_token from CLIP (B, clip_dim)
        """
        # Get task-specific features from ResNet
        resnet_features = self.resnet_encoder(x, condition)

        if return_cls_token:
            # Get semantic cls_token from CLIP
            clip_cls_token = self.clip_encoder(x)
            return resnet_features, clip_cls_token

        return resnet_features
