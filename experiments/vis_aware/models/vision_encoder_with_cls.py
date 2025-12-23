import torch
import torch.nn as nn
from mode.models.perceptual_encoders.pretrained_resnets import FiLMLayer
from timm import create_model


class FiLMResNet50WithCls(nn.Module):
    """
    ResNet50 with FiLM conditioning that also returns cls token.
    The cls token is the feature before global pooling, which represents
    the visual semantics of the image.
    """
    def __init__(self, condition_dim, cls_embed_dim=2048):
        super(FiLMResNet50WithCls, self).__init__()
        self.resnet = create_model('resnet50', pretrained=True, num_classes=0)

        self.film1 = FiLMLayer(256, condition_dim)
        self.film2 = FiLMLayer(512, condition_dim)
        self.film3 = FiLMLayer(1024, condition_dim)
        self.film4 = FiLMLayer(2048, condition_dim)

        self.cls_embed_dim = cls_embed_dim
        self.cls_token_proj = nn.Linear(2048, cls_embed_dim)

    def forward(self, x, condition, return_cls_token=False):
        """
        Args:
            x: Input images (B, C, H, W)
            condition: Language condition (B, D) or (B, 1, D)
            return_cls_token: Whether to return cls token

        Returns:
            features: Pooled features (B, 2048)
            cls_token (optional): Cls token representing visual features (B, cls_embed_dim)
        """
        if len(condition.shape) == 3:
            condition = condition.squeeze(1)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.act1(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.film1(x, condition)

        x = self.resnet.layer2(x)
        x = self.film2(x, condition)

        x = self.resnet.layer3(x)
        x = self.film3(x, condition)

        x = self.resnet.layer4(x)
        x = self.film4(x, condition)

        if return_cls_token:
            cls_token = x.mean(dim=[2, 3])
            cls_token = self.cls_token_proj(cls_token)

        x = self.resnet.global_pool(x)
        x = x.flatten(1)

        if return_cls_token:
            return x, cls_token
        return x


class FiLMResNet34WithCls(nn.Module):
    """ResNet34 with FiLM conditioning that also returns cls token"""
    def __init__(self, condition_dim, cls_embed_dim=512):
        super(FiLMResNet34WithCls, self).__init__()
        self.resnet = create_model('resnet34', pretrained=True, num_classes=0)

        self.film1 = FiLMLayer(64, condition_dim)
        self.film2 = FiLMLayer(128, condition_dim)
        self.film3 = FiLMLayer(256, condition_dim)
        self.film4 = FiLMLayer(512, condition_dim)

        self.cls_embed_dim = cls_embed_dim
        self.cls_token_proj = nn.Linear(512, cls_embed_dim)

    def forward(self, x, condition, return_cls_token=False):
        if len(condition.shape) == 3:
            condition = condition.squeeze(1)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.act1(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.film1(x, condition)

        x = self.resnet.layer2(x)
        x = self.film2(x, condition)

        x = self.resnet.layer3(x)
        x = self.film3(x, condition)

        x = self.resnet.layer4(x)
        x = self.film4(x, condition)

        if return_cls_token:
            cls_token = x.mean(dim=[2, 3])
            cls_token = self.cls_token_proj(cls_token)

        x = self.resnet.global_pool(x)
        x = x.flatten(1)

        if return_cls_token:
            return x, cls_token
        return x
