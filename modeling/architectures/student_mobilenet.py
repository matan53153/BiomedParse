# modeling/architectures/student_mobilenet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
# FCNHead could also be an option for lighter heads
# from torchvision.models.segmentation.fcn import FCNHead

from ..utils import configurable
from .build import register_model
from ..BaseModel import BaseModel

logger = logging.getLogger(__name__)

class MobileNetBackbone(nn.Module):
    """
    MobileNetV3-Large backbone without the final classification layer.
    """
    def __init__(self, backbone_name='mobilenet_v3_large', pretrained=True):
        super().__init__()
        if backbone_name == 'mobilenet_v3_large':
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.mobilenet_v3_large(weights=weights)
            # The output features before the classifier are in backbone.features
            self.features = backbone.features
            # The last layer of features has 960 channels
            self.out_channels = 960
        else:
            raise ValueError(f"Unsupported MobileNet backbone: {backbone_name}")
        # Note: MobileNetV3 doesn't have distinct named layers like ResNet (layer1, layer2...).
        # It returns the final feature map from self.features.

    def forward(self, x):
        # Forward through the feature extractor part
        x = self.features(x)
        # Return features in a dictionary with the key 'out'
        return {'out': x}


class StudentMobileNetSegmentation(nn.Module):
    """
    Student model using a MobileNetV3 backbone and a DeepLabV3 head.
    Produces per-pixel segmentation logits.
    """
    @configurable
    def __init__(self, *, backbone: nn.Module, head: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg):
        student_cfg = cfg['MODEL']['STUDENT']
        backbone_name = student_cfg.get('BACKBONE_NAME', 'mobilenet_v3_large')
        pretrained = student_cfg.get('PRETRAINED', True)
        num_classes = student_cfg.get('NUM_CLASSES', 16)

        backbone = MobileNetBackbone(backbone_name=backbone_name, pretrained=pretrained)
        # Use DeepLabHead, ensure in_channels matches backbone's output
        head = DeepLabHead(in_channels=backbone.out_channels, num_classes=num_classes)

        return {
            "backbone": backbone,
            "head": head,
            "num_classes": num_classes,
        }

    def forward(self, input_tensor):
        # The input_tensor is already on the correct device and is float
        input_shape = input_tensor.shape[-2:]

        # Forward through MobileNet backbone features
        features = self.backbone(input_tensor) # Backbone might return feature dict or tensor

        # Check if backbone output is a dict and has 'out', otherwise assume it's the tensor itself
        if isinstance(features, dict):
            if 'out' not in features:
                raise KeyError(f"Backbone output dict missing expected key 'out'. Keys found: {list(features.keys())}")
            if not isinstance(features['out'], torch.Tensor):
                 raise TypeError(f"Expected backbone feature 'out' to be a Tensor, but got {type(features['out'])}")
            head_input = features['out']
        elif isinstance(features, torch.Tensor):
            head_input = features
        else:
            # Add handling for other unexpected types if necessary
            raise TypeError(f"Unexpected backbone output type: {type(features)}")

        # Forward through DeepLabV3 head
        # Pass the appropriate tensor to the head
        logits = self.head(head_input) # Use the extracted or direct tensor

        # Upsample logits to original input size
        logits = F.interpolate(
            logits,
            size=input_shape,
            mode="bilinear",
            align_corners=False
        )

        # Ensure output is a dict with 'sem_seg_logits' key
        # The DeepLabHead likely returns a dict already, but double-check
        # Update: DeepLabHead returns a dict like {'out': logits_tensor}
        # We need {'sem_seg_logits': logits_tensor}
        # Update 2: Let's return just the tensor for now, pipeline handles wrapping
        # Correction: Head returns the tensor directly now after previous fix.
        # Let's wrap it in the expected dict format here.
        return {"sem_seg_logits": logits}


@register_model
def student_mobilenet_segmentation(cfg, **kwargs):
    """
    Factory function to build the StudentMobileNetSegmentation model.
    Wraps the core model in BaseModel for consistency.
    """
    if 'STUDENT' not in cfg['MODEL']:
         logger.warning("MODEL.STUDENT config not found. Adding default student config for MobileNet.")
         cfg['MODEL']['STUDENT'] = {
             'BACKBONE_NAME': 'mobilenet_v3_large',
             'PRETRAINED': True,
             'NUM_CLASSES': 16
         }

    # Create the core model using the from_config method
    core_model = StudentMobileNetSegmentation.from_config(cfg)
    model = BaseModel(cfg, core_model)
    return model 