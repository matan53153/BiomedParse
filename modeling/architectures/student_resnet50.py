import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from ..utils import configurable
from .build import register_model
from ..BaseModel import BaseModel # Import BaseModel for wrapping

logger = logging.getLogger(__name__)

class ResNetBackbone(nn.Module):
    """
    ResNet backbone without the final classification layer (avgpool and fc).
    Optionally returns intermediate features.
    """
    def __init__(self, backbone_name='resnet50', pretrained=True, return_intermediate_layers=False):
        super().__init__()
        if backbone_name == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = models.resnet50(weights=weights)
            self.out_channels = 2048
        elif backbone_name == 'resnet101':
            weights = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = models.resnet101(weights=weights)
            self.out_channels = 2048
        else:
            raise ValueError(f"Unsupported ResNet backbone: {backbone_name}")

        # Remove final layers
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1 # Output stride 4 (ends with 256 channels for resnet50)
        self.layer2 = backbone.layer2 # Output stride 8 (ends with 512 channels for resnet50)
        self.layer3 = backbone.layer3 # Output stride 16 (ends with 1024 channels for resnet50)
        self.layer4 = backbone.layer4 # Output stride 32 (ends with 2048 channels for resnet50)

        self.return_intermediate_layers = return_intermediate_layers
        if self.return_intermediate_layers:
             # Example: Provide features compatible with some heads, adjust keys/layers as needed
             self.feature_info = {
                 'layer1': 512, # Example channel dim, adjust based on actual layer output
                 'layer2': 1024, # Example channel dim
                 'layer3': 2048, # Example channel dim
             }

    def forward(self, x):
        # Standard ResNet forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out) # Main output feature map

        if self.return_intermediate_layers:
            # Example: Return features from multiple stages if needed by a more complex head
            # Adjust keys and selected layers as necessary
            out = {}
            out['layer1'] = layer2_out # Example assignment
            out['layer2'] = layer3_out # Example assignment
            out['layer3'] = layer4_out # Example assignment
            return out
        else:
            # Default: Return only the final feature map (layer4 output)
            return layer4_out


class StudentResNetSegmentation(nn.Module):
    """
    Student model using a ResNet backbone and a DeepLabV3 head.
    Produces per-pixel segmentation logits.
    """
    @configurable
    def __init__(self, *, backbone: nn.Module, head: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.num_classes = num_classes # Store num_classes

    @classmethod
    def from_config(cls, cfg):
        # Extract student-specific config
        student_cfg = cfg['MODEL']['STUDENT']
        backbone_name = student_cfg.get('BACKBONE_NAME', 'resnet50')
        pretrained = student_cfg.get('PRETRAINED', True)
        num_classes = student_cfg.get('NUM_CLASSES', 16) # Default to 16 if not specified

        backbone = ResNetBackbone(backbone_name=backbone_name, pretrained=pretrained)
        head = DeepLabHead(in_channels=backbone.out_channels, num_classes=num_classes)

        return {
            "backbone": backbone,
            "head": head,
            "num_classes": num_classes,
        }

    def forward(self, batched_inputs, mode='default'):
        # Assuming batched_inputs is similar structure, containing 'image'
        # Get image tensor
        if isinstance(batched_inputs, list):
            # Get device from a reliable parameter within the head
            device = next(self.head.parameters()).device
            images = [x["image"].to(device) for x in batched_inputs]
            # TODO: Add normalization based on student config/ImageNet defaults
            # images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            # TODO: Handle ImageList creation/padding if needed by backbone/head
            # images = ImageList.from_tensors(images, size_divisibility=32) # Example size divisibility
            input_tensor = torch.stack(images, dim=0) # Simple stacking for now
            input_tensor = input_tensor.float() # Convert to float
        elif isinstance(batched_inputs, torch.Tensor):
            # Get device from a reliable parameter within the head
            device = next(self.head.parameters()).device
            input_tensor = batched_inputs.to(device)
            input_tensor = input_tensor.float() # Convert to float
        else:
            raise TypeError(f"Unsupported input type: {type(batched_inputs)}")

        input_shape = input_tensor.shape[-2:]

        # Forward through backbone
        features = self.backbone(input_tensor) # Returns final feature map

        # Forward through head
        # DeepLabHead expects a dictionary {'out': features} <- This seems incorrect for torchvision's head
        # Pass the features tensor directly
        head_output = self.head(features)

        # Upsample logits to input size
        head_output = F.interpolate(head_output, size=input_shape, mode="bilinear", align_corners=False)

        # Return in a dictionary format, using a key like 'sem_seg_logits'
        # to distinguish from potential query-based 'pred_logits'
        results = {"sem_seg_logits": head_output}

        # During training, return dictionary for criterion
        if self.training:
            return {"pred_logits": head_output} # Use standard key
        else:
            # During evaluation, always return the raw logits tensor
            # Log a warning if the mode isn't one explicitly handled by SemSegEvaluator, but return tensor anyway.
            if mode not in ['sem_seg', 'default_eval']:
                 logger = logging.getLogger(__name__)
                 logger.warning(f"StudentResNet50 model received unexpected evaluation mode: '{mode}'. Returning logits tensor directly.")
            return head_output


@register_model
def get_student_resnet50_segmentation(cfg, **kwargs):
    """
    Factory function to build the StudentResNetSegmentation model.
    Wraps the core model in BaseModel for consistency.
    """
    # Ensure student config exists
    if 'STUDENT' not in cfg['MODEL']:
         # Add a default student config if missing (adjust defaults as needed)
         logger.warning("MODEL.STUDENT config not found. Adding default student config for ResNet50.")
         cfg['MODEL']['STUDENT'] = {
             'BACKBONE_NAME': 'resnet50',
             'PRETRAINED': True,
             'NUM_CLASSES': 16 # Example default
         }

    core_model = StudentResNetSegmentation(cfg)
    # Wrap in BaseModel, passing the main config `opt` (cfg in this context)
    model = BaseModel(cfg, core_model)
    return model 