# In /scratch/gpfs/km4074/BiomedParse/modeling/vision/backbone/resnet.py

import torch
import torch.nn as nn
import torchvision.models as models
from detectron2.modeling import ShapeSpec # Assuming ShapeSpec is needed as in base class

from .build import register_backbone
from .backbone import Backbone # Import the base Backbone class

# TODO: Potentially import normalization layers if needed (e.g., FrozenBatchNorm2d from detectron2)


class ResNetBackbone(Backbone):
    """
    A ResNet backbone implementation wrapping torchvision's ResNet.
    Outputs features from specified stages.
    """
    def __init__(self, resnet_model, out_features):
        super().__init__()
        self.resnet_model = resnet_model
        self._out_features = out_features

        # TODO: Define self._out_feature_channels and self._out_feature_strides
        # These depend on the ResNet architecture and the chosen out_features
        # Example for ResNet18 and ["res2", "res3", "res4", "res5"]:
        self._out_feature_channels = {"res2": 64, "res3": 128, "res4": 256, "res5": 512}
        self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}

        # Remove the final classification layer (fc) if it exists
        if hasattr(self.resnet_model, 'fc'):
            del self.resnet_model.fc


    def forward(self, x):
        """
        Extract features from the specified ResNet stages.
        """
        features = {}
        x = self.resnet_model.conv1(x)
        x = self.resnet_model.bn1(x)
        x = self.resnet_model.relu(x)
        x = self.resnet_model.maxpool(x)

        x = self.resnet_model.layer1(x) # res2 (stride 4)
        if "res2" in self._out_features:
            features["res2"] = x

        x = self.resnet_model.layer2(x) # res3 (stride 8)
        if "res3" in self._out_features:
            features["res3"] = x

        x = self.resnet_model.layer3(x) # res4 (stride 16)
        if "res4" in self._out_features:
            features["res4"] = x

        x = self.resnet_model.layer4(x) # res5 (stride 32)
        if "res5" in self._out_features:
            features["res5"] = x

        return features

    # output_shape is inherited from the base Backbone class if channels/strides are set

@register_backbone
def build_resnet_backbone(config):
    """
    Builds a ResNet backbone based on the provided configuration.
    """
    resnet_cfg = config['MODEL']['BACKBONE']['RESNETS']
    depth = resnet_cfg['DEPTH']
    out_features = resnet_cfg['OUT_FEATURES']
    # norm = resnet_cfg['NORM'] # TODO: Handle normalization layer replacement if needed
    # pretrained = config.get('WEIGHT', True) # Check if pretrained weights should be loaded

    # Select the appropriate ResNet model from torchvision
    if depth == 18:
        resnet_model = models.resnet18(pretrained=True) # Load pretrained ResNet-18
        # TODO: Handle normalization layer replacement based on 'norm' config
    elif depth == 50:
        resnet_model = models.resnet50(pretrained=True) # Example for ResNet-50
    # Add other depths (34, 101, 152) if needed
    else:
        raise ValueError(f"Unsupported ResNet depth: {depth}")

    # Wrap the torchvision model in our Backbone compatible class
    backbone = ResNetBackbone(resnet_model, out_features)
    return backbone
