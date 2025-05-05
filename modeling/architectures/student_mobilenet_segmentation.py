import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mobilenetv3 import mobilenet_v3_large
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
import logging

# Import necessary components for registration
from .build import register_model
from ..BaseModel import BaseModel 
from ..criterion.pixel_criterion import PixelCriterion # Import PixelCriterion

logger = logging.getLogger(__name__)

class MobileNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        mobilenet = mobilenet_v3_large(pretrained=pretrained)
        self.features = mobilenet.features

    def forward(self, x):
        x = self.features(x)
        return {'out': x}


class StudentMobileNetSegmentation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # *** Correctly Read NUM_CLASSES from the config dictionary ***
        try:
            # Use the path defined in student_mobilenet.yaml
            self.num_classes = config['MODEL']['SEM_SEG_HEAD']['NUM_CLASSES'] 
            logger.info(f"Initializing DeepLabHead with num_classes = {self.num_classes} from config.")
        except KeyError as e:
            logger.error(f"Could not find NUM_CLASSES in config at MODEL.SEM_SEG_HEAD.NUM_CLASSES. Error: {e}")
            self.num_classes = 16 # Defaulting to 16, but ideally config should exist
            logger.warning(f"Defaulting num_classes to {self.num_classes}. Ensure config is correct.")

        self.pretrained = config['MODEL']['STUDENT']['PRETRAINED']
        
        self.backbone = MobileNetBackbone(pretrained=self.pretrained)
        backbone_out_channels = 960
        
        # Create the DeepLabHead with the correct number of classes
        self.head = DeepLabHead(backbone_out_channels, self.num_classes)
        
        self._init_weights(self.head)

        # Pixel-wise loss criterion - Pass the dictionary config
        self.pixel_criterion = PixelCriterion(config)
        
        # Set device using dictionary access
        self.device = torch.device(config.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.to(self.device)

    def _init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch, is_training=True):
        """ 
        Simplified forward pass.
        Args:
            batch: Input batch (dict or list of dicts).
            is_training (bool): Training mode flag.
        Returns:
            Dict: Losses if training, or segmentation output if evaluating.
        """
        # Handle Detectron2 list-of-dicts format
        if isinstance(batch, list):
             # Ensure float conversion happens here
             images = torch.stack([x["image"].to(self.device) for x in batch]).float()
             if is_training:
                 targets = torch.stack([x["sem_seg"].to(self.device, dtype=torch.long) for x in batch])
             else:
                 targets = None
        elif isinstance(batch, dict):
             # Ensure float conversion happens here too
             images = batch["image"].to(self.device).float()
             if is_training:
                 targets = batch.get("sem_seg")
                 if targets is not None:
                     targets = targets.to(self.device, dtype=torch.long)
             else:
                 targets = None
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        features = self.backbone(images) # Now images should be float32
        # *** Add Debug Print right after backbone ***
        print(f"DEBUG: Type RIGHT AFTER backbone call: {type(features)}")
        if isinstance(features, torch.Tensor):
            print(f"DEBUG: Shape RIGHT AFTER backbone call: {features.shape}")

        # *** Add Debug Print right before head ***
        print(f"DEBUG: Type RIGHT BEFORE head call: {type(features)}")
        # *** Try passing features tensor directly to DeepLabHead ***
        outputs = self.head(features) # Pass the features tensor directly

        if is_training:
            if targets is None:
                logger.warning("Targets are None in training mode. Cannot compute loss.")
                return {'loss_sem_seg_ce': torch.tensor(0.0, device=self.device, requires_grad=True)}
            # *** Implement loss computation using PixelCriterion ***
            losses = self.pixel_criterion(outputs, targets)
            return losses
        else:
            # Return raw logits for evaluation, evaluator handles argmax
            return {"sem_seg": outputs}

if __name__ == '__main__':
    config = {
        'MODEL': {
            'STUDENT': {
                'NUM_CLASSES': 16, 
                'PRETRAINED': True
            },
            'SEM_SEG_HEAD': {
                'NUM_CLASSES': 16
            },
            'IGNORE_LABEL': 255
        }
    }
    
    model = StudentMobileNetSegmentation(config)
    model.eval() 
    
    dummy_input = torch.rand(1, 3, 512, 512)
    
    dummy_target = torch.randint(0, config['MODEL']['STUDENT']['NUM_CLASSES'], (1, 512, 512), dtype=torch.long)
    dummy_batch_train = {'image': dummy_input, 'sem_seg': dummy_target}

    dummy_batch_eval = [{'image': dummy_input, 'height': 512, 'width': 512}]

    with torch.no_grad():
        output_train = model(dummy_batch_train, is_training=True)
        print("Train mode output keys:", output_train.keys())
        if 'sem_seg' in output_train:
            print("Train logits shape:", output_train['sem_seg'].shape)
    
    with torch.no_grad():
        output_eval = model(dummy_batch_eval, is_training=False)
        print("\nEvaluate mode output keys:", output_eval.keys())
        if 'sem_seg' in output_eval:
            print("Evaluate logits shape:", output_eval['sem_seg'].shape)

# Factory function for building and registering the model
@register_model
def student_mobilenet_segmentation(cfg, **kwargs):
    """
    Factory function to build the StudentMobileNetSegmentation model.
    Wraps the core model in BaseModel for consistency.
    """
    # Ensure the necessary config section exists
    if 'STUDENT' not in cfg['MODEL']:
         logger.warning("MODEL.STUDENT config not found. Using default config for MobileNet.")
         # Define a default config if missing (adjust defaults as needed)
         cfg['MODEL']['STUDENT'] = {
             'NUM_CLASSES': 16, 
             'PRETRAINED': True
         }
         # Ensure IGNORE_LABEL is also present if needed by the model/loss
         if 'IGNORE_LABEL' not in cfg['MODEL']:
             cfg['MODEL']['IGNORE_LABEL'] = 255

    # Create the core model instance using the config
    core_model = StudentMobileNetSegmentation(cfg)
    
    # Wrap the core model in BaseModel (if required by your framework)
    model = BaseModel(cfg, core_model)
    return model