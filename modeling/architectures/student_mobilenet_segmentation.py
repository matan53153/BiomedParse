import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mobilenetv3 import mobilenet_v3_large
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
import logging

# Import necessary components for registration
from .build import register_model
from ..BaseModel import BaseModel 

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
        self.num_classes = config['MODEL']['STUDENT']['NUM_CLASSES']
        self.pretrained = config['MODEL']['STUDENT']['PRETRAINED']
        
        self.backbone = MobileNetBackbone(pretrained=self.pretrained)
        backbone_out_channels = 960
        
        self.head = DeepLabHead(backbone_out_channels, self.num_classes)
        self._init_weights(self.head)

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

    def forward(self, batch, **kwargs):
        mode = kwargs.get('mode', 'train') 

        if mode == 'train':
            # Handle both dict and list batch formats during training
            if isinstance(batch, list):
                # Stack images and targets from the list of dicts
                images = torch.stack([item['image'] for item in batch])
                # Check if 'sem_seg' exists in items before stacking
                if all('sem_seg' in item for item in batch):
                    targets = torch.stack([item['sem_seg'] for item in batch])
                else:
                    targets = None 
                    logger.warning("Some items in the training batch list are missing 'sem_seg'.")
            elif isinstance(batch, dict):
                images = batch['image']
                targets = batch.get('sem_seg', None)
            else:
                logger.error(f"Unexpected batch type during training: {type(batch)}")
                # Depending on desired behavior, raise error or return dummy loss
                raise TypeError(f"Unexpected batch type during training: {type(batch)}")

            # Ensure input tensor is float32
            if images.dtype != torch.float32:
                images = images.to(torch.float32)
            
            input_shape = images.shape[-2:]
            
            features = self.backbone(images) 
            head_input = features['out'] 
            logger.debug(f"Shape input to DeepLabHead: {head_input.shape}") # Log shape
            x = self.head(head_input) 
            
            logits = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

            # Always return logits during training; loss calculation is handled by the criterion
            return {'sem_seg_logits': logits}

        elif mode in ['evaluate', 'grounding_refcoco']: 
            if isinstance(batch, list):
                images = torch.stack([x['image'] for x in batch])
            elif isinstance(batch, dict):
                images = batch['image']
            else:
                logger.error(f"Unexpected batch type during evaluation: {type(batch)}")
                return {"error": f"Unexpected batch type: {type(batch)}"} 

            if images.dtype != torch.float32:
                images = images.to(torch.float32)

            input_shape = images.shape[-2:]

            features = self.backbone(images)
            head_input = features['out']
            logger.debug(f"Shape input to DeepLabHead: {head_input.shape}") # Log shape
            x = self.head(head_input)
            
            logits = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            
            result = {'sem_seg_logits': logits}
            return result
            
        else:
            logger.error(f"Unsupported mode: {mode}")
            return {"error": f"Unsupported mode: {mode}"}

if __name__ == '__main__':
    config = {
        'MODEL': {
            'STUDENT': {
                'NUM_CLASSES': 16, 
                'PRETRAINED': True
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
        output_train = model(dummy_batch_train, mode='train')
        print("Train mode output keys:", output_train.keys())
        if 'sem_seg_logits' in output_train:
            print("Train logits shape:", output_train['sem_seg_logits'].shape)
    
    with torch.no_grad():
        output_eval = model(dummy_batch_eval, mode='evaluate')
        print("\nEvaluate mode output keys:", output_eval.keys())
        if 'sem_seg_logits' in output_eval:
            print("Evaluate logits shape:", output_eval['sem_seg_logits'].shape)
        elif 'error' in output_eval:
            print("Evaluate mode error:", output_eval['error'])

    with torch.no_grad():
        output_grounding = model(dummy_batch_eval, mode='grounding_refcoco')
        print("\nGrounding_refcoco mode output keys:", output_grounding.keys())
        if 'sem_seg_logits' in output_grounding:
            print("Grounding logits shape:", output_grounding['sem_seg_logits'].shape)
        elif 'error' in output_grounding:
            print("Grounding mode error:", output_grounding['error'])

    with torch.no_grad():
        output_unsupported = model(dummy_batch_eval, mode='invalid_mode')
        print("\nUnsupported mode output:", output_unsupported)

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