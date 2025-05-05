import logging
import torch

logger = logging.getLogger(__name__)

def hook_opt(opt):
    # Check if we're using a student model
    if 'MODEL' in opt and 'NAME' in opt['MODEL'] and 'student_mobilenet_segmentation' in opt['MODEL']['NAME']:
        # For student model, we don't need to modify ATTENTION_ARCH
        logger.info("Student model detected, skipping ATTENTION_ARCH modifications")
        return opt
        
    # Skip ATTENTION_ARCH modifications if it's not present
    if 'ATTENTION_ARCH' not in opt or opt['ATTENTION_ARCH'] is None:
        logger.warning("ATTENTION_ARCH not found in config, skipping related modifications")
        return opt
        
    try:
        grounding_flag = opt['REF']['INPUT']['SPATIAL']
    except:
        grounding_flag = False

    if grounding_flag and 'ATTENTION_ARCH' in opt and opt['ATTENTION_ARCH'] is not None:
        if 'SELF_ATTENTION' in opt['ATTENTION_ARCH'] and opt['ATTENTION_ARCH']['SELF_ATTENTION'] is not None:
            if 'queries' in opt['ATTENTION_ARCH']['SELF_ATTENTION']:
                opt['ATTENTION_ARCH']['SELF_ATTENTION']['queries']['grounding'] = ['queries_grounding', 'tokens_grounding', 'tokens_spatial']

    try:
        spatial_flag = opt['STROKE_SAMPLER']['EVAL']['GROUNDING']
    except:
        spatial_flag = False

    if spatial_flag and 'ATTENTION_ARCH' in opt and opt['ATTENTION_ARCH'] is not None:
        if 'SELF_ATTENTION' in opt['ATTENTION_ARCH'] and opt['ATTENTION_ARCH']['SELF_ATTENTION'] is not None:
            if 'queries' in opt['ATTENTION_ARCH']['SELF_ATTENTION']:
                opt['ATTENTION_ARCH']['SELF_ATTENTION']['queries']['spatial'] = ['queries_spatial', 'tokens_spatial', 'memories_spatial', 'tokens_grounding']

    return opt

# HACK for evalution 
def hook_metadata(metadata, name):
    return metadata

# HACK for evalution 
def hook_switcher(model, name):
    # Check if this is a student model
    is_student_model = False
    if hasattr(model, 'opt') and 'MODEL' in model.opt and 'NAME' in model.opt['MODEL']:
        is_student_model = 'student_mobilenet_segmentation' in model.opt['MODEL']['NAME']
    
    # For student model, we only do semantic segmentation
    if is_student_model:
        logger.info(f"Student model detected for dataset {name}, setting semantic_on=True, instance_on=False, panoptic_on=False")
        if hasattr(model.model, 'semantic_on'):
            model.model.semantic_on = True
        if hasattr(model.model, 'instance_on'):
            model.model.instance_on = False
        if hasattr(model.model, 'panoptic_on'):
            model.model.panoptic_on = False
        return
    
    # Standard model handling
    mappings = {}
    if name in ['cityscapes_fine_sem_seg_val', 'scannet_21_val_seg', 'scannet_38_val_seg', 'scannet_41_val_seg', 'sunrgbd_37_val_seg', 'context_59_val_seg', 'context_459_val_seg', 'voc_2012_val_seg', 'bdd10k_val_sem_seg', 'ade20k_full_sem_seg_val']:
        mappings = {'SEMANTIC_ON': True, 'INSTANCE_ON': False, 'PANOPTIC_ON': False}
    elif name in ['cityscapes_fine_instance_seg_val'] or 'seginw' in name:
        mappings = {'SEMANTIC_ON': False, 'INSTANCE_ON': True, 'PANOPTIC_ON': False}
    elif name in ['cityscapes_fine_panoptic_val', 'scannet_21_panoptic_val', 'bdd10k_40_panoptic_val']:
        mappings = {'SEMANTIC_ON': True, 'INSTANCE_ON': False, 'PANOPTIC_ON': True}
    elif name in ['coco_2017_val_panoptic_with_sem_seg', 'ade20k_panoptic_val', 'coco_2017_test-dev']:
        mappings = {'SEMANTIC_ON': True, 'INSTANCE_ON': True, 'PANOPTIC_ON': True}
    # Handle biomedical datasets - for all biomed datasets, we do semantic segmentation only
    elif 'biomed' in name:
        logger.info(f"Biomedical dataset detected: {name}, setting semantic_on=True, instance_on=False, panoptic_on=False")
        mappings = {'SEMANTIC_ON': True, 'INSTANCE_ON': False, 'PANOPTIC_ON': False}
    else:
        if name not in ["med_sam_train", "med_sam_test", "vlp_val", "vlp_captioning_val", "vlp_val2017", "vlp_captioning_val2017", "imagenet_val", "refcocog_val_google", "phrasecut_val", "phrasecut_test", "refcocop_val_unc", "refcoco_val_unc", "refcocog_val_umd", "pascalvoc_val_Point", "grounding_coco_entity_val", "vlp_coco_entity_val"]:
            logger.warning(f"Unknown dataset: {name}, defaulting to semantic segmentation only")
            mappings = {'SEMANTIC_ON': True, 'INSTANCE_ON': False, 'PANOPTIC_ON': False}

    # Apply the mappings to the model
    for key, value in mappings.items():
        if key == 'SEMANTIC_ON' and hasattr(model.model, 'semantic_on'):
            model.model.semantic_on = value
        if key == 'INSTANCE_ON' and hasattr(model.model, 'instance_on'):
            model.model.instance_on = value
        if key == 'PANOPTIC_ON' and hasattr(model.model, 'panoptic_on'):
            model.model.panoptic_on = value