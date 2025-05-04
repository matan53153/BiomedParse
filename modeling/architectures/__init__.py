from .xdecoder_model import *
from .seem_model_v0 import *
from .seem_model_v1 import get_seem_model
from .seem_model_demo import *
from .student_resnet50 import get_student_resnet50_segmentation
from .student_vit import get_student_vit_segmentation
from .student_mobilenet import get_student_mobilenet_segmentation

def build_model(args):
    """
    Build the whole model architecture, defined by `MODEL.META_ARCHITECTURE`.
    Note that it does not load any weights from `MODEL.WEIGHTS`.
    """
    meta_arch = args['MODEL']['NAME']
    if meta_arch == 'seem_model_v1':
        model = get_seem_model(args)
    elif meta_arch == 'student_resnet50_segmentation':
        model = get_student_resnet50_segmentation(args)
    elif meta_arch == 'student_vit_segmentation':
        model = get_student_vit_segmentation(args)
    elif meta_arch == 'student_mobilenet_segmentation':
        model = get_student_mobilenet_segmentation(args)
    else:
        raise NotImplementedError
    return model