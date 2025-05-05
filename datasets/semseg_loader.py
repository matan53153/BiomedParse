import numpy as np
from PIL import Image
import scipy.io

def load_semseg(filename, loader_type):
    if loader_type == 'PIL':
        # Open image and convert to grayscale ('L') before converting to numpy
        semseg = np.array(Image.open(filename).convert('L'), dtype=int)
    elif loader_type == 'MAT':
        semseg = scipy.io.loadmat(filename)['LabelMap']
    return semseg