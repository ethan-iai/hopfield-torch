import PIL
import torch
import numpy as np

from PIL.Image import Image
from skimage.filters import threshold_mean
from matplotlib import pyplot as plt

def binarize(pic):
    """
    Args:
        pic (PIL Image or numpy.ndarray): Image to be binarized.

    Returns:
        numpy.ndarray: Binarized image.
    """
    if isinstance(pic, Image):
        pic = np.asarray(pic)

    thres = threshold_mean(pic)
    pic = np.sign(pic - thres).astype(np.float32)

    return pic

def linear_unscale(tensor):
    return (tensor + 1.0) / 2.0 

class Binarize(object):

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be binarized.

        Returns:
            numpy.ndarray: Binarized image.
        """
        return binarize(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Flatten(object):

    def __call__(self, tensor):
        return torch.flatten(tensor)

    def __repr__(self):
        return self.__class__.__name__ + '()'