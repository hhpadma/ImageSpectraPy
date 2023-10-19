import cv2
from cv2 import fastNlMeansDenoising
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_gradient_magnitude


def is_grayscale(img: np.ndarray) -> bool:
    """
    Check if an image is grayscale.

    Parameters:
    - img: np.ndarray, the image to check.

    Returns:
    - bool: True if the image is grayscale, False otherwise.
    """
    return len(img.shape) == 2 or img.shape[2] == 1


def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale.

    Parameters:
    - img: np.ndarray, the image to convert.

    Returns:
    - np.ndarray: the converted grayscale image.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_to_power_of_two(img: np.ndarray, size: int = 1024) -> np.ndarray:
    """
    Resize an image to the next power of two.

    Parameters:
    - img: np.ndarray, the image to resize.
    - size: int, the desired size of the resized image.

    Returns:
    - np.ndarray: the resized image.
    """
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def make_square(img: np.ndarray) -> np.ndarray:
    """
    Make an image square by cropping the longer side.

    Parameters:
    - img: np.ndarray, the image to make square.

    Returns:
    - np.ndarray: the square image.
    """
    h, w = img.shape[:2]
    if h == w:
        return img
    elif h > w:
        diff = (h - w) // 2
        return img[diff:h-diff, :]
    else:
        diff = (w - h) // 2
        return img[:, diff:w-diff]


def prepare_image(img: np.ndarray, size: int = 1024) -> np.ndarray:
    """
    Prepare an image for analysis by ensuring it is grayscale, square, and resized to a power of two.

    Parameters:
    - img: np.ndarray, the image to prepare.
    - size: int, the desired size of the prepared image.

    Returns:
    - np.ndarray: the prepared image.
    """
    if not is_grayscale(img):
        img = convert_to_grayscale(img)
    img = make_square(img)
    img = resize_to_power_of_two(img, size)
    return img

def remove_noise(img, h=10, templateWindowSize=7, searchWindowSize=21):
    return fastNlMeansDenoising(img, h=h, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)

