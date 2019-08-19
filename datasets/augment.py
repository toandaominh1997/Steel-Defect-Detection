from torchvision import transforms
import numpy as np 
import cv2
from PIL import Image
from albumentations import (Normalize, Compose)

class ResizeImage:
    def __init__(self, height):
        self.height = height 
    def __call__(self, image):
        image = np.array(image)
        h, w = image.shape[:2]
        new_w = int(self.height/h*w)
        image = cv2.resize(image, (new_w, self.height), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(image)
def train_transforms():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform


def test_transforms():
    transform = transforms.Compose([
        transforms.ToTensor()

    ])
    return transform