
import torch
from torchvision import transforms as T
from utils.misc import is_list_or_tuple
from PIL import ImageFilter
import random
import numpy as np
import cv2
def get_normalize(norm_mean = (0.485, 0.456, 0.406), norm_std = (0.228, 0.224, 0.225)):
    return T.Compose([T.ToTensor(), T.Normalize(mean=norm_mean, std=norm_std)])
from data.FCJitter import FCJitter
class PILRandomGaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = torch.rand(1) <= self.prob
        if not do_it:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max)))


class CVGaussianBlur:
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        prob = np.random.random_sample()
        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return sample

class ApplyTransform(object):

    def __init__(self, transformations, views=1):

        self.transformations = transformations

        if is_list_or_tuple(transformations):
            self.views = len(transformations)
        else:
            self.views = views

    def __call__(self, x):
        if is_list_or_tuple(self.transformations):
            return [transformation(x) for transformation in self.transformations]
        else:
            if self.views == 1:
                return self.transformations(x)
            else:
                return [self.transformations(x) for i in range(self.views)]


class CCTransforms:
    def __init__(self, crop_size=224, s=0.5, blur=False, validation=False):
        self.validation = validation
        self.s = s
        self.blur = blur
        if validation:
            self.transforms = T.Compose(
                [T.Resize((crop_size, crop_size)), T.ToTensor()])
        else:
            augmentations = [T.RandomResizedCrop(size=crop_size),
                             T.RandomHorizontalFlip(),
                             T.RandomApply([T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)],p=0.8),
                             #FCJitter(0.1 * s, 0.1 * s, 0.1* s, 0.025 * s),
                             FCJitter(0.01 * s, 0, 0, 0),
                             T.RandomGrayscale(p=0.2)]
            if blur:
                if crop_size == 224:
                    augmentations.append(CVGaussianBlur(kernel_size=23))
                else:
                    augmentations.append(PILRandomGaussianBlur())
            augmentations += [T.ToTensor()]
            self.transforms = T.Compose(augmentations)

    def __call__(self, x):
        if self.validation:
            return self.transforms(x)
        else:
            return torch.stack([self.transforms(x), self.transforms(x)])