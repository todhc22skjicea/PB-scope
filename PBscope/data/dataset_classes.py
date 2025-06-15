
import torch
from torch.utils.data import Dataset
import collections.abc
import numpy as np
from PIL import Image
from torchvision import transforms as T

class PBDataset(Dataset):
    def __init__(self, data, annotations=None, transformations=None, data_reader=None):

        self.transformations = transformations or T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.data_reader = data_reader or PBReader(data, annotations)

    def __len__(self):
        return self.data_reader.__len__()

    def __getitem__(self, index):
        image, annotation = self.data_reader.__getitem__(index)
        if self.transformations is not None:
            image = self.transformations(image)
        return index, image, annotation

    def _get_data_type(self, data):
        data_type = None
        if isinstance(data, torch.Tensor) or isinstance(data, np.ndarray):
            data_type = "tensor"
        elif isinstance(data, collections.abc.Sequence):
            if isinstance(data[0], str):
                data_type = "files"
        assert data_type is not None

        return data_type

class PBReader:
    def __init__(self, data, annotations=None):
        self.data = data
        self.annotations=annotations
        self.tensor2PIL = T.ToPILImage()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        if isinstance(image, str):
            image = load_image(image)
        elif isinstance(image, torch.Tensor):
            image = self.tensor2PIL(image)
        else:
            image = Image.fromarray(image)

        if self.annotations is not None:
            annotation = int(self.annotations[index])
        else:
            annotation = None

        return image, annotation

def load_image(path):
    """Loads an image from given path"""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


