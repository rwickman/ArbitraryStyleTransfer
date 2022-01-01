import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn as nn
from PIL import Image, ImageFile
from pathlib import Path
import random

from conf import *


imsize = 256 if torch.cuda.is_available() else 128  # use small size if no gpu

class Random90Rot:
    def __init__(self, p=0.5):
        # Rotation probability
        self.p = p

    def __call__(self, x):
        if random.random() <= self.p:
            rot_dir = random.choice([-1, 1])
            x = torch.rot90(x, rot_dir, [1,2])
        return x


class ConditionalResize:
    """Resize a tensor if it less than a given size."""
    def __init__(self, min_size=256):
        self._min_size = min_size
        
    def __call__(self, tensor):
        if tensor.shape[1] < self._min_size or tensor.shape[2] < self._min_size:
            if tensor.shape[1] < tensor.shape[2]:
                new_width = self._min_size
                new_height =  int(tensor.shape[2] / tensor.shape[1] * new_width)
            else:
                new_height = self._min_size
                new_width =  int(tensor.shape[1] / tensor.shape[2] * new_height)
            #t = transforms.Resize([new_width, new_height])
            t = transforms.Resize([new_width, new_height])
            tensor = t(tensor)#F.interpolate(tensor, size=(1, new_width, new_height))

        return tensor 

class RandomResizeOrCrop:
    """Either crop and resize or only resize."""
    def __init__(self, imsize, p=0.5):
        """Args
            p: float giving probability of only resizing.
        """
        self.p = p
        # Resizes to given size
        self.resize = transforms.Resize((imsize, imsize))

        # Only resizes if needed
        self.cond_resize = ConditionalResize(imsize)
        self.rand_crop_resize = transforms.RandomResizedCrop(imsize)

    def __call__(self, x):
        if random.random() < self.p:
            x = self.resize(x)
        else:
            x = self.cond_resize(x)
            x = self.rand_crop_resize(x)

        return x 
            
class RandomBlur:
    def __init__(self, p=0.1, blur_size=7):
        self.p = p
        self.blur = transforms.GaussianBlur(blur_size)
    
    def __call__(self, x):
        if random.random() <= self.p:
            x = self.blur(x)
        
        return x            



def get_transform(crop=True):
    if crop:
        return transforms.Compose([
            transforms.ToTensor(),
            RandomResizeOrCrop(imsize),
            #AddGaussianNoise(0, 0.1),
            Random90Rot(0.5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 
                                                            0.4, 
                                                            0.4, 
                                                            0.1)], p=0.25),
            # transforms.RandomAutocontrast(0.1),
            # transforms.RandomAdjustSharpness(1.5, 0.25),
            RandomBlur(0.05),
            transforms.RandomGrayscale(p=0.001),
        ])  # transform it into a torch tensor
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            # Resize(512),
            transforms.Resize((imsize, imsize)),  # scale imported image
            ])  # transform it into a torch tensor

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = get_transform(False)(image).unsqueeze(0)
    return image.to(device, torch.float)

def infinite_sampler(n):
    i = 0
    perm = torch.randperm(n).tolist()

    while True:
        yield perm[i]
        i += 1
        if i >= n:
            i = 0
            perm = torch.randperm(n).tolist()


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(infinite_sampler(self.num_samples))

    def __len__(self):
        return 2 ** 31

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.paths = []
        for d in self.root:
            self.paths += list(Path(d).glob('*'))
        random.shuffle(self.paths)
        self.transform = transform

    def __getitem__(self, idx):
        while True:
            try:
                path = self.paths[idx]
                img = Image.open(str(path)).convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                return img
            except:
                idx = torch.randint(0, len(self.paths), ())
        

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'
