import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn as nn
from PIL import Image, ImageFile
from pathlib import Path
import random
from PIL import ImageCms
from skimage import io, color
from conf import *
from model_util import rgb2lab


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
    def __init__(self, imsize, p=0.90):
        """Args
            p: float giving probability of only resizing.
        """
        self.p = p
        # Resizes to given size
        self.resize = transforms.Resize((imsize[0], imsize[1]))

        # Only resizes if needed
        self.cond_resize = ConditionalResize(min(imsize[0], imsize[1]))
        self.rand_crop_resize = transforms.RandomResizedCrop((imsize[0], imsize[1]))

    def __call__(self, x):
        if random.random() < self.p:
            x = self.resize(x)
        else:
            x = self.cond_resize(x)
            x = self.rand_crop_resize(x)

        return x 
            
class RandomBlur:
    def __init__(self, p=0.1, blur_sizes=[3,5,7,9]):
        self.p = p
        self.blur_sizes = blur_sizes
    
    def __call__(self, x):
        if random.random() <= self.p:
            blur_fac = random.choice(self.blur_sizes)
            self.blur = transforms.GaussianBlur(blur_fac)
            x = self.blur(x)


        return x          


class ImageTransform:
    def __init__(self, batch_size, use_transform=True):
        self.batch_size = batch_size
        self.transform = get_transform(use_transform)
        self.num_in_batch = 0

    def reset(self):
        # Create transform again
        self.num_in_batch = 0
        rand_h = random.choice(img_sizes)
        
        rand_w = random.choice(img_sizes)

        
        self.transform = get_transform(True, (rand_h, rand_w))

    def __call__(self, img):
        if self.num_in_batch >= self.batch_size * 2:
            self.reset()
        # print("self.num_in_batch", self.num_in_batch)
        self.num_in_batch += 1

        return self.transform(img)




def get_transform(crop=True, imsize=(256, 256)):

    if crop:
        return transforms.Compose([
            transforms.ToTensor(),
            #AddGaussianNoise(0, 0.1),
            Random90Rot(0.25),
            transforms.RandomHorizontalFlip(0.25),
            transforms.RandomVerticalFlip(0.25),

            transforms.RandomApply([transforms.ColorJitter(0.4, 
                                                            0.10, 
                                                            0.4, 
                                                            0.10)], p=0.25),
            RandomResizeOrCrop(imsize),
            # transforms.RandomAutocontrast(0.1),
            # transforms.RandomAdjustSharpness(1.5, 0.25),
            RandomBlur(0.05),
            transforms.RandomGrayscale(p=0.001),
        ])  # transform it into a torch tensor
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            # Resize(512),
            transforms.Resize((imsize, 256)),  # scale imported image
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
    def __init__(self, content_root, style_root, transform=None):
        super().__init__()
        self.content_paths = self._get_paths(content_root)
        self.style_paths =  self._get_paths(style_root)
        self.transform = transform
    
    def _get_paths(self, root):
        paths = []
        for d in root:
            paths += list(Path(d).glob('*'))
        
        random.shuffle(paths)
        return paths

    def _get_item(self, paths):
        idx = torch.randint(0, len(paths), ()) 
        while True:
            try:
                path = paths[idx]
                img = Image.open(str(path)).convert("RGB")
                # # img = self.rgb2lab_trans(img)
                # img = ((color.rgb2lab(img) / 100) + 1) / 2
                if self.transform is not None:
                    img = self.transform(img).float()

                # img = rgb2lab(img.unsqueeze(0)).squeeze(0)
                return img
            except Exception as e:
                print("e:", e)
                idx = torch.randint(0, len(paths), ())  
      
    def __getitem__(self, idx):
        content_img = self._get_item(self.content_paths)
        style_img = self._get_item(self.style_paths)
        return content_img, style_img

    def __len__(self):
        return len(self.content_paths) + len(self.style_paths)

    def name(self):
        return 'FlatFolderDataset'

class FlatFolderDatasetAE(data.Dataset):
    def __init__(self, content_root, transform=None):
        super().__init__()
        self.content_paths = self._get_paths(content_root)
        self.transform = transform
    
    def _get_paths(self, root):
        paths = []
        for d in root:
            paths += list(Path(d).glob('*'))
        
        random.shuffle(paths)
        return paths

    def _get_item(self, paths):
        idx = torch.randint(0, len(paths), ()) 
        while True:
            try:
                path = paths[idx]
                img = Image.open(str(path)).convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                return img
            except:
                idx = torch.randint(0, len(paths), ())  
      
    def __getitem__(self, idx):
        content_img = self._get_item(self.content_paths)
        return content_img

    def __len__(self):
        return len(self.content_paths)

    def name(self):
        return 'FlatFolderDataset'