from pathlib import Path

from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.data as data
import torch.nn as nn
import random
from models import *

unloader = transforms.ToPILImage()  # reconvert into PIL image

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated


# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

def get_transform(crop=True):
    if crop:
        return transforms.Compose([
            transforms.Resize((imsize, imsize)),  # scale imported image
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomAutocontrast(),
            transforms.RandomCrop(256),
            transforms.ToTensor()])  # transform it into a torch tensor
    else:
        return transforms.Compose([
            transforms.Resize((imsize, imsize)),  # scale imported image
            transforms.ToTensor()])  # transform it into a torch tensor

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
    def __init__(self, root, transform):
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
                img = self.transform(img)
                return img
            except:
                idx = torch.randint(0, len(self.paths), ())
        

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

# Load images
content_file = "temp_dataset/content_test/dog.jpg"
#content_file = "temp_dataset/content_test/n04254680_15943.jpg"
style_file = "temp_dataset/style_test/fractal_style_6.jpg"

style_img = image_loader(style_file)
content_img = image_loader(content_file)

#content_dir = ["temp_dataset/content/", "/media/data/code/stylegan/data/new_dataset/creeper_dataset/", "/media/data/datasets/unlabeled2017/"] 
#style_dir = ["/media/data/code/stylegan/data/new_dataset/creeper_dataset/", "/media/data/datasets/wikiart_smaller", "temp_dataset/style/"]
content_dir = ["temp_dataset/content/", "/media/data/code/stylegan/data/portraits_smaller_4/", "/media/data/datasets/unlabeled2017/"] 
style_dir = ["/media/data/code/stylegan/data/new_dataset/creeper_dataset/", "/media/data/datasets/wikiart_smaller", "temp_dataset/style/"] 
# content_dir = ["temp_dataset/content/"]
# style_dir = ["temp_dataset/style/"]
# content_dir = ["temp_dataset/content_test/"]
# style_dir = ["temp_dataset/style_test/"]


batch_size = 16
num_workers = 8

content_dataset = FlatFolderDataset(content_dir, get_transform(True))
style_dataset = FlatFolderDataset(style_dir, get_transform(True))

content_iter = iter(data.DataLoader(
    content_dataset,
    batch_size=batch_size,
    sampler = InfiniteSamplerWrapper(content_dataset),
    num_workers=num_workers))
style_iter = iter(data.DataLoader(
    style_dataset, 
    batch_size=batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=num_workers))

fig, axs = plt.subplots(1, 2, figsize=(15,10))
plt.ion()

# Load models
ast = AST().to(device)
ast_optim = optim.Adam(ast.parameters(), lr=8e-5)
dis = Discriminator().to(device)
dis_optim = optim.Adam(dis.parameters(), lr=5e-5)


def save(model, ast_optim, dis, dis_optim):
    model_dict = {
        "ast" : model.state_dict(),
        "ast_optim": ast_optim.state_dict(),
        "dis" : dis.state_dict(),
        "dis_optim" : dis_optim.state_dict()
    }
    torch.save(model_dict, "models/dec.pth")

def load(model, ast_optim, dis, dis_optim):
    model_dict = torch.load("models/dec.pth")
    model.load_state_dict(model_dict["ast"])
    ast_optim.load_state_dict(model_dict["ast_optim"])
    dis.load_state_dict(model_dict["dis"])
    dis_optim.load_state_dict(model_dict["dis_optim"])


should_load = True

if should_load:
    load(ast, ast_optim, dis, dis_optim)

print("NUM DEC PARAMETERS: ", sum(p.numel() for p in ast._dec.parameters()))

num_iter = 10000000
for j in range(num_iter):
    content_imgs = next(content_iter).to(device)
    style_imgs = next(style_iter).to(device)
    with torch.no_grad():
        t_cs_map, style_map, t, t_cs, content_map = ast(content_imgs, style_imgs)


    
    # Compute the discriminator loss
    # print("content_map[0].shape", content_map[0].shape, t.shape, style_map[-1].shape, t_cs_map[-1].shape)
    dis_true = dis(torch.cat((content_map[-2].detach(), style_map[-2].detach()), 0))
    true_loss = discriminator_loss(dis_true, torch.ones(batch_size*2, 1, dtype=torch.float32, device=device) - 0.2)
    dis_fake = dis(t_cs_map[-2].detach())
    #dis_true = dis(style_map[-2].detach())

    #true_loss = discriminator_loss(dis_true, torch.ones(batch_size, 1, dtype=torch.float32, device=device) - 0.2)
    fake_loss = discriminator_loss(dis_fake, torch.zeros(batch_size, 1, device=device))

    dis_loss = true_loss + fake_loss
    dis_optim.zero_grad()
    dis_loss.backward()
    nn.utils.clip_grad.clip_grad_norm_(dis.parameters(), 0.2)
    dis_optim.step()
    
    t_cs_map, style_map, t, _, content_map = ast(content_imgs, style_imgs)
    dis_fake = dis(t_cs_map[-2])

    fake_loss = discriminator_loss(dis_fake, torch.ones(batch_size, 1, device=device))
    #print("t_cs_map[-1].shape", t_cs_map[-1].shape, t.shape)
    
    # print("t_cs_map[-1]", t_cs_map[-1])
    # print("t", t)

    # print("t_cs_map[-1]", t_cs_map[-1].mean(), t_cs_map[-1].std())
    # print("t", t.mean(), t.std())
    # print("content_map[0]", content_map[0].mean(), content_map[0].std())
    
    #content_loss = compute_content_loss(t_cs_map[-1], t)
    #content_loss += compute_content_loss(t_cs_map[-1], content_map[0]) * 0.01
    
    # Compute content loss
    for i in range(1, len(t_cs_map)):
        if i == 1:
            content_loss = compute_content_loss(t_cs_map[i], content_map[i])
        else:
            content_loss += compute_content_loss(t_cs_map[i], content_map[i])


    # Compute style loss
    for i in range(len(t_cs_map) - 1):
        # if i >= 4:
        #     break
        if i == 0:
            style_loss = compute_style_loss(t_cs_map[i], style_map[i])
        else:
            style_loss += compute_style_loss(t_cs_map[i], style_map[i])

    loss = content_loss + style_loss * 5 + fake_loss * 1e-2


    ast_optim.zero_grad()
    loss.backward()
    nn.utils.clip_grad.clip_grad_norm_(ast.parameters(), 1.0)
    # print(ast._dec._decoder_blocks[3]._conv._layers[0].weight.grad)
    
    # print(ast._dec._img_out.weight.grad)
    ast_optim.step()

    if (j +1) % 64 == 0:
        
        print("dis_true", dis_true)
        print("dis_fake", dis_fake)
        save(ast, ast_optim, dis, dis_optim)
        print("content_loss", content_loss)
        print("style_loss", style_loss)
        print("dis_loss", dis_loss)
        print("fake_loss", fake_loss)

        print("content_img.max()", content_img.max())
        print("content_img.min()", content_img.min())
        print("style_img.max()", style_img.max())
        print("style_img.min()", style_img.min())
        print("t_cs[0].max()", t_cs[0].max())
        print("t_cs[0].min()", t_cs[0].min())
        
        # print("content_img", style_img)
        
        content_file = "temp_dataset/content_test/zoro.jpg"
        #content_file = "temp_dataset/content_test/dog.jpg"
        #content_file = "temp_dataset/content/aang.jpg"
        #content_file = "temp_dataset/content_test/IMG_2836.JPG"
        
        
        #style_file = "temp_dataset/style_test/336a26f4e3c09f31f3262875fa13faa8c.jpg"
        #style_file = "temp_dataset/style_test/sketch.jpg"
        #style_file = "temp_dataset/style_test/fractal_style_6.jpg"
        #style_file = "temp_dataset/content_test/IMG_2836.JPG"
        #style_file = "temp_dataset/content_test/zoro.jpg"
        #style_file = "temp_dataset/style/vaporwave_style_4.jpeg"
        style_file = "temp_dataset/style_test/thumb-1920-1180547.png"
        axs[0].imshow(t_cs[0].detach().cpu().permute((1,2,0)))
        style_img = image_loader(style_file)
        content_img = image_loader(content_file)
        
        #axs[0].imshow(content_img.detach().cpu()[0].permute((1,2,0)))
        with torch.no_grad():
            _, _, _, t_cs, content_map = ast(content_img, style_img)
        
        print("t_cs[0].max()", t_cs[0].max())
        print("t_cs[0].min()", t_cs[0].min())
        axs[1].imshow(t_cs[0].detach().cpu().permute((1,2,0)))
        # with torch.no_grad():
        #     _, _, _, t_cs, content_map = ast(content_img, style_img, alpha=0.1)
        

        plt.draw()
        plt.pause(0.01)
        plt.show()
        
        # plt.pause(0.1)


