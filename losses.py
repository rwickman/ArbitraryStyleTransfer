import torch.nn.functional as F
import torch
import torch.nn as nn
from conf import *

from model_util import channel_stats

class EarthMoversDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # input has dims: (Batch x Bins)
        bins = x.size(1)
        r = torch.arange(bins)
        s, t = torch.meshgrid(r, r)
        tt = (t >= s).to(device)

        cdf_x = torch.matmul(x, tt.float())
        cdf_y = torch.matmul(y, tt.float())

        return torch.sum(torch.square(cdf_x - cdf_y), dim=1)

def phi_k(x, L, W):
    return torch.sigmoid((x + (L / 2)) / W) - torch.sigmoid((x - (L / 2)) / W)


def compute_pj(x, mu_k, K, L, W):
    # we assume that x has only one channel already
    # flatten spatial dims
    x = x.reshape(x.size(0), 1, -1)
    x = x.repeat(1, K, 1)  # construct K channels

    # apply activation functions
    return phi_k(x - mu_k, L, W)


class HistLayerBase(nn.Module):
    def __init__(self):
        super().__init__()

        self.K = 256
        self.L = 1 / self.K  # 2 / K -> if values in [-1,1] (Paper)
        self.W = self.L / 2.5

        self.mu_k = (self.L * (torch.arange(self.K) + 0.5)).view(-1, 1).to(device)


class SingleDimHistLayer(HistLayerBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N = x.size(1) * x.size(2)
        pj = compute_pj(x, self.mu_k, self.K, self.L, self.W)
        return pj.sum(dim=2) / N

class SoftHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(SoftHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        

    def forward(self, x):
        self.centers = float(self.min) + self.delta * (torch.arange(x.shape[2], device=device).float() + 0.5)
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.sigmoid(self.sigma * (x + self.delta/2)) - torch.sigmoid(self.sigma * (x - self.delta/2))
        x = x.sum(dim=1)
        return x

#soft_hist = SoftHistogram(255, 0, 1, 3).to(device)
hist=  SingleDimHistLayer().to(device)
earth_movers = EarthMoversDistanceLoss().to(device)
# def compute_hist_loss(t_cs, style_map):

#     t_cs_hist = soft_hist(t_cs)
#     style_map_hist = soft_hist(style_map)
#     return F.mse_loss(t_cs_hist, style_map_hist)
def compute_hist_loss(t_cs, style_map):

    t_cs_hist = hist(t_cs)
    style_map_hist = hist(style_map)
    return earth_movers(t_cs_hist, style_map_hist).mean()


def tv_loss(img):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    loss = (h_variance + w_variance)
    return loss

def gram_matrix(tensor):
    B, C, H, W = tensor.shape
    x = tensor.view(B, C, H*W)
    x_t = x.transpose(1, 2)
    return  torch.bmm(x, x_t) / (C*H*W)

# def gram_matrix(input):
#     a, b, c, d = input.size()  # a=batch size(=1)
#     # b=number of feature maps
#     # (c,d)=dimensions of a f. map (N=c*d)

#     features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
#     print("features.shape", features.shape)
#     G = torch.mm(features, features.t())  # compute the gram product
#     print("G.shape", G.shape)
#     # we 'normalize' the values of the gram matrix
#     # by dividing by the number of element in each feature maps.
#     return G.div(a * b * c * d)

def compute_content_loss(inp, tgt):
    # assert t.requires_grad is False
    return F.huber_loss(inp, tgt)

def compute_style_loss(t_cs_map, style_map):

    enc_mean, enc_std = channel_stats(t_cs_map)
    style_mean, style_std = channel_stats(style_map)
    mean_loss = F.huber_loss(enc_mean, style_mean) * 1.25
    std_loss = F.huber_loss(enc_std, style_std) * 1.25
    g_c = gram_matrix(t_cs_map)
    g_s = gram_matrix(style_map)
    gram_loss = F.huber_loss(g_c, g_s) * 10

    #print("mean_loss, std_loss, gram_loss", mean_loss, std_loss, gram_loss)
    return mean_loss + std_loss + gram_loss


def discriminator_loss(output, label):
    return F.binary_cross_entropy(output, label)