import torch.nn.functional as F
import torch
import torch.nn as nn
from conf import *

from model_util import channel_stats

# class SoftHistogram(nn.Module):
#     def __init__(self, bins, min, max, sigma):
#         super(SoftHistogram, self).__init__()
#         self.bins = bins
#         self.min = min
#         self.max = max
#         self.sigma = sigma
#         self.delta = float(max - min) / float(bins)
        

#     def forward(self, x):
#         self.centers = float(self.min) + self.delta * (torch.arange(x.shape[2], device=device).float() + 0.5)
#         x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
#         x = torch.sigmoid(self.sigma * (x + self.delta/2)) - torch.sigmoid(self.sigma * (x - self.delta/2))
#         x = x.sum(dim=1)
#         return x

# # soft_hist = SoftHistogram(256, 0, 1, 3).to(device)
# def compute_hist_loss(t_cs, style_map):

#     t_cs_hist = soft_hist(t_cs)
#     style_map_hist = soft_hist(style_map)
#     return F.mse_loss(t_cs_hist, style_map_hist)

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

def compute_content_loss(t_cs_map, t):
    # assert t.requires_grad is False
    return F.huber_loss(t_cs_map, t)

def compute_style_loss(t_cs_map, style_map):
    enc_mean, enc_std = channel_stats(t_cs_map)
    style_mean, style_std = channel_stats(style_map)
    mean_loss = F.mse_loss(enc_mean, style_mean) * 0.1
    std_loss = F.mse_loss(enc_std, style_std) * 0.1
    g_c = gram_matrix(t_cs_map)
    g_s = gram_matrix(style_map)
    gram_loss = F.l1_loss(g_c, g_s) * 100
    #print("mean_loss, std_loss, gram_loss", mean_loss, std_loss, gram_loss)
    return mean_loss + std_loss + gram_loss


def discriminator_loss(output, label):
    return F.binary_cross_entropy(output, label)