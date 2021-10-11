import torch.nn.functional as F
import torch

from model_util import channel_stats


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

def compute_content_loss(t_cs_map, t):
    # assert t.requires_grad is False
    return F.mse_loss(t_cs_map, t)

def compute_style_loss(t_cs_map, style_map):
    enc_mean, enc_std = channel_stats(t_cs_map)
    style_mean, style_std = channel_stats(style_map)

    mean_loss = F.mse_loss(enc_mean, style_mean)
    std_loss = F.mse_loss(enc_std, style_std)
    g_c = gram_matrix(t_cs_map)
    g_s = gram_matrix(style_map)
    gram_loss = F.l1_loss(g_c, g_s) * 100
    #print(mean_loss, std_loss, gram_loss)
    return mean_loss + std_loss + gram_loss


def discriminator_loss(output, label):
    return F.binary_cross_entropy(output, label)