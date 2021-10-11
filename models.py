import torch
import torch.nn as nn

from model_util import channel_stats
from conf import *
from losses import *
from mobilenetv2 import MobileNetV2, InvertedResidual, DepthWiseConv
import torchvision.models as models
import torch.nn.functional as F


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()
        # #self.target_mean = gram_matrix(target_feature).detach()
        # self.target_mean, self.target_std = channel_stats(target_feature.detach())

    def forward(self, content_map, style_map):
        style_std, style_mean = channel_stats(style_map)
        content_mean, content_std = channel_stats(content_map)
        # Standardize the content feature map
        content_map = (content_map - content_mean) / content_std

        # Shift statistics to target
        content_map = content_map * style_std + style_mean
        return content_map 
    

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

class Encoder(nn.Module):
    def __init__(self, exporting=False):
        super().__init__()
        if not exporting:
            norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
            norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        else:
            norm_mean = torch.tensor([0.485, 0.456, 0.406])
            norm_std = torch.tensor([0.229, 0.224, 0.225])
        self.norm_layer = Normalization(norm_mean, norm_std)
        #self.mob_net = models.mobilenet_v2(True).features#MobileNetV2()

        model = models.mobilenet_v2(True).features[:-2]
        print(models.mobilenet_v2(True).features)
        # * downsamples: 0, 2, 4, 7, 14
        # * content: 4
        # * style: 1, 2, 4, 7, 14
        blocks = []
        blocks += [model[:2]]
        blocks += [model[2]]
        blocks += [model[3:5]]
        blocks += [model[5:8]]
        blocks += [model[8:15]]
        blocks += [model[15:]]

        self.mob_net = nn.ModuleList(blocks)
        print("self.mob_net", self.mob_net)
        #self.mob_net.load_state_dict(torch.load("models/mobilenetv2.pth"))
        
        # norm_layer.name="norm"
        # self._vgg_layers = []
        # i = 0
        # for layer in vgg.children():
        #     if isinstance(layer, nn.Conv2d):
        #         # if i > 0:
        #         #     ref_layer = nn.ReflectionPad2d((1,1,1,1))
        #         #     ref_layer.name = f"ref_{i}"
        #         #     self._vgg_layers.append(ref_layer)

        #         i += 1
                
        #         name = 'conv_{}'.format(i)
        #         # layer.padding = (0, 0)

        #     elif isinstance(layer, nn.ReLU):
        #         name = 'relu_{}'.format(i)
        #         # The in-place version doesn't play very nicely with the ContentLoss
        #         # and StyleLoss we insert below. So we replace with out-of-place
        #         # ones here.
        #         layer = nn.ReLU(inplace=False)
        #     elif isinstance(layer, nn.MaxPool2d):
        #         name = 'pool_{}'.format(i)
        #     elif isinstance(layer, nn.BatchNorm2d):
        #         name = 'bn_{}'.format(i)
        #     else:
        #         raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        #     layer.name = name
        #     self._vgg_layers.append(layer)
        
        # self._vgg_layers = nn.ModuleList(self._vgg_layers)
        # print("self._vgg_layers", self._vgg_layers)


    def forward(self, x, out_layers=[]):
        """Get features maps.
        
        Args:
            x: the input
            out_layers: names of layers whose outputs are needed.
        """
        outs = []
        #print("BEFORE ENC STATS", x.mean(), x.std())
        x = self.norm_layer(x)
        #print("AFTER ENC STATS", x.mean(), x.std())
        #outs = self.mob_net(x, out_layers)
        # print("x.shape", x.shape)
        for i, layer in enumerate(self.mob_net):
            x = layer(x)
            # print("i, x.shape", i, x.shape)
            if i in out_layers:
                outs.append(x)            
        
        return outs



class DecoderBlock(nn.Module):
    """A block of the Decoder"""
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), upsample=False):
        super().__init__()
        self._ref_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self._conv = DepthWiseConv(in_channels, out_channels, 1, 6, use_norm=False)
        # self._conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # self._act = nn.ReLU()
        self._should_upsample = upsample
        if self._should_upsample:
            self._ref_out = nn.ReflectionPad2d((1,1,1,1))
            self._upsample_2 = DepthWiseConv(out_channels, out_channels, 1, 1, use_norm=False)#nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=0)
            self._upsample_3 = nn.Upsample(scale_factor=2, mode='nearest')

    
    def forward(self, x):
        
        # print("")
        #print("DEC x.shape", x.shape)
        #x= self._ref_pad(x)
        x = self._conv(x)
        #print("DEC x.shape", x.shape)
        # x = self._act(x)
        if self._should_upsample:
            x_1 = self._upsample_3(x)
            x_2 = self._upsample_2(x_1)
            x = x_1 + x_2
        
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Create the decoder block modules
        self._decoder_blocks = []
        
        for i, conv_shape in enumerate(decoder_conv_shapes[:-1]):
            # Create the block and upsample only if dimensions differ
            should_upsample = conv_shape[0] != conv_shape[1] and i + 2 < len(decoder_conv_shapes) 
            block = DecoderBlock(
                conv_shape[0],
                conv_shape[1],
                upsample=should_upsample)
            
            self._decoder_blocks.append(block)
        
        # Create the trainable module list
        self._decoder_blocks = nn.ModuleList(self._decoder_blocks)

        # Create the output layers
        self._ref_out = nn.ReflectionPad2d((1,1,1,1))
        self._img_out = nn.Conv2d(decoder_conv_shapes[-1][0], decoder_conv_shapes[-1][1], kernel_size=(3,3))
        #self._img_out_2 = nn.Conv2d(3, 3, 1, 1, 0)

    def forward(self, x):
        for block in self._decoder_blocks:
            #print("DEC x.shape", x.shape)
            x = block(x)

        x = self._ref_out(x)
        x = self._img_out(x)

        #print("x.min()", x.min(), x.max())
        #x = self._img_out_2(x)
        return x



class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(160, 64, kernel_size=3)
        self.act_1 = nn.LeakyReLU(0.2)
        self.conv_2 = nn.Conv2d(64, 32, kernel_size=3)
        self.act_2 = nn.LeakyReLU(0.2)
        self.conv_3 = nn.Conv2d(32, 16, kernel_size=3)
        self.act_3 = nn.LeakyReLU(0.2)
        self.fc_out = nn.Linear(64,1)
        self.act_out = nn.Sigmoid()
    
    def forward(self, t_cs_map):
        batch_size = t_cs_map.shape[0]
        x = self.conv_1(t_cs_map)
        x = self.act_1(x)
        x = self.conv_2(x)
        x = self.act_2(x)
        x = self.conv_3(x)
        x = self.act_3(x)
        return self.act_out(self.fc_out(x.reshape(batch_size, -1)))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(160, 64, kernel_size=3)
        self.act_1 = nn.LeakyReLU(0.2)
        self.conv_2 = nn.Conv2d(64, 32, kernel_size=3)
        self.act_2 = nn.LeakyReLU(0.2)
        self.conv_3 = nn.Conv2d(32, 16, kernel_size=3)
        self.act_3 = nn.LeakyReLU(0.2)
        self.fc_out = nn.Linear(64,1)
        self.act_out = nn.Sigmoid()
    
    def forward(self, t_cs_map):
        batch_size = t_cs_map.shape[0]
        x = self.conv_1(t_cs_map)
        x = self.act_1(x)
        x = self.conv_2(x)
        x = self.act_2(x)
        x = self.conv_3(x)
        x = self.act_3(x)
        return self.act_out(self.fc_out(x.reshape(batch_size, -1)))
        #content_map = self._enc(content_img, self._content_layers)


class UnNormalization(nn.Module):
    def __init__(self, exporting):
        super().__init__()


    def forward(self, x):
        return sel

class AST(nn.Module):
    #def __init__(self, vgg, style_layers=["relu_3", "relu_11"], content_layers=["relu_11"]):
    def __init__(self, style_layers=[0, 1, 2, 3, 4, 5], content_layers=[2,3,4.5], exporting=False):
        super().__init__()
        self._style_layers = style_layers
        self._content_layers = content_layers
        self._exporting = exporting
        self._enc = Encoder(self._exporting).eval()
        self._enc.requires_grad_(False)

        #self._enc.eval()
        self._adain = AdaIN()
        self._conv = DepthWiseConv(320, 160, 1, 2, use_norm=False)
        self._conv_2 = DepthWiseConv(160, 160, 1, 1, use_norm=False)

        if not exporting:
            self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
            self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)
        else:
            self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
            self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        #self._norm = nn.InstanceNorm2d(160)
        # self._conv = nn.Conv2d(320, 160, 3, 1, 0)
        # self._ref_out = nn.ReflectionPad2d((1,1,1,1))

        self._dec = Decoder()

    def forward(self, content_img, style_img, alpha=1.0):
        #print("content_img.shape", content_img.shape)
        content_map = self._enc(content_img, self._style_layers)
        style_map = self._enc(style_img, self._style_layers)
        
        # Perform AdaIN on the feature maps
        #print("torch.cat((content_map[0], style_map[-1])", torch.cat((content_map[0], style_map[-1]), 1).shape)
        #t = self._adain(content_map[0], style_map[-1])
        style_map[-2] = alpha * style_map[-2] + (1.0-alpha) * content_map[-2]

        x = torch.cat((content_map[-2], style_map[-2]), 1)
        #t = self._ref_out(x)
        t_1 = self._conv(x) 
        t = self._conv_2(t_1)
        t = t + t_1
        #t = self._norm(t)
        # print("t.shape", t.shape)
        #t = self._adain(t, style_map[-1])
        #t = self._adain(content_map[0], style_map[-1])
        # t = alpha * t + (1.0-alpha) * content_map[-1]
        # t = content_map[0]
        #print("content_map[0].shape", content_map[0].shape)


        
        # Transform back to image space
        t_cs = self._dec(t)
        t_cs = t_cs * self.norm_std + self.norm_mean
        t_cs = torch.clip(t_cs, 0, 1)
        #print("t_cs.shape", t_cs.shape)

        # Encode it again
        t_cs_map = self._enc(t_cs, self._style_layers)
        #print("t_cs_map.shape", t_cs_map[-1].shape) 
        # Compute discriminator prediction

        if self._exporting:
            return t_cs
        else:
            return t_cs_map, style_map, t, t_cs, content_map
    def save(self):
        torch.save(self._dec.state_dict(), "models/dec.pth")
    
    def load(self):
        dec_dict = torch.load("models/dec.pth")
        self._dec.load_state_dict(dec_dict)







 

    
        



    
# decoder = nn.Sequential(
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 256, (3, 3)),
#     nn.ReLU(),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 128, (3, 3)),
#     nn.ReLU(),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 128, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 64, (3, 3)),
#     nn.ReLU(),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 64, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 3, (3, 3)),
# ).to(device)

# vgg = models.vgg19(pretrained=True).features.to(device).eval()
# enc = Encoder(vgg)

# x = torch.randn(1, 3, 512, 512, device=device)
# # y = torch.randn(1, 512, 32, 32, device=device)
# img_enc = enc(x, ["conv_11"])[0]
# print("img_enc.shape", img_enc.shape)
# d = Decoder().to(device=device)
# print("d(x).shape", d(img_enc).shape)
# print("decoder(x).shape", decoder(img_enc).shape)