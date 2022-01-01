import torch
import torch.nn as nn

from model_util import channel_stats
from conf import *
from losses import *
from mobilenetv2 import MobileNetV2, InvertedResidual, DepthWiseConv, conv_3x3_bn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import random

class AddGaussianNoise:
    def __init__(self, mean=0., std=0.01, p=0.9):
        self.std = std
        self.mean = mean
        self.p = p
        
    def __call__(self, tensor):
        if random.random() > self.p:
            tensor =  tensor + torch.randn(tensor.size()) * self.std + self.mean
            #tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
            tensor = torch.clip(tensor, 0, 1)

            return tensor
        else:
            return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


 

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

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x

class Encoder(nn.Module):
    def __init__(self, exporting=False, use_inst_norm=False):
        super().__init__()
        # if not exporting:
        #     norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        #     norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        # else:
        #     norm_mean = torch.tensor([0.485, 0.456, 0.406])
        #     norm_std = torch.tensor([0.229, 0.224, 0.225])
        # self.norm_layer = Normalization(norm_mean, norm_std)
        #self.mob_net = models.mobilenet_v2(True).features#MobileNetV2()

        
        # print(models.mobilenet_v2(True).features)
        # * downsamples: 0, 2, 4, 7, 14
        # * content: 4
        # * style: 1, 2, 4, 7, 14

        # blocks = []
        # if use_inst_norm:
        #     model = models.mobilenet_v2(False).features[:-2]
        #     #model[0][1] = nn.InstanceNorm2d(model[0][1].num_features, affine=True, track_running_stats=True)
        #     model[0][2] = nn.LeakyReLU(0.2, inplace=True)

        #     #model[1].conv[0][1] = nn.InstanceNorm2d(model[1].conv[0][1].num_features, affine=True, track_running_stats=True)
        #     model[1].conv[0][2] = nn.LeakyReLU(0.2, inplace=True)
        #     #model[1].conv[2] = nn.InstanceNorm2d(model[1].conv[2].num_features, affine=True, track_running_stats=True)
            
        #     for i in range(2, len(model)):
        #         #model[i].conv[0][1] = nn.InstanceNorm2d(model[i].conv[0][1].num_features, affine=True, track_running_stats=True)
        #         #model[i].conv[1][1] = nn.InstanceNorm2d(model[i].conv[1][1].num_features, affine=True, track_running_stats=True)
        #         if i + 1 < len(model):
        #             pass
        #             #model[i].conv[3] = nn.InstanceNorm2d(model[i].conv[3].num_features, affine=True, track_running_stats=True)
        #         else:
        #             model[i].conv[3] = Identity()
        #         model[i].conv[0][2] = nn.LeakyReLU(0.2, inplace=True)
        #         model[i].conv[1][2] = nn.LeakyReLU(0.2, inplace=True)
        # else:
        #     model = models.mobilenet_v2(True).features[:-2]
        
        # blocks += [model[:2]]
        # blocks += [model[2]]
        # blocks += [model[3:5]]
        # blocks += [model[5:8]]
        # blocks += [model[8:15]]
        #blocks += [model[15:]]    
        
        #print("model[14].conv[1][1]", model[14].conv[1][0])
        #model[14].conv[1][0].stride = (1,1)
        
        

        blocks = []
        blocks.append(conv_3x3_bn(enc_conv_shapes[0][0], enc_conv_shapes[0][1], enc_conv_shapes[0][2]))
        for in_ch, out_ch, stride in enc_conv_shapes[1:-1]:
            #print(in_ch, out_ch, stride)
            blocks.append(DepthWiseConv(in_ch, out_ch, stride, EXPAND_RATIO, use_norm=True))

        blocks.append(DepthWiseConv(in_ch, out_ch, stride, EXPAND_RATIO, use_norm=False))
        self.mob_net = nn.ModuleList(blocks)
        # print("self.mob_net", self.mob_net)
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
    

    def forward(self, x, out_layers=[], auto_enc=False):
        """Get features maps.
        
        Args:
            x: the input
            out_layers: names of layers whose outputs are needed.
            auto_enc: Indicate training auto-encoder
        """
        # print("ENC x.shape", x.shape)
        if auto_enc:
            for i, layer in enumerate(self.mob_net):
                x = layer(x)
                #print("ENC x.shape", x.shape)
            return x
        else:
            outs = []
            #print("BEFORE ENC STATS", x.mean(), x.std())
            #print("AFTER ENC STATS", x.mean(), x.std())
            #outs = self.mob_net(x, out_layers)
            # print("x.shape", x.shape)
            for i, layer in enumerate(self.mob_net):
                x = layer(x)
                # print("i, x.shape", i, x.shape)
                if i in out_layers:
                    outs.append(x)            
            
            return outs

class PretrainedEncoder(nn.Module):
    def __init__(self, content_layers=['conv_1', 'conv_3', 'conv_5', 'conv_9', 'conv_13', 'relu_13']):
        super().__init__()
        norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self._content_layers = set(content_layers)
        vgg =  models.vgg19(pretrained=True).features
        norm_layer = Normalization(norm_mean, norm_std)
        norm_layer.name="norm"
        self._vgg_layers = []
        self._vgg_layers.append(norm_layer)

        i = 0
        for layer in vgg.children():
            if isinstance(layer, nn.Conv2d):
                # if i > 0:
                #     ref_layer = nn.ReflectionPad2d((1,1,1,1))
                #     ref_layer.name = f"ref_{i}"
                #     self._vgg_layers.append(ref_layer)

                i += 1
                
                name = 'conv_{}'.format(i)
                # layer.padding = (0, 0)

            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            layer.name = name
            
            self._vgg_layers.append(layer)
        
        self._vgg_layers = nn.ModuleList(self._vgg_layers)

        #print("self._vgg_layers", self._vgg_layers)

    def forward(self, x):
        output_layers = []
        for layer in self._vgg_layers:
            x = layer(x)

            if layer.name in self._content_layers:
                output_layers.append(x)
            if len(output_layers) == len(self._content_layers):
                return output_layers

        return output_layers

class DecoderBlock(nn.Module):
    """A block of the Decoder"""
    def __init__(self, in_channels, out_channels, stride, kernel_size=(3,3), upsample=False, expand_ratio=6):
        super().__init__()
        self._ref_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self._conv = DepthWiseConv(in_channels, out_channels, stride, expand_ratio, use_norm=False)
        # self._conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # self._act = nn.ReLU()
        self._should_upsample = upsample
        if self._should_upsample:
            self._ref_out = nn.ReflectionPad2d((1,1,1,1))
            self._upsample_2 = DepthWiseConv(out_channels, out_channels, 1, EXPAND_RATIO, use_norm=False)#nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=0)
            #self._upsample_2 = nn.Conv2d(out_channels, out_channels, 3, 1, 0)#DepthWiseConv(out_channels, out_channels, 1, 6, use_norm=True)
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
            # x_1 = self._ref_out(x_1)
            x = self._upsample_2(x_1)

        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Create the decoder block modules
        self._decoder_blocks = []
        
        for i, conv_shape in enumerate(decoder_conv_shapes[:-1]):
            if i + 1 == len(decoder_conv_shapes) - 1:
                expand_ratio = 1
            else:
                expand_ratio = 6

            # Create the block and upsample only if dimensions differ
            should_upsample = conv_shape[0] != conv_shape[1] and i + 2 < len(decoder_conv_shapes) 
            block = DecoderBlock(
                conv_shape[0],
                conv_shape[1],
                conv_shape[2],
                upsample=should_upsample,
                expand_ratio=EXPAND_RATIO)
            
            self._decoder_blocks.append(block)
        
        # Create the trainable module list
        self._decoder_blocks = nn.ModuleList(self._decoder_blocks)

        # Create the output layers
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self._ref_out = nn.ReflectionPad2d((1,1,1,1))
        self._img_out = nn.Conv2d(decoder_conv_shapes[-1][0], decoder_conv_shapes[-1][1], kernel_size=(3,3))
        #self._img_out_2 = nn.Conv2d(3, 3, 1, 1, 0)
        # Squish output between 0 and 1
        self.last_act = nn.Hardtanh(0.0, 1.0)

    def forward(self, x):
        for block in self._decoder_blocks:
            #print("DEC x.shape", x.shape)
            x = block(x)
        
        x = self.act(x)
        x = self._ref_out(x)
        x = self._img_out(x)
        x = self.last_act(x)

        #print("x.min()", x.min(), x.max())
        #x = self._img_out_2(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(use_inst_norm=True)
        self.decoder = Decoder()
    
    def forward(self, x):
        enc_x = self.encoder(x, auto_enc=True)
        dec_x = self.decoder(enc_x)
        return dec_x
    
    


# class Discriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv_1 = nn.Conv2d(160, 64, kernel_size=3)
#         self.act_1 = nn.LeakyReLU(0.2)
#         self.conv_2 = nn.Conv2d(64, 32, kernel_size=3)
#         self.act_2 = nn.LeakyReLU(0.2)
#         self.conv_3 = nn.Conv2d(32, 16, kernel_size=3)
#         self.act_3 = nn.LeakyReLU(0.2)
#         self.fc_out = nn.Linear(64,1)
#         self.act_out = nn.Sigmoid()
    
#     def forward(self, t_cs_map):
#         batch_size = t_cs_map.shape[0]
#         x = self.conv_1(t_cs_map)
#         x = self.act_1(x)
#         x = self.conv_2(x)
#         x = self.act_2(x)
#         x = self.conv_3(x)
#         x = self.act_3(x)
#         return self.act_out(self.fc_out(x.reshape(batch_size, -1)))




class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self._mobnet = MobileNetV2(num_classes=1)
        #print()

        self._mobnet.features[0][1] = nn.InstanceNorm2d(32)
        self._mobnet.conv[1] = nn.InstanceNorm2d(1280)
        # self._mobnet.conv.append(nn.Dropout(0.2))
        self._mobnet.features.append(nn.Dropout(0.2))


        # self.conv_3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        # self.act_3 = nn.LeakyReLU(0.2)
        self.act_out = nn.Sigmoid()
    
    def forward(self, t_cs_map):
        batch_size = t_cs_map.shape[0] 

        x = self._mobnet.predict_class(t_cs_map)
        return self.act_out(x)
        #content_map = self._enc(content_img, self._content_layers)



class AST(nn.Module):
    #def __init__(self, vgg, style_layers=["relu_3", "relu_11"], content_layers=["relu_11"]):
    def __init__(self, style_layers=[4, 7, 10, 12, 16], content_layers=[4, 7, 10, 12, 16], exporting=False):
        super().__init__()
        self._style_layers = style_layers
        self._content_layers = content_layers
        self._exporting = exporting
        self._enc = Encoder(self._exporting).eval()
        self._enc.requires_grad_(False)
        self._dec = Decoder()
        #self._enc.eval()
        self._adain = AdaIN()
        
        # self._conv = DepthWiseConv(320, 160, 1, 2, use_norm=False)
        # self._conv_2 = DepthWiseConv(160, 160, 1, 1, use_norm=False)
        # self.act = nn.ReLU6()
        # self.act_2 = nn.ReLU6()


        
        #self._norm = nn.InstanceNorm2d(160)
        # self._conv = nn.Conv2d(320, 160, 3, 1, 0)
        # self._ref_out = nn.ReflectionPad2d((1,1,1,1))

      

    def forward(self, content_img, style_img, alpha=1.0):
        #print("content_img.shape", content_img.shape)
        content_map = self._enc(content_img, self._style_layers)
        style_map = self._enc(style_img, self._style_layers)
        # for c_map in content_map:
        #     print("c_map.shape", c_map.shape)
        # print(len(content_map))
        # print(len(style_map))
        # Perform AdaIN on the feature maps
        #print("torch.cat((content_map[0], style_map[-1])", torch.cat((content_map[0], style_map[-1]), 1).shape)
        #t = self._adain(content_map[0], style_map[-1])
        
        # t = self._adain(content_map[-2], style_map[-2])
        #t = alpha * t + (1.0-alpha) * content_map[-1]
        

        #style_map[-2] = alpha * style_map[-1] + (1.0-alpha) * content_map[-1]
        t = self._adain(content_map[-1], style_map[-1])
        #t = content_map[-1]
        #content_adain = self._adain(content_map[-2], style_map[-2])
        #x = torch.cat((content_adain, style_map[-2]), 1)

        #t_1 = self._conv(x)         
        #t_1 = t_1 + content_map[-2]
        #t = self._conv_2(t_1)
        


        # print("t.shape", t.shape)
        #t = self._adain(t, style_map[-1])
        #t = self._adain(content_map[0], style_map[-1])
        # t = alpha * t + (1.0-alpha) * content_map[-1]
        # t = content_map[0]
        #print("content_map[0].shape", content_map[0].shape)


        
        # Transform back to image space
        # t_cs = self.act(t)
        t_cs = self._dec(t)
        #t_cs = t_cs * self.norm_std + self.norm_mean
        #t_cs = self.act_2(t_cs)
        #t_cs 
        #t_cs = torch.clip(t_cs, 0, 1)
        #print("t_cs.max()", t_cs.max(), t_cs.min())
        #t_cs_min = t_cs.min()
        #t_cs = (t_cs - t_cs_min) / (t_cs.max() - t_cs_min)

        #print("t_cs.shape", t_cs.shape)

        # Encode it again
        
        #print("t_cs_map.shape", t_cs_map[-1].shape) 
        # Compute discriminator prediction

        if self._exporting:
            return t_cs
        else:
            # t_cs = (t_cs - t_cs_min) / (t_cs.max() - t_cs_min)
            t_cs_map = self._enc(t_cs, self._style_layers)
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

