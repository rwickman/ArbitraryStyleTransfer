import torch
import torch.nn as nn

from model_util import channel_stats, rgb2lab, lab2rgb
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


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    #size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(feat.size())) / std.expand(feat.size())
    return normalized_feat

class AdaAttN(nn.Module):
    def __init__(self, inp_size):
        super().__init__()
        self.W_q = nn.Conv2d(inp_size, inp_size, 1, 1, 0, bias=False)
        self.W_k = nn.Conv2d(inp_size, inp_size, 1, 1, 0, bias=False)
        self.W_v = nn.Conv2d(inp_size, inp_size, 1, 1, 0, bias=False)
        self.att_act = nn.Softmax(dim=-1)
        self.std_act = nn.ReLU(True)
        self.inst_norm_1 = nn.InstanceNorm2d(inp_size)
        self.inst_norm_2 = nn.InstanceNorm2d(inp_size)
        self.inst_norm = nn.InstanceNorm2d(inp_size)
        self.inp_size = inp_size

    def forward(self, content_map, style_map):
        batch_size, _ , height, width = content_map.size()#torch._shape_as_tensor(content_map)
        batch_size, _ , style_height, style_width = style_map.size()
        
        q = self.W_q(self.inst_norm_1(content_map))
        k = self.W_k(self.inst_norm_2(style_map))
        v = self.W_v(style_map)

        # Reshape to flatten each feature map
        q = q.view(batch_size, self.inp_size, -1).permute(0, 2, 1)
        k = k.view(batch_size, self.inp_size, -1)
        v = v.view(batch_size, self.inp_size, -1).permute(0, 2, 1)

        # Performs attention to every pair-wise feature maps in style and content 
        qk = torch.bmm(q, k)

        att_weights = self.att_act(qk)

        mean = torch.bmm(att_weights, v)

        std = torch.sqrt(self.std_act(torch.bmm(att_weights, v ** 2) - mean ** 2))

        std = std.view(batch_size, -1, width, self.inp_size).permute(0, 3, 1, 2)
        mean = mean.view(batch_size, -1, width, self.inp_size).permute(0, 3, 1, 2)
        #print("content_map.shape", content_map.shape, "std.shape", std.shape, mean.shape)
        # print(len(content_map[0, 0, 0]))
        #std = std.view(batch_size, torch._shape_as_tensor(content_map)[2], -1, 160).permute(0, 3, 1, 2)
        #mean = mean.view(batch_size, torch._shape_as_tensor(content_map)[2], -1, 160).permute(0, 3, 1, 2)
        
        #return self.inst_norm(content_map)
        # print("content_map.shape", content_map.shape)
        # print("self.inst_norm(content_map).shape", self.inst_norm(content_map).shape)
        return std * self.inst_norm(content_map) + mean
        



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

        blocks = []
        blocks.append(conv_3x3_bn(enc_conv_shapes[0][0], enc_conv_shapes[0][1], enc_conv_shapes[0][2]))
        i= 0
        for in_ch, out_ch, stride, kernel_size, expand_ratio in enc_conv_shapes[1:-1]:
            #print(in_ch, out_ch, stride)
            blocks.append(
                DepthWiseConv(
                    in_ch, out_ch, stride, expand_ratio, use_norm=True, kernel_size=kernel_size))
            i += 1

        blocks.append(DepthWiseConv(in_ch, out_ch, stride, EXPAND_RATIO, use_norm=True))
        self.mob_net = nn.ModuleList(blocks)
  

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
    def __init__(self, content_layers=['conv_1', 'conv_3', 'conv_5', 'conv_9', 'conv_13', 'relu_15']):
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
    def __init__(self, in_channels, out_channels, stride, kernel_size=3, upsample=False, expand_ratio=6):
        super().__init__()
        self._ref_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self._conv = DepthWiseConv(in_channels, out_channels, stride, expand_ratio, use_norm=False, kernel_size=kernel_size)
        # self._conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # self._act = nn.ReLU()
        self._should_upsample = upsample
        if self._should_upsample:
            self._ref_out = nn.ReflectionPad2d((1,1,1,1))
            self._upsample_2 = DepthWiseConv(out_channels, out_channels, 1, 1, use_norm=False)#nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=0)
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
            #print("\nx.shape", x.shape)
            x_1 = self._upsample_3(x)
            # x_1 = self._ref_out(x_1)
            x = self._upsample_2(x_1)
            # print("AFTER UP x.shape", x.shape, "\n")
 
        return x

class Decoder(nn.Module):
    def __init__(self, exporting=False):
        super().__init__()
        # Create the decoder block modules
        self._decoder_blocks = []
        self.exporting = exporting
        
        for i, conv_shape in enumerate(decoder_conv_shapes[:-1]):

            # Create the block and upsample only if dimensions differ
            should_upsample = (conv_shape[0] != conv_shape[1] and i + 6 < len(decoder_conv_shapes))
            block = DecoderBlock(
                conv_shape[0],
                conv_shape[1],
                conv_shape[2],
                upsample=should_upsample,
                expand_ratio=conv_shape[4],
                kernel_size=conv_shape[3])
            
            self._decoder_blocks.append(block)
        
        # Create the trainable module list
        self._decoder_blocks = nn.ModuleList(self._decoder_blocks)

        # Create the output layers
        # self.act = nn.Hardswish(inplace=True)
        self._ref_out = nn.ReflectionPad2d((1,1,1,1))
        self._img_out = nn.Conv2d(decoder_conv_shapes[-1][0], decoder_conv_shapes[-1][1], kernel_size=(3,3))
        #self._img_out_2 = nn.Conv2d(3, 3, 1, 1, 0)
        # Squish output between 0 and 1
        self.last_act = nn.Hardtanh(0.0, 1.0)

    def forward(self, x):
        for block in self._decoder_blocks:
            #print("DEC x.shape", x.shape)
            x = block(x)
        
        # x = self.act(x)
        x = self._ref_out(x)
        x = self._img_out(x)
        # x = torch.clip(x, 0, 1)
        if self.exporting:
            x = self.last_act(x)

        #print("x.min()", x.min(), x.max())
        #x = self._img_out_2(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(use_inst_norm=True)
        self.ada_out = DepthWiseConv(enc_out_channels*2, enc_out_channels, 1, EXPAND_RATIO, use_norm=False, use_identity=False)
        self.decoder = Decoder()
    
    def forward(self, x):
        enc_x = self.encoder(x, out_layers=enc_out_layers)

        enc_x = self.ada_out(torch.cat((enc_x[0], enc_x[1]), dim=1))

        dec_x = self.decoder(enc_x)
        # print("x.shape", x.shape)
        # print("enc_x.shape", enc_x.shape)
        # print("dec_x.shape", dec_x.shape)
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
        self._enc = Encoder(self._exporting)
        #self._enc.requires_grad_(False)
        self._dec = Decoder(self._exporting)
        #self._enc.eval()
        #self._adain = AdaIN()
        self.upsample_att_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.ada_att_1 = AdaAttN(enc_out_channels)
        #self.ada_att_2 = AdaAttN(enc_out_channels)
        
        
        #self.ada_out = DepthWiseConv(enc_out_channels*2, enc_out_channels, 1, EXPAND_RATIO, use_norm=False, use_identity=False)

        # self._conv = DepthWiseConv(320, 160, 1, 2, use_norm=False)
        # self._conv_2 = DepthWiseConv(160, 160, 1, 1, use_norm=False)
        # self.act = nn.ReLU6()
        # self.act_2 = nn.ReLU6()


        
        #self._norm = nn.InstanceNorm2d(160)
        # self._conv = nn.Conv2d(320, 160, 3, 1, 0)
        # self._ref_out = nn.ReflectionPad2d((1,1,1,1))

      

    def forward(self, content_img, style_img, alpha=1.0):
        #print("content_img.shape", content_img.shape)
        # if self._exporting:
        #     content_img = rgb2lab(content_img)
        #     style_img = rgb2lab(style_img)
        
        # content_maps = self._enc(content_img, out_layers=enc_out_layers)
        # style_maps = self._enc(style_img, out_layers=enc_out_layers)
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
        
        #t = self._adain(content_map, style_map)

        # t_1 = self.ada_att_1()

        # t = self.ada_att(content_map, style_map)
        # t = self.ada_out(t)
        # t = self.ada_att_2(t, style_map)
        #t = (t_1 + t_2) / 2

        
        
        if not self._exporting:
            stylized_map_1 t = self.encode(content_img, style_img, detach=True, return_maps=True)

            t_return = stylized_map_1
            
            #t_return = self.encode(content_img, style_img, detach=True)

            # t_return = self.ada_att(content_map.detach(), style_map.detach())
            # t_return = self.ada_out(t_return)
            # t_return = self.ada_att_2(t_return, style_map.detach())
            content_maps = self._enc(content_img, out_layers=enc_out_layers)
            content_maps = torch.cat((content_maps[0], content_maps[1]), dim=1)
            content_map = self.ada_out(content_maps)
            t = alpha * t + (1-alpha) * content_map
            # print("content_img.shape", content_img.shape)
            # print("t.shape", t.shape)

            
            org_out = self._dec(content_map)
            #print("org_out.shape", org_out.shape)
            
        else:
            t = self.encode(content_img, style_img)


        #t_return = (t_return_1 + t_return_2) / 2 
 
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

        #org_inp = alpha * t + (1-alpha) * content_map
        
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

        # print("t_cs.shape", t_cs.shape)
        # print("content_img.shape", content_img.shape)

        if self._exporting:
            return t_cs#lab2rgb(t_cs)
        else:
            # t_cs = (t_cs - t_cs_min) / (t_cs.max() - t_cs_min)
            # t_cs_map = self._enc(t_cs, self._style_layers)
            return t_cs, t_return, org_out #t_cs_map, style_map, t, t_cs, content_map
    
    def encode(self, content_img, style_img, detach=False, return_maps=False):
        
        # Get the content and style maps

        if detach:
            self._enc.eval()
            content_maps = self._enc(content_img, out_layers=enc_out_layers)
            style_maps = self._enc(style_img, out_layers=enc_out_layers)
            for i in range(len(enc_out_layers)):
                content_maps[i] = content_maps[i].detach()
                style_maps[i] = style_maps[i].detach()
            
            self._enc.train()
        else:
            content_maps = self._enc(content_img, out_layers=enc_out_layers)
            style_maps = self._enc(style_img, out_layers=enc_out_layers)


        # Get the stylized maps
        stylized_map_1 = self.ada_att_1(content_maps[0], style_maps[0])
        stylized_map_2 = self.ada_att_2(content_maps[1], style_maps[1])

        # Upsample the first stylized map to match other one

        #stylized_map_1 = self.upsample_att_1(stylized_map_1)
        # print("stylized_map_1.shape", stylized_map_1.shape)
        # print("stylized_map_2.shape", stylized_map_2.shape)


        # Add them together and run through model
        stylized_map = torch.cat((stylized_map_1, stylized_map_2), dim=1)
        stylized_map = self.ada_out(stylized_map)

        if return_maps:
            return stylized_map_1, stylized_map_2, stylized_map
        else:

            return stylized_map



    
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

