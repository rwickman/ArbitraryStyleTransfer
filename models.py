import torch
import torch.nn as nn

from model_util import channel_stats
from conf import *
from losses import *


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()
        # #self.target_mean = gram_matrix(target_feature).detach()
        # self.target_mean, self.target_std = channel_stats(target_feature.detach())

    def forward(self, content_map, style_map):
        style_std, style_mean = channel_stats(style_map)
        content_mean, content_std = channel_stats(content_map)
        
        # Standardize the content feature map
        content_map = (content_map - content_mean) / (content_std + 1e-5)

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
    def __init__(self, vgg):
        super().__init__()
        norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        norm_layer = Normalization(norm_mean, norm_std)
        norm_layer.name="norm"
        self._vgg_layers = []
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
        print("self._vgg_layers", self._vgg_layers)

    def forward(self, x, out_layers=[]):
        """Get features maps.
        
        Args:
            x: the input
            out_layers: names of layers whose outputs are needed.
        """
        outs = []
        for layer in self._vgg_layers:
            x = layer(x)
            if layer.name in out_layers:
                outs.append(x)            
        
        return outs



class DecoderBlock(nn.Module):
    """A block of the Decoder"""
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), upsample=False):
        super().__init__()
        self._ref_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self._act = nn.ReLU()
        self._should_upsample = upsample
        if self._should_upsample:
            self._upsample = nn.Upsample(scale_factor=2, mode='nearest')
    
    def forward(self, x):
        x = self._ref_pad(x)

        x = self._conv(x)
        x = self._act(x)
        if self._should_upsample:
            x = self._upsample(x)
        
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


    def forward(self, x):
        for block in self._decoder_blocks:
            x = block(x)

        x = self._ref_out(x)
        x = self._img_out(x)
        
        return x



class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(512, 32, kernel_size=(3,3))
        self.act_1 = nn.LeakyReLU(0.2)
        self.conv_2 = nn.Conv2d(32, 16, kernel_size=(3,3), stride=2)
        self.act_2 = nn.LeakyReLU(0.2)
        self.conv_3 = nn.Conv2d(16, 4, kernel_size=(3,3), stride=2)
        self.act_3 = nn.LeakyReLU(0.2)
        self.fc_out = nn.Linear(144,1)
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



class AST(nn.Module):
    def __init__(self, vgg, style_layers=['relu_1', 'relu_3', 'relu_5', 'relu_11'], content_layers=['relu_11']):
        super().__init__()
        self._style_layers = style_layers
        self._content_layers = content_layers

        self._enc = Encoder(vgg)
        self._enc.requires_grad_(False)
        self._adain = AdaIN()
        self._dec = Decoder()

    def forward(self, content_img, style_img, alpha=1.0):

        content_map = self._enc(content_img, self._content_layers)
        style_map = self._enc(style_img, self._style_layers)
        
        # Perform AdaIN on the feature maps
        t = self._adain(content_map[0], style_map[-1])

        t = alpha * t + (1.0-alpha) * content_map[0]
        print("content_map[0].shape", content_map[0].shape)


        
        # Transform back to image space
        t_cs = self._dec(t)

        # Encode it again
        t_cs_map = self._enc(t_cs, self._style_layers)

        # Compute discriminator prediction


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