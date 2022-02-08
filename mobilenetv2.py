"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch
import torch.nn as nn
import math
from conf import *


__all__ = ['mobilenetv2']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False, padding_mode="reflect"),
        #nn.BatchNorm2d(oup),
        nn.Hardswish(True)
    )


def conv_1x1_bn(inp, oup):
    print("conv_1x1_bn oup", oup)
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.Dropout(0.2),
        nn.Hardswish(True)
    )

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                nn.Hardtanh(0.0, 1.0)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze (Each channel represented by one value)
        y = self.avg_pool(x).view(b, c)

        # Excite (Runs aggregate value for each channel through, gives value bentween [0,1] for each channel)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Reshape(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.pos_enc = nn.Parameter(torch.randn(1, num_channels * 4, 1, 1))

        self.num_channels = num_channels
    
    def forward(self, x):
        x = x + self.pos_enc
        x = x.view(x.shape[0], self.num_channels, x.shape[2] * 2, x.shape[3] * 2)
        return x

class DepthWiseConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, kernel_size=3, use_norm=False, padding=0, use_identity=True, use_relu=False):
        super().__init__()
        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup and use_identity
        
        self._layers = []
        
        if expand_ratio == 1:
            # dw
            self._layers.append(nn.ReflectionPad2d((1, 1, 1, 1)))
            self._layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 0, groups=hidden_dim, bias=False))
            if use_norm:
                self._layers.append(nn.BatchNorm2d(hidden_dim, affine=True, track_running_stats=True))
            self._layers.append(nn.Hardswish(True))

            # Squeeze-and-Excitation
            self._layers.append(SELayer(hidden_dim))

            # pw-linear
            self._layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            if use_norm:
                self._layers.append(nn.BatchNorm2d(oup, affine=True, track_running_stats=True))
            
        else:
            #pw
            self._layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            if use_norm:
                self._layers.append(nn.BatchNorm2d(hidden_dim, affine=True, track_running_stats=True))
            self._layers.append(nn.Hardswish(True))
            
            # dw
            #self._layers.append(nn.ReflectionPad2d((1, 1, 1, 1)))
            # if stride == 2:
            #     self._layers.append(nn.Conv2d(hidden_dim, hidden_dim, 2, stride, 0, groups=hidden_dim, bias=False))    
            # else:
            #     self._layers.append(nn.Conv2d(hidden_dim, hidden_dim*4, 2, stride*2, 0, groups=hidden_dim, bias=False))
            #     self._layers.append(Reshape(hidden_dim))
            
            # self._layers.append(nn.ReflectionPad2d((1, 1, 1, 1)))
            self._layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False, padding_mode="reflect"))
            if use_norm:
                self._layers.append(nn.BatchNorm2d(hidden_dim, affine=True, track_running_stats=True))
            
            self._layers.append(nn.Hardswish(True))
            
            # Squeeze-and-Excitation
            self._layers.append(SELayer(hidden_dim))
            
            
            # pw-linear
            self._layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            if use_norm:
                self._layers.append(nn.BatchNorm2d(oup, affine=True, track_running_stats=True))
            
        self._layers = nn.ModuleList(self._layers)
        self._initialize_weights()
    
    def forward(self, x):
        org_x = x
        #print("\n")
        for layer in self._layers:
            # if isinstance(layer, nn.BatchNorm2d):
            #     continue
            # else:    
            x = layer(x)
        if self.identity:
            x = x + org_x
        # print("x.shape", x.shape, "\n")

        return x


    def _initialize_weights(self):
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()      

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish(True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish(True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish(True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        #print(x.shape)
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)




class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        print(input_channel)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.ModuleList(layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x, out_layers):
        layer_outputs = []
        for i, layer in enumerate(self.features):
            #print("x.shape", x.shape, i)
            x = layer(x)
            if i in out_layers:
                layer_outputs.append(x)

        # print("x.shape", x.shape)
        # layer_outputs.append(x)

        # x = self.features(x)
        # # for i, layer in enumerate(self.features):
        # #     #print(i)
        # #     x = layer(x)
        # #     if i in out_layers:
        # #         layer_outputs.append(x)


        
        # x = self.conv(x)
        # print("AFTER LAST CONV x.shape", x.shape)
        # x = self.avgpool(x)
        # print("AFTER AVG POOL", x.shape)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        # print("AFTER CLASS x.shape", x.shape)
        # import torch.nn.functional as F
        # print(F.softmax(x).argmax(dim=1))
        return layer_outputs
    
    def predict_class(self, x):
        for i, layer in enumerate(self.features):
            #print("x.shape", x.shape, i)
            x = layer(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(**kwargs)

# import torch
# from PIL import Image
# import torchvision.transforms as transforms
# mob_net = mobilenetv2()
# from collections import OrderedDict
# #mob_net.load()
# weights = "models/mobilenetv2.pth"


# mob_net.load_state_dict(torch.load(weights))
# mob_net.eval()
# norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1) * 255
# norm_std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1) * 255
# input_size=160
# t = transforms.Compose([
#             transforms.Resize(int(input_size / 0.875)),
#             transforms.CenterCrop(input_size),
#             transforms.ToTensor()])  # transform it into a torch tensor

# img = t(Image.open("temp_dataset/content/n04254680_15943.JPEG").convert("RGB"))
# #img = t(Image.open("temp_dataset/content_test/dog.jpg").convert("RGB"))

# print(img.shape)
# #img = (img - norm_mean) / norm_std
# print(img.shape)
# out = mob_net(img.unsqueeze(0), [])
# import torch.nn.functional as F
# s = F.softmax(out)
# print(s.argmax())
# print(s[0, 463])
# print(s[0, 805])
# print(s[0, 804])
# print(s[0, 806])
