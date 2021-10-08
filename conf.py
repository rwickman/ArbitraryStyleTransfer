import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
decoder_conv_shapes = [(512, 256), (256, 256), (256, 256), (256, 256), (256, 128), (128, 128), (128, 64), (64, 64), (64 , 3)]
