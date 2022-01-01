import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
#decoder_conv_shapes = [(512, 256), (256, 256), (256, 256), (256, 256), (256, 128), (128, 128), (128, 64), (64, 64), (64 , 3)]
#decoder_conv_shapes = [(320, 160), (160,160), (160,160), (160,96), (96,96), (96, 96), (96, 64), (64, 64), (64, 64), (64, 64), (64, 32), (32, 32), (32, 32),  (32, 24), (24, 24), (24,16), (16, 3)]
#decoder_conv_shapes = [(160, 160), (160, 96), (96, 64), (64,32), (32,24), (24, 16), (16, 8), (8, 3)]
#decoder_conv_shapes = [(160, 160, 1), (160, 160, 1), (160,96, 1), (96,96, 1), (96, 96, 1), (96, 64, 1), (64, 64, 1), (64, 64, 1), (64, 32, 1), (32, 32, 1), (32, 32, 1), (32, 32, 1), (32, 24 ,1), (24, 24, 1), (24, 24, 1), (24,16, 1), (16, 16, 1), (16, 8, 1), (8, 3, 1)]
EXPAND_RATIO = 3
enc_conv_shapes = [(3, 32, 2), (32, 16, 1), (16, 32, 2), (32, 32, 1), (32, 32, 1), (32, 32, 1), (32, 32, 1), (32, 64, 2), (64, 64, 1), (64, 64, 1), (64, 64, 1), (64, 96, 1), (96, 96, 1), (96, 96, 1), (96, 160, 2), (160, 160, 1), (160, 160, 1)]
decoder_conv_shapes = [(160, 160, 1), (160, 160, 1), (160, 96, 1), (96, 96, 1), (96, 96, 1), (96, 64, 1), (64, 64, 1), (64, 64, 1), (64, 64, 1), (64, 32, 1), (32, 32, 1), (32, 32, 1), (32, 32, 1), (32, 32, 1), (32, 16, 1), (16, 32, 1), (32, 3, 1)]


#content_dir = ["temp_dataset/content/", "/media/data/code/stylegan/data/new_dataset/creeper_dataset/", "/media/data/datasets/unlabeled2017/"] 
#style_dir = ["/media/data/code/stylegan/data/new_dataset/creeper_dataset/", "/media/data/datasets/wikiart_smaller", "temp_dataset/style/"] 
# content_dir = ["temp_dataset/content/"]
# style_dir = ["temp_dataset/style/"]
# content_dir = ["temp_dataset/content_test/"]
# style_dir = ["temp_dataset/style_test/"]

content_dir = ["temp_dataset/content/", "/media/data/code/stylegan/data/kaggle_imgs_smaller_2/", "/media/data/code/stylegan/data/portraits_smaller_4/", "/media/data/datasets/unlabeled2017/", "/media/data/datasets/landscape", "/media/data/datasets/flickr_dataest/flickr30k_images/flickr30k_images/"] 
style_dir = ["/media/data/code/stylegan/data/kaggle_imgs_smaller_2/", "/media/data/code/stylegan/data/portraits_smaller_4/", "/media/data/datasets/wikiart_smaller", "temp_dataset/style/", "/media/data/datasets/landscape"]