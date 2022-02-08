import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
img_sizes = [96, 128, 160]
#img_sizes = [64, 96]
#img_sizes = [128, 192, 256, 320, 512]

imsize = 320 if torch.cuda.is_available() else 128
#decoder_conv_shapes = [(512, 256), (256, 256), (256, 256), (256, 256), (256, 128), (128, 128), (128, 64), (64, 64), (64 , 3)]
#decoder_conv_shapes = [(320, 160), (160,160), (160,160), (160,96), (96,96), (96, 96), (96, 64), (64, 64), (64, 64), (64, 64), (64, 32), (32, 32), (32, 32),  (32, 24), (24, 24), (24,16), (16, 3)]
#decoder_conv_shapes = [(160, 160), (160, 96), (96, 64), (64,32), (32,24), (24, 16), (16, 8), (8, 3)]
#decoder_conv_shapes = [(160, 160, 1), (160, 160, 1), (160,96, 1), (96,96, 1), (96, 96, 1), (96, 64, 1), (64, 64, 1), (64, 64, 1), (64, 32, 1), (32, 32, 1), (32, 32, 1), (32, 32, 1), (32, 24 ,1), (24, 24, 1), (24, 24, 1), (24,16, 1), (16, 16, 1), (16, 8, 1), (8, 3, 1)]

# enc_conv_shapes = [(3, 32, 2), (32, 16, 1), (16, 32, 2), (32, 32, 1), (32, 32, 1), (32, 32, 1), (32, 32, 1), (32, 64, 2), (64, 64, 1), (64, 64, 1), (64, 64, 1), (64, 96, 1), (96, 96, 1), (96, 96, 1), (96, 128, 2), (128, 128, 1), (128, 128, 1)]
# decoder_conv_shapes = [(128, 128, 1), (128, 128, 1), (128, 96, 1), (96, 96, 1), (96, 96, 1), (96, 64, 1), (64, 64, 1), (64, 64, 1), (64, 64, 1), (64, 32, 1), (32, 32, 1), (32, 32, 1), (32, 32, 1), (32, 32, 1), (32, 16, 1), (16, 32, 1), (32, 3, 1)]

#enc_conv_shapes = [(3, 32, 2), (32, 32, 1), (32, 32, 2), (32, 32, 1), (32, 32, 1), (32, 64, 2), (64, 64, 1), (64, 64, 1), (64, 64, 1), (64, 96, 1), (96, 96, 1), (96, 96, 1), (96, 128, 2), (128, 128, 1), (128, 128, 1)]
# cfgs = [
#     [3,   1,  16, 0, 0, 1],
#     [3,   4,  24, 0, 0, 2],
#     [3,   3,  24, 0, 0, 1],
#     [5,   3,  40, 1, 0, 2],
#     [5,   3,  40, 1, 0, 1],
#     [5,   3,  40, 1, 0, 1],
#     [3,   6,  80, 0, 1, 2],
#     [3, 2.5,  80, 0, 1, 1],
#     [3, 2.3,  80, 0, 1, 1],
#     [3, 2.3,  80, 0, 1, 1],
#     [3,   6, 112, 1, 1, 1],
#     [3,   6, 112, 1, 1, 1],
#     [5,   6, 160, 1, 1, 2],
#     [5,   6, 160, 1, 1, 1],
#     [5,   6, 160, 1, 1, 1]
# ]


# enc_conv_shapes = [
#     (3, 16, 2),
#     (16, 16, 1),
#     (16, 24, 2), 
#     (24, 24, 1), 
#     (24, 40, 1), 
#     (40, 40, 2), 
#     (40, 40, 1), 
#     (40, 48, 1), 
#     (48, 48, 1), 
#     (48, 96, 1), 
#     (96, 96, 1), 
#     (96, 96, 1), 
#     (96, 96, 1),
#     (96, 96, 1)]
# cfgs = [
#     # k, t, c, SE, HS, s 
#     [3,   1,  16, 0, 0, 1],
#     [3,   4,  24, 0, 0, 2],
#     [3,   3,  24, 0, 0, 1],
#     [5,   3,  40, 1, 0, 2],
#     [5,   3,  40, 1, 0, 1],
#     [5,   3,  40, 1, 0, 1],
#     [3,   6,  80, 0, 1, 2],
#     [3, 2.5,  80, 0, 1, 1],
#     [3, 2.3,  80, 0, 1, 1],
#     [3, 2.3,  80, 0, 1, 1],
#     [3,   6, 112, 1, 1, 1],
#     [3,   6, 112, 1, 1, 1],
#     [5,   6, 160, 1, 1, 2],
#     [5,   6, 160, 1, 1, 1],
#     [5,   6, 160, 1, 1, 1]
# ]

EXPAND_RATIO = 3
expand_ratios = [1, 6, 6, 6, 6, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4]
kernel_sizes = [3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

enc_conv_shapes = [
    # c_in, c_out, s, k, t
    (3, 16, 1, 3, 1),
    (16, 16, 1, 3, 6),
    (16, 24, 2, 3, 6), 
    (24, 24, 1, 3, 6), 
    (24, 40, 2, 5, 6), 
    (40, 40, 1, 5, 4), 
    (40, 40, 1, 5, 4), 
    (40, 80, 2, 3, 4), 
    (80, 80, 1, 3, 4), 
    (80, 80, 1, 3, 4), 
    (80, 96, 1, 5, 4), 
    (96, 96, 1, 5, 3), 
    (96, 128, 1, 3, 3),
    (128, 128, 1, 3, 3),
    (128, 128, 1, 3, 3)]

decoder_conv_shapes = [
    # c_in, c_out, s, k, t 
    (128, 128, 1, 3, 3),
    (128, 128, 1, 3, 3), 
    (128, 96, 1, 3, 3), 
    (96, 96, 1, 5, 3), 
    (96, 80, 1, 5, 4), 
    (80, 80, 1, 3, 4),
    (80, 80, 1, 3, 4), 
    (80, 40, 1, 3, 4), 
    (40, 40, 1, 5, 4), 
    (40, 40, 1, 5, 4),
    (40, 24, 1, 5, 6),
    (24, 24, 1, 3, 6),
    (24, 16, 1, 3, 6),
    (16, 16, 1, 3, 6),
    (16, 3, 1)]


enc_out_layers = [12, 14]
enc_out_channels = 128
#content_dir = ["temp_dataset/content/", "/media/data/code/stylegan/data/new_dataset/creeper_dataset/", "/media/data/datasets/unlabeled2017/"] 
#style_dir = ["/media/data/code/stylegan/data/new_dataset/creeper_dataset/", "/media/data/datasets/wikiart_smaller", "temp_dataset/style/"] 
# content_dir = ["temp_dataset/content/"]
# style_dir = ["temp_dataset/style/"]
# content_dir = ["temp_dataset/content_test/"]
# style_dir = ["temp_dataset/style_test/"]

content_dir = ["temp_dataset/content/", "temp_dataset/style/", "/media/data/code/stylegan/data/final_dataset_square", "/media/data/datasets/unlabeled2017/", "/media/data/datasets/landscape", "/media/data/datasets/flickr_dataest/flickr30k_images/flickr30k_images/", "/media/data/datasets/flowers", "/media/data/datasets/wikiart", "/media/data/datasets/stylegan_anime_results"] 
style_dir = ["temp_dataset/content/", "/media/data/datasets/wikiart_smaller", "temp_dataset/style/", "/media/data/datasets/landscape", "/media/data/datasets/flowers", "/media/data/datasets/stylegan_anime_results", "/media/data/datasets/wikiart"]

#content_dir = ["temp_dataset/content_test/", "temp_dataset/style_test/"] 
#style_dir = ["temp_dataset/content_test/", "temp_dataset/style_test/"]


#style_dir = ["temp_dataset/content/", "/media/data/code/stylegan/data/final_dataset_square/", "/media/data/datasets/wikiart_smaller", "temp_dataset/style/", "/media/data/datasets/landscape", "/media/data/datasets/flowers", "/media/data/datasets/wikiart", "/media/data/datasets/stylegan_anime_results"]
# style_dir = ["temp_dataset/style/", "/media/data/datasets/stylegan_anime_results"]

#style_dir = ["/media/data/code/stylegan/data/kaggle_imgs_smaller_2/", "/media/data/code/stylegan/data/portraits_smaller_4/", "/media/data/datasets/wikiart_smaller", "temp_dataset/style/"]
