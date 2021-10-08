def channel_stats(img):
    img_mean = img.mean(dim=(2,3), keepdim=True)
    img_std = img.std(dim=(2,3), keepdim=True)
    # print("img_std", img_std)
    # print("img_mean", img_mean)
    return img_mean, img_std
