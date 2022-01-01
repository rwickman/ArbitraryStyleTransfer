
import argparse, os
import torch
import torch.optim as optim
import torch.nn as nn
import json
import matplotlib.pyplot as plt
import numpy as np

from data_loader import *
from models import AutoEncoder, Encoder, PretrainedEncoder
from conf import *
from losses import compute_content_loss


class AutoencoderTrainer:
    def __init__(self, args, content_iter, val_loader, content_layers=[2,3,4.5]):
        self.args = args
        self.val_loader = val_loader
        self.content_iter = content_iter
        self.content_layers = content_layers
        
        self.model = AutoEncoder().to(device)
        self.pretrained_mobnet = PretrainedEncoder().to(device).eval()
        self.ae_optim = optim.Adam(self.model.parameters(), lr=self.args.lr)#, betas=[0.5, 0.99])
        
        self.loss_fn = nn.HuberLoss()

        self.save_file = os.path.join(self.args.save_dir, "ae.pth")
        self.train_dict_file = os.path.join(self.args.save_dir, "train_dict.json")
        self.train_dict = {
            "train_loss" : [],
            "val_loss": [],
            "perp_loss" : []
        }
        print("NUM AutoEncoder PARAMETERS: ", sum(p.numel() for p in self.model.parameters()))
        
        if self.args.load:
            self.load()
            self.ae_optim.param_groups[0]["betas"] = (0.8, 0.9)
            self.ae_optim.param_groups[0]["lr"] = self.args.lr
            print("self.ae_optim", self.ae_optim)

    def save(self):
        """Save the AutoEncoder and optimizer."""
        
        # Save the model
        model_dict = {
            "AE" : self.model.state_dict(),
            "optim" : self.ae_optim.state_dict()
        }
            
        torch.save(model_dict, self.save_file)

        
        # Save the training and validation results
        with open(self.train_dict_file, "w") as f:
            json.dump(self.train_dict, f)



    def load(self):
        """Load the trained auto-encoder."""
        model_dict = torch.load(self.save_file)

        self.model.load_state_dict(model_dict["AE"])
        self.ae_optim.load_state_dict(model_dict["optim"])
        
        with open(self.train_dict_file) as f:
            self.train_dict = json.load(f)

    def validate(self):
        total_val_loss = 0
        
        val_imgs = next(self.val_loader).to(device)
        self.model.eval()
        recon_imgs = self.model(val_imgs)
        total_val_loss += (val_imgs - recon_imgs).abs().mean()
        
        self.train_dict["val_loss"].append(total_val_loss.item() / val_imgs.shape[0])
        self.model.train()


    def train(self):
        fig, axs = plt.subplots(1, 2, figsize=(16,6))
        plt.ion()
        for cur_iter in range(self.args.train_iter):
            if (cur_iter + 1) % 16  == 0:
                self.save()
                print(f"Number of images trained on { self.args.batch_size * cur_iter}")
                axs[0].imshow(content_imgs[0].detach().cpu().permute((1,2,0)))
                axs[1].imshow(recon_imgs[0].detach().cpu().permute((1,2,0)))
                plt.draw()
                plt.pause(0.01)
                plt.show()
                
                self.validate()


            
            content_imgs = next(self.content_iter).to(device)
            recon_imgs = self.model(content_imgs.clone())
            self.ae_optim.zero_grad()
            recon_loss = self.loss_fn(recon_imgs, content_imgs)
            
            # Get perceptual loss 
            content_maps = self.pretrained_mobnet(content_imgs)
            recon_maps = self.pretrained_mobnet(recon_imgs)
            #print("len(content_maps)", len(content_maps))
            

            for i in range(len(content_maps)):
                #print(f"content_maps: range: [{content_maps[i].min()},{content_maps[i].max()}], mean and std, {content_maps[i].mean()}, {content_maps[i].std()}")
                #print(f"recon_maps: range: [{recon_maps[i].min()},{recon_maps[i].max()}], mean and std, {recon_maps[i].mean()}, {recon_maps[i].std()}")
                # if i < 2:
                #     content_weight = 0.1
                # else:
                #     content_weight = 1.0
                content_weight = 1.0

                if i == 0:
                    content_loss = compute_content_loss(content_maps[i].detach(), recon_maps[i]) * content_weight
                else:
                    content_loss = content_loss + compute_content_loss(content_maps[i].detach(), recon_maps[i]) * content_weight
            
            #print(content_loss, recon_loss)
            self.train_dict["train_loss"].append(recon_loss.item() )
            self.train_dict["perp_loss"].append(content_loss.item())
            print("recon_loss", recon_loss * self.args.recon_lam)
            print("content_loss", content_loss * self.args.perp_lam)
            loss = self.args.recon_lam * recon_loss + self.args.perp_lam * content_loss
            loss.backward()

            # print(self.model.encoder.mob_net[0][0].weight.grad.max(), self.model.encoder.mob_net[0][0].weight.grad.min(), self.model.encoder.mob_net[0][0].weight.grad.mean(), self.model.encoder.mob_net[0][0].weight.grad.std())
            # print(self.model.encoder.mob_net[3]._layers[3], self.model.encoder.mob_net[3]._layers[3].pos_enc.grad.max())

            self.ae_optim.step()

    def get_distr(self, num_samples=16):
        enc_sum = None
        self.model.eval()
        for i in range(num_samples):
            content_imgs = next(self.content_iter).to(device)
            content_imgs.requires_grad_(False)
            enc_imgs = self.model.encoder(content_imgs, auto_enc=True)

            if enc_sum is None:
                enc_sum = enc_imgs.sum(axis=0)
            else:
                enc_sum  = enc_sum + enc_imgs.sum(axis=0)

        enc_mean = (enc_sum / (self.args.batch_size * num_samples)).sum(axis=0)

        
        return enc_mean

    def interpolate(self, img_1, img_2, alpha=0.5):
        img_enc_1 = self.model.encoder(img_1, auto_enc=True)
        img_enc_2 = self.model.encoder(img_2, auto_enc=True)

        # Interpolate the encodings
        print("img_enc_1")
        img_enc_inter = alpha * img_enc_1 + (1-alpha) * img_enc_2

        # Decode the interpolation
        img_inter = self.model.decoder(img_enc_inter)

        return img_inter



def main(args):
    content_dir = ["/media/data/code/stylegan/data/kaggle_imgs_smaller_2/", "/media/data/code/stylegan/data/portraits_smaller_4/", "/media/data/datasets/unlabeled2017/", "/media/data/datasets/landscape", "/media/data/datasets/flickr_dataest/flickr30k_images/flickr30k_images/"] 
    style_dir = ["/media/data/code/stylegan/data/kaggle_imgs_smaller_2/", "/media/data/code/stylegan/data/portraits_smaller_4/", "/media/data/datasets/wikiart_smaller", "temp_dataset/style/", "/media/data/datasets/landscape"]
    content_dir = content_dir + style_dir

    content_dataset = FlatFolderDataset(content_dir, get_transform(True))


    content_iter = iter(data.DataLoader(
        content_dataset,
        batch_size=args.batch_size,
        sampler = InfiniteSamplerWrapper(content_dataset),
        num_workers=8))

    val_dir = ["temp_dataset/content/", "temp_dataset/content_test/", "temp_dataset/style_test/"]
    val_dataset = FlatFolderDataset(val_dir, get_transform(False))
    #val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = iter(data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler = InfiniteSamplerWrapper(val_dataset),
        num_workers=8))
    trainer = AutoencoderTrainer(args, content_iter, val_loader)
    trainer.train()
    
    # enc_mean = trainer.get_distr()

    # enc_mean = enc_mean.flatten().detach().cpu().numpy()
    # enc_cov = np.identity(enc_mean.shape[0])

    # sampled_imgs = np.random.multivariate_normal(enc_mean.flatten(), enc_cov, size = 3*160)
    # sampled_imgs = torch.tensor(sampled_imgs)
    # sampled_imgs = sampled_imgs.reshape(3, 160, 8, 8)
    # sampled_imgs = trainer.model.decoder(sampled_imgs.float().to(device))
    # #print(sampled_imgs.shape)
    # fig, axs = plt.subplots(1, 3)

    # axs[0].imshow(sampled_imgs[0].detach().cpu().squeeze(0).permute((1,2,0)))
    # axs[1].imshow(sampled_imgs[1].detach().cpu().squeeze(0).permute((1,2,0)))
    # axs[2].imshow(sampled_imgs[2].detach().cpu().squeeze(0).permute((1,2,0)))
    # plt.show()


    # import numpy as np
    # from PIL import Image
    # t = get_transform(False)
    # img_1 = t(Image.open("temp_dataset/content/aang.jpg")).to(device).unsqueeze(0)
    # img_2 = t(Image.open("temp_dataset/content/Boruto.jpg")).to(device).unsqueeze(0)
    # print(img_1.min(), img_1.max(), img_1.shape)
    
    # fig, axs = plt.subplots(1, 3)
    # inter_img_1 = trainer.interpolate(img_1, img_2, 1.0)
    # inter_img_2 = trainer.interpolate(img_1, img_2, 0.5)
    # inter_img_3 = trainer.interpolate(img_1, img_2, 0.0)

    # axs[0].imshow(inter_img_1.detach().cpu().squeeze(0).permute((1,2,0)))
    # axs[1].imshow(inter_img_2.detach().cpu().squeeze(0).permute((1,2,0)))
    # axs[2].imshow(inter_img_3.detach().cpu().squeeze(0).permute((1,2,0)))
    # plt.show()
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_iter", type=int, default=2048,
            help="Number of train iteration (batches of examples).")
    parser.add_argument("--batch_size", type=int, default=16,
            help="Number of train iteration (batches of examples).")
    parser.add_argument("--lr", type=float, default=2e-4,
            help="Learning rate.")
    parser.add_argument("--save_dir", default="models/auto_encoder/",
            help="Directory to save the model.")
    parser.add_argument("--load", action="store_true",
            help="Load model.")
    parser.add_argument("--recon_lam", type=float, default=100.0,
            help="Reconstruction loss weight.")
    parser.add_argument("--perp_lam", type=float, default=0.01,
            help="Reconstruction loss weight.")


    main(parser.parse_args())

