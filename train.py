import argparse
import os
import random
from pathlib import Path
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.data as data
import torch.nn as nn
import json


from models import *
from data_loader import *

unloader = transforms.ToPILImage()  # reconvert into PIL image

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated



# desired size of the output image

R1_LAM = 5
def r1_loss(pred_real, real_sample):
    grad_real = torch.autograd.grad(outputs=pred_real.sum(), inputs=real_sample, create_graph=True)[0]
    reg_loss = R1_LAM * grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return reg_loss


fig, axs = plt.subplots(1, 3, figsize=(15,10))
plt.ion()

class ASTTrainer:
    def __init__(self, args):
        self.args = args
        # Load models
        self.ast = AST().to(device)
        
        #self.pretrained_enc = PretrainedEncoder().to(device).eval()

        if not self.args.load:
            self.load_ae()
            
        self.ast_optim = optim.Adam(self.ast.parameters(), lr=self.args.lr, betas=[0.5, 0.99])
        # self.dis = Discriminator().to(device)
        # self.dis_optim = optim.Adam(self.dis.parameters(), lr=self.args.dis_lr, betas=[0.5, 0.99])

        num_workers = 8

        content_dataset = FlatFolderDataset(content_dir, get_transform(True))
        style_dataset = FlatFolderDataset(style_dir, get_transform(True))

        self.content_iter = iter(data.DataLoader(
            content_dataset,
            batch_size=self.args.batch_size,
            sampler = InfiniteSamplerWrapper(content_dataset),
            num_workers=num_workers))
        self.style_iter = iter(data.DataLoader(
            style_dataset, 
            batch_size=self.args.batch_size,
            sampler=InfiniteSamplerWrapper(style_dataset),
            num_workers=num_workers))

        self.train_dict = {
            "content_loss" : [],
            "style_loss" : []
            # "fake_loss" : [],
            # "dis_loss" : []
        }

        self.save_file = os.path.join(self.args.save_dir, "ast.pth")
        self.train_dict_file = os.path.join(self.args.save_dir, "ast_train_dict.json")

        if self.args.load:
            self.load()



    def save(self):
        model_dict = {
            "ast" : self.ast.state_dict(),
            "ast_optim": self.ast_optim.state_dict()
            # "dis" : self.dis.state_dict(),
            # "dis_optim" : self.dis_optim.state_dict()
        }
        
        torch.save(model_dict, self.save_file)
        
        # Save the training and validation results
        with open(self.train_dict_file, "w") as f:
            json.dump(self.train_dict, f)


    def load(self):
        model_dict = torch.load(self.save_file)
        self.ast.load_state_dict(model_dict["ast"])

        # for i in range(len(model._dec._decoder_blocks)):
        #     if decoder_conv_shapes[i][0] != decoder_conv_shapes[i][1] and i + 2 < len(decoder_conv_shapes):
        #         print("model._dec._decoder_blocks[i]", model._dec._decoder_blocks[i])
        #         print("model._dec._decoder_blocks[i]._upsample_2", model._dec._decoder_blocks[i]._upsample_2)
        #         model._dec._decoder_blocks[i]._upsample_2 = DepthWiseConv(decoder_conv_shapes[i][1], decoder_conv_shapes[i][1], 1, 1, use_norm=True).to(device)

        self.ast_optim.load_state_dict(model_dict["ast_optim"])
        # self.dis.load_state_dict(model_dict["dis"])
        # self.dis_optim.load_state_dict(model_dict["dis_optim"])

        with open(self.train_dict_file) as f:
            self.train_dict = json.load(f)

    def load_ae(self):
        """Load the pretrained auto-encoder model into the AST model."""
        model_dict = torch.load(self.args.ae_model)
        
        
        ae = AutoEncoder()
        ae.load_state_dict(model_dict["AE"])
        self.ast._enc.load_state_dict(ae.encoder.state_dict())
        self.ast._dec.load_state_dict(ae.decoder.state_dict())

    def train(self):
        print("NUM AST PARAMETERS: ", sum(p.numel() for p in self.ast.parameters()))
        #print("NUM Discriminator PARAMETERS: ", sum(p.numel() for p in self.dis.parameters()))

        for j in range(self.args.train_iter):
            content_imgs = next(self.content_iter).to(device)
            style_imgs = next(self.style_iter).to(device)
            t_cs_map, style_map, t, t_cs, content_map = self.ast(content_imgs, style_imgs)


            
            # Compute the discriminator loss
            # print("content_map[0].shape", content_map[0].shape, t.shape, style_map[-1].shape, t_cs_map[-1].shape)
            #real_sample = torch.cat((content_imgs, style_imgs), 0)
            # real_sample = content_imgs#torch.cat((content_imgs, style_imgs), 0)
            # if (j + 1) % 8 == 0:
            #     import time
            #     start_time = time.time()
            #     real_sample.requires_grad = True
            #     dis_true = self.dis(real_sample)
            #     r1_dis_loss = r1_loss(dis_true, real_sample)
            #     print("r1_dis_loss", r1_dis_loss, time.time() - start_time)
            # else:
            #     dis_true = self.dis(real_sample)
            #     r1_dis_loss = 0

            # #true_loss = discriminator_loss(dis_true, torch.ones(batch_size*2, 1, dtype=torch.float32, device=device) - 0.2)
            # true_loss = discriminator_loss(dis_true, torch.ones(self.args.batch_size, 1, dtype=torch.float32, device=device) - 0.2)
            # dis_fake = self.dis(t_cs.detach())

            # #true_loss = discriminator_loss(dis_true, torch.ones(batch_size, 1, dtype=torch.float32, device=device) - 0.2)
            # fake_loss = discriminator_loss(dis_fake, torch.zeros(self.args.batch_size, 1, device=device))
            
            # dis_loss = true_loss + fake_loss + r1_dis_loss
            # self.dis_optim.zero_grad()
            # dis_loss.backward()

            # self.dis_optim.step()
            
            # self.train_dict["dis_loss"].append(dis_loss.detach().item())

            
            t_cs_map, style_map, t, t_cs, content_map = self.ast(content_imgs, style_imgs)
            #dis_fake = self.dis(t_cs)

            #fake_loss = discriminator_loss(dis_fake, torch.ones(self.args.batch_size, 1, device=device))
            #print("t_cs_map[-1].shape", t_cs_map[-1].shape, t.shape)
            
            # print("t_cs_map[-1]", t_cs_map[-1])
            # print("t", t)
            # print("t_cs_map[-1]", t_cs_map[-1].mean(), t_cs_map[-1].std())
            # print("t", t.m+ean(), t.std())
            # print("content_map[0]", content_map[0].mean(), content_map[0].std())
            
            #content_loss = compute_content_loss(t_cs_map[-1], t)
            #content_loss += compute_content_loss(t_cs_map[-1], content_map[0]) * 0.01
            
            # Compute content loss
            for i in range(len(t_cs_map)):
                if i == 0:
                    content_loss = compute_content_loss(t_cs_map[i], content_map[i].detach())
                else:
                    content_loss = content_loss + compute_content_loss(t_cs_map[i], content_map[i].detach()) 


            # Compute style loss
            for i in range(len(t_cs_map)):

                if i == 0:
                    style_loss = compute_style_loss(t_cs_map[i], style_map[i].detach())
                else:
                    style_loss = style_loss + compute_style_loss(t_cs_map[i], style_map[i].detach())


            #print("adversarial loss", self.args.dis_lam  * fake_loss)

            loss = self.args.content_lam * content_loss  + self.args.style_lam * style_loss #+ self.args.dis_lam  * fake_loss

            self.ast_optim.zero_grad()
            loss.backward()
            #nn.utils.clip_grad.clip_grad_norm_(ast.parameters(), 1.0)
            # print(ast._dec._decoder_blocks[3]._conv._layers[0].weight.grad)
            
            # print(ast._dec._img_out.weight.grad)
            self.ast_optim.step()

            self.train_dict["content_loss"].append(content_loss.detach().item())
            self.train_dict["style_loss"].append(style_loss.detach().item())
            #self.train_dict["fake_loss"].append(fake_loss.detach().item())
            


            if (j +1) % 16 == 0:
                print("content_loss", self.args.content_lam * content_loss)
                print("style_loss", self.args.style_lam * style_loss)
                
                # print("dis_true", dis_true)
                # print("dis_fake", dis_fake)
                self.save()
                # print("content_loss", content_loss)
                # print("style_loss", style_loss * 0.5)
                # print("dis_loss", dis_loss)
                # print("fake_loss", fake_loss * 1e-2)
                # print("hist_loss", hist_loss * 100000)

                # print("content_img.max()", content_img.max())
                # print("content_img.min()", content_img.min())
                # print("style_img.max()", style_img.max())
                # print("style_img.min()", style_img.min())
                # print("t_cs[0].max()", t_cs[0].max())
                # print("t_cs[0].min()", t_cs[0].min())
                
                # print("content_img", style_img)
                
                # content_file = "temp_dataset/content_test/shoto.jpg"
                #content_file = "temp_dataset/content_test/dog.jpg"
                content_file = "temp_dataset/content/aang.jpg"
                #content_file = "temp_dataset/content_test/IMG_2836.JPG"
                
                
                #style_file = "temp_dataset/style_test/336a26f4e3c09f31f3262875fa13faa8c.jpg"
                # style_file = "temp_dataset/style_test/sketch.jpg"
                #style_file = "temp_dataset/style_test/fractal_style_6.jpg"
                #style_file = "temp_dataset/content_test/IMG_2836.JPG"
                #style_file = "temp_dataset/content_test/zoro.jpg"
                # style_file = "temp_dataset/style/vaporwave_style_4.jpeg"
                #style_file = "temp_dataset/style_test/thumb-1920-1180547.png"
                style_file = "temp_dataset/style/blue_and_red_fire_style.jpg"
                style_img = image_loader(style_file)
                content_img = image_loader(content_file)
                # with torch.no_grad():
                #     _, _, _, t_cs, content_map = ast(content_img, style_img, alpha=0.75)

                axs[0].imshow(content_imgs[0].detach().cpu().permute((1,2,0)))
                axs[1].imshow(style_imgs[0].detach().cpu().permute((1,2,0)))
                axs[2].imshow(t_cs[0].detach().cpu().permute((1,2,0)))
                #axs[0].imshow(content_img.detach().cpu()[0].permute((1,2,0)))

                # axs[0].imshow(t_cs[0].detach().cpu().permute((1,2,0)))
                # with torch.no_grad():
                #     _, _, _, t_cs, content_map = ast(content_img, style_img)
                
                # print("t_cs[0].max()", t_cs[0].max())
                # print("t_cs[0].min()", t_cs[0].min())
                # axs[1].imshow(t_cs[0].detach().cpu().permute((1,2,0)))
                
                
                # with torch.no_grad():
                #     _, _, _, t_cs, content_map = ast(content_img, style_img, alpha=0.1)
                

                plt.draw()
                plt.pause(0.01)
                plt.show()
                
                # plt.pause(0.1)


def main(args):
    trainer = ASTTrainer(args)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_iter", type=int, default=2048000,
            help="Number of train iteration (batches of examples).")
    parser.add_argument("--batch_size", type=int, default=16,
            help="Number of train iteration (batches of examples).")
    parser.add_argument("--lr", type=float, default=2e-4,
            help="Learning rate.")
    parser.add_argument("--dis_lr", type=float, default=1e-5,
            help="Learning rate for the discriminator.")
    parser.add_argument("--dis_lam", type=float, default=1e-3,
            help="Weight for discriminator.")
    parser.add_argument("--content_lam", type=float, default=1e-3,
            help="Weight for content loss.")
    parser.add_argument("--style_lam", type=float, default=1e-3,
            help="Weight for content loss.")
    parser.add_argument("--r1_lam", type=float, default=5.0,
            help="Weight for content loss.")
    parser.add_argument("--save_dir", default="models/ast/",
            help="Directory to save the model.")
    parser.add_argument("--ae_model", default="models/auto_encoder/ae.pth",
            help="Directory to save the model.")
    parser.add_argument("--load", action="store_true",
            help="Load model.")
    parser.add_argument("--recon_lam", type=float, default=100.0,
            help="Reconstruction loss weight.")
    parser.add_argument("--perp_lam", type=float, default=0.01,
            help="Reconstruction loss weight.")


    main(parser.parse_args())

