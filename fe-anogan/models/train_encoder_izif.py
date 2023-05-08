import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import torch.nn as nn
import numpy as np

from models_encoder import *
from model import Generator, Discriminator, Encoder
from tools import load_svhn, load_usps, load_mnist, get_datasets_full_length_loader, Dataset
from data_preprocessing import get_datasets



def main(opt):
    if type(opt.seed) is int:
        torch.manual_seed(opt.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if opt.type == "full_batching":
        train_dataloader, _ = get_datasets("dataset", opt.batch_size)

    # load data
    if opt.type == "mnist":
        train_dataset, _ = load_mnist("dataset", training_label=opt.training_label)
    elif opt.type == "svhn":
        train_dataset, _ = load_svhn("dataset", training_label=opt.training_label)
        print(len(train_dataset))
    elif opt.type == "usps":
        train_dataset, _ = load_usps("dataset", training_label=opt.training_label)

    if opt.type != "skip" and opt.type != "full_batching":
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
   
    # get features
    if opt.fe == "lenet":
        torch_features = get_features(train_dataloader)
        unsqueezed_features = torch_features.unsqueeze(1)
        features_dataloader = DataLoader(unsqueezed_features, batch_size=opt.batch_size, shuffle=True)
    elif opt.fe == "resnet18":
        torch_features = get_features_resnet18(train_dataloader, "weights/resnet18_default_val_no_svhn.pth.tar")
        # unsqueezed_features = torch_features.unsqueeze(1)
        features_dataloader = DataLoader(torch_features, batch_size=opt.batch_size, shuffle=True)
    elif opt.fe == "resnet50":
        torch_features = get_features_resnet50(train_dataloader, "weights/resnet50_pretrained_val.pth.tar")
        # unsqueezed_features = torch_features.unsqueeze(1)
        features_dataloader = DataLoader(torch_features, batch_size=opt.batch_size, shuffle=True)
        # print("dataloader OK")
    elif opt.fe == "pca":
        train_loader, _ = get_datasets_full_length_loader("dataset", training_label=opt.training_label)
        features,labels = get_features_pca(train_loader)
        new_f = torch.as_tensor(np.asarray([x for x in features], dtype=np.float32))
        features_dataset = Dataset(new_f, labels)#, transform=transforms.ToTensor())
        features_dataset.data = features_dataset.data.unsqueeze(1)
        features_dataloader = DataLoader(features_dataset.data, batch_size=opt.batch_size, shuffle=True)
     
    generator = Generator(opt)
    discriminator = Discriminator(opt)
    encoder = Encoder(opt)

    if opt.fe is not None: 
        # features_dataloader = torch.save("train_features_dataloader.pth")
        train_encoder_izif(opt, generator, discriminator, encoder, features_dataloader, device)
    #else:
    #    train_encoder_izif(opt, generator, discriminator, encoder, train_dataloader, device)


def train_encoder_izif(opt, generator, discriminator, encoder,
                       dataloader, device, kappa=1.0):
    generator.load_state_dict(torch.load("results/generator"))
    discriminator.load_state_dict(torch.load("results/discriminator"))

    generator.to(device).eval()
    discriminator.to(device).eval()
    encoder.to(device)

    criterion = nn.MSELoss()

    optimizer_E = torch.optim.Adam(encoder.parameters(),
                                   lr=opt.lr, betas=(opt.b1, opt.b2))

    os.makedirs("results/images_e", exist_ok=True)

    padding_epoch = len(str(opt.n_epochs))
    padding_i = len(str(len(dataloader)))

    batches_done = 0
    for epoch in range(opt.n_epochs):
        for i, imgs in enumerate(dataloader):

            # Configure input
            real_imgs = imgs.to(device)
            optimizer_E.zero_grad()

            z = encoder(real_imgs)
            fake_imgs = generator(z)

            real_features = discriminator.forward_features(real_imgs)
            fake_features = discriminator.forward_features(fake_imgs)

            loss_imgs = criterion(fake_imgs, real_imgs)
            loss_features = criterion(fake_features, real_features)
            e_loss = loss_imgs + kappa * loss_features

            e_loss.backward()
            optimizer_E.step()

            if i % opt.n_critic == 0:
                batches_done += opt.n_critic
    torch.save(encoder.state_dict(), "results/encoder")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fe", type=str, default=None,
                        help="name of encoder")
    parser.add_argument("--type", type=str, default="None",
                        help="name of data")
    parser.add_argument("--n_epochs", type=int, default=200,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1,
                        help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="number of training steps for "
                             "discriminator per iter")
    parser.add_argument("--sample_interval", type=int, default=400,
                        help="interval betwen image samples")
    parser.add_argument("--training_label", type=int, default=0,
                        help="label for normal images")
    parser.add_argument("--split_rate", type=float, default=0.8,
                        help="rate of split for normal training data")
    parser.add_argument("--seed", type=int, default=None,
                        help="value of a random seed")
    parser.add_argument("--name", type=str, default="None",
                        help="name of experiment")
    opt = parser.parse_args()

    main(opt)
