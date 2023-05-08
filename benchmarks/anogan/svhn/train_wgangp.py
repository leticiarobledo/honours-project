import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from fanogan.train_wgangp import train_wgangp

from model import Generator, Discriminator
from tools import SimpleDataset, load_svhn
import torchvision


def main(opt):
    if type(opt.seed) is int:
        torch.manual_seed(opt.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    (x_train, y_train), _ = load_svhn("dataset",
                                       training_label=opt.training_label,
                                       split_rate=opt.split_rate)
    tf = torchvision.transforms.Compose([
                                # transforms.ToPILImage(), # only used in the Anogan part
                                transforms.Resize((28, 28), 
                                interpolation=transforms.InterpolationMode.BICUBIC),
                                transforms.Lambda(lambda x: transforms.functional.adjust_sharpness(x, sharpness_factor=2.0)),
                                #torchvision.transforms.CenterCrop(29),
                                #torchvision.transforms.Resize(28),
                                transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: transforms.functional.adjust_contrast(x, contrast_factor=2)),
                                transforms.Normalize((0.5,), (0.5,))
                                ])
    train = SimpleDataset(x_train, y_train,transform=tf)
    train_dataloader = DataLoader(train, batch_size=opt.batch_size,
                                  shuffle=True)

    generator = Generator(opt)
    discriminator = Discriminator(opt)

    train_wgangp(opt, generator, discriminator, train_dataloader, device)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
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