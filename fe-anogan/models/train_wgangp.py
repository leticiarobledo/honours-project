import torch
import os
import torch.autograd as autograd
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models_encoder import *
from model import Generator, Discriminator
from tools import load_svhn, load_usps, load_mnist, get_datasets_full_length_loader, Dataset
from data_preprocessing import get_datasets


def get_features_resnet18_pretrained(train_dataloader):
    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Identity()
    for param in model.parameters(): param.requires_grad = False
    model.eval()
    from tqdm.autonotebook import tqdm
    features = []
    for batch in tqdm(iter(train_dataloader), total=len(train_dataloader)):
        x, y = batch
        feat = model(x)
        reshaped = feat.reshape(-1,16,16)
        img_size = 16
        features.extend(reshaped)
    return img_size, torch.as_tensor(np.asarray(features, dtype=np.float32))


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

    if opt.fe is not None: 
        #features_dataloader = torch.load("train_features_dataloader.pth")
        train_wgangp(opt, generator, discriminator, features_dataloader, device)
    #else:
    #   train_wgangp(opt, generator, discriminator, train_dataloader, device)


def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand(*real_samples.shape[:2], 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    d_interpolates = D(interpolates)
    fake = torch.ones(*d_interpolates.shape, device=device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=fake, create_graph=True,
                              retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.shape[0], -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_wgangp(opt, generator, discriminator,
                 dataloader, device, lambda_gp=10):
    generator.to(device)
    discriminator.to(device)
    print("pushed to device")
    optimizer_G = torch.optim.Adam(generator.parameters(),
                                   lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(),
                                   lr=opt.lr, betas=(opt.b1, opt.b2))

    os.makedirs("results/images", exist_ok=True)

    padding_epoch = len(str(opt.n_epochs))
    padding_i = len(str(len(dataloader)))
    batches_done = 0
    for epoch in range(opt.n_epochs):
        print("epoch: " + str(epoch))
        for i, imgs in enumerate(dataloader):

            real_imgs = imgs.to(device)
            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = torch.randn(imgs.shape[0], opt.latent_dim, device=device)

            fake_imgs = generator(z)
            real_validity = discriminator(real_imgs)
            fake_validity = discriminator(fake_imgs.detach())
            gradient_penalty = compute_gradient_penalty(discriminator,
                                                        real_imgs.data,
                                                        fake_imgs.data,
                                                        device)
            # Adversarial loss
            d_loss = (-torch.mean(real_validity) + torch.mean(fake_validity)
                      + lambda_gp * gradient_penalty)

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator and output log every n_critic steps
            if i % opt.n_critic == 0:
                fake_imgs = generator(z)
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

                batches_done += opt.n_critic

    torch.save(generator.state_dict(), "results/generator")
    torch.save(discriminator.state_dict(), "results/discriminator")




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
