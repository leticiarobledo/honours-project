import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.model_zoo import tqdm

from model import Generator, Discriminator, Encoder
import numpy as np
from models_encoder import get_features, get_features_pca, get_features_resnet18
from tools import load_svhn, load_usps, load_mnist, get_datasets_full_length_loader, Dataset
from data_preprocessing import get_datasets


def test_anomaly_detection(opt, generator, discriminator, encoder,
                           dataloader, device, kappa=1.0):
    print("started test")
    generator.load_state_dict(torch.load("results/generator"))
    discriminator.load_state_dict(torch.load("results/discriminator"))
    encoder.load_state_dict(torch.load("results/encoder"))

    generator.to(device).eval()
    discriminator.to(device).eval()
    encoder.to(device).eval()

    criterion = nn.MSELoss()
    if opt.type == "skip":
        path = "results/"+ str(opt.training_label) + "_" + str(opt.n_epochs) +"_score.csv"
    else:
        path = "results/"+ opt.type + "_" + opt.fe + "/" + str(opt.training_label) + "_" + str(opt.n_epochs) +"_score.csv"
    
    with open(path, "w") as f:
        f.write("label,img_distance,anomaly_score,z_distance\n")

    for (img, label) in tqdm(dataloader):

        real_img = img.to(device)

        real_z = encoder(real_img)
        fake_img = generator(real_z)
        fake_z = encoder(fake_img)

        real_feature = discriminator.forward_features(real_img)
        fake_feature = discriminator.forward_features(fake_img)

        # Scores for anomaly detection
        img_distance = criterion(fake_img, real_img)
        loss_feature = criterion(fake_feature, real_feature)
        anomaly_score = img_distance + kappa * loss_feature

        z_distance = criterion(fake_z, real_z)

        with open(path, "a") as f:
            f.write(f"{label.item()},{img_distance},"
                    f"{anomaly_score},{z_distance}\n")


def main(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if opt.type == "full_batching":
        _, test_dataloader = get_datasets("dataset", opt.batch_size)

    # load data
    if opt.type == "mnist":
        _, test_dataset = load_mnist("dataset", training_label=opt.training_label)
    elif opt.type == "svhn":
        _, test_dataset = load_svhn("dataset", training_label=opt.training_label)
    elif opt.type == "usps":
        _, test_dataset = load_usps("dataset", training_label=opt.training_label)

    if opt.type != "skip" and opt.type != "full_batching":
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # get features
    if opt.fe == "lenet":
        torch_features = get_features(test_dataloader)
        unsqueezed_features = torch_features.unsqueeze(1)
        # match features to original labels
        features = Dataset(unsqueezed_features, test_dataset.labels)
        features_dataloader = DataLoader(features, batch_size=1, shuffle=False)
    elif opt.fe == "resnet18":
        torch_features = get_features_resnet18(test_dataloader, "weights/resnet18_default_val_no_svhn.pth.tar")
        #unsqueezed_features = torch_features.unsqueeze(1)
        features = Dataset(torch_features, test_dataset.labels)
        features_dataloader = DataLoader(features, batch_size=1, shuffle=False)
    elif opt.fe == "resnet50":
        print("resnet50")
        torch_features = get_features_resnet18(test_dataloader,"weights/resnet50_pretrained_val.pth.tar")
        # unsqueezed_features = torch_features.unsqueeze(1)
        features = Dataset(torch_features, test_dataset.labels)
        features_dataloader = DataLoader(features, batch_size=1, shuffle=False)
    elif opt.fe == "pca":
        _, test_loader = get_datasets_full_length_loader("dataset", training_label=opt.training_label)
        features,labels = get_features_pca(test_loader)
        new_f = torch.as_tensor(np.asarray([x for x in features], dtype=np.float32))
        features_dataset = Dataset(new_f, labels)#, transform=transforms.ToTensor())
        features_dataset.data = features_dataset.data.unsqueeze(1)
        features_dataloader = DataLoader(features_dataset, batch_size=1, shuffle=False)
    
    generator = Generator(opt)
    discriminator = Discriminator(opt)
    encoder = Encoder(opt)

    if opt.fe is not None:
        print("start with featues")
        #features_dataloader = torch.load("features_dataloader.pth")
        print("train loaded")
        test_anomaly_detection(opt, generator, discriminator, encoder,
                           features_dataloader, device)
    #else:
    #    test_anomaly_detection(opt, generator, discriminator, encoder,test_dataloader, device)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fe", type=str, default=None,
                        help="name of encoder")
    parser.add_argument("--type", type=str, default="None",
                        help="name of data")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1,
                        help="number of image channels")
    parser.add_argument("--training_label", type=int, default=0,
                        help="label for normal images")
    parser.add_argument("--split_rate", type=float, default=0.8,
                        help="rate of split for normal training data")
    parser.add_argument("--name", type=str, default="None",
                        help="name of experiment")
    parser.add_argument("--n_epochs", type=int, default=1,
                        help="naum of epochs")
    opt = parser.parse_args()

    main(opt)