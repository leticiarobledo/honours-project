import torch
from torch import nn  # All neural network modules
import numpy as np
import torchvision.models as models
import torch.nn as nn
import sklearn.decomposition

import torch.nn as nn
import torch.nn.functional as F

import resnet_5_layers as largerRN
import resnet_default

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

########################
####    LE NET      ####
########################
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=4,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=4,
            stride=1,
            padding=0,
        )
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=144,
            kernel_size=4,
            stride=1,
            padding=0,
        )
        self.linear1 = nn.Linear(144, 84)
        self.linear2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(
            self.conv3(x)
        )  # num_examples x 120 x 1 x 1 --> num_examples x 120
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    print("Loaded...")

def get_features(train_dataloader):
    model = LeNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_checkpoint(torch.load("weights/lenet_model.pth.tar"), model)
    model.eval()
    model.to(device)
    model.linear1 = nn.Identity()
    model.linear2 = nn.Identity()
    all_predictions = []
    for x, y in train_dataloader:
        x = model(x)
        i = x.detach().cpu().numpy()
        res = np.reshape(i, (12,12))
        all_predictions.append(res)
    return torch.as_tensor(np.asarray(all_predictions, dtype=np.float32))

def get_features_pca(train_dataloader):
    images_all, labels = next(iter(train_dataloader))
    images_flat = images_all[:, 0].reshape(-1, 784).numpy()
    pca = sklearn.decomposition.PCA(n_components=150)
    images_flat_hat = pca.inverse_transform(pca.fit_transform(images_flat))
    images_flat_hat = images_flat_hat.reshape(-1, 28,28)
    return images_flat_hat, labels
    
def get_features_resnet18(train_dataloader, model_weights):
    model = resnet_default.resnet18()
    load_checkpoint(torch.load(model_weights, map_location ='cpu'), model)
    model.fc = nn.Identity(in_features=512, out_features=512)
    model.eval()
    all_predictions = []
    for j, _ in train_dataloader:
        #print(j.shape)
        x = model(j)
        #print(x[0].shape)
        ##########################
        e = torch.flatten(x[0]) #   grad_fn=<ViewBackward0>)
        ##########################
        i = e.detach().cpu().numpy()
        # i = i/i.max() # these are not normalized
        i = i.reshape(-1,16,16)
        #print(i.shape)
        all_predictions.append(i)
    #print(len(all_predictions))
    return torch.as_tensor(np.asarray(all_predictions, dtype=np.float32))

def get_features_resnet50(train_dataloader, model_weights):
    model = models.resnet50(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model.avgpool = nn.Identity()

    load_checkpoint(torch.load(model_weights, map_location ='cpu'), model)
    model.fc = nn.Identity()

    model.eval()
    all_predictions = []
    with torch.no_grad():
        for j, _ in train_dataloader:
            x = model(j)
            # print(x[0].shape)
            ##########################
            e = torch.flatten(x[0])
            # print("e:")
            # print(e.shape)
            ##########################
            i = e.detach().cpu().numpy()
            res = np.reshape(i, (-1,32,32))
            #print("res:")
            #print(res.shape)
            # print(len(res))
            all_predictions.append(res)
        #print(len(all_predictions))
    print("inference OK")
    return torch.as_tensor(np.asarray(all_predictions, dtype=np.float32))


