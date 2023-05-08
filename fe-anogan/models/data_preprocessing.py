"""
Preprocesses data:
Loads mnist, svhn, usps as Dataset class. Data and Labels attributes are tensors with [28,28] size.
All 10 classes are used (parameter `training_label` is leftover), only parameter used is "path"

Composes class of all 3 datasets, with upsampling to the size of SVHN on USPS and MNIST.
Only separates on training and test sets
"""

import torch
from torch.utils.data import WeightedRandomSampler, DataLoader
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.utils import shuffle

class Dataset(torch.utils.data.Dataset):
    def __init__(self,data,labels, transforms=None):
        self.data = data
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]
        if self.transforms:
            data = self.transforms(data)
        return data, labels

    def __len__(self):
        return len(self.data)

def load_mnist(path):
    tf = torchvision.transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,)),
                             ])
    train = torchvision.datasets.MNIST(path, train=True, download=False, 
                            transform=tf)
    test = torchvision.datasets.MNIST(path, train=False, download=False, 
                            transform=tf)
    train_loader = DataLoader(train, batch_size=len(train))
    test_loader = DataLoader(test, batch_size=len(test))
    train = Dataset(next(iter(train_loader))[0], next(iter(train_loader))[1], transforms=tf)
    test = Dataset(next(iter(test_loader))[0], next(iter(test_loader))[1], transforms=tf)
    # normalize
    train.data = train.data/train.data.max()
    test.data = test.data/test.data.max()
    #  NOT SEQUEEZING: squeeze: NOT DONE WHEN USING CONV2D !!!
    # train.data = torch.squeeze(train.data)
    # test.data = torch.squeeze(test.data)

    return train, test

def load_svhn(path):
    split_rate = 0.8
    tf = torchvision.transforms.Compose([
                                #transforms.ToPILImage(),
                                transforms.Lambda(lambda x: transforms.functional.adjust_sharpness(x, sharpness_factor=2.0)),
                                torchvision.transforms.CenterCrop(29),
                                torchvision.transforms.Resize(28),
                                transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: transforms.functional.adjust_contrast(x, contrast_factor=2)),
                                transforms.Normalize((0.5,), (0.5,))
                                ])
    train = torchvision.datasets.SVHN(path, split='train', download=False, transform=tf)
    test = torchvision.datasets.SVHN(path, split='test', download=False, transform=tf)
    train_loader = DataLoader(train, batch_size=len(train))
    test_loader = DataLoader(test, batch_size=len(train))
    train = Dataset(next(iter(train_loader))[0], next(iter(train_loader))[1], transforms=None)
    test = Dataset(next(iter(test_loader))[0], next(iter(test_loader))[1], transforms=None)
    # squeeze: NOT DONE WHEN USEING CONV2D !!!
    # train.data = torch.squeeze(train.data, 1)
    # test.data = torch.squeeze(test.data, 1)
    # normalize
    train.data = train.data/train.data.max()
    test.data = test.data/test.data.max()
    return train, test


def open_file(type_of_file):
    import bz2
    with bz2.open(type_of_file) as fp:
        raw_data = [line.decode().split() for line in fp.readlines()]
        tmp_list = [[x.split(":")[-1] for x in data[1:]] for data in raw_data]
        imgs = np.asarray(tmp_list, dtype=np.float32).reshape((-1, 16, 16))
        imgs = ((imgs + 1) / 2 * 255).astype(dtype=np.uint8)
        targets = [int(d[0]) - 1 for d in raw_data]
        x = np.asarray(targets)
        # USE AS NUMPY ARRAYS (DIFF FROM BENCHMARKING VERSION)
        return imgs, x

##### I think should add .float() to be consistent with ENCODERS file
def load_usps(path):
    train_data, train_labels = open_file("dataset/train.bz2")
    test_data, test_labels = open_file("dataset/test.bz2")
    split_rate = 0.8
    tf = torchvision.transforms.Compose([
                                #transforms.ToPILImage(),
                                transforms.Lambda(lambda x: transforms.functional.adjust_sharpness(x, sharpness_factor=2.0)),
                                torchvision.transforms.Resize(28),
                                transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: transforms.functional.adjust_contrast(x, contrast_factor=2)),
                                transforms.Normalize((0.5,), (0.5,))])
    train = Dataset(train_data, train_labels, transforms=tf)
    test = Dataset(test_data, test_labels, transforms=tf)

    # Resize
    from skimage.transform import resize
    train_arr = [resize(train.data[i], (28,28)) for i in range(len(train.data))]
    train_torch_arr = torch.from_numpy(np.array(train_arr))
    train_torch_arr = train_torch_arr.unsqueeze(1).float() # ADD DIM=1 for conv2d

    test_arr = [resize(test.data[i], (28,28)) for i in range(len(test.data))]
    test_torch_arr = torch.from_numpy(np.array(test_arr))
    test_torch_arr = test_torch_arr.unsqueeze(1).float() # ADD DIM=1 for conv2d

    train_dataset = Dataset(train_torch_arr, torch.from_numpy(train.labels))
    test_dataset = Dataset(test_torch_arr, torch.from_numpy(test.labels))
    return train_dataset, test_dataset

def load_features(train,test):
    train_loader = DataLoader(train, batch_size=len(train))
    test_loader = DataLoader(test, batch_size=len(test))
    train = Dataset(next(iter(train_loader))[0], next(iter(train_loader))[1])
    test = Dataset(next(iter(test_loader))[0], next(iter(test_loader))[1])
    return train, test

def get_datasets(path, batch_size):
    mnist_train, mnist_test = load_mnist(path)
    print("mnist")
    svhn_train, svhn_test = load_svhn(path)
    print("svhn")
    usps_train, usps_test = load_usps(path)
    print("usps")
    #mnist_train, mnist_test = load_features(mnist_train_og, mnist_test_og)
    #svhn_train, svhn_test = load_features(svhn_train_og, svhn_test_og)
    #usps_train, usps_test = load_features(usps_train_og, usps_test_og)


    # Data Augmentation
    from sklearn.utils import resample
    mnist_train_upsample = resample(mnist_train.data.numpy(),mnist_train.labels.numpy(),
                                    replace=True, n_samples=len(svhn_train), random_state=42)
    #mnist_train_upsample = Dataset(torch.from_numpy(mnist_train_upsample[0]), torch.from_numpy(mnist_train_upsample[1]))
    
    usps_train_upsample = resample(usps_train.data.numpy(),usps_train.labels.numpy(),
                                    replace=True,n_samples=len(svhn_train), random_state=42)
    #usps_train_upsample = Dataset(torch.from_numpy(usps_train_upsample[0]), torch.from_numpy(usps_train_upsample[1]))
    
    # transform to numpy elements
    mnist_train_upsample = Dataset(mnist_train_upsample[0], mnist_train_upsample[1])
    usps_train_upsample = Dataset(usps_train_upsample[0], usps_train_upsample[1])
    svhn_train_upsample = Dataset(svhn_train.data.detach().cpu().numpy(), svhn_train.labels.detach().cpu().numpy())
    
    # Concat Datasets
    # Train
    concat_train = torch.utils.data.ConcatDataset([mnist_train_upsample, svhn_train_upsample, usps_train_upsample])
    train_loader = DataLoader(concat_train, batch_size=batch_size, shuffle=True) # we are not using sampler because we have augmented the data
    # Test
    # transform to appropriate objects
    m = Dataset(mnist_test.data.detach().cpu().numpy(), mnist_test.labels.detach().cpu().numpy())
    s = Dataset(svhn_test.data.detach().cpu().numpy(), svhn_test.labels.detach().cpu().numpy())
    u = Dataset(usps_test.data.detach().cpu().numpy(), usps_test.labels.detach().cpu().numpy())
    concat_test = torch.utils.data.ConcatDataset([m,s,u])
    test_loader = DataLoader(concat_test, batch_size=1, shuffle=False)

    return train_loader, test_loader