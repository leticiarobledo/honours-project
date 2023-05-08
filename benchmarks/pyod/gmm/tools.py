import torch
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.utils import shuffle


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.transform = transform
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]

        if self.transform:
            data = self.transform(data)

        return data, labels

    def __len__(self):
        return len(self.data)

def split_mnist(train, test, training_label, split_rate):
    _x_train = train.data[train.targets == training_label]
    x_train, x_test_normal = _x_train.split((int(len(_x_train) * split_rate)),
                                            dim=0)
    _y_train = train.targets[train.targets == training_label]
    y_train, y_test_normal = _y_train.split((int(len(_y_train) * split_rate)),
                                            dim=0)
    x_test = torch.cat([x_test_normal,
                        train.data[train.targets != training_label],
                        test.data], dim=0)
    y_test = torch.cat([y_test_normal,
                        train.targets[train.targets != training_label],
                        test.targets], dim=0)
    return x_train, y_train, x_test, y_test

def normalize_and_shuffle(x_train, y_train, x_test, y_test):
    x_train = x_train/x_train.max()
    x_test = x_test/x_test.max()
    shuffled_x_train, shuffled_y_train = shuffle(x_train.detach().cpu().numpy(), 
                                                y_train.detach().cpu().numpy())
    shuffled_x_test, shuffled_y_test = shuffle(x_test.detach().cpu().numpy(),
                                                y_test.detach().cpu().numpy())
    x_train, y_train = torch.from_numpy(shuffled_x_train), torch.from_numpy(shuffled_y_train)
    x_test, y_test = torch.from_numpy(shuffled_x_test), torch.from_numpy(shuffled_y_test)
    return x_train, y_train, x_test, y_test

def binarize(x_train, y_train, x_test, y_test, training_label):
    # # PyOD uses “0” to represent inliers and “1” to represent outliers
    for i in range(len(y_train)):
        y_train[i] = 0

    for entry in range(len(y_test)):
        # inlier = 0
        if y_test[entry] == training_label:
            y_test[entry] = 0
        else:
            # outlier
            y_test[entry] = 1
    
    return x_train, y_train, x_test, y_test

def split_svhn(train, test, training_label, split_rate):
    _x_train = train.data[train.labels == training_label]
    x_train, x_test_normal = _x_train.split((int(len(_x_train) * split_rate)),
                                            dim=0)
    _y_train = train.labels[train.labels == training_label]
    y_train, y_test_normal = _y_train.split((int(len(_y_train) * split_rate)),
                                            dim=0)
    x_test = torch.cat([x_test_normal,
                        train.data[train.labels != training_label],
                        test.data], dim=0)
    y_test = torch.cat([y_test_normal,
                        train.labels[train.labels != training_label],
                        test.labels], dim=0)
    return x_train, y_train, x_test, y_test

def get_grayscale(x_train, y_train, x_test, y_test):
    x_train = torch.squeeze(torchvision.transforms.functional.rgb_to_grayscale(x_train, num_output_channels=1), 1)
    x_test = torch.squeeze(torchvision.transforms.functional.rgb_to_grayscale(x_test, num_output_channels=1), 1)
    return x_train, y_train, x_test, y_test

def load_mnist(path, training_label, download=False, benchmarking=False):
    train = datasets.MNIST(path, train=True, download=download, 
                            transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
    test = datasets.MNIST(path, train=False, download=download, 
                            transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
    
    x_train, y_train, x_test, y_test = split_mnist(train, test, 
                                        training_label=training_label, split_rate=0.8)
    x_train, y_train, x_test, y_test = normalize_and_shuffle(x_train, y_train, x_test, y_test)
    # binarize values
    x_train, y_train, x_test, y_test = binarize(x_train, y_train, x_test, 
                                                y_test, training_label=training_label)

    if benchmarking: # i dont think we're using this anymore
        x_test = x_test.detach().cpu().numpy()
        x_train = x_train.detach().cpu().numpy()
        y_train = y_train.detach().cpu().numpy()
        y_test = y_test.detach().cpu().numpy()
    
    return x_train, x_test, y_train, y_test
    


def load_svhn(path, training_label=1, download=False, benchmarking=False):
    split_rate = 0.8
    train = datasets.SVHN(path, split='train', download=False)
    test = datasets.SVHN(path, split='test', download=False)

    # # PyOD uses “0” to represent inliers and “1” to represent outliers

    train.data, train.labels = torch.from_numpy(train.data), torch.from_numpy(train.labels)
    # train.data = t(train.data)
    train.data = torch.squeeze(train.data, 1)

    test.data, test.labels = torch.from_numpy(test.data), torch.from_numpy(test.labels)
    # test.data = t(test.data)
    test.data = torch.squeeze(test.data, 1)

    x_train, y_train, x_test, y_test = split_svhn(train, test, 
                                            training_label=training_label, split_rate=0.8)
    x_train, y_train, x_test, y_test = normalize_and_shuffle(x_train, y_train, x_test, y_test)
    # binarize values
    x_train, y_train, x_test, y_test = binarize(x_train, y_train, x_test, 
                                                y_test, training_label=training_label)
    x_train, y_train, x_test, y_test = normalize_and_shuffle(x_train, y_train, x_test, y_test)
    x_train, y_train, x_test, y_test = get_grayscale(x_train, y_train, x_test, y_test)
    if benchmarking:
        x_test = x_test.detach().cpu().numpy()
        x_train = x_train.detach().cpu().numpy()
        y_train = y_train.detach().cpu().numpy()
        y_test = y_test.detach().cpu().numpy()

    return x_train, x_test, y_train, y_test


class Dataset(torch.utils.data.Dataset):
    def __init__(self,data,labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]
        return data, labels

    def __len__(self):
        return len(self.data)

def open_file(type_of_file):
    import bz2
    with bz2.open(type_of_file) as fp:
        raw_data = [line.decode().split() for line in fp.readlines()]
        tmp_list = [[x.split(":")[-1] for x in data[1:]] for data in raw_data]
        imgs = np.asarray(tmp_list, dtype=np.float32).reshape((-1, 16, 16))
        imgs = ((imgs + 1) / 2 * 255).astype(dtype=np.uint8)
        targets = [int(d[0]) - 1 for d in raw_data]

        x = np.asarray(targets)
        return torch.from_numpy(imgs), torch.from_numpy(x) 

def load_usps(path, training_label=1, download=False, benchmarking=False):
    x_train, y_train =  open_file("dataset/usps.bz2")
    x_test, y_test = open_file("dataset/usps_test.bz2")
    split_rate = 0.8
    train = Dataset(x_train, y_train)
    test = Dataset(x_test, y_test)

    x_train, y_train, x_test, y_test = split_svhn(train, test, 
                                            training_label=training_label, split_rate=0.8)
    x_train, y_train, x_test, y_test = normalize_and_shuffle(x_train, y_train, x_test, y_test)
    # binarize values
    x_train, y_train, x_test, y_test = binarize(x_train, y_train, x_test, 
                                                y_test, training_label=training_label)
    x_train, y_train, x_test, y_test = normalize_and_shuffle(x_train, y_train, x_test, y_test)
    
    if benchmarking:
        x_test = x_test.detach().cpu().numpy()
        x_train = x_train.detach().cpu().numpy()
        y_train = y_train.detach().cpu().numpy()
        y_test = y_test.detach().cpu().numpy()
    return x_train, x_test, y_train, y_test

