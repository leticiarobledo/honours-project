import torch
from torchvision import datasets, utils
import numpy as np


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


def load_usps(path, training_label=1, split_rate=0.8, download=True):
    train_data, train_labels = open_file("dataset/train.bz2")
    test_data, test_labels = open_file("dataset/test.bz2")

    train = Dataset(train_data, train_labels)
    test = Dataset(test_data, test_labels)

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
    return (x_train, y_train), (x_test, y_test)
