import torch
from torchvision import datasets


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

def load_svhn(path, training_label=1, split_rate=0.8, download=False):
    train = datasets.SVHN(path, split='train', download=download)
    test = datasets.SVHN(path, split='test', download=download)

    # transform to tensor
    train.data = torch.from_numpy(train.data)
    train.labels = torch.from_numpy(train.labels)
    test.data = torch.from_numpy(test.data)
    test.labels = torch.from_numpy(test.labels)

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
