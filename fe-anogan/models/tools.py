import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np

class Dataset(torch.utils.data.Dataset):
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

############        MNIST       #################         
def load_mnist(path, training_label, download=False):
    split_rate = 0.8
    tf = torchvision.transforms.Compose([transforms.ToPILImage(),
                                        # transforms.Resize((350, 350), interpolation=transforms.InterpolationMode.BICUBIC),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
    train = torchvision.datasets.MNIST(path, train=True, download=download)
    test = torchvision.datasets.MNIST(path, train=False, download=download)
    # normalize
    train.data = train.data/train.data.max()
    test.data = test.data/test.data.max()

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
    x_train = x_train.unsqueeze(1)
    x_test = x_test.unsqueeze(1)

    train_dataset = Dataset(x_train, y_train, transform=tf)
    test_dataset = Dataset(x_test, y_test, transform=tf)
    return train_dataset, test_dataset



#############       SVHN        ##############
def load_svhn(path, training_label=1, download=False):
    split_rate = 0.8
    tf = torchvision.transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Lambda(lambda x: transforms.functional.adjust_sharpness(x, sharpness_factor=2.0)),
                                transforms.Resize((28, 28), interpolation=transforms.InterpolationMode.BICUBIC),
                                transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: transforms.functional.adjust_contrast(x, contrast_factor=2)),
                                transforms.Normalize((0.5,), (0.5,))
                                ])
    train = datasets.SVHN(path, split='train', download=download, transform=tf)
    test = datasets.SVHN(path, split='test', download=download, transform=tf)

    # transform to tensor
    train.data = torch.from_numpy(train.data)
    train.labels = torch.from_numpy(train.labels)
    test.data = torch.from_numpy(test.data)
    test.labels = torch.from_numpy(test.labels)
    
    train.data = train.data/train.data.max()
    test.data = test.data/test.data.max()

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

    #x_train = x_train.unsqueeze(1)
    #x_test = x_test.unsqueeze(1)
    train_dataset = Dataset(x_train, y_train, transform=tf)
    test_dataset = Dataset(x_test, y_test, transform=tf)

    return train_dataset, test_dataset



######################      USPS       ########################
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

# it is GRAYSCALE and 16x16 !!
def load_usps(path, training_label=1, split_rate=0.8, download=False):
    train_data, train_labels = open_file("dataset/train.bz2")
    test_data, test_labels = open_file("dataset/test.bz2")
    split_rate = 0.8
    tf = torchvision.transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Lambda(lambda x: transforms.functional.adjust_sharpness(x, sharpness_factor=2.0)),
                                transforms.Resize((28, 28), 
                                interpolation=transforms.InterpolationMode.BICUBIC),
                                transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: transforms.functional.adjust_contrast(x, contrast_factor=2)),
                                transforms.Normalize((0.5,), (0.5,))])
    train = Dataset(train_data, train_labels, transform=tf)
    test = Dataset(test_data, test_labels, transform=tf)
    # normalize
    train.data = train.data/train.data.max()
    test.data = test.data/test.data.max()

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

    train_dataset = Dataset(x_train, y_train, transform=tf)
    test_dataset = Dataset(x_test, y_test, transform=tf)
    return train_dataset, test_dataset


##############      1-CLASS ALL-DATASETS        ########################

def load_features(train,test):
    train_loader = DataLoader(train, batch_size=len(train))
    test_loader = DataLoader(test, batch_size=len(test))
    train = Dataset(next(iter(train_loader))[0], next(iter(train_loader))[1])
    test = Dataset(next(iter(test_loader))[0], next(iter(test_loader))[1])
    return train, test

def get_datasets(path, batch_size):
    # Dataloader only loads batch size samples
    mnist_train_og, mnist_test_og = load_mnist(path, training_label=1)
    svhn_train_og, svhn_test_og = load_svhn(path,training_label=1)
    usps_train_og, usps_test_og = load_usps(path, training_label=1)
    mnist_train, mnist_test = load_features(mnist_train_og, mnist_test_og)
    svhn_train, svhn_test = load_features(svhn_train_og, svhn_test_og)
    usps_train, usps_test = load_features(usps_train_og, usps_test_og)

    # Data Augmentation
    from sklearn.utils import resample
    frequency = round(1.2 * len(svhn_train))
    mnist_train_upsample = resample(mnist_train.data.numpy(),mnist_train.labels.numpy(),
                                    replace=True, n_samples=frequency, random_state=42)
    #mnist_train_upsample = Dataset(torch.from_numpy(mnist_train_upsample[0]), torch.from_numpy(mnist_train_upsample[1]))
    
    usps_train_upsample = resample(usps_train.data.numpy(),usps_train.labels.numpy(),replace=True,n_samples=frequency, random_state=42)
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
    test_loader = DataLoader(concat_test, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader





#########        PCA  EXPERIMENTS       ##################
def get_datasets_full_length_loader(path, training_label):
    mnist_train_og, mnist_test_og = load_mnist(path, training_label)
    svhn_train_og, svhn_test_og = load_svhn(path,training_label)
    usps_train_og, usps_test_og = load_usps(path, training_label)
    mnist_train, mnist_test = load_features(mnist_train_og, mnist_test_og)
    svhn_train, svhn_test = load_features(svhn_train_og, svhn_test_og)
    usps_train, usps_test = load_features(usps_train_og, usps_test_og)

    # Data Augmentation
    from sklearn.utils import resample
    frequency = round(1.2 * len(svhn_train))
    mnist_train_upsample = resample(mnist_train.data.numpy(),mnist_train.labels.numpy(), replace=True, n_samples=frequency, random_state=42)
    usps_train_upsample = resample(usps_train.data.numpy(),usps_train.labels.numpy(),replace=True,n_samples=frequency, random_state=42)

    # transform to numpy elements
    mnist_train_upsample = Dataset(mnist_train_upsample[0], mnist_train_upsample[1])
    usps_train_upsample = Dataset(usps_train_upsample[0], usps_train_upsample[1])
    svhn_train_upsample = Dataset(svhn_train.data.detach().cpu().numpy(), svhn_train.labels.detach().cpu().numpy())
    
    # Concat Datasets
    # Train
    concat_train = torch.utils.data.ConcatDataset([mnist_train_upsample, svhn_train_upsample, usps_train_upsample])
    train_loader = DataLoader(concat_train, batch_size=len(concat_train), shuffle=True) # we are not using sampler because we have augmented the data
    # Test
    # transform to appropriate objects
    m = Dataset(mnist_test.data.detach().cpu().numpy(), mnist_test.labels.detach().cpu().numpy())
    s = Dataset(svhn_test.data.detach().cpu().numpy(), svhn_test.labels.detach().cpu().numpy())
    u = Dataset(usps_test.data.detach().cpu().numpy(), usps_test.labels.detach().cpu().numpy())
    concat_test = torch.utils.data.ConcatDataset([m,s,u])
    test_loader = DataLoader(concat_test, batch_size=len(concat_test), shuffle=True)

    return train_loader, test_loader

