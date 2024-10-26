from .poisoned_dataset import CIFAR10Poison, MNISTPoison, ImageFolderPoison
from torchvision import datasets, transforms
import torch 
import os 

def build_init_data(dataname, download, dataset_path):
    if dataname == 'MNIST':
        train_data = datasets.MNIST(root=dataset_path, train=True, download=download)
        test_data  = datasets.MNIST(root=dataset_path, train=False, download=download)
    elif dataname == 'CIFAR10':
        train_data = datasets.CIFAR10(root=dataset_path, train=True,  download=download)
        test_data  = datasets.CIFAR10(root=dataset_path, train=False, download=download)
    return train_data, test_data

def build_poisoned_training_set(is_train, args):
    transform, detransform = build_transform(args.dataset, is_train)
    # print("Transform = ", transform)

    if args.dataset == 'CIFAR10':
        trainset = CIFAR10Poison(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    elif args.dataset == 'MNIST':
        trainset = MNISTPoison(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    elif args.dataset == 'TinyImageNet':
        trainset = ImageFolderPoison(args, args.data_path, train=is_train, transform=transform)
        nb_classes = 200
    else:
        raise NotImplementedError()

    assert nb_classes == args.nb_classes
    # print("Number of the class = %d" % args.nb_classes)
    # print(trainset)

    return trainset, nb_classes


def build_testset(is_train, args):
    transform, detransform = build_transform(args.dataset)
    # print("Transform = ", transform)

    if args.dataset == 'CIFAR10':
        testset_clean = datasets.CIFAR10(args.data_path, train=is_train, download=True, transform=transform)
        testset_poisoned = CIFAR10Poison(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    elif args.dataset == 'MNIST':
        testset_clean = datasets.MNIST(args.data_path, train=is_train, download=True, transform=transform)
        testset_poisoned = MNISTPoison(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    elif args.dataset == 'TinyImageNet':
        testset_poisoned = ImageFolderPoison(args, args.data_path, train=is_train, transform=transform)
        testset_clean = testset_poisoned
        nb_classes = 200
    else:
        raise NotImplementedError()

    assert nb_classes == args.nb_classes
    # print("Number of the class = %d" % args.nb_classes)
    # print(testset_clean, testset_poisoned)

    return testset_clean, testset_poisoned

def build_transform(dataset, is_train=False):
    if dataset == "CIFAR10":
        # mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    elif dataset == "MNIST":
        mean, std = (0.5,), (0.5,)
    else:
        raise NotImplementedError()
    
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    transform = transforms.Compose(
        [transforms.ToTensor(),
        #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            normalize])
    train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
                ])

    # transform = transforms.Compose([
    #    transforms.ToTensor(),
    #    transforms.Normalize(mean, std)
    #    ])
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    detransform = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()) # you can use detransform to recover the image
    
    if is_train:
        return train_transform, detransform
    return transform, detransform
