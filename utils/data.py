import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from . import autoaugment
from . import ops

from typing import List
from torchvision.datasets import ImageFolder

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

class iISAR(iData):
    use_path = True
    train_trsf = [
        transforms.CenterCrop(224),
        transforms.Resize((128, 128)),
    ]
    test_trsf = [
        transforms.CenterCrop(224),
        transforms.Resize((128, 128)),
    ]
    common_trsf = [
        transforms.ToTensor(),
    ]

    class_order = [
        30, 0, 11, 16, 9, 6,
        21, 13, 5, 10,
        27, 32, 20, 2,
        7, 15, 26, 29,
        14, 31, 18, 28,
        24, 33, 4, 12,
        17, 22, 1, 3,
        23, 8, 25, 19
    ]

    def download_data(self):
        train_dir = [
            "./data/isar/50_2GHz",
        ]
        test_dir = [
            "./data/isar/55_2GHz",
        ]

        train_dset = ImageFolderConcat(train_dir) #允许训练数据分散在多个不同的文件夹里 这里只写了一个
        test_dset = ImageFolderConcat(test_dir)

        self.classes = train_dset.classes
        self.class_to_idx = train_dset.class_to_idx

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs) # 分别取出图片路径和标签
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class ImageFolderConcat(ImageFolder):
    def __init__(self, root: List[str]):
        data_list = [ImageFolder(path) for path in root]
        self.classes = data_list[0].classes # 路径只写了一个 所以这里取[0] 按顺序返回所有类名
        self.class_to_idx = data_list[0].class_to_idx # 给所有类对应的顺序索引
        self.imgs = [] # [路径 标签]
        for data in data_list:
            self.imgs.extend(data.imgs)

class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100_AA(iCIFAR100):
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        autoaugment.CIFAR10Policy(),
        transforms.ToTensor(),
        ops.Cutout(n_holes=1, length=16),
    ]


class iCIFAR10_AA(iCIFAR10):
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        autoaugment.CIFAR10Policy(),
        transforms.ToTensor(),
        ops.Cutout(n_holes=1, length=16),
    ]


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
