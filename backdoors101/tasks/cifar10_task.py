import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms

from models.resnet_cifar import resnet18
from tasks.task import Task
import torch

# 加载数据和模型
class Cifar10Task(Task):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

    def load_data(self):
        self.load_cifar_data()
    # 已看懂
    def load_cifar_data(self):
        if self.params.transform_train:                                             # 如果要进行转换，则进行下面处理
            transform_train = transforms.Compose([
                # 随机裁剪图像为32x32大小，并在周围填充4个像素
                transforms.RandomCrop(32, padding=4),
                # 做一个随机的上下翻转
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # 使用预先定义的归一化操作对图像进行标准化处理
                self.normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ])
        self.train_dataset = torchvision.datasets.CIFAR10(                          # 下载数据集
            root=self.params.data_path,
            train=True,
            download=True,
            transform=transform_train)
        if self.params.poison_images:                                               # 如果有中毒图像，就删除
            self.train_loader = self.remove_semantic_backdoors()
        else:
            self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=self.params.batch_size,
                                           shuffle=True,
                                           num_workers=0)
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=self.params.data_path,
            train=False,
            download=True,
            transform=transform_test)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.params.test_batch_size,
                                      shuffle=False, num_workers=0)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return True

    # 目的是创建一个残差网络，输出看有多少个类。
    def build_model(self) -> nn.Module:
        if self.params.pretrained:                                                              # 预加载模型
            model = resnet18(pretrained=True)
            # model is pretrained on ImageNet changing classes to CIFAR
            model.fc = nn.Linear(512, len(self.classes))
        else:
            model = resnet18(pretrained=False, num_classes=len(self.classes))
        return model
    # 如果最开始就有中毒图片就删除（未采用任何检测方法）
    def remove_semantic_backdoors(self):                                                            # 移除中毒图像
        """
        Semantic backdoors still occur with unmodified labels in the training
        set. This method removes them, so the only occurrence of the semantic
        backdoor will be in the
        :return: None
        """
        all_images = set(range(len(self.train_dataset)))                                            # 获取训练集中所有图像的索引范围，并将其转换为集合
        unpoisoned_images = list(all_images.difference(set(self.params.poison_images)))             # 从所有图像索引集合中排除受污染图像的索引集合，以获取未受污染的图像索引集合
        self.train_loader = DataLoader(self.train_dataset,                                          # 使用未受污染的图像索引集合创建一个新的数据加载器（DataLoader）
                                       batch_size=self.params.batch_size,                           # 这将用于训练模型，仅包含未受污染的图像
                                       sampler=SubsetRandomSampler(unpoisoned_images))
