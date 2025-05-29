import os, sys
import random
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import  transforms, datasets
from torch import nn
import numpy as np
import time
from functools import wraps
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

def note_print(*args, **kwargs):
    """
    用醒目的红色在cmd输出
    """
    print("\033[0;32m", *args, "\033[0m", **kwargs) # 绿色
    # print("\033[0;40m", *args, "\033[0m", **kwargs) # 红色


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # NOTE: 直接使用time.time()似乎会有多进程同时执行的误差
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Time taken by {func.__name__}: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f} (hh:mm:ss)")
        return result
    return wrapper

def seed_torch(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


class TinyImageNet_load(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)
        # test 数据集也需要targets，用于split_class_data
        self.targets = [target for _, target in self.images]    # NOTE: 添加fix参数

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            # images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(self.train_dir, d))]  # INFO:修改为val_dir
            raise ValueError("Python version is too low")
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(val_image_dir, d))]  
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        # test 数据集 targets 不需要改变
        # raise NotImplementedError("label if got from samples, however, it should be got from targets (for random label)")
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt


class vggface_dataset(torch.utils.data.Dataset):
    def __init__(self, config, transform, mode=None, train=None):    # transorms在dataloader中加入
        if mode == 'pretrain' and train==True:
            self.samples = config["pretrain_train_samples"]
        elif mode == 'pretrain' and train==False:
            self.samples = config["pretrain_test_samples"]
        elif mode == 'finetune' and train==True:
            self.samples = config["finetune_train_samples"]
        elif mode == 'finetune' and train==False:
            self.samples = config["finetune_test_samples"]
        else:
            raise ValueError('mode must be one of [pretrain, finetune]')
        self.targets = [s[1] for s in self.samples]
        self.transform = transform

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        # raise NotImplementedError("label if got from samples, however, it should be got from targets (for random label)")
        img_path, _ = self.samples[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        # NOTE: due to random label method, samples.label is the original label, and targets is the noisy label
        # get label from targets
        label = self.targets[index]
        return img, label


class Identity:
    def __call__(self, x):
        return x


def get_transforms(dataset_name, model_name, wo_dataaug): # 默认有数据增广
    resize_transform = Identity()   if "my" in model_name or model_name == "vgg16" \
                                    else transforms.Resize((224,224))   # INFO: "resnet18_origin"不需要resize
    if dataset_name in ["cifar10", "cifar100"]:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            resize_transform,
            transforms.ToTensor(),          # totensor只改变hwc，不改变rgb的顺序
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        transform_test = transforms.Compose([
            resize_transform,
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    elif dataset_name == "vggface": # 只支持resnet
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 使用针对imagenet的归一化参数
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif dataset_name == "tiny_imagenet":
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4),
            resize_transform,
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
        ])

        transform_test = transforms.Compose([
            resize_transform,
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    if wo_dataaug:
        transform_train = transform_test
    return transform_train, transform_test


def get_dataset(dataset_name, transform_train, transform_test, path=Path("~/data").expanduser()):
    if dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(root=path, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)
    elif dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100(root=path, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root=path, train=False, download=True, transform=transform_test)
    elif dataset_name == 'tiny_imagenet':
        # data_dir = os.path.join(path, 'tiny-imagenet-200') 
        # INFO: 如果尝试赋值变量，会导致data_dir变为局部变量，无法在函数外部访问。可以读data_dir变量，修改为tinyimagenet_dir
        tinyimagenet_dir = os.path.join(path, 'tiny-imagenet-200')
        train_dataset = datasets.ImageFolder(root=os.path.join(tinyimagenet_dir, 'train'), transform=transform_train)
        test_dataset = TinyImageNet_load(tinyimagenet_dir, train=False, transform=transform_test)
    elif dataset_name == 'vggface':
        config_path = 'config/vggface_sample.yaml'
        sample_config = OmegaConf.load(config_path)

        train_dataset = vggface_dataset(sample_config, transform_train, mode='finetune', train=True)
        test_dataset = vggface_dataset(sample_config, transform_test, mode='finetune', train=False)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return train_dataset, test_dataset


def create_dataset(dir_path, pretrain_class_num=100, finetune_class_num=10):
    label_counts = []
    for subfolder in ["train", "test"]:
        subfolder_path = os.path.join(dir_path, subfolder)
        for class_name in os.listdir(subfolder_path):   # class_name是局部文件/文件夹名
            class_path = os.path.join(subfolder_path, class_name)
            if os.path.isdir(class_path):
                num_images = len(os.listdir(class_path))
                label_counts.append((subfolder+ "/" + class_name, num_images))

    # 按照图像数量从大到小排序，选择其中图片超过阈值的类别
    label_counts.sort(key=lambda x: x[1], reverse=True)
    class_num = pretrain_class_num + finetune_class_num

    chosen_classes = [x[0] for x in label_counts if x[1]>= 500]
    assert len(chosen_classes) >= class_num
    chosen_classes = chosen_classes[:class_num]

    print(f"文件夹数目：{len(chosen_classes)}，每个文件夹图片数：{ 500 }")

    # 创建新的数据集，只包含所选类
    pretrain_train_samples, pretrain_test_samples, finetune_train_samples, finetune_test_samples = [], [], [], []
    for i, class_name in enumerate(chosen_classes):
        class_path = os.path.join(dir_path, class_name)
        assert len(os.listdir(class_path)) >= 500, f"Class {class_name} does not have enough images"

        for j, img_name in enumerate(os.listdir(class_path)): # j表示每个类已经添加的图像数量
            img_path = os.path.join(class_path, img_name)
            if i< pretrain_class_num:
                if j < 100:
                    pretrain_test_samples.append((img_path, i))
                elif j < 500:
                    pretrain_train_samples.append((img_path, i))
                else:
                    break
            else:
                if j < 100:
                    finetune_test_samples.append((img_path, i - pretrain_class_num))    # 设置从0开始
                elif j < 500:
                    finetune_train_samples.append((img_path, i - pretrain_class_num))
                else:
                    break

    config_dict = {
        "pretrain_train_samples": pretrain_train_samples,
        "pretrain_test_samples": pretrain_test_samples,
        "finetune_train_samples": finetune_train_samples,
        "finetune_test_samples": finetune_test_samples,
    }
    return config_dict


def get_dataloader(trainset, testset, batch_size, num_workers, shuffle=True):
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)  

    return train_loader, test_loader


def split_class_data(dataset, forget_class_index, num_forget):
    """
    从数据集中筛选出 num_forget 个 forget_class 的数据， forget_class 是 forget_class_index 中的类别。
    NOTE: 目前只支持单类别遗忘。
    
    Returns:
        forget_index: 忘记类别的样本索引
        remain_index: 保留样本的索引，包含所有非 forget_class 和部分 forget_class 样本
        class_remain_index: 保留的 forget_class 样本的索引
    """
    # 提取数据集的标签，假设 dataset 的每个元素是 (数据, 标签) 的形式

    # targets = torch.tensor([target for _, target in dataset])
    targets = dataset.targets   # 与上面的写法等价，相比for循环能加快速度
    targets = torch.tensor(targets)

    forget_class_index = torch.tensor(forget_class_index)   # 不会影响外面的forget_class_index
    # 找出所有属于 forget_class 的样本索引
    # forget_class_indices = torch.nonzero(targets == forget_class).flatten()
    mask = torch.isin(targets, forget_class_index)
    forget_class_indices = torch.nonzero(mask).flatten()
    
    # 确保找到至少一个 forget_class 的样本
    assert forget_class_indices.numel() > 0, f"No samples found for class in {forget_class_indices}"

    # 如果指定的 num_forget 比 forget_class 的样本数量多，调整 num_forget
    num_forget = min(num_forget, forget_class_indices.numel())

    # 获取需要遗忘的样本索引
    forget_index = forget_class_indices[:num_forget]

    # 获取需要保留的 forget_class 样本的索引
    class_remain_index = forget_class_indices[num_forget:]

    # 找出所有非 forget_class 的样本索引
    remain_index = torch.nonzero(~mask).flatten()

    # 将剩余的 forget_class 样本索引添加到保留索引中
    remain_index = torch.cat((remain_index, class_remain_index))

    # 将结果转换为列表，便于后续使用
    return forget_index.tolist(), remain_index.tolist(), class_remain_index.tolist()


def get_unlearn_loader(trainset, testset, forget_class_index, batch_size, num_forget, num_workers, repair_num_ratio=0.01):
    """
    将train dataset上指定数量的遗忘类拿出来
    将test dataset上所有的遗忘类拿出来
    class_remain_index：在指定了满足num_forget个train中图片的下标后，剩下遗忘类别的图片下标
    """
    train_forget_index, train_remain_index, class_remain_index = split_class_data(trainset, forget_class_index, num_forget=num_forget)    # trainset, 4, 5000
    assert isinstance(train_forget_index, list)
    test_forget_index, test_remain_index, _ = split_class_data(testset, forget_class_index, num_forget=len(testset))  # testset, 4, 1000
    # 保证将目标类全部遗忘
    # NOTE: numforget 有点奇怪，只限制训练集不限制测试集

    ######################################################
    # # 保证相同的seed，则每次index的结果都是一致的，下面的代码是一个简单的验证
    ######################################################
    # data_indices={
    #     'train_forget_index': train_forget_index,
    #     'train_remain_index': train_remain_index,
    #     'class_remain_index': class_remain_index,
    #     'test_forget_index': test_forget_index,
    #     'test_remain_index': test_remain_index
    # }
    # # torch.save(data_indices, 'checkpoints/cifar10_data_indices.pth')
    # data_indices_old = torch.load('checkpoints/cifar10_data_indices.pth')
    # for index,index_old in zip(data_indices.values(), data_indices_old.values()):
    #     assert index == index_old
    # print('data_indices check pass!')
    #####################################################

    repair_class_index = random.sample(class_remain_index, int(repair_num_ratio * len(class_remain_index))) # 空列表

    train_forget_sampler = SubsetRandomSampler(train_forget_index)  # 5000。 对列表进行封装便于采样
    train_remain_sampler = SubsetRandomSampler(train_remain_index)  # 45000

    repair_class_sampler = SubsetRandomSampler(repair_class_index)

    test_forget_sampler = SubsetRandomSampler(test_forget_index)  # 1000
    test_remain_sampler = SubsetRandomSampler(test_remain_index)  # 9000

    train_forget_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                      sampler=train_forget_sampler,
                                                      num_workers=num_workers) # 在每个epoch中进行不放回的随机抽样
    train_remain_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                      sampler=train_remain_sampler,
                                                      num_workers=num_workers)

    repair_class_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                      sampler=repair_class_sampler,
                                                      num_workers=num_workers)

    test_forget_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size,
                                                     sampler=test_forget_sampler,
                                                      num_workers=num_workers)
    test_remain_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size,
                                                     sampler=test_remain_sampler,
                                                      num_workers=num_workers)

    return train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
           train_forget_index, train_remain_index, test_forget_index, test_remain_index


def inf_generator(iterable):
    """
    一个永远不会用完的迭代器，实际上你可以使用itertools.cycle(iterable)来实现
    """
    note_print("inf_generator is deprecated, use itertools.cycle(iterable) instead")
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()   # yield就是return
        except StopIteration:
            iterator = iterable.__iter__()
