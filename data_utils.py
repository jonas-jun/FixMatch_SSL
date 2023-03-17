import os
import torch
import pytorch_lightning as pl
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from randaugment import RandAugmentMC

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

class atmDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage: str) -> None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_train = get_transform(mean=mean, std=std, in_size=self.args.in_size, train=True)
        transform_val = get_transform(mean=mean, std=std, in_size=self.args.in_size, train=False)

        if stage == 'fit':
            self.dset_label = None
            self.dset_unlabel = None
            self.dset_val = None
        elif stage == 'test':
            self.dset_test = None

    def train_dataloader(self):
        loader_label = DataLoader(self.dset_label, batch_size=self.args.batch_size, num_workers=self.args.n_workers, shuffle=True, pin_memory=True)
        loader_unlabel = DataLoader(self.dset_unlabel, batch_size=(self.args.batch_size*self.args.ul_ratio), num_workers=self.args.n_workers, shuffle=True, pin_memory=True)
        return {'label': loader_label, 'unlabel': loader_unlabel}

    def val_dataloader(self):
        loader_val = DataLoader(self.dset_val, batch_size=self.args.batch_size*self.args.ul_ratio, num_workers=self.args.n_workers, shuffle=False, pin_memory=True)
        return loader_val
    
    def test_dataloader(self):
        loader_test = DataLoader(self.dset_test, batch_size=self.args.batch_size, num_workers=self.args.n_workers, shuffle=False, pin_memory=True)
        return loader_test

class TransformFixMatch:
    def __init__(self, mean, std, in_size):
        self.weak = transforms.Compose([
            transforms.Resize((in_size, in_size), interpolation=BICUBIC),
            transforms.RandomHorizontalFlip(),
            ])
        self.strong = transforms.Compose([
            transforms.Resize((in_size, in_size), interpolation=BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10)
            ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

class ImageFolderDataset(Dataset):
    def __init__(self, dir, transform=None, limit=False):
        self.data_dir = dir
        self.cls_dict = self.build_cls_to_idx()
        self.file_paths, self.labels, self.count_dict = self.load_dataset()
        if limit:
            self.file_paths = self.file_paths[:limit]
            self.labels = self.labels[:limit]
        self.transform = transform

    def build_cls_to_idx(self) -> dict:
        self.classes = sorted(os.listdir(self.data_dir)) # torchvision.datasets.ImageFolder와 동일하게 구현하려면 sorted 필요
        cls_dict = dict()
        for idx, cls_ in enumerate(self.classes):
            cls_dict[cls_] = idx
        return cls_dict
    
    def load_dataset(self):
        file_paths, labels = list(), list()
        count_dict = dict()
        for cls_ in self.cls_dict:
            temp_files = list(map(lambda x: os.path.join(self.data_dir, cls_, x), os.listdir(os.path.join(self.data_dir, cls_))))
            num = len(temp_files)
            count_dict[cls_] = num
            file_paths += temp_files
            labels += [self.cls_dict[cls_] for _ in range(num)]
        return file_paths, labels, count_dict

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path, label = self.file_paths[idx], self.labels[idx]
        img = Image.open(file_path).convert('RGB')
        img = self.transform(image=np.array(img))['image']
        return img, label, file_path

class Label_Dataset(ImageFolderDataset):
    def __getitem__(self, idx):
        img, target = self.file_paths[idx], self.labels[idx]
        img = Image.open(img).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return idx, img, target

class UnLabel_Dataset(Dataset):
    def __init__(self, dir, transform, limit=False):
        self.dir = dir
        self.files = os.listdir(self.dir)
        self.files = list(map(lambda x: os.path.join(self.dir, x), self.files))
        self.transform = transform
        if limit:
            self.files = self.files[:limit]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = self.files[idx]
        img = Image.open(img).convert('RGB')
        if self.transform:
            img_weak, img_strong = self.transform(img)
        return idx, img_weak, img_strong

def get_transform(mean, std, in_size, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((in_size, in_size), interpolation=BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    else:
        return transforms.Compose([transforms.Resize((in_size, in_size), interpolation=BICUBIC),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])