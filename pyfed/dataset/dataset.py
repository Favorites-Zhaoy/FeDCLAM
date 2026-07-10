import os
import random
import numpy as np
from glob import glob
from pathlib import Path

import torch
from torch.utils.data import Dataset

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None


def _image_paths_for_split(sitedir, split, valid_ratio=0.2):
    """Return image paths without ever using test data as validation."""
    if split not in ('train', 'valid', 'test'):
        raise ValueError(f"Unsupported split: {split}")

    split_dir = os.path.join(sitedir, split)
    if split == 'test':
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Missing test split directory: {split_dir}")
        return sorted(glob(f"{split_dir}/image/*.npy"))

    if split == 'valid' and os.path.isdir(split_dir):
        return sorted(glob(f"{split_dir}/image/*.npy"))

    train_dir = os.path.join(sitedir, 'train')
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Missing train split directory: {train_dir}")

    train_paths = sorted(glob(f"{train_dir}/image/*.npy"))
    has_separate_valid = os.path.isdir(os.path.join(sitedir, 'valid'))
    if split == 'train' and has_separate_valid:
        return train_paths

    if len(train_paths) <= 1:
        return train_paths if split == 'train' else []

    valid_count = int(round(valid_ratio * len(train_paths)))
    valid_count = min(max(1, valid_count), len(train_paths) - 1)
    if split == 'train':
        return train_paths[:-valid_count]
    return train_paths[-valid_count:]


class Prostate(Dataset):
    def __init__(self, site, base_path, train_ratio=0.6, split='train', transform=None):
        channels = {'BIDMC': 3, 'HK': 3, 'I2CVB': 3, 'BMC': 3, 'RUNMC': 3, 'UCL': 3}
        assert site in list(channels.keys())
        self.split = split
        self.train_ratio = train_ratio

        images, labels = [], []
        sitedir = os.path.join(base_path, site)

        sample_list = sorted(os.listdir(sitedir))
        sample_list = [x for x in sample_list if 'segmentation.nii.gz' in x.lower()]

        for sample in sample_list:
            if sitk is None:
                raise ImportError("SimpleITK is required to read prostate volumes")
            sampledir = os.path.join(sitedir, sample)
            if os.path.getsize(sampledir) < 1024 * 1024 and sampledir.endswith("nii.gz"):
                imgdir = os.path.join(sitedir, sample[:6] + ".nii.gz")
                label_v = sitk.ReadImage(sampledir)
                image_v = sitk.ReadImage(imgdir)
                label_v = sitk.GetArrayFromImage(label_v)
                label_v[label_v > 1] = 1
                image_v = sitk.GetArrayFromImage(image_v)
                image_v = convert_from_nii_to_png(image_v)
                
                for i in range(1, label_v.shape[0] - 1):
                    label = np.array(label_v[i, :, :])
                    if np.all(label == 0):
                        continue
                    image = np.array(image_v[i - 1:i + 2, :, :])
                    image = np.transpose(image, (1, 2, 0))

                    labels.append(label)
                    images.append(image)
            else:
                print(f"Skipping {sampledir} due to size")
        labels = np.array(labels).astype(int)
        images = np.array(images)

        index_path = f"data/prostate/{site}-index.npy"
        if not os.path.exists(index_path):
            # index = np.random.permutation(len(images)).tolist()
            # this was causing potential data leakage between train and test
            index = np.arange(len(images))
            np.save(index_path, index)
        else:
            index = np.load(f"data/prostate/{site}-index.npy").tolist()

        labels = labels[index]
        images = images[index]

        trainlen = int(max(self.train_ratio * len(labels), 32))
        vallen = int(0.2 * len(labels))
        testlen = int(0.2 * len(labels))

        if split == 'train':
            self.images, self.labels = images[:trainlen], labels[:trainlen]
        elif split == 'valid':
            self.images, self.labels = images[trainlen:trainlen + vallen], labels[trainlen:trainlen + vallen]
        else:
            self.images, self.labels = images[-testlen:], labels[-testlen:]

        self.transform = transform
        self.channels = channels[site]
        self.labels = self.labels.astype(np.int_).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            if self.split == 'train':
                R1 = RandomRotate90()
                image, label = R1(image, label)
                R2 = RandomFlip()
                image, label = R2(image, label)

            image = np.transpose(image, (2, 0, 1))
            image = torch.Tensor(image)

            label = self.transform(label)

        return image, label
    

class ProstatePre(Dataset):
    def __init__(self, site, base_path, train_ratio, split='train', transform=None):
        channels = {'BIDMC': 3, 'HK': 3, 'I2CVB': 3, 'BMC': 3, 'RUNMC': 3, 'UCL': 3}
        assert site in list(channels.keys())
        self.split = split

        protocol = getattr(self, 'protocol', None)
        datadir = os.path.join(base_path, f"{site}_data")

        split_name = 'train' if split == 'train' else 'valid' if split == 'valid' else 'test'
        split_data = np.load(f"{datadir}/{split_name}.npy", allow_pickle=True).item()
        self.images, self.labels = split_data['images'], split_data['labels']

        self.transform = transform
        self.channels = channels[site]

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            if self.split == 'train':
                R1 = RandomRotate90()
                image, label = R1(image, label)
                R2 = RandomFlip()
                image, label = R2(image, label)

            image = np.transpose(image, (2, 0, 1))
            image = torch.Tensor(image)

            label = self.transform(label)

        return image, label


def convert_from_nii_to_png(img):
    high = np.quantile(img, 0.99)
    low = np.min(img)
    img = np.where(img > high, high, img)
    lungwin = np.array([low * 1., high * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg = (newimg * 255).astype(np.uint8)
    return newimg


class RandomRotate90:
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            factor = random.randint(0, 4)
            img = np.rot90(img, factor)
            if mask is not None:
                mask = np.rot90(mask, factor)
        return img.copy(), mask.copy()


class RandomFlip:
    def __init__(self, prob=0.75):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d) if cv2 is not None else np_flip(img, d)
            if mask is not None:
                mask = cv2.flip(mask, d) if cv2 is not None else np_flip(mask, d)

        return img, mask


def np_flip(array, direction):
    if direction == 0:
        return np.flip(array, axis=0).copy()
    if direction == 1:
        return np.flip(array, axis=1).copy()
    return np.flip(np.flip(array, axis=0), axis=1).copy()


class Fundus(Dataset):
    def __init__(self, site, base_path, train_ratio=0.6, split='train', transform=None):
        self.split = split
        self.images, self.labels = [], []
        sitedir = os.path.join(base_path, site)
        assert os.path.exists(sitedir)
        image_paths = _image_paths_for_split(sitedir, split, valid_ratio=1 - train_ratio)
        self.sample_paths = image_paths
        for image_path in image_paths:
            image_path_obj = Path(image_path)
            mask_path = str(image_path_obj.parent.parent / 'mask' / image_path_obj.name)
            assert (os.path.exists(mask_path))
            img = np.load(image_path).transpose((2, 0, 1))
            mask = np.load(mask_path)
            img = torch.from_numpy(img) / 255.
            _mask = []
            for _c in range(2):
                _mask.append((mask > _c).copy())
            mask = torch.from_numpy(np.array(_mask)).float()
            self.images.append(img)
            self.labels.append(mask)

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)

        return image, label


class Nuclei(Dataset):
    def __init__(self, site, base_path, train_ratio=0.7, split='train', transform=None):
        self.split = split
        self.transform = transform
        self.images, self.labels = [], []
        sitedir = os.path.join(base_path, site)
        assert os.path.exists(sitedir)
        image_paths = _image_paths_for_split(sitedir, split, valid_ratio=1 - train_ratio)
        self.sample_paths = image_paths
        for image_path in sorted(image_paths):
            image_path_obj = Path(image_path)
            mask_path = str(image_path_obj.parent.parent / 'mask' / image_path_obj.name)
            assert os.path.exists(mask_path)
            img = np.load(image_path)
            mask = np.load(mask_path)
            if img.ndim == 2:
                img = np.repeat(img[..., None], 3, axis=2)
            if img.shape[-1] == 4:
                img = img[..., :3]
            img = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.
            mask = (mask > 0).astype(np.int64)
            mask = torch.from_numpy(mask).unsqueeze(0).long()
            self.images.append(img)
            self.labels.append(mask)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)

        return image, label


class Polyp(Dataset):
    def __init__(self, site, base_path, train_ratio=0.7, split='train', transform=None):
        assert site in ['client1', 'client2', 'client3', 'client4']
        self.split = split
        self.train_ratio = train_ratio
        self.transform = transform
        self.images, self.labels = [], []
        sitedir = os.path.join(base_path, site, 'data_npy')
        assert os.path.exists(sitedir)
        sample_list = sorted(os.listdir(sitedir))
        for sample in sample_list:
            sampledir = os.path.join(sitedir, sample)
            data = np.load(sampledir)
            image = data[..., 0:3]
            label = data[..., 3:]
            self.images.append(image)
            self.labels.append(label)
        
        trainlen = int(max(self.train_ratio * len(self.labels), 32))
        vallen = int(0.1 * len(self.labels))
        testlen = int(0.2 * len(self.labels))
        if self.split == "train":
            self.images, self.labels = self.images[:trainlen], self.labels[:trainlen]
        elif self.split == "valid":
            self.images, self.labels = self.images[trainlen:trainlen + vallen], self.labels[trainlen:trainlen + vallen]
        else:
            self.images, self.labels = self.images[-testlen:], self.labels[-testlen]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)            

        return image, label
