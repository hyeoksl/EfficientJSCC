import os
import io
import numpy as np
from PIL import Image

import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader

__all__ = ['loadData', 'load_dataloaders']

class addContinuousNoise(object):
    def __init__(self, image_size, clip_range, quantization_step):
        assert isinstance(image_size, tuple)
        assert isinstance(clip_range, tuple)
        assert isinstance(quantization_step, (int, float))
        self.image_size = image_size
        self.clip_range = clip_range
        self.quantization_step = quantization_step

    def __call__(self, image):
        noise = torch.rand(self.image_size) * self.quantization_step - (self.quantization_step/2)
        image.add_(noise)
        image.clamp_(min=self.clip_range[0], max=self.clip_range[1])
        return image
    

class DynamicImageDataset(Dataset):
    """지정된 디렉토리 내의 이미지 파일들을 동적으로 스캔하여 로드하는 데이터셋"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith(valid_extensions)
        ]
        self.image_paths.sort() # 일관된 순서를 위해 정렬

        if not self.image_paths:
            raise FileNotFoundError(f"No valid images found in the directory: {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image


def loadData(root_dir, transform_type='train', patch_size=(256, 256)):
    """
    폴더 경로와 transform 타입을 입력받아 Dataset 객체를 반환
    """
    if transform_type == 'train':
        transform = v2.Compose([
            v2.RandomResize(640, 1200),
            v2.RandomCrop(patch_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            addContinuousNoise(
                image_size=patch_size,
                clip_range=(0, 1),
                quantization_step=1/256
            )
        ])
    elif transform_type == 'test':
        transform = torchvision.transforms.Compose([
            v2.RandomCrop(patch_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
    elif transform_type == 'valid':
        transform = torchvision.transforms.Compose([
            v2.CenterCrop(patch_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
    else:
        raise ValueError(f"transform_type '{transform_type}' is not supported. Use 'train', 'test', or 'valid'.")
    
    dataset = DynamicImageDataset(root_dir=root_dir, transform=transform)
    return dataset


def load_dataloaders(train_dir, valid_dir, train_batch_size, valid_batch_size, num_workers, patch_size=(256, 256)):
    """
    데이터셋 경로를 입력받아 DataLoader를 생성
    """
    train_ds = loadData(root_dir=train_dir, transform_type='train', patch_size=patch_size)
    valid_ds = loadData(root_dir=valid_dir, transform_type='valid', patch_size=patch_size)
    
    train_loader_kwargs = {
        "batch_size": train_batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": False,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 3

    train_dl = DataLoader(
        train_ds,
        **train_loader_kwargs,
    )
    
    valid_dl = DataLoader(
        valid_ds,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )
    
    return train_dl, valid_dl
