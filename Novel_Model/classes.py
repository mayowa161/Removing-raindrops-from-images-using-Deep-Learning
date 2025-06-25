import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from PIL import Image
import os


class RaindropDataset(Dataset):
    def __init__(self, rain_dir, clean_dir, mask_dir, transform=None):
        """
        Args:
            rain_dir  (str): Directory with rain-degraded images.
            clean_dir (str): Directory with the clean (ground-truth) images.
            mask_dir  (str): Directory with the corresponding raindrop masks (grayscale).
            transform (callable, optional): Optional transform to apply 
                                            to the images and masks.
        """
        super(RaindropDataset, self).__init__()
        self.rain_dir = rain_dir
        self.clean_dir = clean_dir
        self.mask_dir = mask_dir

        self.rain_filenames  = sorted(os.listdir(rain_dir))
        self.clean_filenames = sorted(os.listdir(clean_dir))
        self.mask_filenames  = sorted(os.listdir(mask_dir))

        # Optional transform to apply to all images
        self.transform = transform

        # Quick consistency check (optional)
        if not (len(self.rain_filenames) == len(self.clean_filenames) == len(self.mask_filenames)):
            raise ValueError("Mismatch in number of files among rain/clean/mask directories.")

    def __len__(self):
        return len(self.rain_filenames)
    
    def __getitem__(self, idx):
        # Build full paths
        rain_path  = os.path.join(self.rain_dir,  self.rain_filenames[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_filenames[idx])
        mask_path  = os.path.join(self.mask_dir,  self.mask_filenames[idx])

        # Load images
        # Typically: rain & clean are RGB; mask is grayscale
        rain_img  = Image.open(rain_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")
        mask_img  = Image.open(mask_path).convert("L")  # grayscale

        # Apply transforms if provided
        if self.transform:
            rain_img  = self.transform(rain_img)
            clean_img = self.transform(clean_img)
            mask_img  = self.transform(mask_img)

        # Return all three
        return rain_img, clean_img, mask_img
    

