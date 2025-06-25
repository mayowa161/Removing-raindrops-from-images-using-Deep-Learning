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

class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x
    

class DownSample(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(DownSample, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride)

    def forward(self, x):
        return self.pool(x)
    
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)
        
    def forward(self, x):
        return self.up(x)
    
class Mask_Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(Mask_Generator, self).__init__()

        # Define generator architecture with blocks
        self.cbr1 = CBR(in_channels, 64)
        self.cbr2 = CBR(64, 64)
        self.down1 = DownSample()
        self.cbr3 = CBR(64, 128)
        self.cbr4 = CBR(128, 128)
        self.down2 = DownSample()
        self.cbr5 = CBR(128, 256)
        self.cbr6 = CBR(256, 256)
        self.down3 = DownSample()
        self.cbr7 = CBR(256, 512)
        self.up1 = UpSample(512, 256)
        self.cbr8 = CBR(256, 256)
        self.cbr9 = CBR(256, 256)
        self.up2 = UpSample(256, 128)
        self.cbr10 = CBR(128, 128)
        self.cbr11 = CBR(128, 128)
        self.up3 = UpSample(128, 64)
        self.cbr12 = CBR(64, 64)
        self.cbr13 = CBR(64, 64)
        self.conv1 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x1 = self.cbr1(x)
        x2 = self.cbr2(x1)
        x3 = self.down1(x2)
        x4 = self.cbr3(x3)
        x5 = self.cbr4(x4)
        x6 = self.down2(x5)
        x7 = self.cbr5(x6)
        x8 = self.cbr6(x7)
        x9 = self.down3(x8)
        x10 = self.cbr7(x9)
        x11 = self.up1(x10)
        sc1 = x11 + x8
        x12 = self.cbr8(sc1)
        x13 = self.cbr9(x12)
        x14 = self.up2(x13)
        sc2 = x14 + x5
        x15 = self.cbr10(sc2)
        x16 = self.cbr11(x15)
        x17 = self.up3(x16)
        sc3 = x17 + x2
        x18 = self.cbr12(sc3)
        x19 = self.cbr13(x18)
        x20 = self.conv1(x19)
        return x20
    

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


class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(CBL, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.lrelu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out= torch.max(x, dim=1, keepdim=True).values
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out)
    

class RCBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(RCBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention()
        self.CBL = CBL(in_channels, in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
    
    def forward(self, inputs):
        x = self.CBL(inputs)
        x = self.ca(x) * x
        x = self.sa(x) * x
        out = x + self.conv1(inputs)
        return out
    

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.CBL1 = CBL(in_channels, out_channels)
        self.RCBAM = RCBAM(out_channels)
        self.MaxPool = DownSample(kernel_size=2, stride=2)
         
    def forward(self, x):
        x = self.CBL1(x)
        res = self.RCBAM(x)
        x = self.MaxPool(res)
        return x, res
        
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.UpSample = UpSample(in_channels, in_channels // 2)
        self.CBL = CBL(in_channels, out_channels)
        self.RCBAM = RCBAM(out_channels)

    def forward(self, input, res):
        x_res = self.UpSample(input)
        x = torch.cat((x_res, res), dim=1)
        x = self.CBL(x)
        x = self.RCBAM(x)
        x = x + x_res.clone()
        return x
    

class EndToEndModel(nn.Module):
    def __init__(self, mask_in_channels=3, mask_out_channels=1, gen_in_channels=4):
        super(EndToEndModel, self).__init__()


        # Mask Generator
        self.mask_generator = Mask_Generator(mask_in_channels, mask_out_channels)

        # Image Generator

        self.down1 = Down(gen_in_channels, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 1024)

        self.mid_cbl = CBL(1024, 2048)
        self.mid_RCBAM = RCBAM(2048)

        self.up1 = Up(2048, 1024)
        self.up2 = Up(1024, 512)
        self.up3 = Up(512, 256)
        self.up4 = Up(256, 128)
        self.up5 = Up(128, 64)

        self.outconv1_64 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.outconv2_128 = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.outconv3_256 = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

        self.sigmoid = nn.Sigmoid()


    def forward(self, input):

        generated_mask = self.mask_generator(input)

        generator_input = torch.cat((input, generated_mask), dim=1)

        down1, res1 = self.down1(generator_input)
        down2, res2 = self.down2(down1)
        down3, res3 = self.down3(down2)
        down4, res4 = self.down4(down3)
        down5, res5 = self.down5(down4)

        mid = self.mid_cbl(down5)
        mid = self.mid_RCBAM(mid)

        up1 = self.up1(mid, res5)
        up2 = self.up2(up1, res4)
        out3 = self.up3(up2, res3)
        out2 = self.up4(out3, res2)
        out1 = self.up5(out2, res1)

        ms_output1 = self.sigmoid(self.outconv1_64(out1))
        ms_output2 = self.sigmoid(self.outconv2_128(out2))
        ms_output3 = self.sigmoid(self.outconv3_256(out3))

        return generated_mask, ms_output1, ms_output2, ms_output3


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        # mirror the TF structure, using Conv->BN->LeakyReLU blocks
        # final output is a single channel with sigmoid activation

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1),  
            # if you want the sigmoid built in:
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)



  
