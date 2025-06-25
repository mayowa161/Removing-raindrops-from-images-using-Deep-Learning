import torch 
# import torch.nn as nn 
import torch.nn.functional as F 
# import torch.optim as optim
import numpy as np
# from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import Compose, ToTensor
# from PIL import Image
import os
from classes import *
from functions import *
from sklearn.model_selection import KFold
from EndtoEnd import EndToEndModel
from SegFormer import SegFormer
from torchmetrics.functional import structural_similarity_index_measure, peak_signal_noise_ratio

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Training Data
datasetfoldera = 'A_Qian_Cam_A+B/Real'
datasetfolderb = 'B_Quan/Real+Synthetic'
model_folder_paths = '/users/lady6758/Mayowa_4YP_Code/Novel_Model/5FoldCV'

#Define transform of dataset
transform = Compose([
    ToTensor(),
    ]) # Convert to tensors and normalise from [0,1]

# Directories for images and masks
rain_images_a_dir = '/data/lady6758/Parmeet_Datasets/NewDatasets/'+datasetfoldera+'/rain_images'
rain_images_b_dir = '/data/lady6758/Parmeet_Datasets/NewDatasets/'+datasetfolderb+'/rain_images'
y_mask_a_dir = '/data/lady6758/Parmeet_Datasets/NewDatasets/'+datasetfoldera+'/KwonMask'
y_mask_b_dir = '/data/lady6758/Parmeet_Datasets/NewDatasets/'+datasetfolderb+'/KwonMask'
clean_images_a_dir = '/data/lady6758/Parmeet_Datasets/NewDatasets/'+datasetfoldera+'/clean_images'
clean_images_b_dir = '/data/lady6758/Parmeet_Datasets/NewDatasets/'+datasetfolderb+'/clean_images'

# Create datasets
dataset_a = RaindropDataset(rain_images_a_dir, clean_images_a_dir, y_mask_a_dir, transform=transform)
dataset_b = RaindropDataset(rain_images_b_dir, clean_images_b_dir, y_mask_b_dir, transform=transform)

combined_dataset = torch.utils.data.ConcatDataset([dataset_a, dataset_b])

first_rain, first_clean, first_mask = combined_dataset[0]
print(f"Rain shape:  {first_rain.shape}")   # e.g. (3, H, W)
print(f"Clean shape: {first_clean.shape}")  # e.g. (3, H, W)
print(f"Mask shape:  {first_mask.shape}")   # e.g. (1, H, W)

print(f"Number of image-mask pairs in Dataset A: {len(dataset_a)}")
print(f"Number of image-mask pairs in Dataset B: {len(dataset_b)}")
print(f"Total number of image-mask pairs in Combined Dataset: {len(combined_dataset)}")

# Set seed for reproducibility
torch.manual_seed(12345)
generator = torch.Generator()
generator.manual_seed(12345)

# batch_size = 16

# train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, generator=generator) 

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=12345)
indices = list(range(len(combined_dataset)))



def ssim(img1, img2):
    """
    A purely functional SSIM, returning 1 - SSIM for use as a loss.
    Both img1 and img2 should be (N, C, H, W) in [0,1].
    """
    ssim_val = structural_similarity_index_measure(
        img1, 
        img2,
        data_range=1.0,
        kernel_size=(11, 11),
        sigma=1.5,
        k1=0.01,
        k2=0.03
    )
    return ssim_val.mean()

def mse(img1, img2):
    """
    Computes the Mean Squared Error (MSE) between img1 and img2.
    Returns a single scalar if img1 and img2 have the same shape (N, C, H, W).
    """
    return F.mse_loss(img1, img2)

def psnr(img1, img2, max_val=1.0):
    """
    Computes PSNR using TorchMetrics' built-in function.
    """
    return peak_signal_noise_ratio(img1, img2, data_range=max_val)

# Number of folds
num_folds = 5

# We'll assume these functions are defined elsewhere:
#   ssim_metric(img1, img2) -> float
#   mse_metric(img1, img2) -> float
#   psnr_metric(img1, img2, max_val=1.0) -> float

for i in range(num_folds):
    print(f"\n=== Evaluating Fold {i} ===")

    if i == 0:
        continue
    if i == 1:
        continue
    # if i == 2:
    #     continue
    if i == 3:
        continue
    if i ==4:
        continue

    # 1) Load the validation indices for this fold
    val_indices_path = os.path.join(model_folder_paths, f"model{i}", "val.npy")
    if not os.path.exists(val_indices_path):
        print(f"Fold {i} val.npy not found, skipping.")
        continue
    val_indices = np.load(val_indices_path)
    print(f"Loaded {len(val_indices)} validation indices from {val_indices_path}")

    # 2) Build the validation subset/DataLoader
    val_subset = Subset(combined_dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)

    # 3) Load the best model weights for this fold
    best_generator_weights_path = os.path.join(model_folder_paths, f"model{i}", "best_generator_weights.pth")
    if not os.path.exists(best_generator_weights_path):
        print(f"Fold {i} best weights not found at {best_generator_weights_path}, skipping.")
        continue

    best_teacher_weights_path = os.path.join(model_folder_paths, f"model{i}", "best_teacher_weights.pth")
    if not os.path.exists(best_teacher_weights_path):
        print(f"Fold {i} best weights not found at {best_teacher_weights_path}, skipping.")
        continue

    backbone = SegFormer(
    in_channels=3,
    widths=[64,128,320,512],
    depths=[3,4,6,3],
    all_num_heads=[1,2,5,8],
    patch_sizes=[7,3,3,3],
    overlap_sizes=[4,2,2,2],
    reduction_ratios=[64,16,4,1],
    mlp_expansions=[4,4,4,4],
    decoder_channels=256,
    scale_factors=[8,4,2,1],
    num_classes=1
    )

    backbone_cfg = dict(segformer_backbone=backbone, mid_channels=64)

    # Initialize the model and load weights
    Model = EndToEndModel(
    seg_cfg=backbone_cfg,
    in_channels=3,
    mask_channels=1,
    edge_channels=1,
    gan_feats=32
    ).to(device)



    Model.generator.load_state_dict(torch.load(best_generator_weights_path, map_location=device))
    Model.seg_model.teacher.load_state_dict(torch.load(best_teacher_weights_path, map_location=device))
    Model.eval()

    # 4) Lists to store per-sample metrics
    mask_mse_values = []
    mask_ssim_values = []
    gen_psnr_values = []
    gen_ssim_values = []

    # 5) Inference on validation set
    with torch.no_grad():
        for (rain_img, clean_img, true_mask) in val_loader:
            # Move to device
            rain_img = rain_img.to(device)
            clean_img = clean_img.to(device)
            true_mask = true_mask.to(device)

            # forward pass => (generated_mask, ms_output1, ms_output2, ms_output3)
            out = Model(rain_img, use_teacher=True)

            gen_mask = out['mask']
            derained = out['derained']

            # (A) Mask metrics
            # gen_mask: shape [N,1,H,W], true_mask: shape [N,1,H,W]
            # MSE
            batch_mask_mse = mse(gen_mask, true_mask)
            # SSIM
            batch_mask_ssim = ssim(gen_mask, true_mask)

            # Convert to python float or array
            mask_mse_values.append(batch_mask_mse.item())
            mask_ssim_values.append(batch_mask_ssim.item())

            # (B) Generator metrics: compare ms_output1 with the clean image
            # PSNR
            batch_gen_psnr = psnr(derained, clean_img, max_val=1.0)
            # SSIM
            batch_gen_ssim = ssim(derained, clean_img)

            gen_psnr_values.append(batch_gen_psnr.item())
            gen_ssim_values.append(batch_gen_ssim.item())

    # 6) Compute mean ± std for each metric
    def mean_std_str(values):
        arr = np.array(values)
        return f"{arr.mean():.6f} ± {arr.std():.6f}"

    mask_mse_str = mean_std_str(mask_mse_values)
    mask_ssim_str = mean_std_str(mask_ssim_values)
    gen_psnr_str = mean_std_str(gen_psnr_values)
    gen_ssim_str = mean_std_str(gen_ssim_values)

    # 7) Print results
    print(f"Fold {i} Results on Validation Set:")
    print(f"  Mask MSE:   {mask_mse_str}")
    print(f"  Mask SSIM:  {mask_ssim_str}")
    print(f"  Gen PSNR:   {gen_psnr_str}")
    print(f"  Gen SSIM:   {gen_ssim_str}")