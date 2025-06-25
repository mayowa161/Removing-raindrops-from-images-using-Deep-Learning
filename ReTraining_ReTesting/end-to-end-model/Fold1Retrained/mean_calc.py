import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from torchvision.utils import save_image  # For saving the output images
from torchmetrics.functional import structural_similarity_index_measure, peak_signal_noise_ratio

from classes import EndToEndModel, RaindropDataset
from functions import generator_loss, discriminator_loss  # if needed

# ----------------------------------------------------------------------------------
# 1) Device Setup
# ----------------------------------------------------------------------------------
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# ----------------------------------------------------------------------------------
# 2) Data & Model Setup
# ----------------------------------------------------------------------------------
datasetfoldera = 'A_Qian_Cam_A+B'
datasetfolderb = 'B_Quan'
model_folder_path = '/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/end-to-end-model/Fold1Retrained'

transform = Compose([
    ToTensor(),
])

# Directories for images and masks (Test sets)
rain_images_c_dir ='/data/lady6758/Parmeet_Datasets/NewDatasets/OxfordDataSet/50cm/f16.0/rain_images'
rain_images_d_dir = '/data/lady6758/Parmeet_Datasets/NewDatasets/OxfordDataSetMayowa/f8/rain_images/50cm'
y_mask_c_dir      = '/data/lady6758/Parmeet_Datasets/NewDatasets/OxfordDataSet/50cm/f16.0/KwonMask'
y_mask_d_dir      = '/data/lady6758/Parmeet_Datasets/NewDatasets/OxfordDataSetMayowa/f8/KwonMask-50cm'
clean_images_c_dir= '/data/lady6758/Parmeet_Datasets/NewDatasets/OxfordDataSet/50cm/f16.0/clean_images'
clean_images_d_dir='/data/lady6758/Parmeet_Datasets/NewDatasets/OxfordDataSetMayowa/f8/clean_images'

# Create test datasets
test_c = RaindropDataset(rain_images_c_dir, clean_images_c_dir, y_mask_c_dir, transform=transform)
test_d = RaindropDataset(rain_images_d_dir, clean_images_d_dir, y_mask_d_dir, transform=transform)

print(f"Number of test image-mask pairs in Dataset C: {len(test_c)}")
print(f"Number of test image-mask pairs in Dataset D: {len(test_d)}")

# We use batch_size=1 to process each image individually
test_loader_c = DataLoader(test_c, batch_size=1, shuffle=False)
test_loader_d = DataLoader(test_d, batch_size=1, shuffle=False)

# Function to compute SSIM with TorchMetrics
def ssim(img1, img2):
    """
    Uses TorchMetrics structural_similarity_index_measure,
    returning the SSIM in [0, 1].
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
    return ssim_val

def mse(img1, img2):
    """
    Mean Squared Error
    """
    return F.mse_loss(img1, img2)

def psnr(img1, img2, max_val=1.0):
    """
    PSNR using TorchMetrics
    """
    return peak_signal_noise_ratio(img1, img2, data_range=max_val)

# ----------------------------------------------------------------------------------
# 3) Load the trained model
# ----------------------------------------------------------------------------------
fold_1_weights = os.path.join(model_folder_path, "model_weights_final.pth")
if not os.path.exists(fold_1_weights):
    print(f"ERROR: model weights not found at {fold_1_weights}")
    
fold_model = EndToEndModel().to(device)
fold_model.load_state_dict(torch.load(fold_1_weights, map_location=device))
fold_model.eval()

# ----------------------------------------------------------------------------------
# 4) Create result directories (test_a, test_b) with subfolders for images & masks
# ----------------------------------------------------------------------------------
results_root = "./results"

test_c_image_dir = os.path.join(results_root, "test_c", "images")
test_c_mask_dir  = os.path.join(results_root, "test_c", "masks")
os.makedirs(test_c_image_dir, exist_ok=True)
os.makedirs(test_c_mask_dir, exist_ok=True)

test_d_image_dir = os.path.join(results_root, "test_d", "images")
test_d_mask_dir  = os.path.join(results_root, "test_d", "masks")
os.makedirs(test_d_image_dir, exist_ok=True)
os.makedirs(test_d_mask_dir, exist_ok=True)

# ----------------------------------------------------------------------------------
# 5) Metric accumulators
# ----------------------------------------------------------------------------------
mask_mse_values_c = []
mask_mse_values_d = []
mask_ssim_values_c = []
mask_ssim_values_d = []
gen_psnr_values_c = []
gen_psnr_values_d = []
gen_ssim_values_c = []
gen_ssim_values_d = []

def process_test_set(test_loader, image_save_dir, mask_save_dir, dataset_name="test_c"):
    """
    Processes a test set loader, saves images & masks, and calculates metrics.
    Returns the list of mask_mse_values, mask_ssim_values, gen_psnr_values, gen_ssim_values.
    """
    local_mask_mse = []
    local_mask_ssim = []
    local_gen_psnr = []
    local_gen_ssim = []

    with torch.no_grad():
        for idx, (rain_img, clean_img, true_mask) in enumerate(test_loader):
            # Move to device
            rain_img  = rain_img.to(device)
            clean_img = clean_img.to(device)
            true_mask = true_mask.to(device)

            # Forward pass
            gen_mask, out1, out2, out3 = fold_model(rain_img)

            # (A) Mask metrics: compare gen_mask & true_mask
            batch_mask_mse  = mse(gen_mask, true_mask).item()
            batch_mask_ssim = ssim(gen_mask, true_mask).item()

            local_mask_mse.append(batch_mask_mse)
            local_mask_ssim.append(batch_mask_ssim)

            # (B) Generator output metrics: compare out1 & clean_img
            batch_gen_psnr = psnr(out1, clean_img, max_val=1.0).item()
            batch_gen_ssim = ssim(out1, clean_img).item()

            local_gen_psnr.append(batch_gen_psnr)
            local_gen_ssim.append(batch_gen_ssim)

            # Save the output mask and image
            image_out_path = os.path.join(image_save_dir, f"{dataset_name}_image_{idx:05d}.png")
            mask_out_path  = os.path.join(mask_save_dir,  f"{dataset_name}_mask_{idx:05d}.png")

            # Save images
            save_image(out1, image_out_path)
            save_image(gen_mask, mask_out_path)

    return local_mask_mse, local_mask_ssim, local_gen_psnr, local_gen_ssim

# ----------------------------------------------------------------------------------
# 6) Process the two test sets
# ----------------------------------------------------------------------------------
mse_c, ssim_c, psnr_c, genssim_c = process_test_set(
    test_loader_c, 
    test_c_image_dir, 
    test_c_mask_dir, 
    dataset_name="test_c"
)

mse_d, ssim_d, psnr_d, genssim_d = process_test_set(
    test_loader_d, 
    test_d_image_dir, 
    test_d_mask_dir, 
    dataset_name="test_d"
)

mask_mse_values_c  = mse_c
mask_mse_values_d  = mse_d
mask_ssim_values_c = ssim_c
mask_ssim_values_d = ssim_d
gen_psnr_values_c  = psnr_c
gen_psnr_values_d  = psnr_d
gen_ssim_values_c  = genssim_c
gen_ssim_values_d  = genssim_d

# ----------------------------------------------------------------------------------
# 7) Compute final stats (mean ± std) & Print
# ----------------------------------------------------------------------------------
def mean_std_str(values):
    arr = np.array(values)
    return f"{arr.mean():.6f} ± {arr.std():.6f}"

mask_mse_str_c  = mean_std_str(mask_mse_values_c)
mask_mse_str_d  = mean_std_str(mask_mse_values_d)
mask_ssim_str_c = mean_std_str(mask_ssim_values_c)
mask_ssim_str_d = mean_std_str(mask_ssim_values_d)
gen_psnr_str_c  = mean_std_str(gen_psnr_values_c)
gen_psnr_str_d  = mean_std_str(gen_psnr_values_d)
gen_ssim_str_c  = mean_std_str(gen_ssim_values_c)
gen_ssim_str_d  = mean_std_str(gen_ssim_values_d)

print("Results for test set C:")
print(f"  Mask MSE:   {mask_mse_str_c}")
print(f"  Mask SSIM:  {mask_ssim_str_c}")
print(f"  Gen PSNR:   {gen_psnr_str_c}")
print(f"  Gen SSIM:   {gen_ssim_str_c}")

print("Results for test set D:")
print(f"  Mask MSE:   {mask_mse_str_d}")
print(f"  Mask SSIM:  {mask_ssim_str_d}")
print(f"  Gen PSNR:   {gen_psnr_str_d}")
print(f"  Gen SSIM:   {gen_ssim_str_d}")
