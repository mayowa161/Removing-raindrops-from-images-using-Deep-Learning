import os, copy, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, GaussianBlur
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from pathlib import Path
from UnfairGAN import Generator, Discriminator
from RCF import RCF
from EndtoEnd import EndToEndModel
from SegFormer     import SegFormer, SegFormerUnc, SemiSupSegFormer  # ← your classes
from functions     import *
from classes       import RaindropDataset
from sklearn.model_selection import KFold
import numpy as np
import pickle
import kornia as K
import warnings 
from torch.amp import autocast, GradScaler

warnings.filterwarnings("ignore", category=UserWarning)

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
print("torch.cuda.is_available()=", torch.cuda.is_available(), flush=True)
print("torch.version.cuda=", torch.version.cuda, flush=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device", flush=True)


# Training Data
datasetfoldera = 'A_Qian_Cam_A+B/Real'
datasetfolderb = 'B_Quan/Real+Synthetic'
model_folder_paths = '/users/lady6758/Mayowa_4YP_Code/Novel_Model/5FoldCV'
retrained_model_path = '/users/lady6758/Mayowa_4YP_Code/Novel_Model/Fold2Retrained'
os.makedirs(retrained_model_path, exist_ok=True)

#Define transform of dataset
transform = Compose([
    ToTensor(),
])

transform_strong = torch.nn.Sequential(
    K.filters.GaussianBlur2d((3,3), (0.1,2.0))
).to(device)

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


# Combine datasets
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

# Create Dataloader 
batch_size = 8

train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True, generator=generator)     

vgg_extractor = VGG19_FeatureExtractor(requires_grad=False).to(device)

# 2.  MODEL  (SegFormer‑B2 backbone)
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

Model = EndToEndModel(
    seg_cfg=backbone_cfg,
    in_channels=3,
    mask_channels=1,
    edge_channels=1,
    gan_feats=32
).to(device)


i = 2


fold_dir = os.path.join(model_folder_paths, f"model{i}")

print("Train steps per epoch:", len(train_loader), flush=True)

# Load model weights
Model.seg_model.student.load_state_dict(torch.load(os.path.join(fold_dir, "best_teacher_weights.pth")))
Model.seg_model.teacher.load_state_dict(torch.load(os.path.join(fold_dir, "best_teacher_weights.pth")))
Model.generator.load_state_dict(torch.load(os.path.join(fold_dir, "best_generator_weights.pth")))
Model.discriminator.load_state_dict(torch.load(os.path.join(fold_dir, "best_discriminator_weights.pth")))
print("Model weights loaded successfully", flush=True)

generator_params = list(Model.generator.parameters())
discriminator_params = list(Model.discriminator.parameters())
segformer_params = list(Model.seg_model.student.parameters())

ACT_EPS  = 1e-4
max_grad = 5.0

my_dict = {
    'model_loss':[], 'gen_total': [], 'mask_total':[], 'l1':[], 'ssim':[],
    'perceptual':[], 'gen_adv':[], 'mask_loss':[], 'mask_sup':[], 'mask_cons':[],
    'disc_loss':[]
}

optim_G = torch.optim.Adam(generator_params, lr=2e-4, betas=(0.5, 0.999))
optim_D = torch.optim.Adam(discriminator_params, lr=2e-4, betas=(0.5, 0.999))
optim_seg = optim.AdamW(segformer_params, lr=6e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.PolynomialLR(
    optim_seg, total_iters=len(train_loader)*48, power=1.0)

# Define train and val step
def train_step(rain_img, clean_img, mask_img):

    Model.seg_model.train()
    Model.generator.train()
    Model.discriminator.train()

    # Move data to device
    rain_img = rain_img.to(device, non_blocking=True)
    clean_img = clean_img.to(device, non_blocking=True)
    mask_img = mask_img.to(device, non_blocking=True)

    rain_img_strong = transform_strong(rain_img)
    rain_img_strong = rain_img_strong.to(device, non_blocking=True)

    with torch.no_grad():
        edges = Model.edge_model(rain_img)[-1]  # get the last edge map
    # 1) Generate mask and edges
        
    mask, sigma = Model.seg_model.student(rain_img)

    mask = mask.clamp(0, 1)

    # 2) Generate derained image
    derained = Model.generator(rain_img, mask, edges)

    # Create advanced fake image
    mu = 0.5
    eta = torch.rand_like(rain_img).to(device)

    advanced_fake = derained + mu * eta * (clean_img - derained)

    # 3) Discriminator scores
    with torch.no_grad():
        fake_score_G = Model.discriminator(advanced_fake, mask, edges).clamp(ACT_EPS, 1-ACT_EPS)
        real_score_G = Model.discriminator(clean_img, mask_img, edges).clamp(ACT_EPS, 1-ACT_EPS) 

    # 4) Compute losses
    mask_sup, mask_cons = ssl_loss_step(
        Model.seg_model,
        sup_batch=(rain_img, mask_img),
        unsup_batch=(rain_img, rain_img_strong),
        lambda_cons=0.1,
        thresh=0.3
    )

    model_loss, parts = total_generator_loss(
        derained,
        clean_img,
        mask_img,
        mask,
        real_score_G,
        fake_score_G,
        vgg_extractor = vgg_extractor,
        mask_sup=mask_sup,
        mask_cons=mask_cons,
    )

    real_score_D = Model.discriminator(clean_img, mask_img, edges).clamp(ACT_EPS, 1-ACT_EPS)
    fake_score_D = Model.discriminator(advanced_fake.detach(), mask.detach(), edges.detach()).clamp(ACT_EPS, 1-ACT_EPS)
    disc_loss = discriminator_loss(real_score_D, fake_score_D)

    
    optim_seg.zero_grad(set_to_none=True)
    optim_G.zero_grad(set_to_none=True)
    optim_D.zero_grad(set_to_none=True)

    model_loss.backward()

    torch.nn.utils.clip_grad_norm_(Model.seg_model.student.parameters(), max_grad)
    torch.nn.utils.clip_grad_norm_(Model.generator.parameters(), max_grad)
    for p in list(Model.seg_model.student.parameters()) + list(Model.generator.parameters()):
        if p.grad is not None:
            p.grad = torch.where(torch.isfinite(p.grad), p.grad, torch.zeros_like(p.grad))

   
    optim_seg.step()
    optim_G.step()
    Model.seg_model._update_teacher()  # keep teacher in sync
    scheduler.step()


    # 5) Discriminator loss
    disc_loss.backward()

    torch.nn.utils.clip_grad_norm_(Model.discriminator.parameters(), max_grad)
    for p in Model.discriminator.parameters():
        if p.grad is not None:
            p.grad = torch.where(torch.isfinite(p.grad), p.grad, torch.zeros_like(p.grad))

    optim_D.step()

    log = {
    'model_loss' : model_loss.detach().item(),
    'gen_total'  : parts['gen_total'].detach().item(),
    'mask_total' : parts['mask_total'].detach().item(),
    'l1'         : parts['l1'].detach().item(),
    'ssim'       : parts['ssim'].detach().item(),
    'perceptual' : parts['per'].detach().item(),
    'gen_adv'    : parts['adv'].detach().item(),
    'mask_loss'  : parts['mask_l'].detach().item(),
    'mask_sup'   : parts['mask_sup'].detach().item(),
    'mask_cons'  : parts['mask_cons'].detach().item(),
    'disc_loss'  : disc_loss.detach().item()
        }
    
    return log    

# Fit function
def fit(train_loader, num_epochs):

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}", flush=True)

        epoch_train_losses = []

        # Training
        Model.seg_model.train()
        Model.generator.train()
        Model.discriminator.train()
        for batch_idx, (rain_img, clean_img, mask_img) in enumerate(train_loader):
            log = train_step(rain_img, clean_img, mask_img)
            epoch_train_losses.append(log)

        # Calculate average losses
        avg_train_loss = np.mean([log['model_loss'] for log in epoch_train_losses])
        avg_gen_loss = np.mean([log['gen_total'] for log in epoch_train_losses])
        avg_total_mask_loss = np.mean([log['mask_total'] for log in epoch_train_losses])
        avg_l1 = np.mean([log['l1'] for log in epoch_train_losses])
        avg_ssim = np.mean([log['ssim'] for log in epoch_train_losses])
        avg_perceptual = np.mean([log['perceptual'] for log in epoch_train_losses])
        avg_gen_adv = np.mean([log['gen_adv'] for log in epoch_train_losses])
        avg_mask_loss = np.mean([log['mask_loss'] for log in epoch_train_losses])
        avg_mask_sup = np.mean([log['mask_sup'] for log in epoch_train_losses])
        avg_mask_cons = np.mean([log['mask_cons'] for log in epoch_train_losses])
        avg_disc_loss = np.mean([log['disc_loss'] for log in epoch_train_losses])

        my_dict['model_loss'].append(avg_train_loss)
        my_dict['gen_total'].append(avg_gen_loss)
        my_dict['mask_total'].append(avg_total_mask_loss)
        my_dict['l1'].append(avg_l1)    
        my_dict['ssim'].append(avg_ssim)
        my_dict['perceptual'].append(avg_perceptual)
        my_dict['gen_adv'].append(avg_gen_adv)
        my_dict['mask_loss'].append(avg_mask_loss)
        my_dict['mask_sup'].append(avg_mask_sup)
        my_dict['mask_cons'].append(avg_mask_cons)
        my_dict['disc_loss'].append(avg_disc_loss)

        # Save the model weights
        print(f"Train Loss: {avg_train_loss:.4f}  "
                f"Gen Loss: {avg_gen_loss:.4f}  "
                f"Total Mask Loss: {avg_total_mask_loss:.4f}  "
                f"L1 Loss: {avg_l1:.4f}  "
                f"SSIM Loss: {avg_ssim:.4f}  "
                f"Perceptual Loss: {avg_perceptual:.4f}  "
                f"Gen Adv Loss: {avg_gen_adv:.4f}  "
                f"Mask Loss: {avg_mask_loss:.4f}  "
                f"Mask Sup Loss: {avg_mask_sup:.4f}  "
                f"Mask Cons Loss: {avg_mask_cons:.4f}  "
                f"Disc Loss: {avg_disc_loss:.4f}  "
                , flush=True)
        
        # Save the model weights every 2 epochs
        if (epoch + 1) % 2 == 0:
            torch.save(Model.seg_model.student.state_dict(), os.path.join(retrained_model_path, f"student_weights_epoch_{epoch+1}.pth"))
            torch.save(Model.seg_model.teacher.state_dict(), os.path.join(retrained_model_path, f"teacher_weights_epoch_{epoch+1}.pth"))
            torch.save(Model.generator.state_dict(), os.path.join(retrained_model_path, f"generator_weights_epoch_{epoch+1}.pth"))
            torch.save(Model.discriminator.state_dict(), os.path.join(retrained_model_path, f"discriminator_weights_epoch_{epoch+1}.pth"))
            print(f"Model weights saved for epoch {epoch+1}", flush=True)
        
    # Save the final model weights
    torch.save(Model.seg_model.student.state_dict(), os.path.join(retrained_model_path, "student_weights.pth"))
    torch.save(Model.seg_model.teacher.state_dict(), os.path.join(retrained_model_path, "teacher_weights.pth"))
    torch.save(Model.generator.state_dict(), os.path.join(retrained_model_path, "generator_weights.pth"))
    torch.save(Model.discriminator.state_dict(), os.path.join(retrained_model_path, "discriminator_weights.pth"))
    print("Model weights saved successfully", flush=True)


# Train the model
num_epochs = 48
fit(train_loader, num_epochs)
        
# Save losses to file
fold_log_path = os.path.join(retrained_model_path, "loss.pkl")
os.makedirs(os.path.dirname(fold_log_path), exist_ok=True)
with open(fold_log_path, 'wb') as fp:
    pickle.dump(my_dict, fp)
    print(f"Losses saved successfully", flush=True)



       







    
