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
# batch_size = 8

# train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, generator=generator)     

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

print("Downloading ImageNet weights…")
local_bin = hf_hub_download("nvidia/mit-b2", "pytorch_model.bin", repo_type="model")
state = torch.load(local_bin, map_location="cpu")
missing, _ = Model.seg_model.student.backbone.encoder.load_state_dict(state, strict=False)
print("✔ ImageNet init   missing keys:", len(missing))

# Save different model weights
student_weights_path = os.path.join(model_folder_paths,'initial_weights', 'initial_student_weights.pth')
# teacher_weights_path = os.path.join(model_folder_paths, 'initial_weights', 'initial_teacher_weights.pth')
generator_weights_path = os.path.join(model_folder_paths, 'initial_weights', 'initial_generator_weights.pth')
discriminator_weights_path = os.path.join(model_folder_paths,'initial_weights', 'initial_discriminator_weights.pth')

# Save the model weights
# torch.save(Model.seg_model.student.state_dict(), student_weights_path)
# # torch.save(Model.seg_model.teacher.state_dict(), teacher_weights_path)
# torch.save(Model.generator.state_dict(), generator_weights_path)
# torch.save(Model.discriminator.state_dict(), discriminator_weights_path)

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=12345)
indices = list(range(len(combined_dataset)))

for i, (train_index, val_index) in enumerate(kf.split(indices)):
    print(f"Fold {i+1}/{n_splits}")
    print(f"Train indices: {train_index}")
    print(f"Validation indices: {val_index}")

    if i == 0:
        continue
    if i == 1:
        continue
    if i == 2:
        continue
    # if i == 3:
    #     continue
    if i == 4:
        continue

    fold_dir = os.path.join(model_folder_paths, f"model{i}")
    os.makedirs(fold_dir, exist_ok=True)
    val_save_path = os.path.join(fold_dir, "val.npy")
    np.save(val_save_path, val_index)
    print(f"Validation indices saved to: {val_save_path}")

    # Reload initial model weights
    Model.seg_model.student.load_state_dict(torch.load(student_weights_path))
    Model.seg_model.teacher.load_state_dict(Model.seg_model.student.state_dict())
    Model.generator.load_state_dict(torch.load(generator_weights_path))
    Model.discriminator.load_state_dict(torch.load(discriminator_weights_path))

    # Create DataLoader for training and validation
    train_batch = 8
    val_batch = 8

    train_subset = Subset(combined_dataset, train_index)
    val_subset = Subset(combined_dataset, val_index)

    train_loader = DataLoader(train_subset, batch_size=train_batch, shuffle=True, num_workers=1, pin_memory=True, persistent_workers=True)
    # train_loader = DataLoader(train_subset, batch_size=train_batch, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=val_batch, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True)
    # val_loader = DataLoader(val_subset, batch_size=val_batch, shuffle=False)

    print(f"Train subset size: {len(train_subset)}")
    print(f"Validation subset size: {len(val_subset)}")

    generator_params = list(Model.generator.parameters())
    discriminator_params = list(Model.discriminator.parameters())
    segformer_params = list(Model.seg_model.student.parameters())

    scaler = GradScaler(device=device, enabled=(device == "cuda"))
    ACT_EPS  = 1e-4
    max_grad = 5.0

    my_dict = {
        'model_loss':[], 'gen_total': [], 'mask_total':[], 'l1':[], 'ssim':[],
        'pereptual':[], 'gen_adv':[], 'mask_loss':[], 'mask_sup':[], 'mask_cons':[],
        'disc_loss':[], 'val_model_loss':[], 'val_disc_loss':[]
    }

    optim_G = torch.optim.Adam(generator_params, lr=2e-4, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(discriminator_params, lr=2e-4, betas=(0.5, 0.999))
    optim_seg = optim.AdamW(segformer_params, lr=6e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
       optim_seg, total_iters=len(train_loader)*50, power=1.0)
    
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

        # optim_G.zero_grad(set_to_none=True)
        # optim_D.zero_grad(set_to_none=True)

        with torch.no_grad():
            edges = Model.edge_model(rain_img)[-1]  # get the last edge map
        # 1) Generate mask and edges
        with autocast(device_type='cuda', dtype=torch.float16):
            
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

        scaler.scale(model_loss).backward(retain_graph=True)
        scaler.unscale_(optim_seg)
        scaler.unscale_(optim_G)
        torch.nn.utils.clip_grad_norm_(Model.seg_model.student.parameters(), max_grad)
        torch.nn.utils.clip_grad_norm_(Model.generator.parameters(), max_grad)
        for p in list(Model.seg_model.student.parameters()) + list(Model.generator.parameters()):
            if p.grad is not None:
                p.grad = torch.where(torch.isfinite(p.grad), p.grad, torch.zeros_like(p.grad))
        scaler.step(optim_seg)
        scaler.step(optim_G)

        scaler.scale(disc_loss).backward()
        scaler.unscale_(optim_D)
        torch.nn.utils.clip_grad_norm_(Model.discriminator.parameters(), max_grad)
        for p in Model.discriminator.parameters():
            if p.grad is not None:
                p.grad = torch.where(torch.isfinite(p.grad), p.grad, torch.zeros_like(p.grad))
        scaler.step(optim_D)
        scaler.update()

        # model_loss.backward()
        # optim_seg.step()
        # optim_G.step()
        Model.seg_model._update_teacher()  # keep teacher in sync
        scheduler.step()


        # 5) Discriminator loss
        # disc_loss.backward()
        # optim_D.step()

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
    
    @torch.no_grad()
    def val_step(rain_img, clean_img, mask_img):
        Model.eval()

        rain_img = rain_img.to(device)
        clean_img = clean_img.to(device)
        mask_img = mask_img.to(device)

        # 1) Generate mask and edges
        with autocast(device_type='cuda', dtype=torch.float16):
            edges = Model.edge_model(rain_img)[-1]  # get the last edge map
            mask, sigma = Model.seg_model.teacher(rain_img)
            # 2) Generate derained image
            derained = Model.generator(rain_img, mask, edges)
            # 3) Discriminator scores
            fake_score_G = Model.discriminator(derained, mask, edges).clamp(ACT_EPS, 1-ACT_EPS)
            real_score_G = Model.discriminator(clean_img, mask_img, edges).clamp(ACT_EPS, 1-ACT_EPS)
            # 4) Compute losses
            model_loss, _ = total_generator_loss(
                derained,
                clean_img,
                mask_img,
                mask,
                real_score_G,
                fake_score_G,
                vgg_extractor = vgg_extractor,
                mask_sup=None,
                mask_cons=None,
            )
            # 5) Discriminator loss
            real_score_D = Model.discriminator(clean_img, mask_img, edges).clamp(ACT_EPS, 1-ACT_EPS)
            fake_score_D = Model.discriminator(derained.detach(), mask.detach(), edges.detach()).clamp(ACT_EPS, 1-ACT_EPS)
            disc_loss = discriminator_loss(real_score_D, fake_score_D)
        log = {
            'val_model_loss' : model_loss.detach().item(),
            'val_disc_loss'  : disc_loss.detach().item()
        }
        return log
    
    # Fit function
    def fit(train_loader, val_loader, num_epochs):
        val_model_best = 1e5
        best_epoch = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs} [Fold {i}]")

            epoch_train_losses = []
            epoch_val_model_losses = []
            epoch_val_disc_losses = []

            # Training
            Model.seg_model.train()
            Model.generator.train()
            Model.discriminator.train()
            for batch_idx, (rain_img, clean_img, mask_img) in enumerate(train_loader):
                log = train_step(rain_img, clean_img, mask_img)
                epoch_train_losses.append(log)

            # Validation
            Model.seg_model.eval()
            Model.generator.eval()
            for batch_idx, (rain_img, clean_img, mask_img) in enumerate(val_loader):
                log = val_step(rain_img, clean_img, mask_img)
                epoch_val_model_losses.append(log['val_model_loss'])
                epoch_val_disc_losses.append(log['val_disc_loss'])
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
            avg_val_model_loss = np.mean(epoch_val_model_losses)
            avg_val_disc_loss = np.mean(epoch_val_disc_losses)


            my_dict['model_loss'].append(avg_train_loss)
            my_dict['gen_total'].append(avg_gen_loss)
            my_dict['mask_total'].append(avg_total_mask_loss)
            my_dict['l1'].append(avg_l1)    
            my_dict['ssim'].append(avg_ssim)
            my_dict['pereptual'].append(avg_perceptual)
            my_dict['gen_adv'].append(avg_gen_adv)
            my_dict['mask_loss'].append(avg_mask_loss)
            my_dict['mask_sup'].append(avg_mask_sup)
            my_dict['mask_cons'].append(avg_mask_cons)
            my_dict['disc_loss'].append(avg_disc_loss)
            my_dict['val_model_loss'].append(avg_val_model_loss)
            my_dict['val_disc_loss'].append(avg_val_disc_loss)

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
                  f"Val Model Loss: {avg_val_model_loss:.4f}  "
                  f"Val Disc Loss: {avg_val_disc_loss:.4f}", flush=True)
            
            

            # Save the model weights
            if avg_val_model_loss < val_model_best:
                val_model_best = avg_val_model_loss
                best_epoch = epoch + 1
                torch.save(Model.seg_model.student.state_dict(), os.path.join(fold_dir, "best_student_weights.pth"))
                torch.save(Model.seg_model.teacher.state_dict(), os.path.join(fold_dir, "best_teacher_weights.pth"))
                torch.save(Model.generator.state_dict(), os.path.join(fold_dir, "best_generator_weights.pth"))
                torch.save(Model.discriminator.state_dict(), os.path.join(fold_dir, "best_discriminator_weights.pth"))
                print(f"Best model weights saved for fold {i} at epoch {epoch+1}", flush=True)
        print(f"Best model for fold {i} saved at epoch {best_epoch} with loss {val_model_best:.4f}", flush=True)

    # Train the model
    num_epochs = 50
    fit(train_loader, val_loader, num_epochs)
            
    # Save losses to file
    fold_log_path = os.path.join(model_folder_paths, f"model{i}", "loss.pkl")
    os.makedirs(os.path.dirname(fold_log_path), exist_ok=True)
    with open(fold_log_path, 'wb') as fp:
        pickle.dump(my_dict, fp)
        print(f"Losses for fold {i} saved successfully", flush=True)

    # Clear memory
    del train_subset
    del val_subset
    del train_loader
    del val_loader
    torch.cuda.empty_cache()
    print("Cleared memory", flush=True)

       







    
