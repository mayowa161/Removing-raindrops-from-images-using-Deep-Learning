import torch 
import torch.nn as nn 
# import torch.nn.functional as F 
import torch.optim as optim
import numpy as np
# from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda
# from PIL import Image
from classes import *
# from torchmetrics import StructuralSimilarityIndexMeasure
from functions import *
from sklearn.model_selection import KFold
import os, pickle

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.is_available()=", torch.cuda.is_available())
print("torch.version.cuda=", torch.version.cuda)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# torch.autograd.set_detect_anomaly(True)

# Training Data
datasetfoldera = 'A_Qian_Cam_A+B/Real'
datasetfolderb = 'B_Quan/Real+Synthetic'
model_folder_paths = '/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/end-to-end-model/5FoldCV'

#Define transform of dataset
transform = Compose([
    ToTensor(),
])

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
batch_size = 16

train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, generator=generator)       

# Define the model

model = EndToEndModel().to(device)
discriminator = Discriminator().to(device)

initial_model_weights = os.path.join(model_folder_paths, "model_initial_weights.pth")
initial_disc_weights = os.path.join(model_folder_paths, "discriminator_initial_weights.pth")

torch.save(model.state_dict(), initial_model_weights)
torch.save(discriminator.state_dict(), initial_disc_weights)

#K-Fold Cross Validation

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=12345)
indices = list(range(len(combined_dataset)))

for i, (train_index, val_index) in enumerate(kf.split(indices)):
    print(f"Fold {i}:")
    print(f"  Train indices: {train_index}")
    print(f"  Val indices:   {val_index}")

    if i == 0:
        continue
    if i == 1:
        continue
    # if i == 2:
    #     continue
    # if i == 3:
    #     continue
    # if i ==4:
    #     continue

    fold_dir = os.path.join(model_folder_paths, f"model{i}")
    os.makedirs(fold_dir, exist_ok=True)
    val_save_path = os.path.join(fold_dir, "val.npy")
    np.save(val_save_path, val_index)
    print(f"Validation indices saved to: {val_save_path}")

    # 1) Reload initial weights 
    model.load_state_dict(torch.load(initial_model_weights, map_location=device))
    discriminator.load_state_dict(torch.load(initial_disc_weights, map_location=device))

    # 2) Build subsets for this fold
    train_subset = Subset(combined_dataset, train_index)
    val_subset   = Subset(combined_dataset, val_index)

    # 3) Create DataLoaders
    train_batch = 16
    val_batch   = 16

    train_loader = DataLoader(train_subset, batch_size=train_batch, shuffle=True)
    val_loader   = DataLoader(val_subset,   batch_size=val_batch,   shuffle=False)

    print("Train steps per epoch:", len(train_loader))
    print("Val   steps per epoch:", len(val_loader))

    # 4) Define criterion & optimizers
    #    (Equivalent to TF's BinaryCrossentropy(from_logits=False) if 
    #     your Discriminator ends in a sigmoid.)
    adv_criterion = nn.BCELoss()

    model_optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    disc_optimizer  = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # 5) Logging dictionary
    my_dict = {
        'model_loss':[], 'mask_loss':[], 'total_gen_loss':[], 'adversarial_loss':[],
        'ssim':[], 'vgg':[], 'ms_loss':[], 'disc_loss':[],
        'val_model_loss':[], 'val_disc_loss':[]
    }

    val_model_best = 1e10  # track best validation model loss
    

    # 6) Define train_step & val_step
    def train_step(rain_img, clean_img, mask_img):
        """
        Train step for one batch, with separate G and D updates:
        1) Forward pass for Generator (while optionally ignoring D's gradients),
            then backward & step for G.
        2) Forward pass for Discriminator, backward & step for D.
        """
        model.train()
        discriminator.train()

        # Move data to device
        rain_img = rain_img.to(device)
        clean_img = clean_img.to(device)
        mask_img = mask_img.to(device)

        ################################################################
        # (A) Generator Update
        ################################################################
        # 1) Zero-out the Generator gradients
        model_optimizer.zero_grad()

        # 2) Forward pass for G
        gen_mask_out, out1, out2, out3 = model(rain_img)

        # 3) Evaluate Discriminator on fake (but do not backprop through D)
        #    We can detach disc_fake_out so we won't compute gradients for D.
        disc_fake_out = discriminator(out1).detach()

        # 4) Compute G losses (mask, multi-scale, SSIM, etc.)
        g_loss, adv_loss, ms_val, ssim_val, vgg_val, mask_val = generator_loss(
            disc_gen_out=disc_fake_out,
            gen_mask_out=gen_mask_out,
            out256=out1,
            out128=out2,
            out64=out3,
            target_image=clean_img,  # ground-truth clean
            target_mask=mask_img,
            adv_criterion=adv_criterion
        )

        # 5) Backprop & step for G
        g_loss.backward()
        model_optimizer.step()

        ################################################################
        # (B) Discriminator Update
        ################################################################
        # 1) Zero-out the Discriminator gradients
        disc_optimizer.zero_grad()

        # 2) Forward pass real & fake
        #    - Real
        disc_real_out = discriminator(clean_img)
        
        #    - Fake (re-run generator or reuse out1; best to re-run so D sees
        #      newly updated G – but do so in no_grad to avoid building the G graph)
        with torch.no_grad():
            _, out1_new, _, _ = model(rain_img)
        disc_fake_out2 = discriminator(out1_new)

        # 3) Compute D loss
        d_loss = discriminator_loss(disc_real_out, disc_fake_out2, adv_criterion)

        # 4) Backprop & step for D
        d_loss.backward()
        disc_optimizer.step()

        return {
            'model_loss': g_loss.item(),
            'mask_loss': mask_val.item(),
            'total_gen_loss': (g_loss - mask_val).item(),
            'adversarial_loss': adv_loss.item(),
            'ssim': ssim_val.item(),
            'vgg': vgg_val.item(),
            'ms_loss': ms_val.item(),
            'disc_loss': d_loss.item()
        }


    @torch.no_grad()
    def val_step(rain_img, clean_img, mask_img):
        """
        Validation step for one batch:
        - no grad
        - forward generator & discriminator
        - compute losses
        """
        model.eval()
        discriminator.eval()

        rain_img = rain_img.to(device)
        clean_img = clean_img.to(device)
        mask_img = mask_img.to(device)

        gen_mask_out, out1, out2, out3 = model(rain_img)
        disc_real_out = discriminator(clean_img)
        disc_fake_out = discriminator(out1)

        g_loss, _, _, _, _, _ = generator_loss(
            disc_fake_out, 
            gen_mask_out,
            out1, out2, out3,
            clean_img,
            mask_img,
            adv_criterion
        )
        d_loss = discriminator_loss(disc_real_out, disc_fake_out, adv_criterion)
        return g_loss.item(), d_loss.item()

    # 7) Fit function
    def fit(train_loader, val_loader, epochs):
        val_model_best = 1000
        best_epoch = 0
        for epoch in range(epochs):
            # accumulators for logging
            epoch_train_losses = []
            epoch_val_gen = []
            epoch_val_disc = []

            print(f"\nEpoch {epoch+1}/{epochs} [Fold {i}]")

            # Training loop
            model.train()
            for batch_idx, (rain_img, clean_img, mask_img) in enumerate(train_loader):
                loss_dict = train_step(rain_img, clean_img, mask_img)
                epoch_train_losses.append(loss_dict)

            # Validation loop
            model.eval()
            for val_idx, (vrain, vclean, vmask) in enumerate(val_loader):
                g_l, d_l = val_step(vrain, vclean, vmask)
                epoch_val_gen.append(g_l)
                epoch_val_disc.append(d_l)

            # Summaries
            avg_train = loss_dict  # last batch’s stats or do mean over entire epoch
            val_model_loss = float(np.mean(epoch_val_gen))
            val_disc_loss  = float(np.mean(epoch_val_disc))

            # log final batch’s training stats
            my_dict['model_loss'].append(avg_train['model_loss'])
            my_dict['mask_loss'].append(avg_train['mask_loss'])
            my_dict['total_gen_loss'].append(avg_train['total_gen_loss'])
            my_dict['adversarial_loss'].append(avg_train['adversarial_loss'])
            my_dict['ssim'].append(avg_train['ssim'])
            my_dict['vgg'].append(avg_train['vgg'])
            my_dict['ms_loss'].append(avg_train['ms_loss'])
            my_dict['disc_loss'].append(avg_train['disc_loss'])
            # log val
            my_dict['val_model_loss'].append(val_model_loss)
            my_dict['val_disc_loss'].append(val_disc_loss)

            print(f"train_gen_loss={avg_train['model_loss']:.4f}   "
                  f"train_disc_loss={avg_train['disc_loss']:.4f}   "
                  f"val_gen_loss={val_model_loss:.4f}   val_disc_loss={val_disc_loss:.4f}")

            # Save partial logs/weights every 5 epochs
            if (epoch+1) % 10 == 0:
                fold_log_path = os.path.join(model_folder_paths, f"model{i}", "loss.pkl")
                os.makedirs(os.path.dirname(fold_log_path), exist_ok=True)
                with open(fold_log_path, 'wb') as fp:
                    pickle.dump(my_dict, fp)
                    print("Dictionary saved successfully to file")

                fold_model_path = os.path.join(model_folder_paths, f"model{i}", "model_weights.pth")
                torch.save(model.state_dict(), fold_model_path)

            # Check best val
            if val_model_loss < val_model_best:
                val_model_best = val_model_loss
                best_epoch = epoch + 1
                best_val_path = os.path.join(model_folder_paths, f"model{i}", "modelweights_val.pth")
                torch.save(model.state_dict(), best_val_path)
                print(f"Saved best val Weights at epoch {best_epoch} with loss {val_model_best:.4f}")

    # 8) Actually train
    fit(train_loader, val_loader, epochs=50)

    # Save final logs
    fold_log_path = os.path.join(model_folder_paths, f"model{i}", "loss.pkl")
    with open(fold_log_path, 'wb') as fp:
        pickle.dump(my_dict, fp)
        print("Dictionary saved successfully to file")

    # Cleanup 
    del train_subset
    del val_subset
    del train_loader
    del val_loader
    torch.cuda.empty_cache()
