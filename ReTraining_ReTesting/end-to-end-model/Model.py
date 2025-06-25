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

# Training Data
datasetfoldera = 'A_Qian_Cam_A+B/Real'
datasetfolderb = 'B_Quan/Real+Synthetic'
model_folder_paths = '/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/end-to-end-model/5FoldCV'
retrained_model_path = '/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/end-to-end-model/Fold1Retrained'

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

i = 1

fold_1_model_path = os.path.join(model_folder_paths, f'model{i}', 'modelweights_val.pth')

# Load fold1 model best performing weights
if os.path.exists(fold_1_model_path):
    model.load_state_dict(torch.load(fold_1_model_path, map_location=device))
    print(f"Fold {i} model weights loaded successfully.")

print("Train steps per epoch: ", len(train_loader))

# 4) Define criterion & optimizers
#    (Equivalent to TF's BinaryCrossentropy(from_logits=False) if 
#     your Discriminator ends in a sigmoid.)
adv_criterion = nn.BCELoss()

model_optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
disc_optimizer  = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


 # 5) Logging dictionary
my_dict = {
    'model_loss':[], 'mask_loss':[], 'total_gen_loss':[], 'adversarial_loss':[],
    'ssim':[], 'vgg':[], 'ms_loss':[], 'disc_loss':[]
}


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


def fit(train_loader, epochs):
    """
    Training loop, no validation, can save weights every certain epochs and then 
    save model and discriminator weights at the end of training.
    """
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # We'll gather the batch-wise loss dictionaries here
        epoch_losses = []

        # Train loop
        model.train()
        for batch_idx, (rain_img, clean_img, mask_img) in enumerate(train_loader):
            loss_dict = train_step(rain_img, clean_img, mask_img)
            epoch_losses.append(loss_dict)

        # ----------------------------------
        # AVERAGE OVER ALL BATCHES THIS EPOCH
        # ----------------------------------
        # (1) Get the keys, e.g. "model_loss", "disc_loss", etc.
        keys = epoch_losses[0].keys()
        
        # (2) Initialize a sum for each key
        avg_loss = {k: 0.0 for k in keys}
        
        # (3) Sum up losses batch by batch
        for batch_loss_dict in epoch_losses:
            for k in keys:
                avg_loss[k] += batch_loss_dict[k]
        
        # (4) Divide by number of batches to get the mean
        num_batches = len(epoch_losses)
        for k in keys:
            avg_loss[k] /= num_batches
        
        # ----------------------------------
        # Now log these averaged losses
        # ----------------------------------
        my_dict['model_loss'].append(avg_loss['model_loss'])
        my_dict['mask_loss'].append(avg_loss['mask_loss'])
        my_dict['total_gen_loss'].append(avg_loss['total_gen_loss'])
        my_dict['adversarial_loss'].append(avg_loss['adversarial_loss'])
        my_dict['ssim'].append(avg_loss['ssim'])
        my_dict['vgg'].append(avg_loss['vgg'])
        my_dict['ms_loss'].append(avg_loss['ms_loss'])
        my_dict['disc_loss'].append(avg_loss['disc_loss'])

        # Print them
        print(f"  train_gen_loss={avg_loss['total_gen_loss']:.4f}   "
              f"train_disc_loss={avg_loss['disc_loss']:.4f}   "
              f"train_mask_loss={avg_loss['mask_loss']:.4f}   "
              f"model_loss={avg_loss['model_loss']:.4f}   ")

        # Save logs/weights periodically
        if (epoch + 1) % 11 == 0:
            model_save_path = os.path.join(retrained_model_path, 'model_weights.pth')
            disc_save_path = os.path.join(retrained_model_path, 'discriminator_weights.pth')
            torch.save(model.state_dict(), model_save_path)
            torch.save(discriminator.state_dict(), disc_save_path)
            
            loss_path = os.path.join(retrained_model_path, 'loss.pkl')
            with open(loss_path, 'wb') as fp:
                pickle.dump(my_dict, fp)
            print(f"Epoch {epoch+1} weights saved successfully.")

    # ----------------------------------
    # Save final model weights
    # ----------------------------------
    model_save_path = os.path.join(retrained_model_path, 'model_weights_final.pth')
    disc_save_path = os.path.join(retrained_model_path, 'discriminator_weights_final.pth')
    torch.save(model.state_dict(), model_save_path)
    torch.save(discriminator.state_dict(), disc_save_path)

    loss_path = os.path.join(retrained_model_path, 'loss_final.pkl')
    with open(loss_path, 'wb') as fp:
        pickle.dump(my_dict, fp)

    print("Final model weights saved successfully.")
    print("Final discriminator weights saved successfully.")
    print("Losses saved successfully.")
    print("Training completed successfully.")


# Run training

epochs = 44
fit(train_loader, epochs)
