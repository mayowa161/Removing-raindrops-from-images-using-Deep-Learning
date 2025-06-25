import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torchmetrics.functional import structural_similarity_index_measure # For the SSIM-based loss (if desired)
import torchvision.transforms as T
import torchvision.models as models




def bilinear_kernel(num_channels: int, upscale: int) -> torch.Tensor:
    """Return a (C, 1, k, k) kernel that performs bilinear up-sampling."""
    kernel_size = upscale
    factor = (upscale + 1) // 2
    center = factor -1 if kernel_size % 2 == 0 else factor - 0.5
    og = torch.arange(kernel_size).float()
    filt = (1 - torch.abs(og - center) / factor).unsqueeze(0)
    filt = filt @ filt.t()
    weight = torch.zeros(num_channels, 1, upscale, upscale)
    weight[:, 0, :, :] = filt
    return weight


def het_mse_loss(pred_mask, sigma, gt_mask):
    """
    Heteroscedastic negative log‑likelihood with pixel‑wise Gaussian noise.
    pred_mask, gt_mask ∈ [0,1];   sigma is log(variance).
    """
    sigma = sigma.clamp(min = -7, max = 7)  # clamp to avoid numerical issues
    mse = (pred_mask - gt_mask).pow(2)           # (B,1,H,W)
    loss = 0.5 * torch.exp(-sigma).clamp(max=100.0) * mse + 0.5 * sigma
    return loss.mean()


# 5. Semi‑supervised loss step  (put in your train loop)
def ssl_loss_step(model,
                  sup_batch,          # (img_l, mask_l)
                  unsup_batch,        # (img_u_w, img_u_s)
                  lambda_cons=0.1,
                  thresh=0.3):
    img_l, mask_l       = sup_batch
    img_u_w, img_u_s    = unsup_batch


    # ---------- supervised -----------------------------------
    pred_l, sigma_l = model.student(img_l)
    sigma_l = sigma_l.clamp(-7,7)
    l_sup = het_mse_loss(pred_l, sigma_l, mask_l)

    # ---------- teacher prediction ---------------------------
    with torch.no_grad():
        pred_t, _ = model.teacher(img_u_w)
        pred_t = pred_t.detach()

    # ---------- student prediction (strong aug) --------------
    pred_s, sigma_s = model.student(img_u_s)

    sigma_s = sigma_s.clamp(-7,7)
    # uncertainty‑weighted L2 consistency
    w     = torch.exp(-sigma_s).detach().clamp(max=100.0)             # (B,1,H,W)
    mask  = (w > thresh).float()
    l_cons= (w * mask * (pred_s - pred_t).pow(2)).mean()
    l_cons = lambda_cons * l_cons

    return l_sup, l_cons


class VGG19_FeatureExtractor(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19_FeatureExtractor, self).__init__()
        vgg_pretrained = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # For instance, slice up to relu4_4:
        self.features = vgg_pretrained.features
        if not requires_grad:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.features(x)

# Normalization for VGG
vgg_norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])

def perceptual_vgg_loss(vgg_extractor, target, gen):
    """
    Both target & gen are Bx3xHxW in [0,1]. 
    We apply standard ImageNet normalization and compare.
    """
    # Normalize
    t_norm = vgg_norm(target)
    g_norm = vgg_norm(gen)
    feat_t = vgg_extractor(t_norm)
    feat_g = vgg_extractor(g_norm)
    return F.mse_loss(feat_g, feat_t)


def ssim_loss(img1, img2):
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
    return 1.0 - ssim_val


def mask_loss(real_mask, gen_mask):
    """
    MSE between real mask and generated mask + SSIM Loss
    """
    mse = F.mse_loss(gen_mask, real_mask)
    ssim = ssim_loss(gen_mask, real_mask)

    return mse + ssim



def discriminator_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """
    Relativistic LSGAN discriminator loss:
    L_D = 0.5 * [mean((Dr - mean(Df) - 1)^2) + mean((Df - mean(Dr) + 1)^2)]
    where Dr = real_logits, Df = fake_logits (raw, pre‑sigmoid scores).
    """
    Dr = real
    Df = fake
    Dr_mean = Dr.mean()
    Df_mean = Df.mean()
    loss_real = ((Dr - Df_mean - 1) ** 2).mean()
    loss_fake = ((Df - Dr_mean + 1) ** 2).mean()
    return 0.5 * (loss_real + loss_fake)

def generator_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """
    Relativistic LSGAN generator loss:
    L_G = 0.5 * [mean((Df - mean(Dr) - 1)^2) + mean((Dr - mean(Df) + 1)^2)]
    """
    Dr = real
    Df = fake
    Dr_mean = Dr.mean()
    Df_mean = Df.mean()
    loss_fake = ((Df - Dr_mean - 1) ** 2).mean()
    loss_real = ((Dr - Df_mean + 1) ** 2).mean()
    return 0.5 * (loss_fake + loss_real)

def total_generator_loss(gen_out: torch.Tensor,
                         real_img: torch.Tensor,
                         real_mask: torch.Tensor,
                         gen_mask: torch.Tensor,
                         real_score: torch.Tensor,
                         fake_score: torch.Tensor,
                         vgg_extractor: torch.nn.Module = VGG19_FeatureExtractor(requires_grad=False),
                         # -------- pre-computed SSL pieces -------------------
                         mask_sup : torch.Tensor = None,  # l_sup_u  (scalar)
                         mask_cons: torch.Tensor = None,  # l_cons   (scalar)
                         # -------- weights -----------------------------------
                         w_l1: float = 0.16,
                         w_ssim: float = 0.84,
                         w_per: float = 0.001,
                         w_adv: float = 0.01):
                         
    """
    UnfairGAN L1 + SSIM + perceptual + relativistic adversarial
    + supervised-mask + consistency-mask.
    Returns (total_loss, dict_of_parts) for easy logging.
    """
    # ------------------------------------------------ core losses
    l1       = w_l1 * F.l1_loss(gen_out, real_img)

    ssim_l   = w_ssim * ssim_loss(real_img, gen_out)

    per_l    = w_per * perceptual_vgg_loss(vgg_extractor, real_img, gen_out)

    gen_adv_l    = w_adv * generator_loss(real_score.detach(),   # detach so D not updated
                              fake_score.detach())
    w_mask_gt = 1.0
    mask_l   = w_mask_gt * mask_loss(real_mask, gen_mask)        # classic GT mask loss

    # ------------------------------------------------ ssl losses (may be None)
    mask_sup  = torch.tensor(0., device=gen_out.device) if mask_sup  is None else mask_sup
    mask_cons = torch.tensor(0., device=gen_out.device) if mask_cons is None else mask_cons

    # ----------------------------------------------- weighted sum
    total = (l1 + ssim_l + per_l + gen_adv_l + (mask_l + mask_sup) + mask_cons)
    gen_total = l1 + ssim_l + per_l + gen_adv_l 
    mask_total = mask_l + mask_sup + mask_cons

    parts = dict(l1=l1, ssim=ssim_l, per=per_l, adv=gen_adv_l, gen_total=gen_total,
                 mask_l=mask_l, mask_sup=mask_sup, mask_cons=mask_cons, mask_total=mask_total)

    return total, parts


def total_discriminator_loss(real: torch.Tensor,
                             fake: torch.Tensor) -> torch.Tensor:
    """
    Just the relativistic discriminator term.
    """
    return discriminator_loss(real, fake)