import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure # For the SSIM-based loss (if desired)
import torchvision.transforms as T
import torchvision.models as models


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


def multiscale_loss(target, out256, out128, out64):
    """
    out1, out2, out3 => Bx3xHxW, each potentially different resolution.
    We do MSE vs target at each scale (resizing target if needed).
    """
    # out1 is e.g. 256x256, out2 ~128x128, out3 ~64x64, etc.
    total = 0.0
    outputs = [out256, out128, out64]
    for o in outputs:
        if o.shape[2:] != target.shape[2:]:
            # Resize the target to match o's spatial dims
            resized_t = F.interpolate(target, size=o.shape[2:], mode='bilinear', align_corners=False)
            total += F.mse_loss(o, resized_t)
        else:
            total += F.mse_loss(o, target)
    return total

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

def generator_loss(
    disc_gen_out,
    gen_mask_out,    # if you want to combine mask MSE in same gen loss
    out256, out128, out64, 
    target_image, 
    target_mask,
    adv_criterion,   # e.g. BCEWithLogitsLoss() or BCELoss() if D outputs sigmoid
    vgg_extractor=None,
    lambda_mask=1.0,
    lambda_ms=1.0,
    lambda_ssim=1.0,
    lambda_vgg=1.0
):
    """
    - disc_gen_out: D's output on the generator's main scale
    - out1, out2, out3: gen’s multi-scale outputs (Bx3xH x W)
    - target_image, target_mask: ground truth
    - adv_criterion: e.g. BCELoss with real=1
    - vgg_extractor: optional VGG19 for perceptual
    """
    # 1) Adversarial part => want disc_gen_out ~ 1
    valid = torch.ones_like(disc_gen_out)
    adversarial_loss = adv_criterion(disc_gen_out, valid)

    # 2) Multi-scale MSE
    ms = multiscale_loss(target_image, out256, out128, out64) / 3.0

    # 3) SSIM
    ssim_l = ssim_loss(target_image, out256)

    # 4) VGG
    if vgg_extractor is not None:
        vgg_l = perceptual_vgg_loss(vgg_extractor, target_image, out256)
    else:
        vgg_l = torch.tensor(0.0, device=out256.device)

    # 5) Mask MSE
    mask_l = mask_loss(target_mask, gen_mask_out)

    # Weighted sum
    g_loss = (adversarial_loss
              + lambda_ms * ms
              + lambda_ssim * ssim_l
              + lambda_vgg * vgg_l
              + lambda_mask * mask_l)

    return g_loss, adversarial_loss, ms, ssim_l, vgg_l, mask_l


def discriminator_loss(disc_real_out, disc_fake_out, adv_criterion):
    """
    disc_real_out => D(x_real)
    disc_fake_out => D(x_gen)
    typical BCE: real=1, fake=0
    """
    valid = torch.ones_like(disc_real_out)
    fake = torch.zeros_like(disc_fake_out)
    real_loss = adv_criterion(disc_real_out, valid)
    fake_loss = adv_criterion(disc_fake_out, fake)
    return real_loss + fake_loss
