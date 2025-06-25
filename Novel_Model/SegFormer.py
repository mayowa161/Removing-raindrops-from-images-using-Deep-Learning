import torch
from einops import rearrange
from torch import nn
from torchvision.ops import StochasticDepth
from typing import List, Iterable
from functions import *
import torch.nn.functional as F
import copy

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = super().forward(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x
    

class OverlapPatchMerging(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, patch_size: int, overlap_size: int
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=patch_size,
                stride=overlap_size,
                padding=patch_size // 2,
                bias=False
            ),
            LayerNorm2d(out_channels)
        )


class EfficientMultiHeadAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 1, num_heads: int = 8):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=reduction_ratio, stride=reduction_ratio
            ),
            LayerNorm2d(channels),
        )
        self.att = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        _, _, h, w = x.shape
        reduced_x = self.reducer(x)
        # attention needs tensor of shape (batch, sequence_length, channels)
        reduced_x = rearrange(reduced_x, "b c h w -> b (h w) c")
        x = rearrange(x, "b c h w -> b (h w) c")
        out = self.att(x, reduced_x, reduced_x)[0]
        # reshape it back to (batch, channels, height, width)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        return out
    


class MixMLP(nn.Sequential):
    def __init__(self, channels: int, expansion: int = 4):
        super().__init__(
            # dense layer
            nn.Conv2d(channels, channels, kernel_size=1),
            # depth wise conv
            nn.Conv2d(
                channels,
                channels * expansion,
                kernel_size=3,
                groups=channels,
                padding=1,
            ),
            nn.GELU(),
            # dense layer
            nn.Conv2d(channels * expansion, channels, kernel_size=1),
        )

class ResidualAdd(nn.Module):
    """Just an util layer"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        x = x + out
        return x

class SegFormerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
        drop_path_prob: float = .0
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    EfficientMultiHeadAttention(channels, reduction_ratio, num_heads),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    MixMLP(channels, expansion=mlp_expansion),
                    StochasticDepth(p=drop_path_prob, mode="batch")
                )
            ),
        )


class SegFormerEncoderStage(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        overlap_size: int,
        drop_probs: List[int],
        depth: int = 2,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
    ):
        super().__init__()
        self.overlap_patch_merge = OverlapPatchMerging(
            in_channels, out_channels, patch_size, overlap_size,
        )
        self.blocks = nn.Sequential(
            *[
                SegFormerEncoderBlock(
                    out_channels, reduction_ratio, num_heads, mlp_expansion, drop_probs[i]
                )
                for i in range(depth)
            ]
        )
        self.norm = LayerNorm2d(out_channels)


def chunks(data: Iterable, sizes: List[int]):
    """
    Given an iterable, returns slices using sizes as indices
    """
    curr = 0
    for size in sizes:
        chunk = data[curr: curr + size]
        curr += size
        yield chunk
        
class SegFormerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        widths: List[int],
        depths: List[int],
        all_num_heads: List[int],
        patch_sizes: List[int],
        overlap_sizes: List[int],
        reduction_ratios: List[int],
        mlp_expansions: List[int],
        drop_prob: float = .0
    ):
        super().__init__()
        # create drop paths probabilities (one for each stage's block)
        drop_probs =  [x.item() for x in torch.linspace(0, drop_prob, sum(depths))]
        self.stages = nn.ModuleList(
            [
                SegFormerEncoderStage(*args)
                for args in zip(
                    [in_channels, *widths],
                    widths,
                    patch_sizes,
                    overlap_sizes,
                    chunks(drop_probs, sizes=depths),
                    depths,
                    reduction_ratios,
                    all_num_heads,
                    mlp_expansions
                )
            ]
        )
        
    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features
    

class SegFormerDecoderBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=scale_factor),
            )


class SegFormerDecoder(nn.Module):
    def __init__(self, out_channels: int, widths: List[int], scale_factors: List[int]):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                SegFormerDecoderBlock(in_channels, out_channels, scale_factor)
                for in_channels, scale_factor in zip(widths, scale_factors)
            ]
        )
    
    def forward(self, features):
        new_features = []
        for feature, stage in zip(features,self.stages):
            x = stage(feature)
            new_features.append(x)
        return new_features


class SegFormerSegmentationHead(nn.Module):
    def __init__(self, channels: int, num_classes: int, num_features: int = 4):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * num_features, channels, kernel_size=1, bias=False),
            nn.ReLU(), # why relu? Who knows
            nn.BatchNorm2d(channels) # why batchnorm and not layer norm? Idk
        )
        self.predict = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, features):
        x = torch.cat(features, dim=1)
        x = self.fuse(x)
        x = self.predict(x)
        return x
    

class SegFormer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        widths: List[int],
        depths: List[int],
        all_num_heads: List[int],
        patch_sizes: List[int],
        overlap_sizes: List[int],
        reduction_ratios: List[int],
        mlp_expansions: List[int],
        decoder_channels: int,
        scale_factors: List[int],
        num_classes: int,
        drop_prob: float = 0.0
    ):

        super().__init__()
        self.encoder = SegFormerEncoder(
            in_channels,
            widths,
            depths,
            all_num_heads,
            patch_sizes,
            overlap_sizes,
            reduction_ratios,
            mlp_expansions,
            drop_prob,
        )
        self.decoder = SegFormerDecoder(decoder_channels, widths[::-1], scale_factors)
        self.head = SegFormerSegmentationHead(
            decoder_channels, num_classes, num_features=len(widths)
        )

        
        mid_ch = decoder_channels // 2
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.up1 = nn.Sequential(
                nn.Conv2d(num_classes + in_channels, mid_ch, 3, padding = 1, padding_mode='reflect', bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.ELU(inplace=True),
            )
        self.up2 = nn.Sequential(
                nn.Conv2d(mid_ch, mid_ch, 3, padding = 1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.ELU(inplace=True),
            )

        self.final_conv = nn.Sequential(
             nn.Conv2d(mid_ch, num_classes, 1),
             nn.Sigmoid()
        )

        self.decoder_channels = decoder_channels

    def forward(self, x):
        features = self.encoder(x)
        features = self.decoder(features[::-1])
        coarse = self.head(features)

        
        mask = self.upsample4(coarse)           # (B,1,H,W)
        mask = torch.cat([mask, x], dim=1)         # (B,1+3,H,W)
        mask = self.up1(mask)                      # (B,mid,H,W)
        mask = self.up2(mask)                      # (B,mid,H,W)
        mask = self.final_conv(mask)               # (B,1,H,W)
        return mask
    

class SegFormerUnc(nn.Module):
    """
    Wrapper that adds a 3‑layer Conv‑ELU uncertainty head on top of a frozen
    (or fine‑tunable) SegFormer backbone.
    """
    def __init__(self,
                 segformer_backbone: SegFormer,
                 mid_channels: int = 64):
        super().__init__()
        self.backbone = segformer_backbone 

        C = segformer_backbone.decoder_channels

        self.unc_head = nn.Sequential(
            nn.Conv2d(C,
                      mid_channels, 3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1)           # σ̂ (B,1,H,W)
        )

     

    def forward(self, x):
        mask = self.backbone(x)          # (B,1,H,W)

        feats      = self.backbone.encoder(x)
        dec_feats  = self.backbone.decoder(feats[::-1])
        # -- uncertainty -----------------------------------------------------
        sigma = self.unc_head(dec_feats[-1])  
        sigma  = F.interpolate(sigma, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        return mask, sigma

class SemiSupSegFormer(nn.Module):
    def __init__(self, backbone_cfg, alpha=0.999, T=2.0):
        super().__init__()
        self.student = SegFormerUnc(**backbone_cfg)      # your wrapper from before
        self.teacher = copy.deepcopy(self.student)       # initial copy
        for p in self.teacher.parameters():              # teacher is frozen
            p.requires_grad_(False)
        self.alpha = alpha
        self.T   = T                                     # temperature for teacher
    
    @torch.no_grad()
    def _update_teacher(self):
        for t, s in zip(self.teacher.parameters(), self.student.parameters()):
            t.data.mul_(self.alpha).add_(s.data, alpha=1.0 - self.alpha)

    def forward(self, x):
        return self.student(x)                           # convenience
    



