import torch
import torch.nn as nn
import torch.nn.functional as F
from SegFormer import SemiSupSegFormer
from UnfairGAN import Generator, Discriminator
from RCF import RCF

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

class EndToEndModel(nn.Module):
    """
    1) semi‑sup SegFormer predicts a rain mask (student or teacher)
    2) RCF predicts an edge map (frozen, inference‑only)
    3) UnfairGAN G & D use both mask & edges to derain and score real/fake
    """
    def __init__(self,
                 seg_cfg,
                 in_channels: int = 3,
                 mask_channels: int = 1,
                 edge_channels: int = 1,
                 gan_feats: int = 32):
        super().__init__()

        # 1) semi‑supervised SegFormer (student + teacher)
        self.seg_model = SemiSupSegFormer(seg_cfg).to(DEVICE)

        # 2) frozen RCF edge detector (inference only)
        self.edge_model = RCF().to(DEVICE)
        self.edge_model.eval()
        for p in self.edge_model.parameters():
            p.requires_grad = False

        # 3) UnfairGAN generator & discriminator
        #    now accept both mask (1‐ch) and edge (1‐ch)
        self.generator = Generator(
            inX_chs=in_channels,
            inRM_chs=mask_channels,
            inED_chs=edge_channels,
            out_chs=in_channels,
            nfeats=gan_feats
        ).to(DEVICE)

        self.discriminator = Discriminator(
            inX_chs=in_channels,
            inRM_chs=mask_channels,
            inED_chs=edge_channels,
            nfeats=gan_feats
        ).to(DEVICE)

        self.unc_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor, use_teacher: bool = False):
        """
        x           – (B,3,H,W) rainy input
        use_teacher – if True, route through SegFormer's teacher branch (for inference)
                      if False, route through the student branch (for training)
        Returns a dict:
           'mask'       (B,1,H,W) predicted rain mask
           'sigma'      (B,1,H,W) uncertainty logits
           'edges'      (B,1,H,W) RCF edge map
           'derained'   (B,3,H,W) generator output
           'real_score' (B,1) discriminator(x, mask, edges)
           'fake_score' (B,1) discriminator(derained, mask, edges)
        """
        # 1) rain mask + uncertainty
        if use_teacher:
            mask, sigma = self.seg_model.teacher(x)
        else:
            mask, sigma = self.seg_model.student(x)

        sigma = self.unc_act(sigma)    

        # 2) edge map (frozen RCF, no grad)
        with torch.no_grad():
            edges = self.edge_model(x)[-1] # last output of RCF

        # 3) derain
        derained = self.generator(x, rm=mask, ed=edges)

        # 4) score real vs fake
        real_score = self.discriminator(x,      rm=mask, ed=edges)
        fake_score = self.discriminator(derained, rm=mask, ed=edges)

        return {
            'mask':       mask,
            'sigma':      sigma,
            'edges':      edges,
            'derained':   derained,
            'real_score': real_score,
            'fake_score': fake_score
        }
