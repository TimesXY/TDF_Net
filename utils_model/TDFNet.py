import torch

from torch import nn
from .encoder import Encoder
from .clutorch import CCLoss
from .TDF_Module import TDFM
from .backbone import SharedBackBoneFPN
from .revtorch import ModalTransition


class TDF_Net(nn.Module):
    def __init__(self, num_class=2, depth=-4, pretrain=True, channel=512):
        super().__init__()

        # the shared backbone network
        self.shared_backbone = SharedBackBoneFPN(pretrain=pretrain, depth=depth)

        # the global-local dual-branch feature extraction module
        self.encoder = Encoder(dim=channel, input_resolution=(28, 28), num_heads=4, window_size=7, shift_size=0,
                               mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                               act_layer=nn.GELU, norm_layer=nn.LayerNorm, dim_head=64, heads=4, dropout=0.,
                               down_sample_factor=4, offset_scale=None, offset_groups=None, offset_kernel_size=6,
                               group_queries=True, group_key_values=True)

        # contrast clustering Loss  - k_class=2, div=8 -
        self.clustering = CCLoss(dim=channel, k_class=2, div=8, temp=1.0)

        # modal transition loss
        self.rev = ModalTransition(in_c=channel, out_c=channel, split_along_dim=1, fix_random_seed=True)

        # trusted dynamic feature fusion modules
        self.fuse_model = TDFM(dim=channel, classes=num_class, annealing_epoch=10, depth=1, heads=4, dim_head=128,
                               mlp_dim=256, dropout=0.)

    def forward(self, x1, x2, x3, y, epoch):
        # the shared backbone network
        x1, x2, x3 = self.shared_backbone(x1, x2, x3)

        # the global-local dual-branch feature extraction module
        x1, x2, x3 = self.encoder(x1, x2, x3)

        # feature dimension transformation
        x1_trans = x1.permute(0, 2, 3, 1)
        x1_trans = x1_trans.view(x1_trans.size(0), -1, x1_trans.size(-1))

        x2_trans = x2.permute(0, 2, 3, 1)
        x2_trans = x2_trans.view(x2_trans.size(0), -1, x2_trans.size(-1))

        x3_trans = x3.permute(0, 2, 3, 1)
        x3_trans = x3_trans.view(x3_trans.size(0), -1, x3_trans.size(-1))

        # contrast clustering Loss
        loss_clu, _ = self.clustering(x1, x2, x3, y)

        # modal transition loss
        loss_mod, _ = self.rev(x1, x2, x3)

        # trusted dynamic feature fusion modules
        evidence, loss_tdf = self.fuse_model(x1_trans, x2_trans, x3_trans, y, epoch)

        # add the losses
        loss_all = 0.5 * loss_clu + 2.5 * loss_mod + loss_tdf

        return evidence, loss_all


if __name__ == '__main__':
    # [batch_size, channel, height, width]
    A1 = torch.randn([4, 3, 224, 224])
    A2 = torch.randn([4, 3, 224, 224])
    A3 = torch.randn([4, 3, 224, 224])
    B0 = torch.tensor([0, 0, 1, 1])

    model = TDF_Net(num_class=2, depth=-4, pretrain=True)
    C1, C2, C3 = model(A1, A2, A3, B0, 300)

    print(C1)
    print(C2.shape)
    print(C3.shape)
