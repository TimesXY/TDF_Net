import torch
import torch.nn as nn

from .swintorch import WSAM
from .deformable import DSAM


def window_reverse_channel(x):
    """
       [b, h * w, c] -> [b, c, h, w]
    """

    # view: [b, h * w, c] -> [b, c, h * w]
    x = x.permute(0, 2, 1)

    # get the shape of the tensor.
    b, c, hw = x.shape
    h = w = int(hw ** 0.5)

    # [b, c, h * w] -> [b, c, h, w]
    x = x.view(b, c, h, w)

    return x


def channel_reverse_window(x):
    """
       [b, c, h, w] -> [b, h * w, c]
    """

    # view: [b, c, h, w] -> [b, h, w, c]
    x = x.permute(0, 2, 3, 1)

    # get the shape of the tensor.
    b, h, w, c = x.shape

    # [b, h, w, c] -> [b, h * w, c]
    windows = x.view(b, h * w, c)

    return windows


class Encoder(nn.Module):
    """
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

        dim_head: dimension of each attention head
        heads: number of attention heads
        dropout: dropout rate
        down_sample_factor: down sample factor
        offset_scale: the scale of offset
        offset_groups: the groups of offset
        offset_kernel_size: offset kernel size
        group_queries: the groups of queries, whether grouped or not
        group_key_values: the groups of key and value, whether grouped or not
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 dim_head=64, heads=4, dropout=0., down_sample_factor=4, offset_scale=None,
                 offset_groups=None, offset_kernel_size=6, group_queries=True, group_key_values=True):
        super().__init__()

        # building modality-independent window self-attention modules
        self.w_msm1 = WSAM(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size,
                           shift_size=shift_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                           attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)

        self.w_msm2 = WSAM(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size,
                           shift_size=shift_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                           attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)

        self.w_msm3 = WSAM(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size,
                           shift_size=shift_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                           attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)

        # building modality-independent deformable self-attention modules
        self.d_sam1 = DSAM(dim, dim_head=dim_head, heads=heads, dropout=dropout, down_sample_factor=down_sample_factor,
                           offset_scale=offset_scale, offset_groups=offset_groups,
                           offset_kernel_size=offset_kernel_size, group_queries=group_queries,
                           group_key_values=group_key_values)

        self.d_sam2 = DSAM(dim, dim_head=dim_head, heads=heads, dropout=dropout, down_sample_factor=down_sample_factor,
                           offset_scale=offset_scale, offset_groups=offset_groups,
                           offset_kernel_size=offset_kernel_size,
                           group_queries=group_queries, group_key_values=group_key_values)

        self.d_sam3 = DSAM(dim, dim_head=dim_head, heads=heads, dropout=dropout, down_sample_factor=down_sample_factor,
                           offset_scale=offset_scale, offset_groups=offset_groups,
                           offset_kernel_size=offset_kernel_size,
                           group_queries=group_queries, group_key_values=group_key_values)

        # Feature dimension conversion module, converting to semantic information
        self.fcn1 = nn.Conv2d(in_channels=2 * dim, out_channels=dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fcn2 = nn.Conv2d(in_channels=2 * dim, out_channels=dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fcn3 = nn.Conv2d(in_channels=2 * dim, out_channels=dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x1, x2, x3):
        # initialize the output tensors
        y1 = None
        y2 = None
        y3 = None

        # check if x1 is not None
        if x1 is not None:
            # through the deformable self-attention modules
            x1_d = self.d_sam1(x1)

            # dimensional transformation of features
            x1_c = channel_reverse_window(x1)

            # through the windows self-attention modules
            x1_w = self.w_msm1(x1_c)

            # dimensional transformation of features
            x1_w = window_reverse_channel(x1_w)

            # concat of features
            y1 = torch.cat((x1_d, x1_w), dim=1)

            # semantic information
            y1 = self.fcn1(y1)

        # check if x2 is not None
        if x2 is not None:
            # through the deformable self-attention modules
            x2_d = self.d_sam2(x2)

            # dimensional transformation of features
            x2_c = channel_reverse_window(x2)

            # through the windows self-attention modules
            x2_w = self.w_msm2(x2_c)

            # dimensional transformation of features
            x2_w = window_reverse_channel(x2_w)

            # concat of features
            y2 = torch.cat((x2_d, x2_w), dim=1)

            # semantic information
            y2 = self.fcn2(y2)

        # check if x3 is not None
        if x3 is not None:
            # through the deformable self-attention modules
            x3_d = self.d_sam3(x3)

            # dimensional transformation of features
            x3_c = channel_reverse_window(x3)

            # through the windows self-attention modules
            x3_w = self.w_msm3(x3_c)

            # dimensional transformation of features
            x3_w = window_reverse_channel(x3_w)

            # concat of features
            y3 = torch.cat((x3_d, x3_w), dim=1)

            # semantic information
            y3 = self.fcn3(y3)

        return y1, y2, y3


if __name__ == '__main__':
    # [batch_size, channel, height, width]
    A1 = torch.rand([32, 96, 14, 14])
    A2 = torch.rand([32, 96, 14, 14])
    A3 = torch.rand([32, 96, 14, 14])

    encoder = Encoder(dim=96, input_resolution=(14, 14), num_heads=4, window_size=7, shift_size=0, mlp_ratio=4.,
                      qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                      norm_layer=nn.LayerNorm, dim_head=64, heads=4, dropout=0., down_sample_factor=4,
                      offset_scale=None, offset_groups=None, offset_kernel_size=6, group_queries=True,
                      group_key_values=True)

    C1, C2, C3 = encoder(A1, A2, A3)

    print(C1.shape)
    print(C2.shape)
    print(C3.shape)

"""
    Global-Local Dual Branch Feature Extraction Module (GLDM):
      全局-局部双分支特征提取模块: 主要由窗口自注意力模块和可变形自注意力模块两部分组成, 输入的特征图同时通过窗口自注意力模块和可变形自注意力模块分别
      聚合局部特征信息和全局特征信息, 这种带有局部和全局感受野的特征提取分支有助于模型学习到更全面的信息表征.
    
    亮点:
      首先, 引入了特征解耦思想, 将多模态信息分解为共有信息和特有信息. 在深度神经网络中, 浅层网络通常能够提取图像的背景和色彩等共有信息. 因此, 通过
      共享的 BackBone 提取不同模态下的浅层共有信息. 然后采用提出的全局-局部双分支特征提取模块分别独立提取各模态的特有信息, 比如DUS的血流信息和USE
      的硬度信息等. GLDM 主要由 DSAM 和 WSAM 两部分组成. 其中, DSAM 能够在全局范围下关注图像中的重要区域, 有利于提取图像的全局特征信息, 比如乳
      腺结节的形状、尺寸和组织变化. WSAM 则能够提取图像的局部特征信息, 从而充分考虑乳腺结节的边缘和纹理等超声图像中的重要细节信息. 
      
      为了提高提取特征的可控制性和可解释性, 对提取的深层特征添加了相关性约束. 本文提出的约束假设如下: 对于多模态图像联合诊断问题, 不同模态的输入应具
      备相同的语义信息, 即: 同一图像下, 不同模态的深层特征的聚类结果应该一致; 同一模态下, 类别相同图像的深层特征的聚类结果应该一致. 
   
    摘要:
      首先，TDFNet 使用共享的 BackBone 来提取跨模态浅层特征. 然后, 引入了全局-局部双分支特征提取模块, 分别利用可变形自注意力模块和窗口自注意力模
      块提取图像的全局信息和局部信息.
      
      完全共享的编码器无法区分不同模态的特定特征, 而完全私有的编码器则会忽略不同模态的共享特征. 此外, 传统的CNN网络只能提取相对较小的感受野内的局部
      信息, 难以提取全局信息和跨模态信息来进行综合诊断. 因此, 本文旨在探索一种更合理的范式来实现多模态医学图像的诊断工作. 
"""
