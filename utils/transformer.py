import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from timm.models.layers import DropPath, to_2tuple


def window_partition(x, window_size: int):
    """
    Split the feature map into non-overlapping windows according to window_size.
    Args:
        x: (b, h, w, c)
        window_size (int): window size(W)
    Returns:
        windows: (num_windows * b, window_size, window_size, c)
    """

    # get the shape of the input tensor.
    b, h, w, c = x.shape

    # reshape of the input tensor
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)

    # permute: [b, h // Wh, Wh, w // Ww, Ww, c] -> [b, h//Wh, w//Ww, Wh, Ww, c]
    # view: [b, h//Wh, w//Ww, Wh, Ww, c] -> [b * num_windows, Wh, Ww, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)

    return windows


def window_reverse(windows, window_size: int, h: int, w: int):
    """
    Reducing the windows to the feature map
    Args:
        windows: (b * num_windows, window_size, window_size, c)
        window_size (int): Window size(W)
        h (int): Height of image
        w (int): Width of image
    Returns:
        x: (b, h, w, c)
    """

    # batch size
    b = int(windows.shape[0] / (h * w / window_size / window_size))

    # view: [b * num_windows, Wh, Ww, c] -> [b, h // Wh, w // Ww, Wh, Ww, c]
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)

    # permute: [b, h // Wh, w // Ww, Wh, Ww, c] -> [b, h // Wh, Wh, w // Ww, Ww, c]
    # view: [b, h // Wh, Wh, w // Ww, Ww, c] -> [b, h, w, c]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)

    return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    Args:
        in_features: dimension of the input features.
        hidden_features: dimension of the hidden features, if None, defaults to in_features.
        out_features: dimension of the output features, if None, defaults to in_features.
        act_layer: activation function, default is GELU.
        drop: dropout probability, default is 0.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        # setting hyper parameters
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # FC Layer: hidden_features_channel = 4 * in_features_channel
        self.fc1 = nn.Linear(in_features, hidden_features)

        # GELU activation layer
        self.act = act_layer()

        # dropout layer
        self.drop = nn.Dropout(drop)

        # FC Layer: out_features_channel = 1/4 * hidden_features_channel = in_features_channel
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        # Input Layer to Hidden Layer
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        # Hidden Layer to Output Layer
        x = self.fc2(x)
        x = self.drop(x)

        return x


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        # setting hyper parameters
        self.dim = dim
        self.window_size = window_size  # [Wh, Ww]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias [(2 * Wh - 1) * (2 * Ww - 1), nH]
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        # create relative position index
        coordinate_h = torch.arange(self.window_size[0])  # The row coordinates of each element within the window.
        coordinate_w = torch.arange(self.window_size[1])  # The column coordinates of each element within the window.
        coordinates = torch.stack(torch.meshgrid([coordinate_h, coordinate_w], indexing="ij"))  # [2, Wh, Ww]
        coordinates_flatten = torch.flatten(coordinates, 1)  # [2, Wh * Ww]

        # Calculate the relative position deviation between each pair of elements [2, Wh * Ww, Wh * Ww]
        relative_coordinates = coordinates_flatten[:, :, None] - coordinates_flatten[:, None, :]

        # makes memory contiguous
        relative_coordinates = relative_coordinates.permute(1, 2, 0).contiguous()  # [Wh * Ww, Wh * Ww, 2]

        # Add window_size[0] - 1 to the row or column coordinates of the relative coordinates.
        relative_coordinates[:, :, 0] += self.window_size[0] - 1
        relative_coordinates[:, :, 1] += self.window_size[1] - 1

        # Multiply the row coordinates of the relative coordinates by 2 * window_size[1] - 1.
        relative_coordinates[:, :, 0] *= 2 * self.window_size[1] - 1

        # Add the row and column coordinates of the relative coordinates to get the relative position index.
        relative_position_index = relative_coordinates.sum(-1)  # [Wh * Ww, Wh * Ww]

        # Put the relative position index into the model cache as it is a fixed value.
        self.register_buffer("relative_position_index", relative_position_index)

        # Define a linear layer that transforms the input vector into a query, key and value.
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Define a dropout layer for attention weights.
        self.attn_drop = nn.Dropout(attn_drop)

        # Define a linear layer for fusing multiple outputs.
        self.proj = nn.Linear(dim, dim)

        # Define a dropout layer for outputting vectors.
        self.proj_drop = nn.Dropout(proj_drop)

        # Initialization of relative position bias
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        # Define a softmax layer for computing the attention weights.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows * b, Wh * Ww, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh * Ww, Wh * Ww) or None
        """

        # Get the dimensions of the input features: [batch_size * num_windows, Wh * Ww, total_embed_dim]
        b_, n, c = x.shape

        # qkv(): -> [batch_size * num_windows, Wh * Ww, 3 * total_embed_dim]
        # reshape: -> [batch_size * num_windows, Wh * Ww, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size * num_windows, num_heads, Wh * Ww, embed_dim_per_head]
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)

        # [batch_size * num_windows, num_heads, Wh * Ww, embed_dim_per_head]
        q, k, v = qkv.unbind(0)

        # transpose: -> [batch_size * num_windows, num_heads, embed_dim_per_head, Wh * Ww]
        # @: multiply -> [batch_size * num_windows, num_heads, Wh * Ww, Wh * Ww]

        # scale in the Attention formula. F: (Q × K^T) / sqrt(dk)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Wh * Ww * Wh * Ww, nH] -> [Wh * Ww, Wh * Ww, nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Wh * Ww, Wh * Ww]

        # add the relative position bias F: (Q × K ^ T) / sqrt(dk) + B
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Wh * Ww, Wh * Ww]
            num_windows = mask.shape[0]  # num_windows

            # attn.view: [batch_size, num_windows, num_heads, Wh * Ww, Wh * Ww]
            # mask.unsqueeze: [1, nW, 1, Wh * Ww, Wh * Ww]
            attn = attn.view(b_ // num_windows, num_windows, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

            # After adding -100, softmax to the irrelevant position, it will become 0
            attn = self.softmax(attn)
        else:
            # F: softmax((Q × K ^ T) / sqrt(dk) + B)
            attn = self.softmax(attn)

        # dropout attention weight
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size * num_windows, num_heads, Wh * Ww, embed_dim_per_head]
        # transpose: -> [batch_size * num_windows, Wh * Ww, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size * num_windows, Wh * Ww, total_embed_dim]

        # get the attention-weighted result. F: softmax((Q × K ^ T) / sqrt(dk) + B)V
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):

        # calculate flops for one window with token length of N
        flops = 0

        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim

        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N

        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)

        # x = self.proj(x)
        flops += N * self.dim * self.dim

        return flops


class SwinTransformerBlock(nn.Module):
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
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        # hyperparameter setting and assignment
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.window_size = window_size
        self.input_resolution = input_resolution
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # if window size is larger than input resolution, we don't partition windows
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        # define the Layer Normalization layer
        self.norm1 = norm_layer(dim)

        #  Window Attention is W-MSA or SW-MSA based on shift_size
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # define the Layer Normalization layer
        self.norm2 = norm_layer(dim)

        # MLP: define the full connectivity layer
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # shifted windows multi-head self attention (W-MSA).
        if self.shift_size > 0:

            # Get the height and width of the input tensor
            h, w = self.input_resolution

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, h, w, 1))

            # The position of each window is calculated by slicing operations h_slices and w_slices.
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))

            # Iterate all windows, assigning the value of the corresponding position in the img_mask.
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # split the feature map into non-overlapping windows according to window_size.
            mask_windows = window_partition(img_mask, self.window_size)  # [nW, window_size, window_size, 1]

            # reshape mask_windows into a tensor of shape (nW, window_size * window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

            # obtain the attention mask
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)

            # Replace elements of attn_mask not equal to 0 with -100.0, and elements of attn_mask equal to 0 with 0.0
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        # Register attn_mask as a buffer for the model so that it does not participate in training
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):

        # image size
        h, w = self.input_resolution

        # the shape of the input features
        b, l_x, c = x.shape
        assert l_x == h * w, "input feature has wrong size"

        # residual connections
        shortcut = x

        # Layer Normalization [A]
        x = self.norm1(x)

        # The conversion dimensions: [batch height width channel]
        x = x.view(b, h, w, c)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))

        # height and width of the filled features
        _, hp, wp, _ = x.shape

        # pairwise operations in shift_window, cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW * b, Wh, Ww, c]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # [nW * b, Wh * Ww, c]

        # W-MSA / SW-MSA [B]
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # [nW * b, Wh * Ww, c]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)  # [nW * b, Wh, Ww, c]

        # Convert the window back after window_partition processing
        shifted_x = window_reverse(attn_windows, self.window_size, hp, wp)  # [b, hp, wp, c]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Remove the data from the previous pad and make it memory contiguous
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()

        # converts to the dimension form in Vision Transformer
        x = x.view(b, h * w, c)

        # residual connection
        x = shortcut + self.drop_path(x)

        # Normalization + MLP [C + D]
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


if __name__ == '__main__':
    # [batch_size, [height width], channel] -> [batch_size, [height width], channel]
    A = torch.rand([32, 196, 96])
    simT = SwinTransformerBlock(dim=96, input_resolution=(14, 14), num_heads=4, window_size=7, shift_size=0,
                                mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                                act_layer=nn.GELU, norm_layer=nn.LayerNorm)
    C = simT(A)
    print(C.shape)
