import torch
import torch.nn.functional as F

from torch import nn, einsum
from einops import rearrange


def exists(val):
    # Determine if a variable exists
    return val is not None


def default(val, d):
    # Default selection of variables
    return val if exists(val) else d


def divisible_by(numer, de_nom):
    # Determine if it is divisible by an integer
    return (numer % de_nom) == 0


def create_grid_like(t, dim=0):
    # parameter assignment
    device = t.device
    h, w = t.shape[-2], t.shape[-1]
    # generate the grids
    grid = torch.stack(torch.meshgrid(torch.arange(w, device=device),
                                      torch.arange(h, device=device),
                                      indexing='xy'), dim=dim)
    # gradient truncation
    grid.requires_grad = False
    grid = grid.type_as(t)
    return grid


def normalize_grid(grid, dim=1, out_dim=-1):
    # normalizes a grid to range from -1 to 1
    h, w = grid.shape[-2:]
    grid_h, grid_w = grid.unbind(dim=dim)
    grid_h = 2.0 * grid_h / max(h - 1, 1) - 1.0
    grid_w = 2.0 * grid_w / max(w - 1, 1) - 1.0
    return torch.stack((grid_h, grid_w), dim=out_dim)


class Scale(nn.Module):
    # normalization using scale
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class CPB(nn.Module):
    # continuous positional bias -- solve the problem of inconsistent training and test image sizes
    def __init__(self, dim, heads, offset_groups, depth):
        super().__init__()

        # parameter assignment
        self.heads = heads
        self.offset_groups = offset_groups

        # MLP - (1)
        self.mlp = nn.ModuleList([])
        self.mlp.append(nn.Sequential(nn.Linear(2, dim), nn.ReLU()))

        # MLP - (2 - depth)
        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(nn.Linear(dim, dim), nn.ReLU()))

        # MLP - (-1)
        self.mlp.append(nn.Linear(dim, heads // offset_groups))

    def forward(self, grid_q, grid_kv):

        # the initial position encoding
        grid_q = rearrange(grid_q, 'h w c -> 1 (h w) c')
        grid_kv = rearrange(grid_kv, 'b h w c -> b (h w) c')
        pos = rearrange(grid_q, 'b i c -> b i 1 c') - rearrange(grid_kv, 'b j c -> b 1 j c')

        # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)
        bias = torch.sign(pos) * torch.log(pos.abs() + 1)

        # assignment of continuous relative position deviation
        for layer in self.mlp:
            bias = layer(bias)

        # continuous relative position bias
        bias = rearrange(bias, '(b g) i j o -> b (g o) i j', g=self.offset_groups)
        return bias


class DSAM(nn.Module):
    def __init__(self, dim, dim_head=64, heads=4, dropout=0., down_sample_factor=4, offset_scale=None,
                 offset_groups=None, offset_kernel_size=6, group_queries=True, group_key_values=True):
        """
        Args: implementation of the deformable attention module.
            param dim: dimension of input features
            param dim_head: dimension of each attention head
            param heads: number of attention heads
            param dropout: dropout rate
            param down_sample_factor: down sample factor
            param offset_scale: the scale of offset
            param offset_groups: the groups of offset
            param offset_kernel_size: offset kernel size
            param group_queries: the groups of queries, whether grouped or not
            param group_key_values: the groups of key and value, whether grouped or not
        """
        super().__init__()

        # whether to scale offset, if no scale is set, select down sample factor
        offset_scale = default(offset_scale, down_sample_factor)

        # offset kernel size must be greater than or equal to the down sample factor
        assert offset_kernel_size >= down_sample_factor, \
            'offset kernel size must be greater than or equal to the down_sample factor'

        # the difference between the offset kernel size and the down sample factor must be divisible by 2.
        assert divisible_by(offset_kernel_size - down_sample_factor, 2)

        # whether the number of groups is set, if not, the number of attention head as the number of groups
        offset_groups = default(offset_groups, heads)
        assert divisible_by(heads, offset_groups)

        # hyperparameter settings
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.offset_groups = offset_groups
        offset_dims = inner_dim // offset_groups

        # build the network to generate the offset
        self.to_offsets = nn.Sequential(
            nn.Conv2d(offset_dims, offset_dims, (offset_kernel_size, offset_kernel_size), groups=offset_dims,
                      stride=(down_sample_factor, down_sample_factor),
                      padding=(offset_kernel_size - down_sample_factor) // 2),
            nn.GELU(),
            nn.Conv2d(offset_dims, 2, (1, 1), bias=False),
            nn.Tanh(),
            Scale(offset_scale))

        # build the network of continuous relative position bias
        self.rel_pos_bias = CPB(dim // 4, offset_groups=offset_groups, heads=heads, depth=2)

        # build multi-head self-attention mechanisms Q K V
        self.to_q = nn.Conv2d(dim, inner_dim, (1, 1), groups=offset_groups if group_queries else 1, bias=False)
        self.to_k = nn.Conv2d(dim, inner_dim, (1, 1), groups=offset_groups if group_key_values else 1, bias=False)
        self.to_v = nn.Conv2d(dim, inner_dim, (1, 1), groups=offset_groups if group_key_values else 1, bias=False)

        # dimension transformation to maintain the initial dimension
        self.to_out = nn.Conv2d(inner_dim, dim, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_v_grid=False):
        # batch size, height, width and number of attention heads
        b, h, w, heads = x.shape[0], x.shape[-2], x.shape[-1], self.heads

        # queries
        q = self.to_q(x)

        # calculate offsets - offset MLP shared across all groups
        def group(t):
            return rearrange(t, 'b (g d) ... -> (b g) d ...', g=self.offset_groups)

        # generate offset based on query
        grouped_queries = group(q)
        offsets = self.to_offsets(grouped_queries)

        # calculate grid + offsets
        grid = create_grid_like(offsets)
        v_grid = grid + offsets

        # normalization and bi-linear interpolation sampling
        v_grid_scaled = normalize_grid(v_grid)
        kv_feats = F.grid_sample(group(x), v_grid_scaled, mode='bilinear', padding_mode='zeros', align_corners=False)
        kv_feats = rearrange(kv_feats, '(b g) d ... -> b (g d) ...', b=b)

        # derive key / values
        k, v = self.to_k(kv_feats), self.to_v(kv_feats)

        # scale queries
        q = q * self.scale

        # split out heads
        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h=heads), (q, k, v))

        # query / key similarity
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # relative positional bias
        grid = create_grid_like(x)
        grid_scaled = normalize_grid(grid, dim=0)
        rel_pos_bias = self.rel_pos_bias(grid_scaled, v_grid_scaled)
        sim = sim + rel_pos_bias

        # numerical stability
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()

        # attention
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        # aggregate and combine heads
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        out = self.to_out(out)

        # output the final result
        if return_v_grid:
            return out, v_grid

        return out


if __name__ == '__main__':
    # [batch_size, channel, height, width]
    A = torch.rand([32, 96, 14, 14])
    deformable_sam = DSAM(96, dim_head=64, heads=4, dropout=0., down_sample_factor=4, offset_scale=None,
                          offset_groups=None, offset_kernel_size=6, group_queries=True, group_key_values=True)
    C = deformable_sam(A)
    print(C.shape)
