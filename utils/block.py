import torch

from torch import nn, einsum
from einops import rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        """
        Layer Normalization and Feed Forward
        dim: dimension of the input tensor
        fn: Attention or Feed Forward Network
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        """
        Feed Forward Network
        dim: dimension of the input tensor
        hidden_dim: dimension of the hidden layer
        dropout: dropout rate
        """
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, dim),
                                 nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, dropout=0.):
        """
        Multi-Head Self Attention Module
        dim: dimension of the input tensor
        heads: number of the heads
        dim_head: the dimension of each head
        dropout: dropout rate
        """
        super().__init__()
        # get the feature dimensions
        inner_dim = dim_head * heads

        # determine if it is an output layer
        project_out = not (heads == 1 and dim_head == dim)

        # hyperparameter assignment
        self.heads = heads
        self.scale = dim_head ** -0.5

        # obtain the query, key and value
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # project layer
        if project_out:
            self.to_out = nn.Sequential(nn.Linear(inner_dim, dim),
                                        nn.Dropout(dropout))
        else:
            self.to_out = nn.Identity()

    def forward(self, x):

        # 超参数赋值
        head = self.heads
        b, n, _, = x.shape

        # obtain the key query and value, (b, n, dim * 3) -> 3 * (b, n, dim)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # (b, n, (h, d)) -> (b h n d)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=head), qkv)

        # obtain the attention score (Q@K)/scale
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # softmax(score)
        attn = self.attend(dots)

        # softmax(score) * value
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # dimensional transformation
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, depth=1, heads=4, dim_head=64, mlp_dim=256, dropout=0.):
        """
        Vision Transformer Block
        dim: dimension of the input tensor
        depth: number of the block
        heads: number of the heads
        dim_head: the dimension of the heads
        mlp_dim: the dimension of the mlp
        dropout: dropout rate
        """
        super().__init__()

        # establishment of blank networks
        self.layers = nn.ModuleList([])

        # add the transformer block
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


if __name__ == '__main__':
    # [batch_size, [height width], channel] -> [batch_size, [height width], channel]
    A = torch.rand([32, 196, 256])
    trans_block = TransformerBlock(dim=256, depth=1, heads=4, dim_head=64, mlp_dim=256, dropout=0.)
    C = trans_block(A)
    print(C.shape)
