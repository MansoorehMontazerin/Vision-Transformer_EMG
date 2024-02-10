import torch
import torch.nn as nn
from layers import*
from functools import partial

class Attention(nn.Module):
    '''Attention Mechanism

    Parameters
    ----------
    dim : int
        The input and out dimension of per token features.

    n_heads : int
        Number of attention heads.

    qkv_bias : bool
        If True, then we add bias to the query, key, and values projections.

    qk_sclae : ??
        ??

    attn_drop : float
        Dropout probability applied to the query, key, and values tensors.

    proj_drop : float
        Dropout probability applied to the output tensor.  

    Attributes
    ----------
    scale : float
        Normalizing constant for dot product.

    qkv : nn.Linear
        Linear projection for the query, key, and value.

    proj : nn.Linear
        Linear mapping that takes in the concatenated output of the all attention
        heads and maps it into a new space.

    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
    '''
    def __init__(self, dim, n_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0, proj_drop=0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        '''Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape`(n_samples, n_patches + 1, dim)`.
            we have n_patches +1: Because there is a class token (cls).

        Returns
        -------
        torch.Tensor
            Shape`(n_samples, n_patches + 1, dim)`.
        '''
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x) # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(
                n_samples, n_tokens, 3, self.n_heads, self.head_dim
                ) # (n_samples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(
                2, 0, 3, 1, 4
                ) # (3, n_samples, n_heads, n_patches + 1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1) # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (
            q @ k_t
        ) * self.scale # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1) # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v # (n_samples, n_heads, n_patches + 1, head_dim)
        weighted_avg = weighted_avg.transpose(
            1, 2
        ) # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2) # (n_samples, n_patches + 1, dim): concatenated attention heads
        x = self.proj(weighted_avg) # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x) # (n_samples, n_patches + 1, dim)

        return x

class Block(nn.Module):
    """Transformer Block.

    Parameters
    ----------
    dim : int 
        Embedding dimension.

    n_heads : int
        Number of attention heads.

    mlp_ratio : float
        Determine the hidden dimension size of the 'MLP' module with respect to 'dim'.

    qkv_bias : bool
        If True, then include bias to query, key, and value projections.

    p, attn_p : float
        Dropout Probability.

    Attributes
    ----------
    norm_1, norm_2 : layerNorm.
        Layer normalization

    attn : Attention
        Attention module.

    mlp : MLP 
        MLP module. 
    """
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, p=0., attn_p=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=1e-6)
        self.attn = Attention(
            dim, 
            n_heads=n_heads,
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            attn_drop=attn_p, 
            proj_drop=p
        )
        self.norm2 = norm_layer(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, 
            hidden_features=hidden_features, 
            out_features=dim, 
            p=p
        )
    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape`(n_samples, n_patches + 1, dim)`.
        
        Returns
        -------
        torch.Tensor
            Shape`(n_samples, n_patches + 1, dim)`.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """Simplified implementation of the Vision Transformer.

    Parameters
    ----------
    img_size : int or tuple
        Size of the image.

    patch_size : int or tuple
        Size of the patch

    in_chans: int
        Number of input channels.

    n_classes : int
        Number of classes.

    embed_dim : int
        Dimentionality of the token/patch embeddings.

    depth : int
        Number of blocks.

    n_heads : int
        Number of attention heads.

    mlp_ratio : float
        Determines the hidden dimension of the 'MLP' module.

    qkv_bias : bool
        If True, include bias to query, key, and values.

    p, attn_p : float
        Dropout probability

    qk_scale : float) 
        override default qk scale of head_dim ** -0.5 if set

    Attributes
    ----------
    patch_embed : PatchEmbed
        Instance of 'PatchEmbed' layer.

    cls_token : nn.Parameter
        Learnable parameter that will represent the first token in the sequence.
        It has 'embed_dim' elements.

    pos_emb : nn.Parameter
        Positional embedding of cls_token + all patches.
        It has '(n_patches + 1)*embed_dim' elements.

    pos_drop : nn.Dropout
        Dropout layer.

    blocks : nn.ModuleList
        List of 'Block' modules.

    norm : nn.LayerNorm
        Layer normalization.
    """
    def __init__(self, img_size,
                       patch_size,
                       in_chans, 
                       n_classes, 
                       embed_dim,
                       depth, 
                       n_heads, 
                       mlp_ratio, 
                       qkv_bias, 
                       p, 
                       attn_p, 
                       qk_scale,
                       norm_layer 
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_emb = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)
        
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, 
                  n_heads=n_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, 
                  qk_scale=qk_scale,
                  p=p,
                  attn_p=attn_p,
                  norm_layer=norm_layer
                  )
                  for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
       
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape`(n_samples, in_chans, img_size, img_size)`.

        Returns
        -------
        logits : torch.Tensor
            logits over all the classes `(n_samples, n_classes)`.
        """
        n_samples = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(n_samples, -1, -1) # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1) # (n_samples, 1 + n_patches, embed_dim)
        x = x + self.pos_emb # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x) 
        cls_token_final = x[:, 0] # just the cls token
        
        x = self.head(cls_token_final)
       
        
        return x,cls_token_final






if __name__ == "__main__":
    x = torch.randn(32, 3, 384, 384)
    block = VisionTransformer()
    out = block(x)
    print()