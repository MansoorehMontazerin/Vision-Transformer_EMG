import torch
from torch import nn as nn
from helpers import to_2tuple

class PatchEmbed(nn.Module):
    """Split images into patches and then embed them.

    Patameters
    ----------
    img_size : int or tuple
        Size of the image.
    
    patch_size : int or tuple
        Size of the patch

    in_chans : int
        Number of input channels.

    embed_dim : int 
        The embedding dimension.

    norm_layer : ??
        ???
    
    Attributes
    -----------
    num_patches : int
        Number of pathces inside of our image

    proj : nn.Conv2d    
        Convolutional layer that does both splitt ing into patches and their embedding.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        '''Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, img_size[0], img_size[1])'.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches, embed_dim)`.
        '''
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x) # (n_samples, embed_dim, grid_size[0], grid_size[1]), BCHW
        x = x.flatten(2) # (n_samples, embed_dim, num_patches), BCN
        x = x.transpose(1, 2) # (n_samples, num_patches, embed_dim), BNC
        x = self.norm(x)
        return x

class MLP(nn.Module):
    """Multilayer perceptron.

    Parameters
    ----------
    in_features : int
        Number of input_features

    hidden_features : int
        Number of nodes in the hidden layer.

    out_features : int
        Number of output features.

    p : float
        Dropout probability.

    Attributes
    ----------
    fc : nn.Linear
        The first linear layer.

    act : nn.GELU
        GELU activation function.

    fc2 : nn.Linear
        The second linear layer.

    drop : nn.Dropout
        Dropout layer. 
    """
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape`(n_samples, n_patches + 1, in_features)`.

        Returns
        -------
        torch.Tensor
            Shape`(n_samples, n_patches + 1, out_features)`.
        """ 
        x = self.fc1(
            x
        ) # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x) # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x) # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x) # (n_samples, n_patches + 1, out_features)
        x = self.drop(x)

        return x



if __name__ == "__main__":
    x  = torch.randn(128, 3, 224, 224*2)
    patch_embed = PatchEmbed(img_size = (224, 224*2))
    out = patch_embed(x)
    ##########################
    x_2 = torch.randn(32, 400, 128)
    mlp = MLP(in_features = 128, hidden_features = 128, out_features = 256
    )
    out = mlp(x_2)
    print()
    