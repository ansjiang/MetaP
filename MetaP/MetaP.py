import torch
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
import math

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class ParallelSum(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        outputs = [fn(x) for fn in self.fns]
        return sum(outputs) 

class MetaP_SPT(nn.Module):
    def __init__(self, *, dim , patch_h , patch_w):
        super().__init__()
        
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_size = patch_h * patch_w
        self.split_patches = Rearrange('b (h p1) (w p2) -> b h w p1 p2', p1 = patch_h, p2 = patch_w)
        self.to_patch_tokens = nn.Sequential(
            Rearrange('b h w p1 p2 -> b h w (p1 p2)', p1 = patch_h, p2 = patch_w) , 
            nn.LayerNorm(self.patch_size),
            nn.Linear(self.patch_size, dim), 
            nn.LayerNorm(dim)
        ) 

    def forward(self, X_data):
        
        X_data_patches = self.split_patches(X_data)
        X_data_tokens = self.to_patch_tokens(X_data_patches) # no overlap
        return X_data_tokens

 
class MetaP(nn.Module) : 
    def __init__(self , *, image_h , image_w, patch_h , patch_w, dim, depth, num_classes , segments , expansion_factor = 4, dropout = 0.): 
        super().__init__() 

        assert (dim % segments) == 0, 'dimension must be divisible by the number of segments'
        height  = image_h // patch_h
        width = image_w // patch_w
        
        s = segments
        self.to_patch_embedding  = MetaP_SPT(dim = dim, patch_h = patch_h , patch_w = patch_w)
        self.mlp_Mixer = nn.Sequential()
        for _ in range(depth) : 
             self.mlp_Mixer.append(
                nn.Sequential(
                    PreNormResidual(dim, nn.Sequential(
                        ParallelSum(
                            nn.Sequential(
                                Rearrange('b h w (c s) -> b c w (h s)', s = s),    
                                nn.Linear(height * s, height * s),
                                Rearrange('b c w (h s) -> b h w (c s)', s = s),
                            ),
                            nn.Sequential(
                                Rearrange('b h w (c s) -> b h c (w s)', s = s),
                                nn.Linear(width * s, width * s),
                                Rearrange('b h c (w s) -> b h w (c s)', s = s),
                            ),
                            nn.Linear(dim, dim)
                        ),
                        nn.Linear(dim, dim),
                        
                    )),
                    PreNormResidual(dim, nn.Sequential(
                        nn.Linear(dim, int(dim * expansion_factor)),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(int(dim * expansion_factor), dim),
                        nn.Dropout(dropout)
                    ))
                )
             )
        
        self.mlp_Mixer.append(nn.LayerNorm(dim))
        self.classfier = nn.Sequential(Reduce('b h w c -> b c', 'mean') , nn.Linear(dim, num_classes) , nn.Softmax(dim=1))
         
    def forward(self , X_data) :
        x  = self.to_patch_embedding(X_data)
        x = self.mlp_Mixer(x)
        x = self.classfier(x)
        return x