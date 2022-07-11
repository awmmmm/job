import torch
from torch import nn
def pool2d(X,mode = 'max'):
    assert mode in ('max','average')
    B,C,in_w,in_b = X.shape
    out_w,out_h = in_w//2,in_b//2
    out = torch.empty((B,C,out_w,out_h))
    if mode == 'max':
        for b in range (B):
            for c in range(C):
                for w in range(out_w):
                    for h in range(out_h):
                        out[b,c,w,h] = torch.max(X[b,c,w*2:w*2+2,h*2:h*2+2])
    return  out

a= torch.tensor([[[[1,2,3,4],
                [5,6,7,8],
                [1,2,3,4],
                [5,6,7,8]]]])
print(pool2d(a))