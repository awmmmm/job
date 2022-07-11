import torch
from torch import nn
out_channel = 4
C = 3
kernel = 3
weight = nn.Parameter(torch.ones((out_channel,C,kernel,kernel)))
# weight.data.normal_(0,0.01)
# nn.init.xavier_normal_(weight)
def conv2d(X,out_channel,kernel,stride=1,pad = 0):
    B ,C  ,W,H = X.shape
    # weight = nn.Parameter(torch.empty((out_channel,C,kernel,kernel)))
    X_pad = torch.zeros((B,C,W+2*pad,H+2*pad))
    X_pad[:,:,1:-1,1:-1] = X
    Out = torch.empty((B,out_channel,(W+2*pad-kernel)//stride+1,(H+2*pad-kernel)//stride+1))
    # for b in range(B):
    for c in range(out_channel):
        for w in range((W + 2 * pad - kernel) // stride + 1):
            for h in range((H + 2 * pad - kernel) // stride + 1):
                Out[:, c, w, h] = torch.sum(X_pad[:,:,w*stride:w*stride+kernel,h*stride:h*stride+kernel]*weight[c,:,:,:])
    return Out

a = torch.ones((1,3,5,5))
b = conv2d(a,out_channel,kernel,stride=1,pad=1)
print(weight.shape)
print(b)

