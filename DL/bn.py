import torch
from torch import nn
# x = torch.zeros((3,4))
# print(x.shape)
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if X.shape == 2:
            #一维
            m = X.mean(dim =0)
            var = torch.mean((X-m.reshape(1,-1))**2,dim = 0)
            # out = (X - m)*gamma/var+beta
        else:
            ##################

            m = X.mean(dim = (0,2,3),keepdim = True)
            var = ((X - m)**2).mean(dim = (0,2,3),keepdim = True)
        X_hat = (X - m) / torch.sqrt(var + eps)
        moving_mean = momentum*moving_mean+(1-momentum)*m
        moving_var = momentum * moving_var + (1 - var) * m
        out = gamma*X_hat+beta
    return out ,moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y

