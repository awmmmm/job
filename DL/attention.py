import torch
import matplotlib.pyplot as plt
import somework
from torch import nn
n_train = 50  # 训练样本数
x_train, _ = torch.sort(torch.rand(n_train) * 5)   # 排序后的训练样本

def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出
x_test = torch.arange(0, 5, 0.1)  # 测试样本
y_truth = f(x_test)  # 测试样本的真实输出
n_test = len(x_test)  # 测试样本数
# def plot_kernel_reg(y_hat):
#     plt.plot(x_test.detach(), y_truth.detach(), 'x', 'y', legend=['Truth', 'Pred'],
#              xlim=[0, 5], ylim=[-1, 5])
#     plt.plot(x_test.detach(),  y_hat.detach(), 'x', 'y', legend=['Truth', 'Pred'],
#              xlim=[0, 5], ylim=[-1, 5])
#     plt.plot(x_train.detach(), y_train.detach(), 'o', alpha=0.5);
#     plt.show()\


# 𝑓(𝑥)=1𝑛∑𝑖=1𝑛𝑦𝑖,

y_hat = torch.repeat_interleave(y_train.mean(), n_test)

#核回归

X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))

# x_train包含着键。attention_weights的形状：(n_test,n_train),
# 每一行都包含着要在给定的每个查询的值（y_train）之间分配的注意力权重
# softmax里边的叫注意力评分函数
# 加完softmax得到注意力权重
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# query = x_test[:,None]
#
# attention_weights = somework.softmax(-0.5*(query-x_train)**2)
y_hat =  torch.mm(attention_weights,y_train.reshape(-1,1))
plt.plot(x_test.detach(), y_truth.detach())
plt.plot(x_test.detach(),  y_hat.detach())
plt.plot(x_train.detach(), y_train.detach())
plt.show()
# test_data = torch.arange(50)
# (默认没有维度好像是会变成行向量)
# a = torch.arange(50).reshape(1,50)
# print((a-test_data).shape)
# print(test_data.shape)
somework.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')


class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # queries和attention_weights的形状为(查询个数，“键－值”对个数)

        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))

        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w) ** 2 / 2, dim=1)

        # values的形状为(查询个数，“键－值”对个数)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)
# 训练
# X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
X_tile = x_train.repeat((n_train, 1))
# print(X_tile)
# Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
Y_tile = y_train.repeat((n_train, 1))
# keys的形状:('n_train'，'n_train'-1)
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
# 操作是只选中值为true的值，每行有个false所以列数减一
# print(keys)
# print((1-torch.eye(50)).type(torch.bool))
# values的形状:('n_train'，'n_train'-1)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)

# 这段代码在做的事情是通过从原始训练数据中每次按列去掉一个值得到('n_train'，'n_train'-1)的key矩阵
# (即每次不同的query会有不同的key)
# 同样的方式获得对应key的value
for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')

keys = x_train.repeat((n_test, 1))
# value的形状:(n_test，n_train)
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plt.plot(x_test.detach(), y_truth.detach())
plt.plot(x_test.detach(),  y_hat.detach())
plt.plot(x_train.detach(), y_train.detach())
plt.show()
somework.show_heatmaps(net.attention_weights.detach().unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')