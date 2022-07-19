import torch
from torch import nn
import matplotlib.pyplot as plt

import somework

T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
# plt.plot(time, x.numpy(),  )
# plt.show()

# tau = 4
# features = torch.empty((T-tau,tau))
# for i in range(T-tau):
#     features[i,:] = x[i:i+tau]
# label = x[tau:].reshape((-1,1))

#效率更高一点
#隐马尔可夫假设未来第T个输出只和前T-tau个输入有关
tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))
batch_size, n_train = 16, 600
# 只有前n_train个样本用于训练
train_iter = somework.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)

# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

# 平方损失。注意：MSELoss计算平方误差时不带系数1/2
loss = nn.MSELoss(reduction='none')


def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        metric = somework.Accumulator(2)
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
            metric.add(float(l.sum()), y.numel())

        print(f'epoch {epoch + 1}, '
              f'loss: {metric[0]/metric[1]:f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)
onestep_preds = net(features)
# plt.plot([time.numpy(), time[tau:].numpy()],[x.detach().numpy(), onestep_preds.detach().numpy()]  )
plt.plot(time.numpy(),x.detach().numpy())
plt.plot(time[tau:].numpy(),onestep_preds.detach().numpy())
plt.show()


multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(
        multistep_preds[i - tau:i].reshape((1, -1)))

plt.plot(time.numpy(),x.detach().numpy())
plt.plot(time[tau:].numpy(),onestep_preds.detach().numpy())
plt.plot(time[n_train + tau:].numpy(),multistep_preds[n_train + tau:].detach().numpy())
plt.show()