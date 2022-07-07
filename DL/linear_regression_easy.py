import torch
import torch.nn as nn
from torch.utils import data
def synthetic_data(w, b, num_examples):
    X = torch.normal(0,1,(num_examples,len(w)))
    y = torch.matmul(X,w)+b
    y += torch.normal(0,0.01,y.size())
    return X,y.reshape(-1,1)
true_w = torch.tensor([2.0,-3.4])
true_b = torch.tensor(4.2)

features , labels = synthetic_data(true_w,true_b,100)
print(labels.shape)
def load_array(data_arrays,batch_size,is_train = True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,is_train)
# X = torch.arange(6,dtype=float).reshape(3,2)
# print(X)
# print(torch.matmul(X,true_w.T)==torch.matmul(X,true_w))
batchsize =10
gen = load_array((features,labels),batchsize,True)
# X,y = next(iter(gen))
# print(X,'\n',y)
net = nn.Sequential(nn.Linear(len(true_w),1))
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)
Loss = nn.MSELoss()

# for p in net[0].state_dict():
#     print(p)
# for p in net[0].parameters():
#     print(p)
trainer = torch.optim.SGD(net.parameters(),lr=0.03)
num_epoch =3
for epoch in range(num_epoch) :
    for X, y in gen:
        # print(X.shape)
        # print(y.shape)

        L = Loss(net(X),y)
        trainer.zero_grad()
        L.backward()
        # print(net[0].weight.grad)
        trainer.step()
    l = Loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)

# Python之可变参数，*参数，**参数，以及传入*参数，**参数解包，*args，**kwargs的理解
# https://blog.csdn.net/cadi2011/article/details/84871401