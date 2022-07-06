import random

import torch
import matplotlib.pyplot as plt

def synthetic_data(w, b, num_examples):
    x = torch.normal(0,1,(num_examples,len(w)))
    y = torch.matmul(x,w) + b
    y += torch.normal(0,1e-2,y.size())
    return x,y.reshape(-1,1)

true_w = torch.tensor([3.0,4.0])
true_b = torch.tensor(2.0)
features, labels = synthetic_data(true_w, true_b, 30)
# print(features)
# print(labels)
# plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy())
# plt.show()
# plt.close()
# print(features[:,1])
def data_iter(batch_size,feature,label):
    num_examples = len(feature)
    indices = list(range(num_examples))
    random.shuffle(indices)
    # print(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size,num_examples)])
        yield feature[batch_indices],label[batch_indices]

batch_size = 10
gen = data_iter(batch_size,features,labels)
X,y = next(gen)
# print(X,'\n',y)
# print(type(data_iter(batch_size,features,labels)))
for X,y in data_iter(batch_size,features,labels):
    print(X,'\n',y)
    break


w = torch.normal(0,1e-2,size=(2,1),requires_grad=True)
b = torch.normal(0,1e-2,[1],requires_grad=True)
# print(b)
# b = torch.zeros(1,requires_grad=True)
# print(b)
# b = torch.zeros_like()
def linereg(X,w,b):
    return torch.matmul(X,w)+b

def squared_loss(y_hat, y):
    return (y.reshape(y_hat.shape)-y_hat)**2/2

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            # print(id(param))
            #这里有个坑，这里tensor应该是重载 ”-=“还有很多其他符号被重载；这里“-=”不会改变内存地址
            param -= lr * param.grad / batch_size
            # print(id(param))
            param.grad.zero_()
# a = torch.matmul(w.T,w)
# a.backward()
# print(w.grad==2*w)
# print(w.grad)
lr = 0.03
num_epochs = 50
net = linereg
loss = squared_loss
print(w,'\n',b)
for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        y_hat = net(X,w,b)
        l = loss(y_hat,y)
        #当y不是标量时，向量y关于向量x的导数的最自然解释是一个矩阵。
        l.sum().backward()
        # print(b.grad)
        sgd([w,b],lr,batch_size)
        # sgd([b],lr,batch_size)
        # sgd([w,b], lr, batch_size)
    with torch.no_grad():
        train_l = squared_loss(net(X,w,b),y)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
print(w,'\n',b)

# b = 3
# print(id(b))
# a = 5
# print(id(a))
# a = a-1
# print(id(a))
#
# a -= 1
# print(id(a))