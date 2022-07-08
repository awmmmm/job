import torch
import torch.nn as nn
import somework
batch_size = 256
num_input = 784
num_output = 10
hidden_layers = 1
num_hidden =384
train_iter,test_iter =somework.load_data_fashion_mnist(batch_size)
para =[]
W_i = nn.Parameter(torch.randn((num_input,num_hidden),requires_grad=True)*0.01)#torch.randn np.random.radn 返回一个标准正态
b_i = nn.Parameter(torch.zeros(num_hidden,requires_grad=True))
para.append(W_i)
para.append(b_i)
for layer in range(hidden_layers - 1):
    W_h = nn.Parameter(torch.randn((num_hidden, num_hidden), requires_grad=True)*0.01)  # torch.randn np.random.radn 返回一个标准正态
    b_h = nn.Parameter(torch.zeros(num_hidden,requires_grad=True))
    para.append(W_h)
    para.append(b_h)

W_o = nn.Parameter(torch.randn((num_hidden,num_output),requires_grad=True)*0.01)#torch.randn np.random.radn 返回一个标准正态
b_o = nn.Parameter(torch.zeros(num_output,requires_grad=True))
para.append(W_o)
para.append(b_o)
# print(len(para))
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(a,X)
# def softmax(input):
#     fenzi = torch.exp(input)
#     fenmu = torch.sum(fenzi,dim=1,keepdim=True)
#     return fenzi/fenmu# 这里应用了广播机制
# W = nn.Parameter(torch.randn((num_input,num_output),requires_grad=True))#torch.randn np.random.radn 返回一个标准正态
def net(X):

    X = X.reshape(-1,num_input)
    for i in range(0,len(para)-2,2):
        X = relu(X@para[i]+para[i+1])
    return somework.softmax(X@para[-2]+para[-1])
#为啥这里死活出现null因为权重初始化均值唯一太大了，后面softmax求自然数e的指数导致出现几百的指数
# def cross_entropy(y_hat,y):
#     return - torch.log(y_hat[range(len(y_hat)),y])
batch_size = 4
test_input = torch.normal(0,1,(batch_size,num_input))
print(test_input)
test_out = net(test_input)
print(test_out)
print(test_out.sum(1))

def updater(batch_size):
    return somework.sgd(para, 0.1, batch_size)

# loss = somework.cross_entropy
num_epoch =15
# for epoch in range(num_epoch):
#     for X,y in train_iter:
#         l = cross_entropy(net(X),y)
#         l.sum().backward()
#         somework.sgd(para,0.1,X.shape[0])
#     with torch.no_grad():
#         X_test, y_test = next(iter(test_iter))
#         test_loss = cross_entropy(net(X_test), y_test)
#         print(test_loss.sum())


somework.train_ch3(net,train_iter,test_iter,somework.cross_entropy,num_epoch,updater)
# batch_size = 4
# test_input = torch.normal(0,1,(batch_size,num_input))
# test_out = net(test_input)
# print(test_out.shape)