import torch
from torch import nn
import somework
batch_size = 256

train_iter,test_iter =somework.load_data_fashion_mnist(batch_size)
net = nn.Sequential(nn.Flatten(),nn.Linear(784,384),nn.ReLU(),nn.Linear(384,10))

def init_weight(m):
    if isinstance(m,nn.Linear):
        nn.init.normal_(m.weight,0.,0.01)
net.apply(init_weight)
updater = torch.optim.SGD(net.parameters(),lr=0.1)
loss = nn.CrossEntropyLoss(reduction='none')
num_epoch =10
somework.train_ch3(net,train_iter,test_iter,loss,num_epoch,updater)