import torch
import somework
import torch.nn as nn
batchsize = 256
train_iter,test_iter = somework.load_data_fashion_mnist(batch_size=batchsize)
net = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(784,10))
def init_weight(m):
    if isinstance(m,nn.Linear):
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weight)
#nn.Flatten() 将张量转至2d,第一维不变
Loss = torch.nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(),lr = 0.3)
num_epoch = 10
def accuracy(y_hat, y):
    pred_label = torch.argmax(y_hat,dim=1)
    count = 0
    for i in range(len(y)):
        if pred_label[i]==y[i]:
            count += 1
    return float(count/len(y))
# X_test,y_test = next(iter(test_iter))
# test_loss = Loss(net(X_test),y_test)
# test_ACC = accuracy(net(X_test),y_test)
# print(f'epoch {0}, loss {float(test_loss):f} ,ACC{test_ACC:f}')
for epoch in range(num_epoch):
    for X,y in train_iter:
        loss = Loss(net(X),y)
        trainer.zero_grad()
        loss.backward()
        trainer.step()
    X_test,y_test = next(iter(test_iter))
    test_loss = Loss(net(X_test),y_test)
    test_ACC = accuracy(net(X_test),y_test)
    print(f'epoch {epoch + 1}, loss {float(test_loss):f} ,ACC{test_ACC:f}')