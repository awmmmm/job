import torch
import somework

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w,true_b = torch.ones((num_inputs,1))*0.01,0.05
features,labels = somework.synthetic_data(true_w,true_b,n_train)
# batch_size = 16
train_iter = somework.load_array((features,labels),batch_size)
test_iter = somework.load_array(somework.synthetic_data(true_w,true_b,n_test),n_test)
w = torch.normal(0,1,true_w.shape,requires_grad=True)
b = torch.zeros(1,requires_grad=True)
def net(X):
    return X@w+b
def l2_penalty(X):
    # return torch.norm(X)
    # return torch.sum(X.pow(2))/2
    return torch.sum(torch.pow(X,2))/2
def MSE_Loss(y_hat,y):
    return torch.pow(y_hat-y.reshape(y_hat.shape),2)/2
def sgd(para,batchsize,lr):
    with torch.no_grad():
        for p in para:
            p -= p.grad*lr/batchsize
            p.grad.zero_()
num_epoch = 100
lamb = 3
for epoch in range(num_epoch):
    for X,y in train_iter:
        l = MSE_Loss(net(X),y) + lamb*l2_penalty(w)
        # l = MSE_Loss(net(X), y)
        l.sum().backward()
        sgd([w,b],X.shape[0],0.003)
    with torch.no_grad():
        X , y = next(iter(test_iter))
        l = MSE_Loss(net(X), y)
        print(f'epoch {epoch}:',float(l.mean()))
print(torch.norm(w))