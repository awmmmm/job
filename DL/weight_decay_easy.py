import torch
import somework
from torch import nn
from torch.utils import data
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w,true_b = torch.ones((num_inputs,1))*0.01,0.05
features,labels = somework.synthetic_data(true_w,true_b,n_train)
# batch_size = 16
train_iter = somework.load_array((features,labels),batch_size)
test_iter = somework.load_array(somework.synthetic_data(true_w,true_b,n_test),n_test)

net = nn.Sequential(nn.Linear(num_inputs,1))
nn.init.normal_(net[0].weight)
net[0].bias.data.zero_()
wd = 3
trainer = torch.optim.SGD([{"params":net[0].weight,'weight_decay': wd},{"params":net[0].bias}],lr = 0.003)
# def init_weight(m):
#     if isinstance(m,nn.Linear):
#         m.weight.data.normal_()
#         # nn.init.normal_(m.weight, 0., 0.01)
# net.apply(init_weight)

# nn.Linear.weight.data.n
trainer = torch.optim.SGD([{"params":net[0].weight,'weight_decay': wd},{"params":net[0].bias}],lr = 0.003)
Loss =nn.MSELoss(reduction='none')
num_epoch = 100

for epoch in range(num_epoch):
    for X,y in train_iter:

        l = Loss(net(X),y)
        # l = MSE_Loss(net(X), y)
        trainer.zero_grad()
        l.mean().backward()
        trainer.step()

    # with torch.no_grad():
    X, y = next(iter(test_iter))
    l = Loss(net(X), y)
    print(f'epoch {epoch}:', float(l.mean()))

print(torch.norm(net[0].weight))