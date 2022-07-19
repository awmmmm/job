import torch
from torch import nn

import math

from torch.nn import functional as F
import somework

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size, num_steps = 32, 35
train_iter, vocab = somework.load_data_time_machine(batch_size, num_steps)
#        * 符号 第二种情况：用在变量的前面。
#
# 1，向函数传递参数，将变量中可迭代对象的元素拆解出来，作为独立的参数第传给函数，如：
# print(*train_iter)

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01
    Wx = nn.Parameter(normal((num_inputs,num_hiddens)))
    Wh = nn.Parameter(normal((num_hiddens,num_hiddens)))
    Bh = nn.Parameter(torch.zeros(num_hiddens,device=device))

    Wo =nn.Parameter(normal((num_hiddens,num_outputs)))
    Bo =nn.Parameter(torch.zeros(num_outputs,device=device))
    params = [Wx,Wh,Bh,Wo,Bo]
    return params
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros([batch_size,num_hiddens],device=device),)

# [下面的rnn函数定义了如何在一个时间步内计算隐状态和输出。]
# 循环神经网络模型通过inputs最外层的维度实现循环， 以便逐时间步更新小批量数据的隐状态H。
# 此外，这里使用 tanh 函数作为激活函数。 如 :numref:sec_mlp所述，
# 当元素在实数上满足均匀分布时， tanh 函数的平均值为0。
# def rnn(inputs, state, params):
#     B,T,D = inputs.shape
#     torch.permute(inputs,[1,0,2])
#     output = []
#     for i in range(T):
#         h = F.tanh(inputs[i,...].reshape(T,D)@params[0]+state[0]@params[1]+params[2])
#         o = h@params[3]+params[4]
#         output.append(o)
#     return torch.cat(output,dim=0),(h,)
def rnn(inputs, state, params):
    # inputs的形状：(时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形状：(批量大小，词表大小)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

class RNNModelScratch: #@save
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)
num_hiddens = 512
net = RNNModelScratch(len(vocab),num_hiddens,device,get_params,init_rnn_state,rnn)
def predict_ch8(prefix, num_preds, net:RNNModelScratch, vocab, device):
    state = net.begin_state(1,device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda :torch.tensor(outputs[-1],device=device).reshape((1, 1))
    #预热
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def grad_clipping(net, theta):  #@save
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    state = None
    metric = somework.Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1])
#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: somework.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 100, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('and he'))

    print(f'困惑度 {ppl:.1f},  {str(device)}')
    # print(predict('time traveller'))
    # print(predict('traveller'))
    print(predict('and'))
    print(predict('he'))


num_epochs, lr = 500, 1
# train_ch8(net, train_iter, vocab, lr, num_epochs, device)
net = RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                      init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, device,
          use_random_iter=True)