import torch
from torch import nn
import somework
from torch.nn import functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size, num_steps = 32, 35
train_iter, vocab = somework.load_data_time_machine(batch_size, num_steps)
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01
    def three():
        # Wx = normal((num_inputs,num_hiddens))
        # Wh = normal((num_hiddens,num_hiddens))
        # B = torch.zeros(num_hiddens,device=device)
        # return Wx,Wh,B
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xh, W_hh, b_h = three()  # 候选隐状态参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    params = [W_xz, W_hz, b_z,W_xr, W_hr, b_r,W_xh, W_hh, b_h,W_hq,b_q]
    for param in params:
        # param.requires_grad_(True)
        param.requires_grad = True
    return params



# 现在我们将[定义隐状态的初始化函数]init_gru_state。
# 与 :numref:sec_rnn_scratch中定义的init_rnn_state函数一样，
# 此函数返回一个形状为（批量大小，隐藏单元个数）的张量，张量的值全部为零。

def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        H_hat = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H +(1 - Z) * H_hat
        #问题在于 H 必须得覆盖原来的H,这样状态才能一直传递
        O_t = (H @ W_hq) + b_q
        outputs.append(O_t)

    return torch.cat(outputs,dim = 0),(H,)
# def gru(inputs, state, params):
#     W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
#     H, = state
#     outputs = []
#     for X in inputs:
#         Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
#         R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
#         H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
#         H = Z * H + (1 - Z) * H_tilda
#         Y = H @ W_hq + b_q
#         outputs.append(Y)
#     return torch.cat(outputs, dim=0), (H,)
num_hiddens = 256
num_epochs, lr = 500, 1
model = somework.RNNModelScratch(len(vocab),num_hiddens,device,get_params,init_gru_state,gru)

somework.train_ch8(model, train_iter, vocab, lr, num_epochs, device)


# 简洁代码
# num_inputs = len(vocab)
# gru_layer = nn.GRU(num_inputs, num_hiddens)
# model = somework.RNNModel(gru_layer, len(vocab))
# model = model.to(device)
# somework.train_ch8(model, train_iter, vocab, lr, num_epochs, device)