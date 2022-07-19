import torch
from torch import nn
import somework

batch_size, num_steps = 32, 35
train_iter, vocab = somework.load_data_time_machine(batch_size, num_steps)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    def normal(shape):
        return torch.randn(shape,device=device)*0.01
    def three():
        return normal((num_inputs,num_hiddens)),normal((num_hiddens,num_hiddens)),torch.zeros(num_hiddens,device=device)

    W_i,H_i,B_i = three()
    W_f,H_f,B_f = three()
    W_o,H_o,B_o = three()
    W_c,H_c,B_c = three()
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    params = [W_i,H_i,B_i,W_f,H_f,B_f,W_o,H_o,B_o,W_c,H_c,B_c,W_hq,b_q]
    for param in params:
        param.requires_grad = True
    return params

def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size,num_hiddens),device=device),
            torch.zeros((batch_size,num_hiddens),device=device))

def lstm(inputs, state, params):
    W_i, H_i, B_i, W_f, H_f, B_f, W_o, H_o, B_o, W_c, H_c, B_c,W_hq , b_q = params
    H,C = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid(X @ W_i + H @ H_i + B_i)
        F = torch.sigmoid(X @ W_f + H @ H_f + B_f)
        O = torch.sigmoid(X @ W_o + H @ H_o + B_o)
        C_tilda = torch.tanh(X @ W_c + H @ H_c + B_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs,dim=0),(H,C)

vocab_size, num_hiddens, device = len(vocab), 256, device
num_epochs, lr = 500, 1
model = somework.RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params,
                            init_lstm_state, lstm)
somework.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

# num_inputs = vocab_size
# lstm_layer = nn.LSTM(num_inputs, num_hiddens)
# model = somework.RNNModel(lstm_layer, len(vocab))
# model = model.to(device)
# somework.train_ch8(model, train_iter, vocab, lr, num_epochs, device)