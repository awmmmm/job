from math import sqrt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from hparams import hparams
from dataset import Tacotron2_Dataset,TextMelCollate

class LinearNorm(torch.nn.Module):
    def __init__(self,in_dim,out_dim,bias=True,w_init_gain = 'linear'):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_dim,out_dim,bias)
        torch.nn.init.xavier_uniform_(self.linear.weight,torch.nn.init.calculate_gain(w_init_gain))

    def forward(self,x):
        return self.linear(x)


class ConvNorm(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=1, stride=1,padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels,out_channels,kernel_size,stride,padding,dilation,bias)
        torch.nn.init.xavier_uniform_(self.conv.weight, torch.nn.init.calculate_gain(w_init_gain))
    def forward(self,x):
        return self.conv(x)

class Postnet(nn.Module):
    def __init__(self,para):
        super(Postnet, self).__init__()
        self.convblock1 = nn.Sequential(ConvNorm(para.n_mel_channels, para.postnet_embedding_dim, kernel_size=5, stride=1,padding=2,
                 dilation=1, bias=True, w_init_gain='tanh'),
                                        nn.BatchNorm1d(para.postnet_embedding_dim),
                                        nn.Tanh(),
                                        nn.Dropout(0.5))
        self.convblock2 = nn.Sequential(
            ConvNorm(para.postnet_embedding_dim, para.postnet_embedding_dim, kernel_size=5, stride=1, padding=2,
                     dilation=1, bias=True, w_init_gain='tanh'),
            nn.BatchNorm1d(para.postnet_embedding_dim),
            nn.Tanh(),
            nn.Dropout(0.5))
        self.convblock3 = nn.Sequential(
            ConvNorm(para.postnet_embedding_dim, para.postnet_embedding_dim, kernel_size=5, stride=1, padding=2,
                     dilation=1, bias=True, w_init_gain='tanh'),
            nn.BatchNorm1d(para.postnet_embedding_dim),
            nn.Tanh(),
            nn.Dropout(0.5))
    def forward(self,x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x
class Encoder(nn.Module):
    '''
    三个卷积+双向LSTM
    '''
    def __init__(self,para):
        super(Encoder, self).__init__()
        self.convblock1 = nn.Sequential(
            ConvNorm(para.symbols_embedding_dim, para.encoder_embedding_dim, kernel_size=5, stride=1, padding=2,
                     dilation=1, bias=True, w_init_gain='relu'),
            nn.BatchNorm1d(para.postnet_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.convblock2 = nn.Sequential(
            ConvNorm(para.encoder_embedding_dim, para.encoder_embedding_dim, kernel_size=5, stride=1, padding=2,
                     dilation=1, bias=True, w_init_gain='relu'),
            nn.BatchNorm1d(para.postnet_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.convblock3 = nn.Sequential(
            ConvNorm(para.encoder_embedding_dim, para.encoder_embedding_dim, kernel_size=5, stride=1, padding=2,
                     dilation=1, bias=True, w_init_gain='relu'),
            nn.BatchNorm1d(para.postnet_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.bilstm = nn.LSTM(para.encoder_embedding_dim,
                              para.encoder_embedding_dim//2,
                              num_layers=1,bias=True,bidirectional=True)

    def forward(self, x, input_lengths):
        # B,C,T
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = x.permute(0,2,1)
        input_lengths = input_lengths.cpu().numpy()
        # B,T,C
        x = nn.utils.rnn.pack_padded_sequence(x,input_lengths,batch_first=True)
        self.bilstm.flatten_parameters()
        o,_ = self.bilstm(x)
        o,_ = nn.utils.rnn.pad_packed_sequence(o,batch_first=True)
        return o
    def inference(self,x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = x.permute(0, 2, 1)
        self.bilstm.flatten_parameters()
        o, _ = self.bilstm(x)
        return o

