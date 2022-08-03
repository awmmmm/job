from math import sqrt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from hparams import hparams
from dataset import Tacotron2_Dataset,TextMelCollate
def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    # ids = torch.arange(0, max_len, out=torch.LongTensor(max_len))
    ids = torch.arange(0, max_len,out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask

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

        self.conv = torch.nn.Conv1d(in_channels,out_channels,
                                    kernel_size,stride,padding,
                                    dilation,bias=bias)
        # self.conv = torch.nn.Conv1d(in_channels, out_channels,
        #                             kernel_size=kernel_size, stride=stride,
        #                             padding=padding, dilation=dilation,
        #                             bias=bias)
        torch.nn.init.xavier_uniform_(self.conv.weight,
                    torch.nn.init.calculate_gain(w_init_gain))
    def forward(self,x):
        return self.conv(x)
class Prenet(nn.Module):
    def __init__(self,in_dim,prenet_dim):
        super(Prenet, self).__init__()
        self.linear1 = nn.Sequential(
            LinearNorm(in_dim,prenet_dim,bias=False),#why no bias?& why don't use relu gain
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.linear2 = nn.Sequential(
            LinearNorm(prenet_dim, prenet_dim, bias=False),  # why no bias?& why don't use relu gain
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    def forward(self,x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, para):
        super(Postnet, self).__init__()

        self.postnet_layer_1 = nn.Sequential(
            ConvNorm(para.n_mel_channels, para.postnet_embedding_dim,
                     kernel_size=5, stride=1, padding=2, dilation=1, w_init_gain='tanh'),
            nn.BatchNorm1d(para.postnet_embedding_dim),
            nn.Tanh(),
            nn.Dropout(0.5)
        )

        self.postnet_layer_2 = nn.Sequential(
            ConvNorm(para.postnet_embedding_dim, para.postnet_embedding_dim,
                     kernel_size=5, stride=1, padding=2, dilation=1, w_init_gain='tanh'),
            nn.BatchNorm1d(para.postnet_embedding_dim),
            nn.Tanh(),
            nn.Dropout(0.5)
        )

        self.postnet_layer_3 = nn.Sequential(
            ConvNorm(para.postnet_embedding_dim, para.postnet_embedding_dim,
                     kernel_size=5, stride=1, padding=2, dilation=1, w_init_gain='tanh'),
            nn.BatchNorm1d(para.postnet_embedding_dim),
            nn.Tanh(),
            nn.Dropout(0.5)
        )

        self.postnet_layer_4 = nn.Sequential(
            ConvNorm(para.postnet_embedding_dim, para.postnet_embedding_dim,
                     kernel_size=5, stride=1, padding=2, dilation=1, w_init_gain='tanh'),
            nn.BatchNorm1d(para.postnet_embedding_dim),
            nn.Tanh(),
            nn.Dropout(0.5)
        )

        self.postnet_layer_5 = nn.Sequential(
            ConvNorm(para.postnet_embedding_dim, para.n_mel_channels,
                     kernel_size=5, stride=1, padding=2, dilation=1, w_init_gain='linear'),
            nn.BatchNorm1d(para.n_mel_channels),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        out = self.postnet_layer_1(x)
        out = self.postnet_layer_2(out)
        out = self.postnet_layer_3(out)
        out = self.postnet_layer_4(out)
        out = self.postnet_layer_5(out)

        return out


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
        # 双向的lstm必须这么搞，不然无法准确计算
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
class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        # attention_weights_cat [B,2,T]
        processed_attention = self.location_conv(attention_weights_cat) #[B,32,T]
        processed_attention = processed_attention.transpose(1, 2)   # [B,T,32]
        processed_attention = self.location_dense(processed_attention) #[B,T,128]
        return processed_attention
class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim,attention_dim,False,'tanh')
        self.key = LinearNorm(embedding_dim,attention_dim,False,'tanh')
        self.locationlayer = LocationLayer(attention_location_n_filters,
                                           attention_location_kernel_size,
                                           attention_dim)
        self.v = LinearNorm(attention_dim,1,bias=False)
        self.score_mask_value = -float('inf')
    def get_alignment_energies(self,query,processed_memeory,
                               attention_weights_cat):
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.locationlayer(attention_weights_cat)
        energies = self.v(torch.tanh(processed_query+processed_memeory+processed_attention_weights))
        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        alignment = self.get_alignment_energies(attention_hidden_state,processed_memory,attention_weights_cat)
        if mask is not None:
            alignment.data.masked_fill_(mask,self.score_mask_value)
        attention_weights = torch.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Decoder(nn.Module):
    def __init__(self,para):
        super(Decoder, self).__init__()
        self.n_mel_channels = para.n_mel_channels
        self.n_frames_per_step = para.n_frames_per_step
        self.encoder_embedding_dim = para.encoder_embedding_dim
        self.attention_rnn_dim = para.attention_rnn_dim
        self.decoder_rnn_dim =para.decoder_rnn_dim
        self.prenet_dim = para.prenet_dim
        self.max_decoder_steps = para.max_decoder_steps
        self.gate_threshold = para.gate_threshold

        self.prenet = Prenet(para.n_frames_per_step*para.n_mel_channels,para.prenet_dim)
        self.attention_rnn = nn.LSTMCell(para.prenet_dim+para.encoder_embedding_dim,
                                         para.attention_rnn_dim)
        self.attention_rnn_dropout = nn.Dropout(0.1)
        self.attention = Attention(para.attention_rnn_dim,para.encoder_embedding_dim,
                                   para.attention_dim,para.attention_location_n_filters,
                                   para.attention_location_kernel_size)
        self.decoder_rnn = nn.LSTMCell(para.attention_rnn_dim+para.encoder_embedding_dim,
                                       para.decoder_rnn_dim)
        self.decoder_rnn_dropout = nn.Dropout(0.1)
        self.linear_projection = LinearNorm(para.decoder_rnn_dim+para.encoder_embedding_dim
                                           ,para.n_mel_channels*para.n_frames_per_step)
        self.gate_linear = LinearNorm(para.decoder_rnn_dim + para.encoder_embedding_dim
                                            , 1,w_init_gain='sigmoid')
        # self.post_net =Postnet(para)
    def get_go_frame(self, memory):
        """
        构造一个全0的矢量作为 decoder 第一帧的输出
        """
        # new创建一个新的Tensor，该Tensor的type和device都和原有Tensor一致，且无内容。

        B = memory.shape[0]
        first_output = memory.data.new(B,self.n_mel_channels*self.n_frames_per_step).zero_()
        return first_output

    def initialize_decoder_states(self, memory, mask):

        B ,MAX_TIME,_ = memory.shape
        self.attention_rnn_hidden = memory.data.new(B,self.attention_rnn_dim).zero_()
        self.attention_rnn_cell = memory.data.new(B, self.attention_rnn_dim).zero_()
        self.decoder_rnn_hidden = memory.data.new(B,self.decoder_rnn_dim).zero_()
        self.decoder_rnn_cell = memory.data.new(B, self.decoder_rnn_dim).zero_()
        self.attention_weights = memory.data.new(B,MAX_TIME).zero_()
        self.attention_weights_cum = memory.data.new(B, MAX_TIME).zero_()
        self.attention_context = memory.data.new(B,self.encoder_embedding_dim).zero_()
        self.memory = memory
        self.processed_memory = self.attention.key(memory)
        self.mask = mask
    

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """

        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)

        decoder_inputs = decoder_inputs.transpose(1,2)

        # (B, T_out, n_mel_channels) -> (B, T_out/3, n_mel_channels*3)

        decoder_inputs = decoder_inputs.reshape(decoder_inputs.size(0),
                                                decoder_inputs.size(1)//self.n_frames_per_step,
                                                -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0,1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0,1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()

        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        mel_outputs = mel_outputs.transpose(1,2)

        return mel_outputs,gate_outputs,alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
               PARAMS
               ------
               decoder_input: previous mel output

               RETURNS
               -------
               mel_output:
               gate_output: gate output energies
               attention_weights:
               """
        cell_input = torch.cat((decoder_input,self.attention_context),dim=1)
        self.attention_rnn_hidden,self.attention_rnn_cell = self.attention_rnn(cell_input,(self.attention_rnn_hidden,self.attention_rnn_cell))
        self.attention_rnn_hidden = self.attention_rnn_dropout(self.attention_rnn_hidden)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention(
            self.attention_rnn_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)
        self.attention_weights_cum += self.attention_weights

        decoder_rnn_cell_input = torch.cat((self.attention_rnn_hidden,self.attention_context),dim=1)
        self.decoder_rnn_hidden,self.decoder_rnn_cell = self.decoder_rnn(decoder_rnn_cell_input,(self.decoder_rnn_hidden,self.decoder_rnn_cell))
        self.decoder_rnn_hidden = self.decoder_rnn_dropout(self.decoder_rnn_hidden)
        decoder_rnn_hidden_attention_context = torch.cat((self.decoder_rnn_hidden,self.attention_context),dim=1)
        decoder_output = self.linear_projection(decoder_rnn_hidden_attention_context)
        gate_output = self.gate_linear(decoder_rnn_hidden_attention_context)
        return decoder_output,gate_output,self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
                PARAMS
                ------
                memory: Encoder outputs
                decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
                memory_lengths: Encoder output lengths for attention masking.

                RETURNS
                -------
                mel_outputs: mel outputs from the decoder
                gate_outputs: gate outputs from the decoder
                alignments: sequence of attention weights from the decoder

                 mel_outputs, gate_outputs, alignments = self.decoder(
                    encoder_outputs, mels, memory_lengths=text_lengths)

                """
        decoder_input = self.get_go_frame(memory).unsqueeze(0)#1,B,3*80
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)#(T_out, B, n_mel_channels)
        decoder_inputs = torch.cat([decoder_input,decoder_inputs])
        decoder_inputs = self.prenet(decoder_inputs)
        self.initialize_decoder_states(memory,mask=~get_mask_from_lengths(memory_lengths))
        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs)<decoder_inputs.size(0)-1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            decoder_output,gate_output,attention_weights = \
                self.decode(decoder_input)
            gate_outputs.append(gate_output.squeeze(1))
            mel_outputs.append(decoder_output.squeeze(1))
            alignments.append(attention_weights)
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(mel_outputs,gate_outputs,alignments)
        return mel_outputs, gate_outputs, alignments

    def inference(self,memory):
        decoder_input = self.get_go_frame(memory)
        self.initialize_decoder_states(memory, mask=None)
        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            decoder_output, gate_output, attention_weights = \
                self.decode(decoder_input)
            # decoder_input = decoder_output
            gate_outputs.append(gate_output)
            mel_outputs.append(decoder_output.squeeze(1))
            alignments.append(attention_weights)
            if torch.sigmoid(gate_output) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("已达到最大编码步数")
                break
            decoder_input = decoder_output
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)
        return mel_outputs, gate_outputs, alignments

class Tacotron2(nn.Module):
    def __init__(self,para):
        super(Tacotron2, self).__init__()
        self.n_frames_per_step = para.n_frames_per_step
        self.n_mel_channels = para.n_mel_channels
        self.embedding = nn.Embedding(para.n_symbols, para.symbols_embedding_dim)
        std = sqrt(2.0 / (para.n_symbols + para.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)#why?
        self.encoder = Encoder(para)
        self.decoder = Decoder(para)
        self.postnet = Postnet(para)

    def parse_output(self, outputs, output_lengths=None):
        # mask = ~get_mask_from_lengths(output_lengths)
        max_len = outputs[0].size(-1)
        # ids = torch.arange(0, max_len, out=torch.LongTensor(max_len))
        ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
        mask = (ids < output_lengths.unsqueeze(1)).bool()
        mask = ~mask
        mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
        mask = mask.permute(1, 0, 2)

        outputs[0].data.masked_fill_(mask, 0.0)
        outputs[1].data.masked_fill_(mask, 0.0)
        outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, text_inputs, text_lengths, mels, output_lengths):
        # 进行 text 编码
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        # 得到encoder输出
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        # 得到 decoder 输出
        mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, mels, memory_lengths=text_lengths)

        gate_outputs = gate_outputs.unsqueeze(2).repeat(1, 1, self.n_frames_per_step)
        gate_outputs = gate_outputs.view(gate_outputs.size(0), -1)

        # 进过postnet 得到预测的 mel 输出
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, alignments], output_lengths)
    def inference(self, inputs):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = [mel_outputs, mel_outputs_postnet, gate_outputs, alignments]

        return outputs


if __name__ == "__main__":

    para = hparams()
    m_Dataset = Tacotron2_Dataset(para)
    collate_fn = TextMelCollate(para.n_frames_per_step)

    m_DataLoader = DataLoader(m_Dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)

    m_model = Tacotron2(para)

    for e_epoch in range(5):

        for i, batch_samples in enumerate(m_DataLoader):

            text_in = batch_samples[0]
            text_lengths = batch_samples[1]
            mel_in = batch_samples[2]
            mel_lengths = batch_samples[4]

            outputs = m_model(text_in, text_lengths, mel_in, mel_lengths)
            print(outputs[0])

            m_model.eval()
            eval_outputs = m_model.inference(text_in[0].unsqueeze(0))
            print(eval_outputs)
            m_model.train()
            if i > 5:
                break


