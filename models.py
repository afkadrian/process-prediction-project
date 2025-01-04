import torch.nn as nn
import torch
import math
import torch.nn.init as init

# http://nlp.seas.harvard.edu/2018/04/03/attention.html#embeddings-and-softmax
# TODO study the example with padding_idx part at https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
# TODO https://github.com/pytorch/pytorch/blob/bac4cfd54d44aa0fbc574e6561b878cb406762cc/torch/nn/modules/sparse.py#L22
# From now on input/output is always a tuple!
# or further attributes should be concatenated as an extra (last) dim of input tensor x
# https://walkwithfastai.com/tab.ae
class Embedding(nn.Module):
    def __init__(self, d_model, vocab_size, dropout_prob=0.0, attributes_meta=None, time_attribute_concatenated=False, pad_token=None):
        super().__init__()

        self.d_model = d_model
        self.attributes_meta = attributes_meta
        self.time_attribute_concatenated = time_attribute_concatenated

        if 1 in self.attributes_meta.keys() and not self.time_attribute_concatenated:
            self.activity_label = nn.Embedding(vocab_size, self.d_model, padding_idx=pad_token)
            self.time_attribute = nn.Linear(1, self.d_model)
        elif 1 in self.attributes_meta.keys() and self.time_attribute_concatenated:
            self.activity_label = nn.Embedding(vocab_size, self.d_model-1, padding_idx=pad_token)
        elif 1 not in self.attributes_meta.keys():
            self.activity_label = nn.Embedding(vocab_size, self.d_model, padding_idx=pad_token)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x): # input is always a tuple
        if 1 in self.attributes_meta.keys() and not self.time_attribute_concatenated:
            return self.dropout(self.activity_label(x[0].long()).squeeze(2) + self.time_attribute(x[1])) * math.sqrt(self.d_model)
        elif 1 in self.attributes_meta.keys() and self.time_attribute_concatenated:
            return self.dropout(torch.cat((self.activity_label(x[0].long()).squeeze(2), x[1]), dim=-1)) * math.sqrt(self.d_model)
        elif 1 not in self.attributes_meta.keys():
            return self.dropout(self.activity_label(x[0].long()).squeeze(2)) * math.sqrt(self.d_model)


class ManualEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size, dropout_prob=0.0, attributes_meta=None, time_attribute_concatenated=False):
        super().__init__()

        self.d_model = d_model
        self.attributes_meta = attributes_meta
        self.time_attribute_concatenated = time_attribute_concatenated

        if 1 in self.attributes_meta.keys() and not self.time_attribute_concatenated:
            self.activity_label = nn.Linear(vocab_size, self.d_model)
            self.time_attribute = nn.Linear(1, self.d_model)
        elif 1 in self.attributes_meta.keys() and self.time_attribute_concatenated:
            self.activity_label = nn.Linear(vocab_size, self.d_model-1)
        elif 1 not in self.attributes_meta.keys():
            self.activity_label = nn.Linear(vocab_size, self.d_model)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x): # input is always a tuple
        if 1 in self.attributes_meta.keys() and not self.time_attribute_concatenated:
            return self.dropout(self.activity_label(x[0]).squeeze(2) + self.time_attribute(x[1])) * math.sqrt(self.d_model)
        elif 1 in self.attributes_meta.keys() and self.time_attribute_concatenated:
            return self.dropout(torch.cat((self.activity_label(x[0]).squeeze(2), x[1]), dim=-1)) * math.sqrt(self.d_model)
        elif 1 not in self.attributes_meta.keys():
            return self.dropout(self.activity_label(x[0]).squeeze(2)) * math.sqrt(self.d_model)


# http://nlp.seas.harvard.edu/2018/04/03/attention.html#model-architecture
# TODO Weight sharing https://arxiv.org/abs/1608.05859 & https://arxiv.org/abs/1706.03762
# TODO Wrapping sigmoid() could be beneficial for the time_attribute
class Readout(nn.Module):
    def __init__(self, d_model, vocab_size, dropout_prob=0.0, attributes_meta=None, time_attribute_concatenated=False):
        super().__init__()

        self.attributes_meta = attributes_meta
        self.time_attribute_concatenated = time_attribute_concatenated

        if 1 in self.attributes_meta.keys() and not self.time_attribute_concatenated:
            self.activity_label = nn.Linear(d_model, vocab_size)
            self.time_attribute = nn.Linear(d_model, 1)
        elif 1 in self.attributes_meta.keys() and self.time_attribute_concatenated:
            self.activity_label = nn.Linear(d_model-1, vocab_size)
        elif 1 not in self.attributes_meta.keys():
            self.activity_label = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        if 1 in self.attributes_meta.keys() and not self.time_attribute_concatenated:
            return self.dropout(self.activity_label(x)), self.dropout(self.time_attribute(x))
        elif 1 in self.attributes_meta.keys() and self.time_attribute_concatenated:
            return self.dropout(self.activity_label(x[:, :, :-1])), self.dropout(x[:, :, -1:])
        elif 1 not in self.attributes_meta.keys():
            return (self.dropout(self.activity_label(x)),)  # output is always a tuple


class SequentialDecoder(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_layers,
                 dropout_prob,
                 vocab_size,
                 attributes_meta,
                 time_attribute_concatenated,
                 pad_token,
                 nb_special_tokens,
                 architecture=None):
        super().__init__()

        self.vocab_size = vocab_size + nb_special_tokens

        if architecture is not None:
            self.architecture = architecture

        self.value_embedding = Embedding(d_model=hidden_size,
                                         vocab_size=self.vocab_size,
                                         dropout_prob=dropout_prob,
                                         attributes_meta=attributes_meta,
                                         time_attribute_concatenated=time_attribute_concatenated,
                                         pad_token=pad_token)

        self.cell = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_prob)

        self.readout = Readout(d_model=hidden_size,
                               vocab_size=self.vocab_size,
                               dropout_prob=dropout_prob,
                               attributes_meta=attributes_meta,
                               time_attribute_concatenated=time_attribute_concatenated)

    def forward(self, x, init_hidden=None):
        if init_hidden is not None:
            return self.readout(self.cell(self.value_embedding(x), init_hidden)[0])
        else:
            return self.readout(self.cell(self.value_embedding(x))[0])

class SequentialAutoEncoder(nn.Module):
    # TODO implement SequentialDecoder.Readout.weights = SequentialDecoder.Embedding.weights = SequentialEncoder.Embedding.weights
    def __init__(self,
                 hidden_size,
                 num_layers,
                 dropout_prob,
                 vocab_size,
                 attributes_meta,
                 time_attribute_concatenated,
                 pad_token,
                 nb_special_tokens):
        super().__init__()

        self.encoder = SequentialEncoder(hidden_size,
                                         num_layers,
                                         dropout_prob,
                                         vocab_size,
                                         attributes_meta,
                                         time_attribute_concatenated,
                                         pad_token,
                                         nb_special_tokens)
        self.decoder = SequentialDecoder(hidden_size,
                                         num_layers,
                                         dropout_prob,vocab_size,
                                         attributes_meta,
                                         time_attribute_concatenated,
                                         pad_token,
                                         nb_special_tokens)

    def forward(self, prefix, suffix):
        # During training it is teacher forcing / supervised learning / closed loop
        # During inference it is open loop
        return self.decoder(suffix, self.encoder(prefix)[1])
    
class SequentialEncoder(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_layers,
                 dropout_prob,
                 vocab_size,
                 attributes_meta,
                 time_attribute_concatenated,
                 pad_token,
                 nb_special_tokens):
        super().__init__()

        self.vocab_size = vocab_size + nb_special_tokens

        self.value_embedding = Embedding(d_model=hidden_size,
                                         vocab_size=self.vocab_size,
                                         dropout_prob=dropout_prob,
                                         attributes_meta=attributes_meta,
                                         time_attribute_concatenated=time_attribute_concatenated,
                                         pad_token=pad_token)

        self.cell = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_prob)

    def forward(self, x):
        return self.cell(self.value_embedding(x))
    
    
class SequentialDiscriminator(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_layers,
                 dropout_prob,
                 vocab_size,
                 attributes_meta,
                 time_attribute_concatenated,
                 pad_token,
                 nb_special_tokens):
        super().__init__()

        self.vocab_size = vocab_size + nb_special_tokens

        self.value_embedding = ManualEmbedding(d_model=hidden_size,
                                               vocab_size=self.vocab_size,
                                               dropout_prob=dropout_prob,
                                               attributes_meta=attributes_meta,
                                               time_attribute_concatenated=time_attribute_concatenated)

        self.cell = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_prob)

        self.readout = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        return self.dropout(self.readout(self.cell(self.value_embedding(x))[0]))
    
# credits to https://github.com/litanli/wavenet-time-series-forecasting/blob/master/wavenet_pytorch.py
class DilatedCausalConv1d(nn.Module):
    def __init__(self, hyperparams: dict, dilation_factor: int, in_channels: int):
        super().__init__()

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight.data)
                init.zeros_(m.bias.data)

        self.dilation_factor = dilation_factor
        self.dilated_causal_conv = nn.Conv1d(in_channels=in_channels,
                                             out_channels=hyperparams['nb_filters'],
                                             kernel_size=hyperparams['kernel_size'],
                                             dilation=dilation_factor)
        self.dilated_causal_conv.apply(weights_init)

        self.skip_connection = nn.Conv1d(in_channels=in_channels,
                                         out_channels=hyperparams['nb_filters'],
                                         kernel_size=1)
        self.skip_connection.apply(weights_init)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.layer_norm = nn.LayerNorm(hyperparams['nb_filters'])

    def forward(self, x):
        x1 = self.leaky_relu(self.dilated_causal_conv(x))
        x2 = x[:, :, self.dilation_factor:]
        x2 = self.skip_connection(x2)
        return self.layer_norm((x1 + x2).transpose(1, 2)).transpose(1, 2)

class WaveNet(nn.Module):
    def __init__(self,
                 hidden_size,
                 n_layers=4,
                 dropout_prob=0.0,
                 pad_token=None,
                 sos_token=None,
                 eos_token=None,
                 mask_token=None,
                 vocab_size=None,
                 attributes_meta=None,
                 time_attribute_concatenated=False,
                 nb_special_tokens=None,
                 architecture=None):
        super().__init__()

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight.data)
                init.zeros_(m.bias.data)

        in_channels = hidden_size
        hyperparams = {'nb_layers': n_layers,
                       'nb_filters': hidden_size,
                       'kernel_size': 2}

        if architecture is not None:
            self.architecture = architecture

        self.dropout_prob = dropout_prob
        self.hidden_size = hidden_size
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.mask_token = mask_token
        self.attributes_meta = attributes_meta
        self.time_attribute_concatenated = time_attribute_concatenated
        self.nb_special_tokens = nb_special_tokens
        self.vocab_size = vocab_size + self.nb_special_tokens

        self.dilation_factors = [2 ** i for i in range(0, hyperparams['nb_layers'])]
        self.in_channels = [in_channels] + [hyperparams['nb_filters'] for _ in range(hyperparams['nb_layers'])]
        self.dilated_causal_convs = nn.Sequential(
            *[DilatedCausalConv1d(hyperparams, self.dilation_factors[i], self.in_channels[i]) for i in
              range(hyperparams['nb_layers'])])
        for dilated_causal_conv in self.dilated_causal_convs:
            dilated_causal_conv.apply(weights_init)

        self.output_layer = nn.Conv1d(in_channels=self.in_channels[-1],
                                      out_channels=self.hidden_size,
                                      kernel_size=1)
        self.output_layer.apply(weights_init)
        self.leaky_relu = nn.LeakyReLU(0.1)

        self.value_embedding = Embedding(d_model=self.hidden_size,
                                         vocab_size=self.vocab_size,
                                         dropout_prob=self.dropout_prob,
                                         attributes_meta=self.attributes_meta,
                                         time_attribute_concatenated=self.time_attribute_concatenated,
                                         pad_token=self.pad_token)
        self.readout = Readout(d_model=self.hidden_size,
                               vocab_size=self.vocab_size,
                               dropout_prob=self.dropout_prob,
                               attributes_meta=self.attributes_meta,
                               time_attribute_concatenated=self.time_attribute_concatenated)

        receptive_field = 2 ** (hyperparams['nb_layers'] - 1) * hyperparams['kernel_size']
        print('receptive_field: ' + str(receptive_field))
        self.left_padding = receptive_field - 1

    def forward(self, x, left_padding=None):
        x = self.value_embedding(x)
        x = x.transpose(1, 2)

        if left_padding is None:
            x = nn.functional.pad(x, (self.left_padding, 0), mode='constant', value=0)
        else:
            if left_padding > 0:
                x = nn.functional.pad(x, (left_padding, 0), mode='constant', value=0)

        x = self.dilated_causal_convs(x)
        x = self.leaky_relu(self.output_layer(x))

        x = x.transpose(1, 2)
        x = self.readout(x)
        return x