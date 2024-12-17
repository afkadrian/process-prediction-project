import torch.nn as nn
import torch
import math

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
