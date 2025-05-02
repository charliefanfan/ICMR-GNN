from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from layers import *
from utils import *
from torch.autograd import Variable
import sys
import math

class TLDynamic_SS(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.x_h = 1
        self.f_h = data.m
        self.m = data.m
        self.d = data.d
        self.w = args.window
        self.h = args.horizon
        self.adjs = data.adjs
        self.cuda = args.cuda
        self.dropout = args.dropout
        self.n_hidden = args.n_hidden
        self.rawdat=data.rawdat
        self.months = data.months
        self.embed=300
        self.vocab_path = ''
        if os.path.exists(self.vocab_path):
            self.vocab = pkl.load(open(self.vocab_path, 'rb'))                                           

        self.adj_log_file = open(os.path.join(args.log_dir, 'adjacency_indices1.txt'), 'a')


        self.Wb = Parameter(torch.Tensor(self.m, self.m))
        self.wb = Parameter(torch.Tensor(1))
        self.fc = nn.Linear(self.m, self.n_hidden)
        self.k = args.k
        self.conv = nn.Conv1d(1, self.k, self.w)
        long_kernel = self.w // 2

        self.n_spatial = 10        
        self.conv1 = GraphConvLayer(self.n_hidden, self.n_hidden)
        self.conv2 = GraphConvLayer(self.n_hidden, self.n_hidden)

        if args.rnn_model == 'LSTM':
            self.lstm = nn.LSTM(input_size=self.x_h, hidden_size=self.n_hidden, num_layers=args.n_layer, dropout=args.dropout, batch_first=True, bidirectional=args.bi)
        elif args.rnn_model == 'GRU':
            print("test for GRu ")
            self.rnn = nn.GRU(input_size=self.x_h, hidden_size=self.n_hidden, num_layers=args.n_layer, dropout=args.dropout, batch_first=True, bidirectional=args.bi)
        elif args.rnn_model == 'RNN':
            self.rnn = nn.RNN(input_size=self.x_h, hidden_size=self.n_hidden, num_layers=args.n_layer, dropout=args.dropout, batch_first=True, bidirectional=args.bi)
        else:
            raise LookupError('only support LSTM, GRU and RNN')
        
        if args.rnn_model_1 == 'LSTM':
            self.lstm = nn.LSTM(input_size=self.embed, hidden_size=self.n_hidden, num_layers=args.n_layer, dropout=args.dropout, batch_first=True, bidirectional=args.bi)
        elif args.rnn_model_1 == 'GRU':
            self.rnn = nn.GRU(input_size=self.embed, hidden_size=self.n_hidden, num_layers=args.n_layer, dropout=args.dropout, batch_first=True, bidirectional=args.bi)
        elif args.rnn_model_1 == 'RNN':
            self.rnn = nn.RNN(input_size=self.embed, hidden_size=self.n_hidden, num_layers=args.n_layer, dropout=args.dropout, batch_first=True, bidirectional=args.bi)
        else:
            raise LookupError('only support LSTM, GRU and RNN')

        hidden_size = (int(args.bi) + 1) * self.n_hidden
        self.out = nn.Linear(hidden_size, 1)

        self.residual_window = 0
        self.ratio = 1.0
        if self.residual_window > 0:
            self.residual_window = min(self.residual_window, args.window)
            self.residual = nn.Linear(self.residual_window, 1)
        self.init_weights()
        
        self.n_vocab = len(self.vocab)
        self.embedding = nn.Embedding(self.n_vocab, self.embed, padding_idx=self.n_vocab - 1)



    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, x,x2, feat=None,month=None):

        '''
        Args:  x: (batch, time_step, m)
               x2: (batch, seq_len, m)
               x3: (batch, time_step, m) time_series post  
               x4: (batch, time_step, m) time_series comment 
            feat: [batch, window, dim, m]
        Returns: (batch, m)
        '''
        b, w, m = x.size()
        orig_x = x
        rawdat=self.rawdat

        x3=x2.long()
        x3 = x3.reshape(-1, 64)
        out = self.embedding(x3) 
        out, _ = self.lstm(out)
        last_hid_2 = out[:, -1, :]
        last_hid_2 = last_hid_2.view(-1, self.m, self.n_hidden)
        out_nlp = last_hid_2  
       

       

        min_adj_index = w
        max_adj_index = rawdat.shape[0]-1
        adj_index = min_adj_index

        adjs = []

        for i in range(b):
            month_str = month[i]
            try:
                month_index = self.months.index(month_str)
            except ValueError:
                raise ValueError(f"Month '{month_str}' not found in data.months")
            
            current_adj = self.adjs[month_index]
            if self.cuda:
                current_adj = current_adj.cuda()
            adjs.append(current_adj)
            self.adj_log_file.write(f"Batch {i}: Adjacency matrix index {month_index}\n")

        adjs = torch.stack(adjs, dim=0)

        node_features = out_nlp  # [batch, m, hidden_dim]
        attention_scores = torch.matmul(node_features, node_features.transpose(-1, -2))  # [batch, m, m]
        attention_matrix = F.softmax(attention_scores, dim=-1)


        x = out_nlp
        x = F.relu(self.conv1(x, adjs))
        x = F.dropout(x, self.dropout, training=self.training)
        out = self.out(out)

        out = torch.squeeze(out)

        if self.residual_window > 0:
            z = orig_x[:, -self.residual_window:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.residual_window)
            z = self.residual(z)
            z = z.view(-1, self.m)
            out = out * self.ratio + z
        
        return out, None
    









