# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from modify_gnn import *
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import os, itertools, random, argparse, time, datetime
import numpy as np
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score,explained_variance_score
from math import sqrt
import scipy.sparse as sp
from scipy.stats.stats import pearsonr
import csv


import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import shutil
import logging
import glob
import time
from tensorboardX import SummaryWriter
from modify_data import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp

# Training settings
ap = argparse.ArgumentParser()
ap.add_argument('--dataset', type=str, default='update_transposed_final_frm', help="Dataset string")
ap.add_argument('--dataset2', type=str, default='transposed_final_comment-2', help="Dataset string")
ap.add_argument('--sim_mat', type=str, default='final_follow', help="adjacency matrix filename (*-adj.txt)")
ap.add_argument('--n_layer', type=int, default=1, help="number of layers (default 1)")
ap.add_argument('--n_hidden', type=int, default=40, help="rnn hidden states (could be set as any value)") 
ap.add_argument('--seed', type=int, default=32, help='random seed')
ap.add_argument('--epochs', type=int, default=1500, help='number of epochs to train')
ap.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
ap.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
ap.add_argument('--dropout', type=float, default=0.2, help='dropout rate usually 0.2-0.5.')
ap.add_argument('--batch', type=int, default=8, help="batch size")
ap.add_argument('--check_point', type=int, default=1, help="check point")
ap.add_argument('--shuffle', action='store_true', default=True, help="not used, default false")
ap.add_argument('--train', type=float, default=.8, help="Training ratio (0, 1)")
ap.add_argument('--val', type=float, default=.1, help="Validation ratio (0, 1)")
ap.add_argument('--test', type=float, default=.1, help="Testing ratio (0, 1)")
ap.add_argument('--model', default='LSTNet', choices=['cola_gnn','CNNRNN_Res','RNN','AR','ARMA','VAR','GAR','SelfAttnRNN','lstnet','stgcn','dcrnn'], help='')
ap.add_argument('--rnn_model', default='GRU', choices=['LSTM','RNN','GRU'], help='')
ap.add_argument('--rnn_model_1', default='LSTM', choices=['LSTM','RNN','GRU'], help='')
ap.add_argument('--mylog', action='store_false', default=True,  help='save tensorboad log')
ap.add_argument('--cuda', action='store_true', default=False,  help='')

ap.add_argument('--window', type=int, default=3, help='')

ap.add_argument('--horizon', type=int, default=1, help='leadtime default 1')
ap.add_argument('--save_dir', type=str,  default='save',help='dir path to save the final model')
ap.add_argument('--gpu', type=int, default=3,  help='choose gpu 0-10')
ap.add_argument('--lamda', type=float, default=0.01,  help='regularize params similarities of states')
ap.add_argument('--bi', action='store_true', default=False,  help='bidirectional default false')
ap.add_argument('--patience', type=int, default=100, help='patience default 100')
ap.add_argument('--k', type=int, default=10,  help='kernels')
ap.add_argument('--hidsp', type=int, default=15,  help='spatial dim')
ap.add_argument('--log_dir', type=str, default='/Users/ching-haofan/Downloads/TLSS-main', help='Path to log directory')


args = ap.parse_args() 
print('--------Parameters--------')
print(args)
print('--------------------------')

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


args.cuda = args.cuda and torch.cuda.is_available() 
logger.info('cuda %s', args.cuda)

time_token = str(time.time()).split('.')[0] # tensorboard model
log_token = '%s.%s.w-%s.h-%s.%s' % (args.model, args.dataset, args.window, args.horizon, args.rnn_model)

if args.mylog:
    print("save the result")
    tensorboard_log_dir = 'tensorboard/%s' % (log_token)
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    writer = SummaryWriter(tensorboard_log_dir)
    shutil.rmtree(tensorboard_log_dir)
    logger.info('tensorboard logging to %s', tensorboard_log_dir)


MAX_VOCAB_SIZE = 10000 
UNK, PAD = '<unk>', '<pad>'  
MAX_VOCAB_SIZE = 10000  

def process_text(text):
    pattern = re.compile(r'<a href=[^>]+>@[^<]+</a>')
    text = pattern.sub('', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.encode('ascii', 'ignore').decode('ascii')   
    text = text.translate(str.maketrans('', '', string.punctuation))   
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def build_vocab(file_path,min_freq=0, max_size=MAX_VOCAB_SIZE):
    df = pd.read_csv(file_path).fillna("<PAD>")
    tokenizer = lambda x: x.split(' ')
    vocab_dic = {}
    lines = df["content"].to_list()

    for line in lines:
        if not line:
            continue
        content = process_text(line.lower())
        for word in tokenizer(content):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def convert_test(file_path, source_file_path):
    df = pd.read_csv(file_path).fillna("<PAD>")
    vocab_path = "vocab.pkl"
    if os.path.exists(vocab_path):
        vocab = pkl.load(open(vocab_path, 'rb'))
    else:
        vocab = build_vocab(source_file_path, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(vocab_path, 'wb'))

    def words_to_integers(text):
        return [vocab[word] for word in text.split(" ")]

    columns_to_convert = df.columns[1:] 
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    df[columns_to_convert] = df[columns_to_convert].applymap(words_to_integers)


    def pad_or_truncate(lst, max_length=64):
        if len(lst) > max_length:
            return lst[:max_length]  
        return lst + [vocab['<pad>']] * (max_length - len(lst))  
    df[columns_to_convert] = df[columns_to_convert].applymap(pad_or_truncate)

    df = df[columns_to_convert]
    rows, cols = df.shape

    max_length = max(df.applymap(len).max())
    matrix_3d = np.zeros((rows, cols, max_length), dtype=int)

    for i in range(rows):
        for j in range(cols):
            matrix_3d[i, j, :len(df.iloc[i, j])] = df.iloc[i, j]
    
    reshaped_matrix = matrix_3d.transpose(0, 2, 1)
    reshaped_matrix = reshaped_matrix.transpose(2, 0, 1)
    print(reshaped_matrix.shape)
    # [54, 1644, 64]
    return reshaped_matrix

file = "/Users/ching-haofan/Downloads/TLSS-main/data/new_3.csv"
source_file_path = "/Users/ching-haofan/Downloads/TLSS-main/data/filtered_combined_posts_comments_final.csv"
reshaped_matrix=convert_test(file, source_file_path)
args.reshaped_matrix = reshaped_matrix


############Model Start##########
TLdata_loader = TLData_Loader(args,args.reshaped_matrix)

if args.model == 'cola_gnn':
    TLmodel = TLDynamic_SS(args, TLdata_loader)
elif args.model == 'ARMA':
    TLmodel = ARMA(args, TLdata_loader)
elif args.model == 'RNN':
    TLmodel = RNN(args, TLdata_loader)
elif args.model == 'CNNRNN_Res':
    TLmodel = CNNRNN_Res(args, TLdata_loader)
elif args.model == 'LSTNet':
    TLmodel = LSTNet(args, TLdata_loader)
#TLmodel = TLDynamic_SS(args, TLdata_loader)

logger.info('model %s', TLmodel)
if args.cuda:
    TLmodel.cuda()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, TLmodel.parameters()), lr=args.lr, weight_decay=args.weight_decay)
pytorch_total_params = sum(p.numel() for p in TLmodel.parameters() if p.requires_grad)
print('#params:',pytorch_total_params)

def evaluate(TLdata_loader, data, tag='val'):
    TLmodel.eval()
    total = 0.
    n_samples = 0.
    total_loss = 0.
    y_true, y_pred = [], []
    batch_size = args.batch
    y_pred_mx = []
    y_true_mx = []
    for inputs in TLdata_loader.get_batches(data, batch_size, False):
        X,Adj, Y,month,NLP = inputs[0],inputs[1], inputs[2],inputs[3],inputs[4]
        output, _ = TLmodel(X,NLP,Adj,month=month)

        #loss_train = F.l1_loss(output, Y) # mse_loss
        loss_train = F.mse_loss(output, Y)
        total_loss += loss_train.item()
        n_samples += (output.size(0) * TLdata_loader.m)

        y_true_mx.append(Y.data.cpu())
        y_pred_mx.append(output.data.cpu())


    y_pred_mx = torch.cat(y_pred_mx)
    y_true_mx = torch.cat(y_true_mx)

    y_true_states = y_true_mx.numpy() * (TLdata_loader.max - TLdata_loader.min ) * 1.0 + TLdata_loader.min  
    y_pred_states = y_pred_mx.numpy() * (TLdata_loader.max - TLdata_loader.min ) * 1.0 + TLdata_loader.min 


    y_true = np.reshape(y_true_states,(-1))
    y_pred = np.reshape(y_pred_states,(-1))

    v_rmse = sqrt(mean_squared_error(y_true, y_pred))
    v_mae = mean_absolute_error(y_true, y_pred)
    v_r2 = r2_score(y_true, y_pred)
    v_evs = explained_variance_score(y_true, y_pred)
    r2_states = np.mean(r2_score(y_true_states, y_pred_states, multioutput='raw_values'))


    y_true_1 = y_true_mx.numpy().flatten()  
    y_pred_1 = y_pred_mx.numpy().flatten()  
    
    

    df = pd.DataFrame({'True Values': y_true_1, 'Predicted Values': y_pred_1})
    df.to_csv(f'{tag}_predictions.csv', index=False)
    np.savetxt('/Users/ching-haofan/Downloads/TLSS-main/ypred_ar_hosp_1_15.txt', y_pred, delimiter=',')
    np.savetxt('/Users/ching-haofan/Downloads/TLSS-main/ytrue_ar_hosp_1_15.txt', y_true, delimiter=',')

    return total_loss / n_samples, v_rmse, v_mae, v_r2, v_evs,r2_states

def train(TLdata_loader, data):
    TLmodel.train()
    total_loss = 0.
    n_samples = 0.
    batch_size = args.batch

    for inputs in TLdata_loader.get_batches(data, batch_size, True):
        X,Adj,Y,month,NLP = inputs[0], inputs[1],inputs[2],inputs[3],inputs[4]
        optimizer.zero_grad()
        output, _ = TLmodel(X,Adj) 


        if Y.size(0) == 1:
            Y = Y.view(-1)
        loss_train = F.mse_loss(output, Y)
        total_loss += loss_train.item()
        loss_train.backward()
        optimizer.step()
        n_samples += (output.size(0) * TLdata_loader.m)
    return float(total_loss / n_samples)

 
bad_counter = 0
best_epoch = 0
best_val = 1e+20
try:
    print('begin training')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_loss = train(TLdata_loader, TLdata_loader.train)
        val_loss, mae, rmse, r2, evs,r2_states = evaluate(TLdata_loader, TLdata_loader.val)
        print('Epoch {:3d}|time:{:5.2f}s|train_loss {:5.8f}|val_loss {:5.8f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_loss))

        if args.mylog:
            writer.add_scalars('data/loss', {'train': train_loss}, epoch )
            writer.add_scalars('data/loss', {'val': val_loss}, epoch)
            writer.add_scalars('data/mae', {'val': mae}, epoch)
            writer.add_scalars('data/rmse', {'val': rmse}, epoch)
            writer.add_scalars('data/R2', {'val': r2}, epoch)
            writer.add_scalars('data/EVS', {'val': evs}, epoch)
            writer.add_scalars('data/R2_states', {'val': r2_states}, epoch)

       
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            bad_counter = 0
            model_path = '%s/%s.pt' % (args.save_dir, log_token)
            with open(model_path, 'wb') as f:
                torch.save({'model_state_dict': TLmodel.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, f)
            print('Best validation epoch:', epoch, time.ctime())
            test_loss, test_mae, test_rmse, test_r2, test_evs,r2_states = evaluate(TLdata_loader, TLdata_loader.test, tag='test')
            print('TEST MAE {:5.4f} RMSE {:5.4f} R2 {:5.4f} EVS {:5.4f} R2s {:5.4f} '.format(test_mae, test_rmse, test_r2, test_evs,r2_states))
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early, epoch', epoch)

model_path = '%s/%s.pt' % (args.save_dir, log_token)
test_loss, test_mae, test_rmse, test_r2, test_evs ,r2_states= evaluate(TLdata_loader, TLdata_loader.test, tag='test')
print('Final evaluation')
print('TEST MAE {:5.4f} RMSE {:5.4f} R2 {:5.4f} EVS {:5.4f} R2s {:5.4f}'.format(test_mae, test_rmse, test_r2, test_evs,r2_states))

