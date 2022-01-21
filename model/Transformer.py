import torch
import numpy as np
import pandas as pd
import torch.utils.data as Data
from torch.optim import lr_scheduler
from torch.utils.data import sampler
import os
import torch.nn as nn
import math
import random


def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    # print('get_attn_pad_mask')

    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # 补充长度的层，id编码为-1
    pad_attn_mask = seq_k.data.eq(-1).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        # print('ScaledDotProductAttention')
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
    def forward(self, input_Q, input_K, input_V, attn_mask):
        # print('MultiHeadAttention')
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention(self.d_k)(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).cuda()(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.LeakyReLU(),
            nn.Linear(d_ff, d_model, bias=False),
            nn.LeakyReLU()
        )
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)

        return nn.LayerNorm(self.d_model).cuda()(output + residual) # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        # print('EncoderLayer')
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''

        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_layers, n_heads):
        super(Encoder, self).__init__()
        # self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, d_k, d_v, n_heads) for _ in range(n_layers)])

    def forward(self, enc_inputs, layer_id):
        # print('Encoder')
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = enc_inputs # [batch_size, src_len, d_model]
        # enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(layer_id, layer_id) # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            # print(enc_outputs.shape,11231)
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class FEL(nn.Module):
    def __init__(self, d_model):
        super(FEL, self).__init__()
        self.d_model = d_model
        self.embedding_fc1 = nn.Sequential(
            nn.Linear(110, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU()
        )
        self.embedding_fc2 = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.LeakyReLU(),
            nn.Linear(d_model*2, d_model),
            nn.LeakyReLU()
        )
        self.embedding_fc3 = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.LeakyReLU(),
            nn.Linear(d_model*2, d_model)
        )
    def forward(self, x):
        # print('FEL')
        '''
        enc_inputs: [batch_size, max_layer_length, max_vec_length]
        不同的层，经过不同的全连接层进行embedding
        '''
        res = self.embedding_fc1(x)
        x = self.embedding_fc2(res)
        x = nn.LayerNorm(self.d_model).cuda()(x + res)

        res = x
        x = self.embedding_fc3(res)
        x = nn.LayerNorm(self.d_model).cuda()(x + res)

        return x

class Transformer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_layers, n_heads, max_layer_length):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_k = d_k
        self.d_v = d_v
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_layer_length = max_layer_length
        self.embedding = FEL(d_model).cuda()
        self.encoder = Encoder(d_model, d_ff, d_k, d_v, n_layers, n_heads).cuda()
        self.projection = nn.Linear(d_model, 1).cuda()
        self.energy_fc = nn.Linear(max_layer_length, 1).cuda()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, enc_inputs, layer_id):
        '''
        enc_inputs: [batch_size, max_layer_length, max_vec_length]
        id_inputs : [batch_size, max_layer_length]
        '''
        enc_inputs = self.embedding(enc_inputs) # [batch_size, max_layer_length, d_model]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs, layer_id) # [batch_size, max_layer_length, d_model]
        
        enc_outputs = torch.sum(enc_outputs , dim=1) / self.max_layer_length
        enc_outputs = self.projection(enc_outputs) # [batch_size, max_layer_length, 1]
        
        return torch.squeeze(enc_outputs, dim=1), enc_self_attns
