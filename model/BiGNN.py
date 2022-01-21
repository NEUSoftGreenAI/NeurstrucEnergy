import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    
    def __init__(self, in_features, out_features, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, h, adj):
        batch_size = h.shape[0]
        
        Wh = torch.matmul(h, self.W) # h.shape: (Batch, N, in_features), Wh.shape: (Batch, N, out_features)
        # score(x,y) = Wx * a1 + Wy * a2
        a1 = self.a[0:self.out_features].repeat(batch_size,1,1) # shape [Batch, out_features, 1]
        a2 = self.a[self.out_features:2*self.out_features].repeat(batch_size,1,1) # shape [Batch, out_features, 1]
        Wh_a1 = torch.matmul(Wh, a1) # shape [Batch,N,1]
        Wh_a2 = torch.matmul(Wh, a2) # shape [Batch,N,1]
        e = self.leakyrelu(torch.matmul(Wh_a1, Wh_a2.transpose(1,2))) # shape [Batch,N,N]
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        zeros = torch.zeros_like(e)
        attention = torch.where(adj > 0, attention, zeros)
        h_prime = torch.matmul(attention, Wh)

        if self.concat: 
            return F.elu(h_prime)
        else:       
            return h_prime      

class Attention(nn.Module):
    def __init__(self,in_features, out_features, nheads, concat=True):
        self.in_features = in_features
        self.out_features = out_features
        super(Attention, self).__init__()
        self.attentions = [GraphAttentionLayer(in_features, out_features, concat=True).cuda() for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.reshape_fc = nn.Linear(nheads * out_features, out_features).cuda()

    def forward(self, x, adj):
        '''
        enc_inputs: [batch_size, max_layer_length, max_vec_length]
        不同的层，经过不同的全连接层进行embedding
        '''
        res = x
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        # print(x.shape,nheads,in)
        x = self.reshape_fc(x)
        if x.shape[2] == res.shape[2]:
          x = nn.LayerNorm(self.out_features).cuda()(x + res)
        else:
          x = nn.LayerNorm(self.out_features).cuda()(x)
        return x

class FEL(nn.Module):
    def __init__(self,in_features,out_features):
        super(FEL, self).__init__()
        self.out_features = out_features
        self.embedding_fc1 = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(),
            nn.Linear(out_features, out_features),
            nn.LeakyReLU()
        )
        self.embedding_fc2 = nn.Sequential(
            nn.Linear(out_features, out_features*2),
            nn.LeakyReLU(),
            nn.Linear(out_features*2, out_features),
            nn.LeakyReLU()
        )
        self.embedding_fc3 = nn.Sequential(
            nn.Linear(out_features, out_features*2),
            nn.LeakyReLU(),
            nn.Linear(out_features*2, out_features)
        )
    def forward(self, x):
        # print('FEL')
        '''
        enc_inputs: [batch_size, max_layer_length, max_vec_length]
        不同的层，经过不同的全连接层进行embedding
        '''
        res = self.embedding_fc1(x)
        x = self.embedding_fc2(res)
        x = nn.LayerNorm(self.out_features).cuda()(x + res)

        res = x
        x = self.embedding_fc3(res)
        x = nn.LayerNorm(self.out_features).cuda()(x + res)
        
        return x

class BiGNN(nn.Module):
    def __init__(self, nfeat, nhid, reverse_hidden, nheads):
        
        super(BiGNN, self).__init__()
        self.nhid = nhid
        self.attention = Attention(nhid, nhid, concat=True, nheads=nheads)
        self.attention1 = Attention(nhid, nhid, concat=True, nheads=nheads)
        self.attention2 = Attention(nhid, nhid, concat=True, nheads=nheads)
        self.out_att = GraphAttentionLayer(nhid, nhid, concat=False).cuda()
        trans_hid = reverse_hidden
        self.attention_trans = Attention(nhid, trans_hid, concat=True, nheads=nheads)
        self.attention1_trans = Attention(trans_hid, trans_hid, concat=True, nheads=nheads)
        self.attention2_trans = Attention(trans_hid, trans_hid, concat=True, nheads=nheads)
        self.out_att_trans = GraphAttentionLayer(trans_hid, trans_hid, concat=False).cuda()

        self.projection = nn.Linear(nhid + trans_hid, 1).cuda()
        self.branch_param = nn.Parameter(torch.Tensor([0.01]))

        self.embedding = FEL(nfeat,nhid).cuda()
        self.leakyrelu = nn.LeakyReLU()


    def forward(self, x, adj, layer_id):
        x = self.embedding(x)
        adj_trans = adj.transpose(1,2)
        
        x = self.attention(x, adj)
        x = self.attention1(x, adj)
        x = self.attention2(x, adj)
        x = self.leakyrelu(self.out_att(x, adj))

        '''
        You can obtain the GAT utilized in the structure features extraction experiment by moving the reverse aggregators.
        '''
        
        x_trans = self.attention_trans(x, adj_trans)
        x_trans = self.attention1_trans(x_trans, adj_trans)
        x_trans = self.attention2_trans(x_trans, adj_trans)
        x_trans = self.leakyrelu(self.out_att_trans(x_trans, adj_trans))
        
        x = torch.cat([x,x_trans],dim=2)
        batch_size, length = x.shape[0],x.shape[1]
        x = torch.sum(x,dim=1) / length
        x = self.projection(x)
        
        return torch.squeeze(x, dim=1)
