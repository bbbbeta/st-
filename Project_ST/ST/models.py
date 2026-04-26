import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class SelfAttention(nn.Module):
    def __init__(self, dropout):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        queries = q
        keys = k
        values = v
        n, d = queries.shape
        scores = torch.mm(queries, keys.t()) / math.sqrt(d)
        att_weights = F.softmax(scores, dim=1)
        att_emb = torch.mm(self.dropout(att_weights), values)
        return att_weights, att_emb

# class SelfAttention(nn.Module):
#     def __init__(self, dropout):
#         super(SelfAttention, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, q, k, v):
#         chunk_size = 2048
#         n_points = q.shape[0]
#         outputs = []
#         for i in range(0, n_points, chunk_size):
#             end = min(i + chunk_size, n_points)
#             q_chunk = q[i:end]
#             scores = torch.mm(q_chunk, k.t()) / math.sqrt(q.shape[1])
#             att_weights_chunk = F.softmax(scores, dim=1)
#             att_weights_chunk = self.dropout(att_weights_chunk)
#             out_chunk = torch.mm(att_weights_chunk, v)
#             outputs.append(out_chunk)
#             del scores, att_weights_chunk, out_chunk
#         att_emb = torch.cat(outputs, dim=0)
#         return None, att_emb

class MLP(nn.Module):
    def __init__(self, z_emb, dropout_rate, use_llm_gene=False):
        super(MLP, self).__init__()
        self.use_llm_gene = use_llm_gene
        self.mlpx = nn.Sequential(
            nn.Linear(z_emb, z_emb),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.mlpi = nn.Sequential(
            nn.Linear(z_emb, z_emb),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        if use_llm_gene:
            self.mlpl = nn.Sequential(
                nn.Linear(z_emb, z_emb),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
            )

    def forward(self, z_x, z_y, z_l=None):
        q_x = self.mlpx(z_x)
        q_i = self.mlpi(z_y)
        if self.use_llm_gene and z_l is not None:
            q_l = self.mlpl(z_l)
            return q_x, q_i, q_l
        return q_x, q_i, None

class MultiGCN(nn.Module):
    def __init__(self, nfeatX, nfeatI, hidden_dims, nfeatL=None, use_llm_gene=False):
        super(MultiGCN, self).__init__()
        self.use_llm_gene = use_llm_gene
        self.GCNA1_1 = GraphConvolution(nfeatX, hidden_dims[0])
        self.GCNA1_3 = GraphConvolution(hidden_dims[0], hidden_dims[1])
        self.GCNA2_1 = GraphConvolution(nfeatI, hidden_dims[0])
        self.GCNA2_3 = GraphConvolution(hidden_dims[0], hidden_dims[1])
        if use_llm_gene and nfeatL is not None and nfeatL > 0:
            self.GCNA3_1 = GraphConvolution(nfeatL, hidden_dims[0])
            self.GCNA3_3 = GraphConvolution(hidden_dims[0], hidden_dims[1])

    def forward(self, x, i, a, l=None):
        emb1 = self.GCNA1_1(x, a)
        emb1 = self.GCNA1_3(emb1, a)
        emb2 = self.GCNA2_1(i, a)
        emb2 = self.GCNA2_3(emb2, a)
        if self.use_llm_gene and l is not None and hasattr(self, 'GCNA3_1'):
            emb3 = self.GCNA3_1(l, a)
            emb3 = self.GCNA3_3(emb3, a)
            return emb1, emb2, emb3
        return emb1, emb2, None

class ZINBdecoder(torch.nn.Module):
    def __init__(self, nhid1, nfeat):
        super(ZINBdecoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid1, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1,  nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)


    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]

class ST(nn.Module):
    def __init__(self, nfeatX, nfeatI, hidden_dims, nfeatL=None, use_llm_gene=False,
                 weight_gene=20.0, weight_image=1.0, weight_llm=15.0, weight_fusion=10.0):
        super(ST, self).__init__()
        self.use_llm_gene = use_llm_gene
        self.weight_gene = weight_gene
        self.weight_image = weight_image
        self.weight_llm = weight_llm
        self.weight_fusion = weight_fusion

        self.mgcn = MultiGCN(nfeatX, nfeatI, hidden_dims, nfeatL, use_llm_gene)
        self.attlayer1 = SelfAttention(dropout=0.1)
        self.attlayer2 = SelfAttention(dropout=0.1)
        if use_llm_gene:
            self.attlayer3 = SelfAttention(dropout=0.1)

        if use_llm_gene:
            self.fc = nn.Linear(hidden_dims[1] * 3, hidden_dims[1])
        else:
            self.fc = nn.Linear(hidden_dims[1] * 2, hidden_dims[1])

        self.mlp = MLP(hidden_dims[1], dropout_rate=0.1, use_llm_gene=use_llm_gene)
        self.ZINB = ZINBdecoder(hidden_dims[1], nfeatX)

    def forward(self, x, i, a, l=None):
        emb_x, emb_i, emb_l = self.mgcn(x, i, a, l)
        att_weights_x, att_emb_x = self.attlayer1(emb_x, emb_x, emb_x)
        att_weights_i, att_emb_i = self.attlayer2(emb_i, emb_i, emb_i)

        if self.use_llm_gene and emb_l is not None:
            att_weights_l, att_emb_l = self.attlayer3(emb_l, emb_l, emb_l)
            q_x, q_i, q_l = self.mlp(emb_x, emb_i, emb_l)
            emb_con = torch.cat([q_x, q_i, q_l], dim=1)
            z_xi = self.fc(emb_con)
            z_I = (self.weight_gene * att_emb_x + self.weight_image * att_emb_i +
                   self.weight_llm * att_emb_l + self.weight_fusion * z_xi)
        else:
            q_x, q_i, _ = self.mlp(emb_x, emb_i, None)
            emb_con = torch.cat([q_x, q_i], dim=1)
            z_xi = self.fc(emb_con)
            z_I = 20 * att_emb_x + 1 * att_emb_i + 10 * z_xi

        [pi, disp, mean] = self.ZINB(z_I)
        if self.use_llm_gene and emb_l is not None:
            return z_I, q_x, q_i, q_l, emb_x, emb_i, pi, disp, mean
        return z_I, q_x, q_i, None, emb_x, emb_i, pi, disp, mean
