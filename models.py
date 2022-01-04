import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import util2
class Embedding_init(nn.Module):
    @staticmethod
    def init_emb(row, col):
        w = torch.empty(row, col)
        torch.nn.init.normal_(w)
        w = torch.nn.functional.normalize(w)
        entities_emb = nn.Parameter(w)
        return entities_emb

class OverAll(nn.Module):
    def __init__(self, node_size, node_hidden,
                 rel_size, rel_hidden,
                 triple_size,
                 rel_matrix,
                 ent_matrix,
                 dropout_rate=0., depth=2,
                 ):
        super(OverAll, self).__init__()
        self.dropout_rate = dropout_rate

        self.e_encoder = GraphAttention(node_size, rel_size, triple_size, depth=depth)
        self.r_encoder = GraphAttention(node_size, rel_size, triple_size, depth=depth)
        self.ent_adj = self.get_spares_matrix_by_index(ent_matrix, (node_size, node_size))
        self.rel_adj = self.get_spares_matrix_by_index(rel_matrix, (node_size, rel_size))

        self.ent_emb = self.init_emb(node_size, node_hidden)
        self.rel_emb = self.init_emb(rel_size, node_hidden)

    # get prepared
    @staticmethod
    def get_spares_matrix_by_index(index, size):
        index = torch.LongTensor(index)
        adj = torch.sparse.FloatTensor(torch.transpose(index, 0, 1),
                                       torch.ones_like(index[:, 0], dtype=torch.float), size)
        # dim ??
        return torch.sparse.softmax(adj, dim=1)

    @staticmethod
    def init_emb(row, col):
        w = torch.empty(row, col)
        torch.nn.init.normal_(w)
        w = torch.nn.functional.normalize(w)
        entities_emb = nn.Parameter(w)
        return entities_emb


    def forward(self, inputs):
        # inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix, train_pairs]
        ent_feature = torch.matmul(self.ent_adj, self.ent_emb)
        rel_feature = torch.matmul(self.rel_adj, self.rel_emb)

        adj_input = inputs[0]
        index_input = inputs[1]
        val_input = inputs[2]

        opt = [self.rel_emb, adj_input, index_input, val_input]
        out_feature_ent = self.e_encoder([ent_feature] + opt)
        out_feature_rel = self.r_encoder([rel_feature] + opt)
        out_feature = torch.cat((out_feature_ent, out_feature_rel), dim=-1)
        out_feature = F.dropout(out_feature, p=self.dropout_rate)
        return out_feature


class GraphAttention(nn.Module):
    def __init__(self, node_size,  rel_size, triple_size,
                 activation=torch.tanh, use_bias=True,
                 attn_heads=1,
                 depth=1):
        super(GraphAttention, self).__init__()
        self.node_size = node_size
        self.activation = activation
        self.rel_size = rel_size
        self.triple_size = triple_size
        self.use_bias = use_bias
        self.attn_heads = attn_heads
        self.attn_heads_reduction = 'concat'
        self.depth = depth
        self.attn_kernels = []

        # may have problems, suppose to be dim
        node_F = 128
        rel_F = 128
        self.ent_F = node_F
        ent_F = self.ent_F
        # fuck

        self.gate_kernel = OverAll.init_emb(ent_F*(self.depth+1), ent_F*(self.depth+1))
        self.proxy = OverAll.init_emb(64, node_F * (self.depth + 1))
        if self.use_bias:
            self.bias = OverAll.init_emb(1, ent_F * (self.depth + 1))
        attn_kernels= {}
        for d in range(self.depth):
            # self.attn_kernels.append([])
            for h in range(self.attn_heads):
                attn_kernels["{}{}".format(d,h)]=OverAll.init_emb(node_F, 1)

        self.attn_kernels = nn.ParameterDict(attn_kernels)


    def forward(self, inputs):
        outputs = []
        features = inputs[0]
        rel_emb = inputs[1]
        index = inputs[2]   # adj
        index = torch.tensor(index, dtype=torch.int64)
        # adj = torch.sparse.FloatTensor(torch.LongTensor(index),
        #                                torch.FloatTensor(torch.ones_like(index[:,0])),
        #                                (self.node_size, self.node_size))
        sparse_indices = inputs[3]
        sparse_val = inputs[4]

        features = self.activation(features)
        outputs.append(features)

        for l in range(self.depth):
            features_list = []
            for head in range(self.attn_heads):
                attention_kernel = self.attn_kernels["{}{}".format(l,head)]
                ####
                rels_sum = torch.sparse.FloatTensor(
                    torch.transpose(torch.LongTensor(sparse_indices), 0, 1),
                    torch.FloatTensor(sparse_val),
                    (self.triple_size, self.rel_size)
                )
                rels_sum = torch.matmul(rels_sum, rel_emb)
                neighs = features[index[:, 1]]
                selfs = features[index[:, 0]]
                rels_sum = F.normalize(rels_sum, p=2, dim=1)
                neighs = neighs - 2 * torch.sum(neighs * rels_sum, 1, keepdim=True) * rels_sum

                att1 = torch.squeeze(torch.matmul(rels_sum, attention_kernel), dim=-1)
                att = torch.sparse.FloatTensor(torch.transpose(index, 0, 1), att1, (self.node_size, self.node_size))
                # ??? dim ??
                att = torch.sparse.softmax(att, dim=1)
                # ?
                # print(att1)
                # print(att.data)
                new_features = torch_scatter.scatter_add(torch.transpose(neighs * torch.unsqueeze(att.coalesce().values(), dim=-1),0, 1),
                                                         index[:,0])
                new_features = torch.transpose(new_features, 0,1)
                features_list.append(new_features)

            if self.attn_heads_reduction == 'concat':
                features = torch.cat(features_list)

            features = self.activation(features)
            outputs.append(features)

        outputs = torch.cat(outputs, dim=1)
        proxy_att = torch.matmul(F.normalize(outputs, dim=-1),
                                 torch.transpose(F.normalize(self.proxy, dim=-1), 0, 1))
        proxy_att = F.softmax(proxy_att, dim=-1)  # eq.3
        proxy_feature = outputs - torch.matmul(proxy_att, self.proxy)

        if self.use_bias:
            gate_rate = F.sigmoid(torch.matmul(proxy_feature, self.gate_kernel) + self.bias)
        else:
            gate_rate = F.sigmoid(torch.matmul(proxy_feature, self.gate_kernel))
        outputs = gate_rate * outputs + (1-gate_rate) * proxy_feature
        return outputs




