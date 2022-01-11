from dgl_sample_gcn import *
import dgl
import numpy as np
import torch
import time
from utils import *
from evaluate import evaluate
from tqdm import *
from dgl.dataloading import MultiLayerNeighborSampler


gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

train_pair, dev_pair, adj_matrix, r_index, r_val, adj_features, rel_features = load_data("data/ja_en/", train_ratio=0.30)
adj_matrix = np.stack(adj_matrix.nonzero(), axis=1)
rel_matrix, rel_val = np.stack(rel_features.nonzero(), axis=1), rel_features.data
ent_matrix, ent_val = np.stack(adj_features.nonzero(), axis=1), adj_features.data

node_size = adj_features.shape[0]
rel_size = rel_features.shape[1]
triple_size = len(adj_matrix) # not triple size, but number of diff(h, t)
node_hidden = 128
rel_hidden = 128
batch_size = 1024
dropout_rate = 0.3
lr = 0.005
gamma = 1
depth = 2
device = 'cuda'
print('adj_matrix', adj_matrix)
print('rel_matrix', r_index)

# newly add
fanouts = [10] * depth


def get_embedding(index_a, index_b, vec=None):
    if vec is not None:
        vec = vec.detach().numpy()
    else:
        vec = fuck
    Lvec = np.array([vec[e] for e in index_a])
    Rvec = np.array([vec[e] for e in index_b])
    Lvec = Lvec / (np.linalg.norm(Lvec, axis=-1, keepdims=True) + 1e-5)
    Rvec = Rvec / (np.linalg.norm(Rvec, axis=-1, keepdims=True) + 1e-5)
    return Lvec, Rvec

def align_loss(align_input, embedding):
    def squared_dist(x):
        A, B = x
        row_norms_A = torch.sum(torch.square(A), dim=1)
        row_norms_A = torch.reshape(row_norms_A, [-1, 1])  # Column vector.
        row_norms_B = torch.sum(torch.square(B), dim=1)
        row_norms_B = torch.reshape(row_norms_B, [1, -1])  # Row vector.
        # may not work
        return row_norms_A + row_norms_B - 2 * torch.matmul(A, torch.transpose(B, 0, 1))
    # modified
    left = torch.tensor(align_input[:, 0])
    right = torch.tensor(align_input[:, 1])
    l_emb = embedding[left]
    r_emb = embedding[right]
    pos_dis = torch.sum(torch.square(l_emb - r_emb), dim=-1, keepdim=True)

    l_neg_dis = squared_dist([l_emb, embedding])
    r_neg_dis = squared_dist([r_emb, embedding])

    l_loss = pos_dis - l_neg_dis + gamma
    l_loss = l_loss * (1 - F.one_hot(left, num_classes=node_size) - F.one_hot(right, num_classes=node_size)).to(device)

    r_loss = pos_dis - r_neg_dis + gamma
    r_loss = r_loss * (1 - F.one_hot(left, num_classes=node_size) - F.one_hot(right, num_classes=node_size)).to(device)
    # modified
    with torch.no_grad():
        r_mean = torch.mean(r_loss, dim=-1, keepdim=True)
        r_std= torch.std(r_loss, dim=-1, keepdim=True)
        r_loss.data =(r_loss.data -r_mean)/r_std
        l_mean = torch.mean(l_loss, dim=-1, keepdim=True)
        l_std= torch.std(l_loss, dim=-1, keepdim=True)
        l_loss.data =(l_loss.data -l_mean)/l_std
        # l_loss = (l_loss - torch.mean(l_loss, dim=-1, keepdim=True).detach()) / torch.std(
        #     l_loss, dim=-1, keepdim=True).detach()

    lamb, tau = 30, 10
    # l_loss = K.log(K.sum(K.exp(lamb * l_loss + tau), axis=-1))
    # r_loss = K.log(K.sum(K.exp(lamb * r_loss + tau), axis=-1))
    # l_loss = K.logsumexp(lamb * l_loss + tau, axis=-1)
    # r_loss = K.logsumexp(lamb * r_loss + tau, axis=-1)
    l_loss = torch.logsumexp(lamb * l_loss + tau, dim=-1)
    r_loss = torch.logsumexp(lamb * r_loss + tau, dim=-1)
    return torch.mean(l_loss + r_loss)

def constructRelGraph(r_index, rel_size):
    src, trg = [], []
    for index in r_index:
        src.append(index[1])
        trg.append(index[0])
    return dgl.heterograph({
        ('relation', 'in', 'index'): (torch.tensor(src), torch.tensor(trg)), })


def constructGraph(adj_matrix):
    src, trg = adj_matrix[:, 0], adj_matrix[:, 1]
    print(len(src))
    print(len(r_index))
    # todo src trg
    g = dgl.graph((src, trg))
    # g = dgl.graph(( trg,src))
    return g

g = constructGraph(adj_matrix)
g_r = constructRelGraph(r_index, rel_size)


print('begin')
#inputs = [adj_input, index_input, val_input, rel_adj, ent_adj]

model = overAll(node_size=node_size, node_hidden=node_hidden,
                 rel_size=rel_size, rel_matrix=rel_matrix,
                ent_matrix=ent_matrix, dropout_rate=dropout_rate,
                depth=depth, device=device)
model = model.to(device)
# opt = torch.optim.RMSprop(model.parameters(), lr=lr)
opt = torch.optim.Adam(model.parameters(), lr=lr)
print('model constructed')

evaluater = evaluate(dev_pair)
rest_set_1 = [e1 for e1, e2 in dev_pair]
rest_set_2 = [e2 for e1, e2 in dev_pair]
np.random.shuffle(rest_set_1)
np.random.shuffle(rest_set_2)

# this model merge two KGs into one
# need a different sampler
def dataloader(train_pair, batch_size, node_size):
    pos_node = set(train_pair[:, 0] + train_pair[:, 1])
    neg_node = list(filter(lambda x: x not in pos_node, range(node_size)))
    n_bs = int(batch_size*len(neg_node)/len(train_pair))
    np.random.shuffle(neg_node)
    for i in range(len(train_pair)//batch_size+1):
        l = i * batch_size
        r = l +batch_size if l + batch_size < len(train_pair) else len(train_pair)
        pairs = train_pair[l: r]
        # node ID of two KGs is not overlapping
        l_n = i * n_bs
        r_n = l_n + n_bs if l_n + n_bs < len(neg_node) else len(neg_node)
        neg_sample = neg_node[l_n: r_n]
        yield pairs, neg_sample

def Sampler(g, nodes, fanouts):
    blocks = MultiLayerNeighborSampler(fanouts).sample_blocks(g, nodes)
    return blocks



# here is dual-m
epoch = 20
for turn in range(5):
    for i in trange(epoch):
        np.random.shuffle(train_pair)
        for pairs, neg_node in dataloader(train_pair, batch_size, node_size):
            pos_node = pairs[:, 0]+ pairs[:, 1]
            print(pos_node)
            print(neg_node)
            blocks = Sampler(g, pos_node+neg_node,fanouts)
            output = model(blocks, g_r)
            # print(output)
            loss = align_loss(pairs, neg_node, output)
            print(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if i %10 == 9:
            model.eval()
            Lvec, Rvec = get_embedding(dev_pair[:, 0], dev_pair[:, 1])
            evaluater.test(Lvec, Rvec)
            model.train()
        new_pair = []
    Lvec, Rvec = get_embedding(rest_set_1, rest_set_2, output.cpu())
    A, B = evaluater.CSLS_cal(Lvec, Rvec, False)
    for i, j in enumerate(A):
        if B[j] == i:
            new_pair.append([rest_set_1[j], rest_set_2[i]])

    train_pair = np.concatenate([train_pair, np.array(new_pair)], axis=0)
    for e1, e2 in new_pair:
        if e1 in rest_set_1:
            rest_set_1.remove(e1)

    for e1, e2 in new_pair:
        if e2 in rest_set_2:
            rest_set_2.remove(e2)
    epoch = 5