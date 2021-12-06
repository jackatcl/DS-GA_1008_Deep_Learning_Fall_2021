import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib
import torch
import random


"""
    Q4
"""
def generate_sbm_dataset(n=1000, s=5, K=100, shuffle=True):
    a_s = 12 + s
    b_s = 8 - s
    p = a_s / n
    q = b_s / n
    res = np.zeros((K, n, n))
    Y = np.zeros((K, n))
    Y[:, :n // 2] = 1
    Y[:, n // 2:] = -1
    np.random.default_rng(seed=1).shuffle(Y, axis=1)

    for idx in range(K):
        M = np.zeros((n, n))

        n_half = n // 2

        M[:n_half, :n_half] = np.random.binomial(1, p, (n_half, n_half))
        M[n_half:, n_half:] = np.random.binomial(1, p, (n - n_half, n - n_half))
        M[:n_half, n_half:] = np.random.binomial(1, q, (n_half, n - n_half))
        M[n_half:, :n_half] = np.random.binomial(1, q, (n - n_half, n_half))
        M = np.maximum(M, M.T)
        M = M * (np.ones(n) - np.eye(n))

        # assert np.allclose(M, M.T)
        if shuffle:
            np.random.default_rng(seed=1).shuffle(M, axis=0)
        res[idx, :, :] = M

    return res, Y

train_set_dict = {}

for s in [1, 2, 3, 4, 5]:
    train_set_dict[s] = generate_sbm_dataset(n=1000, s=s, shuffle=False)

def q4_networkx_visualization():
    plt.rcParams["figure.figsize"] = (5, 5)
    def show_graph_with_labels(s, adjacency_matrix, mylabels):
        labels = {idx:str(int(label)) for idx, label in enumerate(mylabels)}
        rows, cols = np.where(adjacency_matrix == 1)

        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.Graph()
        gr.add_edges_from(edges)
        nx.draw(gr,
                node_size=10,
                width=0.1,
                labels=labels,
                with_labels=False,
                cmap=plt.get_cmap('Blues'),
                node_color=mylabels,
               pos=nx.spring_layout(gr))

        n = len(labels)
        a_s = 12 + s
        b_s = 8 - s
        p = a_s / n
        q = b_s / n
        plt.title('Graph with s={}, n={}, p={}, q={}'.format(s, n, p, q))
        plt.savefig('q4_s={}.png'.format(s))
        plt.show()

    for s, (graph, label) in train_set_dict.items():
        show_graph_with_labels(s, graph[0, :, :], label[0, :])

def q4_heatmap_visualization():
    plt.rcParams["figure.figsize"] = (20, 4)
    cmap = matplotlib.colors.ListedColormap(['white', 'blue'])
    fig, axes = plt.subplots(ncols=5,nrows=1, sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        m = train_set_dict[i+1][0].sum(axis=0)
        ax.imshow(m, cmap='Oranges', vmin=0, vmax=10)
        n = train_set_dict[i+1][0].shape[1]
        s = i + 1
        a_s = 12 + s
        b_s = 8 - s
        p = a_s / n
        q = b_s / n
        ax.set_title('Graph with s={},\n n={}, p={}, q={}'.format(s, n, p, q))

    plt.savefig('Q4_heatmap.png')

q4_heatmap_visualization()
q4_networkx_visualization()


"""
    Q5
"""
train_set_dict = {s: generate_sbm_dataset(n=100, s=s) for s in [1, 2, 3, 4, 5]}
test_set_dict = {s: generate_sbm_dataset(n=100, s=s) for s in [1, 2, 3, 4, 5]}

# ensure randomness
assert not np.allclose(train_set_dict[1][0][0, :, :], test_set_dict[1][0][0, :, :])


def overlap(pred, true):
    pred = pred.flatten()
    true = true.flatten()
    N = pred.shape[0]
#     print((pred==true))
    return 2 *((1 / N) * max((pred==true).sum(), (pred==-true).sum()) - .5)

# test
overlap(test_set_dict[1][1], -test_set_dict[1][1])


"""
    Q6
"""
# torch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# SBM Model
class SBM(torch.nn.Module):
    def __init__(self, K=10, V=1000):
        super(SBM, self).__init__()
        self.K = K
        self.alpha = torch.nn.ParameterDict()
        self.beta = torch.nn.ParameterDict()
        self.bn = torch.nn.InstanceNorm1d(V)
        for k in range(K):
            self.alpha[str(k)] = torch.nn.Parameter(torch.rand(1) - .5)
            self.beta[str(k)] = torch.nn.Parameter(torch.rand(1) - .5)

    def forward(self, A):
        x_0 = torch.sum(A, dim=1).to(device)
        x_prev = x_0.to(device)
        V_len = A.shape[1]
        batch_size = A.shape[0]

        for k in range(self.K):
            #             print(k)
            k = str(k)
            x = torch.zeros((batch_size, V_len)).to(device)
            for i in range(V_len):
                x_i_prev = x_prev[:, i]
                s = torch.zeros(batch_size, V_len).to(device)
                s[:, i] += torch.sum(A[:, i] * x_prev, axis=1)
                x_i = self.alpha[k] * x_i_prev + self.beta[k] * s[:, i]
                x_i = torch.nn.functional.relu(x_i)
                x[:, i] = x_i
            x = self.bn(x.reshape(batch_size, 1, -1)).reshape(batch_size, -1)
            x_prev = x

        return x.reshape((batch_size, -1))

# Loss
def sbm_loss(y_pred, y_true, K=10):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    def logit(y_pred, y_true):
        return torch.log(1 + torch.exp(-torch.mul(y_pred, y_true)))

    l1 = logit(y_pred, y_true)
    l2 = logit(y_pred, -y_true)
    ans = 0
    for k in range(K):
        ans += 1 / y_pred.shape[0] * torch.min(torch.sum(l1), torch.sum(l2))

    return ans / K


train_set_dict = {s: generate_sbm_dataset(n=100, s=s, K=100) for s in [1, 2, 3, 4, 5]}
test_set_dict = {s: generate_sbm_dataset(n=100, s=s, K=100) for s in [1, 2, 3, 4, 5]}

# Training
test_set_mat = torch.from_numpy(test_set_dict[5][0]).float().to(device)
test_set_y = torch.from_numpy(test_set_dict[5][1]).float().to(device)
train_set_mat = train_set_dict[5][0]
train_set_y = train_set_dict[5][1]

model = SBM(K=5, V=100).to(device)
optim = torch.optim.SGD(list(model.parameters()), lr=0.1)

loss_list = []
MAX_EPOCH = 20

for epoch in range(MAX_EPOCH):
    print(epoch)
    rdm_seed = random.randint(1, 9999)
    np.random.default_rng(seed=rdm_seed).shuffle(train_set_mat, axis=0)
    np.random.default_rng(seed=rdm_seed).shuffle(train_set_y, axis=0)
    model.train()
    # Train
    batch_size = 1
    idx = 0
    #     print(list(model.parameters()))
    while idx + batch_size <= train_set_y.shape[0]:
        X = torch.from_numpy(train_set_mat[idx:idx + batch_size, :, :]).float().to(device)
        y = torch.from_numpy(train_set_y[idx:idx + batch_size, :]).float().to(device)

        out = model.forward(X)
        loss = sbm_loss(out, y)
        loss.backward()
        optim.step()
        idx += batch_size

    # Eval
    model.eval()
    test_out = model.forward(test_set_mat)
    loss = sbm_loss(test_out, test_set_y)
    loss_list.append(loss.detach().numpy())
    print('loss', loss)


"""
    Q7
"""

# Spectral
s = 5
test_set_mat = test_set_dict[5][0]
test_set_y = test_set_dict[5][1]
spectral_score = []
for idx in range(test_set_mat.shape[0]):
    eig_val, eig_vec = np.linalg.eigh(test_set_mat[idx, :])
    v_2 = eig_vec[-2]
    spectral_score.append(overlap(np.sign(v_2), test_set_y[idx]))

print("Average overlap score for the sample graphs using spectral is: {}".format(sum(spectral_score) / len(spectral_score)))

test_set_mat = torch.from_numpy(test_set_dict[5][0]).float().to(device)
test_set_y = torch.from_numpy(test_set_dict[5][1]).float().to(device)
test_out = model.forward(test_set_mat).detach().numpy()

model_score = []
for idx in range(test_out.shape[0]):
    test_out_1 = test_out[idx,:]
    model_score.append(overlap(np.sign(test_out_1), test_set_y[idx].detach().numpy()))

print("Average overlap score for the sample graphs using spectral is: {}".format(sum(model_score) / len(model_score)))