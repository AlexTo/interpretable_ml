import torch
import numpy as np

def load_adj(num_classes, t=0.4, adj_files=None, add_identity=False):
    _adj_stack = torch.eye(num_classes).type(
        torch.FloatTensor).unsqueeze(-1) if add_identity else torch.Tensor([])

    for adj_file in adj_files:
        if '_emb' in adj_file:
            _adj = gen_emb_A(adj_file)
        else:
            _adj = gen_A(num_classes, adj_file, t)

        _adj = torch.from_numpy(_adj).type(torch.FloatTensor)
        _adj_stack = torch.cat([_adj_stack, _adj.unsqueeze(-1)], dim=-1)

    return _adj_stack.permute(2, 0, 1)


def gen_A(num_classes, adj_file, t=None):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / (_nums + 1e-6)

    if t is not None and t > 0.0:
        _adj[_adj < t] = 0
        _adj[_adj >= t] = 1

    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    return _adj


def gen_emb_A(adj_file, t=None):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    mean_v = _adj.mean()
    std_v = _adj.std()
    t = mean_v - std_v if t is None else t

    if t is not None and t > 0.0:
        _adj[_adj < t] = 0
        _adj[_adj >= t] = 1

    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    return _adj


def transform_adj(A, is_stack=False):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


def batched_target_to_adj(target):
    batch_size = target.size()[0]
    n = target.size()[1]
    starget = torch.reshape(target, [batch_size * n])
    zero = torch.zeros(batch_size * n)
    ztarget = torch.where(starget < 0, starget, zero)
    indices = torch.nonzero(ztarget, as_tuple=True)[0]
    smask_adj = torch.ones((batch_size * n, n))
    smask_adj[indices] = 0
    mask_adj = torch.reshape(smask_adj, [batch_size, n, n])
    mask_adj_r = torch.rot90(mask_adj, 1, [1, 2])
    adj = (mask_adj.bool() & mask_adj_r.bool() & torch.logical_not(
        torch.diag_embed(torch.ones((batch_size, n))).bool())).float()
    return adj


def batched_target_to_nums(target):
    batch_size = target.size()[0]
    n = target.size()[1]
    zero = torch.zeros(batch_size, n)
    ztarget = torch.where(target > 0, target, zero)
    return torch.sum(ztarget, 0)


def normalise_adj(A, t=0.4):
    # Thresholding
    A.masked_fill_(A < t, 0.0)
    A.masked_fill_(A >= t, 1.0)
    # Normalisation
    A = torch.div(torch.mul(A, 0.25), torch.add(A.sum(0, keepdim=True), 1e-6))
    # Add identity matrix
    mask = torch.eye(A.shape[0]).bool()
    A.masked_fill_(mask, 1.0)

    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


def batched_adj_to_freq(adj):
    return torch.sum(adj, 0)
