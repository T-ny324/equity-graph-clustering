import torch
from src.models.layers import GraphConv, normalise_adjacency

def test_permutation_equivariance():
    torch.manual_seed(0)
    N,F,D = 12, 5, 3
    A = torch.rand(N,N); A =((A + A.T) / 2 > 0.6).float(); A.fill_diagonal_(0)
    X = torch.rand(N, F)
    layer = GraphConv(F,D,False)


    perm = torch.randperm(N)
    out_then_perm = layer(X, normalise_adjacency(A))[perm]
    perm_then_out = layer(X[perm], normalise_adjacency(A[perm][:,perm]))

    assert torch.allclose(out_then_perm, perm_then_out, atol=1e-5)
