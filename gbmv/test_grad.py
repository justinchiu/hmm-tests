import torch

import genbmm

device = torch.device("cuda:0")
#device = torch.device("cpu")

C = 8
K = 3

K1K = 2*K+1

# diagonal rep
bA = torch.randn(K1K, C, device=device)
x = torch.randn(C, device=device)

# dense
def to_dense(bA):
    K1K, C = bA.shape
    K = K1K // 2
    A = torch.zeros(C, C, device=device)
    # fill in diagonal of A
    for k in range(K):
        lenK = C - k
        # kth closest super diagonal
        diagonal = bA[K - k - 1]
        A.diagonal(k+1).copy_(diagonal[k+1:])

        # kth closest sub diagonal
        diagonal = bA[k-K]
        A.diagonal(-k-1).copy_(diagonal[k+1:])
    A.diagonal().copy_(bA[K])
    return A

A = to_dense(bA)
y = A.T @ x # x @ A

def clamp(x, l, u):
    return min(u, max(l, x))

# convert to C x K matrix
def to_rows(bA):
    K1K, C = bA.shape
    A = to_dense(bA)
    A_rows = torch.zeros(C, K1K, device=device)
    for c in range(C):
        lower = clamp(c-K, 0, C)
        upper = clamp(c+K+1, 0, C)
        length = upper - lower
        A_rows[c,:length].copy_(A[c, lower:upper])
    return A_rows

A_rows = to_rows(bA)
xb = genbmm.BandedMatrix(x.view(1, C, 1), 0, 0)
Ab = genbmm.BandedMatrix(A_rows.view(1, C, K1K), K, K)
yb = xb.multiply(Ab).data.sum(2).squeeze()

import pdb; pdb.set_trace()
