import torch

Z = 7
X = 13
N = 2
T = 5

log_prior = torch.randn((Z,)).log_softmax(0)
log_trans = torch.randn((Z, Z)).log_softmax(0)
log_emit = torch.randn((X, Z)).log_softmax(0)

xs = torch.stack([torch.arange(n,T+n) for n in range(N)], dim=0)

def forward(
    log_prior, # Z
    log_trans, # Z x Z
    log_emit, # X x Z
    xs, # N x T
):
    N, T = xs.shape
    # log p(x | z)
    emit = log_emit[xs]
    log_evidences = []
    # log p(x0), first evidence does not use transition
    un_lpz0 = log_prior + emit[:,0] # unnormalized posterior
    evidence = un_lpz0.logsumexp(dim=-1)
    log_evidences.append(evidence)
    # log p(z0 | x0)
    alpha = un_lpz0 - evidence[:,None]
    for t in range(1, T):
        # alpha: N x Z_last, lt: Z_next x Z_last, emit: N x T x Z_next
        un_alpha = (alpha[:,None] + log_trans[None] + emit[:,t,:,None]).logsumexp(-1)
        evidence = un_alpha.logsumexp(-1)
        log_evidences.append(evidence)
        alpha = un_alpha - evidence[:,None]
    return torch.stack(log_evidences, dim=1), alpha


Z, alpha = forward(log_prior, log_trans, log_emit, xs)
print(Z)
print(Z.shape)
print(alpha)
print(alpha.shape)
