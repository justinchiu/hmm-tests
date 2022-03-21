
import torch
import torch.nn.functional as F

def logbbmv(x, cA, K):
    K1K = 2 * K + 1
    padded_x = F.pad(x, (K, K), value=float("-inf"))
    unfolded_x = padded_x.unfold(-1, K1K, 1)
    #return (unfolded_x * cA).sum(-1)
    return (unfolded_x + cA).logsumexp(-1)

def bbmv(x, cA, K):
    K1K = 2 * K + 1
    padded_x = F.pad(x, (K, K), value=0)
    unfolded_x = padded_x.unfold(-1, K1K, 1)
    result = torch.einsum("bzk,zk->bz", unfolded_x, cA)
    return result
    #return (unfolded_x + cA).logsumexp(-1)

def zero_grads(params):
    for param in params:
        param.grad.zero_()

def clone_and_zero_grads(params):
    grads = tuple(param.grad.detach().clone() for param in params)
    zero_grads(params)
    return grads

def run_inference(text, params, inference_fn):
    evidence, alpha = inference_fn(text, *params)
    evidence.sum().backward()
    grads = clone_and_zero_grads(params)
    return evidence.detach(), grads

#def 
