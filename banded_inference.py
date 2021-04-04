import torch
import torch_struct

def idx_fn(C, K):
    def range_fn(i, K, C):
        if i < K // 2 :
            return range(0, K)
        elif i > C-K//2-1:
            return range(C-K, C)
        else:
            return range(i - K//2, i + K//2+1)
    idx = [
        range_fn(i, K, C)
        for i in range(C)
    ]
    return torch.LongTensor(idx)


def evidence_ts(
    text,
    start_emb, state_emb, next_state_emb,
    projection,
    preterminal_emb, terminal_emb,
    banded_transition,
):
    C, K = banded_transition.shape
    log_phi_start = start_emb @ projection
    log_phi_w = state_emb @ projection
    log_phi_u = next_state_emb @ projection

    transition_logits = (log_phi_w[:,None] + log_phi_u[None,:]).logsumexp(-1)

    idx = idx_fn(C, K)

    #X = torch.zeros_like(transition_logits)
    #Y = torch.ones_like(banded_transition)
    #Z = X.scatter_add(-1, idx, Y)
    #import pdb; pdb.set_trace()

    transition_logits = transition_logits.scatter_add(-1, idx, banded_transition)
    transition = transition_logits.log_softmax(-1)

    start = (log_phi_start[None,None] + log_phi_u[None,:]).logsumexp(-1).log_softmax(-1)
    emission = (preterminal_emb @ terminal_emb.T).log_softmax(-1)

    log_potentials = torch_struct.LinearChain.hmm(
        transition = transition.T,
        emission = emission.T,
        init = start,
        observations = text,
    )
    evidence = torch_struct.LinearChain().sum(log_potentials)
    return evidence

def evidence_fastbmm(
    text,
    start_emb, state_emb, next_state_emb,
    projection,
    preterminal_emb, terminal_emb,
    banded_transition,
):
    # LOOP_FAST_BMM
    N, T = text.shape
    C, K = banded_transition.shape


    log_phi_start = start_emb @ projection
    log_phi_w = state_emb @ projection
    log_phi_u = next_state_emb @ projection

    start = (log_phi_start[None,None] + log_phi_u[None,:]).logsumexp(-1).log_softmax(-1)
    #transition = (log_phi_w @ log_phi_u.T).softmax(-1)
    emission = (preterminal_emb @ terminal_emb.T).log_softmax(-1)
    # O(CD)
    log_denominator = (log_phi_w + log_phi_u.logsumexp(0, keepdim=True)).logsumexp(-1)
    log_denominator = log_denominator.logaddexp(banded_transition.logsumexp(-1))
    # O(CD)
    normed_log_phi_w = log_phi_w - log_denominator[:,None]

    normed_banded_transition = (banded_transition - log_denominator[:,None]).exp()
    idx = idx_fn(C, K).repeat(N, 1, 1)

    normalized_phi_w = normed_log_phi_w.exp()
    phi_u = log_phi_u.exp()

    # gather emission
    # N x T x C
    p_emit = emission[
        torch.arange(C)[None,None],
        text[:,:,None],
    ]
    alphas = []
    Os = []
    #alpha = start * p_emit[:,0] # {N} x C
    alpha_un = start + p_emit[:,0]
    Ot = alpha_un.logsumexp(-1, keepdim=True)
    alpha = (alpha_un - Ot).exp()
    alphas.append(alpha)
    Os.append(Ot)
    for t in range(T-1):
        gamma = alpha @ normalized_phi_w
        alpha_un = p_emit[:,t+1] + (gamma @ phi_u.T).log()#.scatter_add(-1, idx, banded_transition)
        import pdb; pdb.set_trace()
        alpha_un = alpha_un.scatter_add(-1, idx, (alpha @ normed_banded_transition).log())
        #import pdb; pdb.set_trace()
        Ot = alpha_un.logsumexp(-1, keepdim=True)
        alpha = (alpha_un - Ot).exp()

        alphas.append(alpha)
        Os.append(Ot)
    O = torch.cat(Os, -1)
    evidence = O.sum(-1)
    return evidence
