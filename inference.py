import torch
import torch_struct

def evidence_ts(
    text,
    start_emb, state_emb, next_state_emb,
    projection,
    preterminal_emb, terminal_emb,
):
    log_phi_start = start_emb @ projection
    log_phi_w = state_emb @ projection
    log_phi_u = next_state_emb @ projection

    start = (log_phi_start[None,None] + log_phi_u[None,:]).logsumexp(-1).log_softmax(-1)
    transition = (log_phi_w[:,None] + log_phi_u[None,:]).logsumexp(-1).log_softmax(-1)
    emission = (preterminal_emb @ terminal_emb.T).log_softmax(-1)

    log_potentials = torch_struct.LinearChain.hmm(
        transition = transition.T,
        emission = emission.T,
        init = start,
        observations = text,
    )
    evidence = torch_struct.LinearChain().sum(log_potentials)
    return evidence

def evidence_loop(
    text,
    start_emb, state_emb, next_state_emb,
    projection,
    preterminal_emb, terminal_emb,
):
    C = state_emb.shape[0]
    N, T = text.shape

    log_phi_start = start_emb @ projection
    log_phi_w = state_emb @ projection
    log_phi_u = next_state_emb @ projection

    start = (log_phi_start[None,None] + log_phi_u[None,:]).logsumexp(-1).log_softmax(-1)
    transition = (log_phi_w[:,None] + log_phi_u[None]).logsumexp(-1).log_softmax(-1)
    emission = (preterminal_emb @ terminal_emb.T).log_softmax(-1)
    # gather emission
    # N x T x C
    p_emit = emission[
        torch.arange(C)[None,None],
        text[:,:,None],
    ]
    alphas = []
    #alpha = start * p_emit[:,0] # {N} x C
    alpha = start + p_emit[:,0]
    alphas.append(alpha)
    for t in range(T-1):
        # logbmm
        alpha = (alpha[:,:,None] + transition[None] + p_emit[:,t+1,None,:]).logsumexp(-2)
        #alpha = (alpha @ transition) * p_emit[:,t+1]
        alphas.append(alpha)
    evidence = alpha.logsumexp(-1)
    return evidence

def evidence_loopbmm(
    text,
    start_emb, state_emb, next_state_emb,
    projection,
    preterminal_emb, terminal_emb,
):
    # LOOPBMM
    C = state_emb.shape[0]
    N, T = text.shape

    log_phi_start = start_emb @ projection
    log_phi_w = state_emb @ projection
    log_phi_u = next_state_emb @ projection

    start = (log_phi_start[None,None] + log_phi_u[None,:]).logsumexp(-1).log_softmax(-1)
    transition = (log_phi_w[:,None] + log_phi_u[None]).logsumexp(-1).softmax(-1)
    emission = (preterminal_emb @ terminal_emb.T).log_softmax(-1)
    # gather emission
    # N x T x C
    p_emit = emission[
        torch.arange(C)[None,None],
        text[:,:,None],
    ]
    alphas_bmm = []
    evidences_bmm = []
    alpha_un = start + p_emit[:,0] # {N} x C
    Ot = alpha_un.logsumexp(-1, keepdim=True)
    alpha = (alpha_un - Ot).exp()
    alphas_bmm.append(alpha)
    evidences_bmm.append(Ot)
    for t in range(T-1):
        # logbmm
        #alpha = (alpha[:,:,None] + transition[None] + p_emit[:,t+1,None,:]).logsumexp(-2)
        alpha_un = (alpha @ transition).log() + p_emit[:,t+1]
        Ot = alpha_un.logsumexp(-1, keepdim=True)
        alpha = (alpha_un - Ot).exp()
        alphas_bmm.append(alpha)
        evidences_bmm.append(Ot)
    O = torch.cat(evidences_bmm, -1)
    evidence = O.sum(-1)
    return evidence

def evidence_fast(
    text,
    start_emb, state_emb, next_state_emb,
    projection,
    preterminal_emb, terminal_emb,
):
    C = state_emb.shape[0]
    N, T = text.shape

    # LOOP_FAST
    log_phi_start = start_emb @ projection
    log_phi_w = state_emb @ projection
    log_phi_u = next_state_emb @ projection

    start = (log_phi_start[None,None] + log_phi_u[None,:]).logsumexp(-1).log_softmax(-1)
    #transition = (log_phi_w @ log_phi_u.T).softmax(-1)
    emission = (preterminal_emb @ terminal_emb.T).log_softmax(-1)
    # O(CD)
    log_denominator = (log_phi_w + log_phi_u.logsumexp(0, keepdim=True)).logsumexp(-1)
    # O(CD)
    normed_log_phi_w = log_phi_w - log_denominator[:,None]
    # gather emission
    # N x T x C
    p_emit = emission[
        torch.arange(C)[None,None],
        text[:,:,None],
    ]
    alphas = []
    #alpha = start * p_emit[:,0] # {N} x C
    alpha = start + p_emit[:,0]
    alphas.append(alpha)
    for t in range(T-1):
        # for a single timestep, we project previous alpha, ie posterior over last state
        # given words up to t, compute next alpha by projection to feature space and back

        # logmm: (N,C) @ (C,D)
        # N = batch
        # C = num states
        # D = num features
        logmm = lambda x,y: (x[:,:,None] + y[None]).logsumexp(1)
        beta = logmm(alpha, normed_log_phi_w)
        alpha = p_emit[:,t+1] + logmm(beta, log_phi_u.T)

        #beta = (alpha[:,:,None] + log_phi_w[None] - log_denominator[None,:,None]).logsumexp(-2)
        #alpha = p_emit[:,t+1] + (log_phi_u[None] + beta[:,None]).logsumexp(-1)

        # logbmm
        #alpha = (alpha @ transition) * p_emit[:,t+1]
        alphas.append(alpha)
    evidence = alpha.logsumexp(-1)
    return evidence


def evidence_fastbmm(
    text,
    start_emb, state_emb, next_state_emb,
    projection,
    preterminal_emb, terminal_emb,
):
    # LOOP_FAST_BMM
    C = state_emb.shape[0]
    N, T = text.shape

    log_phi_start = start_emb @ projection
    log_phi_w = state_emb @ projection
    log_phi_u = next_state_emb @ projection

    start = (log_phi_start[None,None] + log_phi_u[None,:]).logsumexp(-1).log_softmax(-1)
    #transition = (log_phi_w @ log_phi_u.T).softmax(-1)
    emission = (preterminal_emb @ terminal_emb.T).log_softmax(-1)
    # O(CD)
    log_denominator = (log_phi_w + log_phi_u.logsumexp(0, keepdim=True)).logsumexp(-1)
    # O(CD)
    normed_log_phi_w = log_phi_w - log_denominator[:,None]

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
        alpha_un = p_emit[:,t+1] + (gamma @ phi_u.T).log()
        Ot = alpha_un.logsumexp(-1, keepdim=True)
        alpha = (alpha_un - Ot).exp()

        alphas.append(alpha)
        Os.append(Ot)
    O = torch.cat(Os, -1)
    evidence = O.sum(-1)
    return evidence
