import torch
import torch.nn.functional as F
import torch_struct

from genbmm import BandedMatrix

from utils import logbbmv

def evidence_ts(
    text,
    start_emb, state_emb, next_state_emb,
    projection,
    preterminal_emb, terminal_emb,
    col_banded_transition,
):
    C, K = col_banded_transition.shape

    # convert to row dense
    cls_banded_transition = BandedMatrix(
        col_banded_transition[None], K // 2, K // 2,
        fill=float("-inf"),
    )
    #banded_transition = cls_banded_transition.transpose().data[0]
    dense_banded_transition = cls_banded_transition.to_dense()[0]
    # want to not affect logits of off diagonals, so +0=*1

    log_phi_start = start_emb @ projection
    log_phi_w = state_emb @ projection
    log_phi_u = next_state_emb @ projection

    transition_logits = (log_phi_w[:,None] + log_phi_u[None,:]).logsumexp(-1)

    transition_logits = transition_logits.logaddexp(dense_banded_transition)
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
    return evidence, None

def evidence_fastbmm(
    text,
    start_emb, state_emb, next_state_emb,
    projection,
    preterminal_emb, terminal_emb,
    col_banded_transition,
):
    # LOOP_FAST_BMM
    N, T = text.shape
    C, K = col_banded_transition.shape

    row_banded_transition = BandedMatrix(
        col_banded_transition[None], K // 2, K // 2,
        fill = float("-inf")
    ).transpose().data[0]

    log_phi_start = start_emb @ projection
    log_phi_w = state_emb @ projection
    log_phi_u = next_state_emb @ projection

    start = (log_phi_start[None,None] + log_phi_u[None,:]).logsumexp(-1).log_softmax(-1)
    emission = (preterminal_emb @ terminal_emb.T).log_softmax(-1)
    # O(CD)
    log_denominator0 = (log_phi_w + log_phi_u.logsumexp(0, keepdim=True)).logsumexp(-1)
    log_denominator = log_denominator0.logaddexp(row_banded_transition.logsumexp(-1))
    # O(CD)
    normed_log_phi_w = log_phi_w - log_denominator[:,None]

    normed_banded_transition = row_banded_transition - log_denominator[:,None]
    normed_col_banded_transition = BandedMatrix(
        normed_banded_transition[None],
        K // 2, K // 2,
        fill = float("-inf"),
    ).transpose().data[0]

    normalized_phi_w = normed_log_phi_w.exp()
    phi_u = log_phi_u.exp()

    """
    # DBG
    cls_banded_transition = BandedMatrix(
        col_banded_transition[None], K // 2, K // 2,
        fill=float("-inf"),
    )
    dense_banded_transition = cls_banded_transition.to_dense()[0]
    transition_logits = (log_phi_w[:,None] + log_phi_u[None,:]).logsumexp(-1)
    transition_logits2 = transition_logits.logaddexp(dense_banded_transition)
    log_transition = transition_logits2.log_softmax(-1)
    transition = transition_logits.softmax(-1)
    # /DBG
    """
    
    # check partition_fn
    #Z_dense = transition_logits.logsumexp(-1)
    #dense_band_logits = dense_banded_transition - Z_dense

    """
    log_dense_banded_transition = BandedMatrix(
        normed_col_banded_transition[None], K // 2, K // 2,
        fill=float("-inf"),
    ).to_dense()[0]
    """

    # gather emission
    # N x T x C
    p_emit = emission[
        torch.arange(C)[None,None],
        text[:,:,None],
    ]
    alphas = []
    Os = []
    alpha_un = start + p_emit[:,0]
    Ot = alpha_un.logsumexp(-1, keepdim=True)
    log_alpha = alpha_un - Ot
    alpha = log_alpha.exp()
    alphas.append(alpha)
    Os.append(Ot)
    for t in range(T-1):
        gamma = alpha @ normalized_phi_w
        alpha_un = (gamma @ phi_u.T).log()

        log_band_alpha = logbbmv(log_alpha, normed_col_banded_transition, K // 2)
        alpha_un1 = alpha_un.logaddexp(log_band_alpha)

        #log_band_alpha2 = (log_alpha[:,:,None] + log_dense_banded_transition[None]).logsumexp(1)
        #alpha_un2 = alpha_un.logaddexp(log_band_alpha2)
        #alpha0 = (alpha @ transition).log()
        #alpha1 = (log_alpha[:,:,None] + log_transition[None]).logsumexp(1)

        alpha_un = p_emit[:,t+1] + alpha_un
        Ot = alpha_un.logsumexp(-1, keepdim=True)
        log_alpha = alpha_un - Ot
        alpha = log_alpha.exp()

        alphas.append(alpha)
        Os.append(Ot)
    O = torch.cat(Os, -1)
    # probably want to use this in actual version
    # return correct alphas by indexing in using lengths.
    #return O, alphas
    evidence = O.sum(-1) # mask here.
    return evidence, alpha
