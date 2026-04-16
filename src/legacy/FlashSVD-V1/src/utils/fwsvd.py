from typing import Tuple, Dict, Optional, Callable
from collections import defaultdict

from tqdm import tqdm

import torch
import torch.linalg as LA
from torch.utils.data import DataLoader
from transformers import BertModel


# this will compute the FWSVD for the FF Layers
def compute_row_sum_svd_decomposition(A: torch.Tensor, weights: Optional[torch.Tensor] = None, rank: Optional[int] = None):
    """Computes FWSVD from https://arxiv.org/pdf/2207.00112.pdf.

    Args: 
      A (torch.Tensor): matrix of size (H, W) to decompose, where H is the hidden dimension, W is the intermediate
      weights (Optional[torch.Tensor]): matrix of size (H, W) or (H,) - Fisher weights.
        If None (default), set to ones.
      rank (Optional[int]): approx. rank in SVD. If None (default), computes
        full-rank decomposition without compression.
    
    Returns:
      left_w (torch.Tensor): matrix [H, r] = I_hat_inv @ Ur @ Sr
      right_w (torch.Tensor): matrix [r, W] = Vr.T
    """
    h, w = A.shape

    if weights is None:
        weights = torch.ones(h)
    
    if weights.ndim > 1:
        weights = weights.sum(dim=1)
    
    i_hat = torch.diag(torch.sqrt(weights + 1e-5))
    i_hat_inv = LA.inv(i_hat)  # actually it's diagonal so we can just take 1 / i_hat

    u, s, v = LA.svd(i_hat @ A, full_matrices=True)
    s = torch.diag(s)  # more convenient form

    if rank is not None:
        u = u[:, :rank]
        s = s[:rank, :rank]
        v = v[:rank]
    else:
        s_tmp = s
        s = torch.zeros_like(A)
        s[:min(h, w), :min(h, w)] = s_tmp

    left_w = i_hat_inv @ (u @ s)
    right_w = v

    return left_w, right_w



# NEW: this help finds the fisher weights of multi-head attention
def estimate_fisher_weights_bert_with_attention(
    model: BertModel,
    dataloader: DataLoader,
    compute_full: bool = False,
    device: str = 'cuda'
):
    """
    Returns six dicts keyed by layer index:
      fisher_q, fisher_k, fisher_v  each of shape [d_model] (or summed to [dh] per head)
      fisher_int, fisher_out         each of shape [d_model] (or [intermediate] for FFN)
    """
    model = model.to(device).train()
    cfg   = model.config
    d_model = cfg.hidden_size
    H       = cfg.num_attention_heads
    dh      = d_model // H
    dint    = cfg.intermediate_size

    # initialize accumulators
    fisher_q   = defaultdict(lambda: torch.zeros(d_model, device=device))
    fisher_k   = defaultdict(lambda: torch.zeros(d_model, device=device))
    fisher_v   = defaultdict(lambda: torch.zeros(d_model, device=device))
    fisher_int = defaultdict(lambda: torch.zeros(d_model, device=device))
    fisher_out = defaultdict(lambda: torch.zeros(dint,   device=device))

    for batch in dataloader:
        # move inputs to device…
        inputs = {k: v.to(device) for k,v in batch.items() if isinstance(v, torch.Tensor)}
        outputs = model(**inputs)
        loss    = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        loss.backward()

        for i in range(cfg.num_hidden_layers):
            # attention: [out_features, in_features] grads => transpose to [in, out]
            q_grad = model.bert.encoder.layer[i].attention.self.query.weight.grad.data.t()  ** 2
            k_grad = model.bert.encoder.layer[i].attention.self.key.weight.grad.data.t()    ** 2
            v_grad = model.bert.encoder.layer[i].attention.self.value.weight.grad.data.t()  ** 2

            # flatten to vector of length d_model
            fisher_q[i]   += q_grad.sum(dim=1) if not compute_full else q_grad
            fisher_k[i]   += k_grad.sum(dim=1) if not compute_full else k_grad
            fisher_v[i]   += v_grad.sum(dim=1) if not compute_full else v_grad

            # FFN intermediate
            int_grad = model.bert.encoder.layer[i].intermediate.dense.weight.grad.data.t() ** 2
            out_grad = model.bert.encoder.layer[i].output.dense.weight.grad.data.t()      ** 2

            fisher_int[i] += int_grad.sum(dim=1) if not compute_full else int_grad
            fisher_out[i] += out_grad.sum(dim=1) if not compute_full else out_grad

        model.zero_grad()

    # normalize each dict to [0,1]
    def normalize(d):
        return {i: v / v.max() for i,v in d.items()}

    return (normalize(fisher_q),
            normalize(fisher_k),
            normalize(fisher_v),
            normalize(fisher_int),
            normalize(fisher_out))
