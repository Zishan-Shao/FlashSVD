from typing import Optional, Callable
import copy

from transformers import BertModel
from torch.utils.data import DataLoader

from .fwsvd import estimate_fisher_weights_bert, replace_dense2fw_bert


def compute_nd_replace_dense(
    model: BertModel,
    dataloader: DataLoader,
    loss_fn: Optional[Callable] = None, 
    rank: int = None, 
    compute_full: bool = False, 
    device: str = 'cpu',
    return_fisher: bool = False,
    use_baseline: bool = False,
    low_rank_method: str = "row-sum-weighted-svd",
) -> BertModel:
    """Replaces Bert-like model dense layers with FWSVD:  
      https://arxiv.org/pdf/2207.00112.pdf.

    Args:
      model (BertModel): Bert-like model from transformers package
      dataloader (DataLoader): torch.utils.data.Dataloader compatible with
        BertModel. Should contain smth like: 
          (input_ids, token_type_ids, attention_mask, labels)
      loss_fn (Callable): currently not supported, assuming model.forward(inputs)
        outputs the loss value.
      rank (int): approx. rank in SVD decomposition
      compute_full (bool): assign False (default), if you want to compute fisher information over
        rows, else compute for each weight
      return_fisher (bool): (optionally) return estimated fisher information.
      use_baseline (bool): do not compute Fisher information (all parameters have importance of 1)
      low_rank_method (str): which method is used to solve weighted low-rank problem:
          "row-sum-weighted-svd" / "weighted-svd" / "nesterov" / "anderson"
    """

    if use_baseline:
        fisher_int = {}
        fisher_out = {}
        for idx in range(model.config.num_hidden_layers):
            fisher_int[idx] = None
            fisher_out[idx] = None
    else:
        fisher_int, fisher_out = estimate_fisher_weights_bert(model, dataloader, loss_fn, compute_full, device)

    model_fw = replace_dense2fw_bert(copy.deepcopy(model), fisher_int, fisher_out, rank, low_rank_method)

    return (model_fw, fisher_int, fisher_out) if return_fisher else model_fw
