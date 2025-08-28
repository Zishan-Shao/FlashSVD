from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class FineTuneDataset(Dataset):
    """Base class for fine-tuning transformers models.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class FWDense(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()

        self.dense1 = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=False)
        self.dense2 = nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.dense2(hidden_states)

        return hidden_states

    def _init_weights(self, left_w: torch.Tensor, right_w: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        """Initialize dense layers with matrices A, B
          computed as (by default) in: https://arxiv.org/pdf/2207.00112.pdf

        Args:
          left_w (torch.Tensor): tensor of shape [H, r] = I_hat_inv @ Ur @ Sr,
            where r is a low-rank approx. in SVD
          right_w (torch.Tensor): tensor o shape [r, W] = Vr.T,
            where r is a low-rank approx. in SVD
          bias (Optional[torch.Tensor]): bias in the initial dense layer
        """
        self.dense1.weight.data.copy_(left_w.T)
        self.dense2.weight.data.copy_(right_w.T)
        self.dense2.bias.data.copy_(bias)
