from __future__ import annotations

import torch


def ensure_autoawq_compatibility() -> None:
    # AutoAWQ 0.2.9 still imports an older transformers symbol that is absent in 4.57.x.
    import transformers.activations as activations

    if not hasattr(activations, "PytorchGELUTanh"):
        class PytorchGELUTanh(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.gelu(x, approximate="tanh")

        activations.PytorchGELUTanh = PytorchGELUTanh
