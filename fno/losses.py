import torch
import torch.nn as nn


class RelL2Loss(nn.Module):
    """
    Error relativo cuadrático L2 para campos continuos:
    loss = mean(||pred - target||_2 / ||target||_2)
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = pred.shape[0]

        # Aplanamos todas las dimensiones excepto el batch
        diff_norms = torch.norm(
            pred.reshape(batch_size, -1) - target.reshape(batch_size, -1),
            p=2,
            dim=1
        )
        target_norms = torch.norm(
            target.reshape(batch_size, -1),
            p=2,
            dim=1
        )

        return torch.mean(diff_norms / target_norms)