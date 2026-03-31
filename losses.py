import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is None:
            self.alpha = None
        else:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, torch.Tensor):
                self.alpha = alpha.clone().detach().float()
            else:
                self.alpha = torch.tensor([float(alpha)], dtype=torch.float32)

    def forward(self, inputs, targets):
        targets = targets.long()
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            if alpha.numel() == 1:
                alpha_t = torch.full_like(ce_loss, alpha.item())
            elif alpha.numel() == inputs.size(1):
                alpha_t = alpha[targets]
            else:
                alpha_t = torch.ones_like(ce_loss)
            ce_loss = alpha_t * ce_loss
        loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss