"""Subject matrices: independent per-example routing, without dropout rescaling."""
import math

import torch
from torch import nn


class SharedSubjectLayer(nn.Module):
    """Map (B, N, T) -> (B, U, T); subject indices are zero-based.

    During training p_personal selects A[s], otherwise the common matrix.
    Evaluation always uses the common matrix (or the trained-slot mean for
    the individual baseline). No held-out slot participates in that mean.
    """

    def __init__(self, initial, trained_subjects, p_personal, mean_at_eval=False):
        super().__init__()
        if initial.ndim != 3 or min(initial.shape) < 1:
            raise ValueError('initial must have shape (S, U, N), with positive dimensions')
        if not math.isfinite(p_personal) or not 0 <= p_personal <= 1:
            raise ValueError('p_personal must be in [0, 1]')
        subjects = list(trained_subjects)
        if (not subjects or len(set(subjects)) != len(subjects)
                or any(type(s) is not int or not 0 <= s < len(initial) for s in subjects)):
            raise ValueError('trained_subjects must be unique zero-based slot indices')
        if mean_at_eval and p_personal != 1:
            raise ValueError('mean_at_eval is reserved for the p_personal=1 baseline')
        self.p_personal = p_personal
        self.mean_at_eval = mean_at_eval
        self.A = nn.Parameter(initial.detach().clone(), requires_grad=p_personal > 0)
        self.shared = nn.Parameter(torch.empty_like(initial[0]), requires_grad=p_personal < 1)
        nn.init.kaiming_uniform_(self.shared, a=math.sqrt(5))
        mask = torch.zeros(len(initial), dtype=torch.bool, device=initial.device)
        mask[subjects] = True
        self.register_buffer('trained_mask', mask)
        self.register_buffer('route_counts', torch.zeros(2, dtype=torch.long, device=initial.device),
                             persistent=False)  # [personal, shared], training only

    def forward(self, x, subjects):
        if x.ndim != 3 or x.shape[1] != self.A.shape[2]:
            raise ValueError('x must have shape (B, N, T) matching the matrix input')
        if subjects.dtype != torch.long or subjects.shape != (x.shape[0],):
            raise ValueError('subjects must be a long tensor with shape (B,)')
        if ((subjects < 0) | (subjects >= len(self.A))).any():
            raise ValueError('subject index out of bounds')
        if not self.training:
            matrix = self.A[self.trained_mask].mean(0) if self.mean_at_eval else self.shared
            return torch.matmul(matrix, x)
        if not self.trained_mask[subjects].all():
            raise ValueError('held-out subjects must never enter the training forward')
        personal = torch.rand(len(subjects), device=x.device) < self.p_personal
        matrices = torch.where(personal[:, None, None], self.A[subjects], self.shared[None])
        with torch.no_grad():
            self.route_counts += torch.stack((personal.sum(), (~personal).sum()))
        return torch.bmm(matrices, x)
