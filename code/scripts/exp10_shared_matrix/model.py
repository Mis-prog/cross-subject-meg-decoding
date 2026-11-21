"""Local LISA adapter; upstream files are never written."""
from unittest.mock import patch
from pathlib import Path

import torch
from torch import nn

from .subject_layer import SharedSubjectLayer


def build_model(hyper, spec, checkpoint=None, device='cpu'):
    from lisa.model import nn_modules
    from lisa.model.load_model import load_from_config

    expected = Path(__file__).resolve().parents[2] / 'LISA/src/lisa/model/nn_modules.py'
    if Path(nn_modules.__file__).resolve() != expected.resolve():
        raise ValueError('imported LISA is not this repository checkout; install code/LISA editable in the active environment')

    class CompatibleConvHead(nn_modules.ConvHead):
        """Same single_conv head, with kernel_size spelled correctly.

        The local annotated upstream copy currently spells it kernel_sie.
        This adapter also works with the pinned, unmodified checkout.
        """

        def __init__(self, n_channels, n_features, pool, head_stride):
            if pool != 'single_conv':
                super().__init__(n_channels, n_features, pool, head_stride)
                return
            nn.Module.__init__(self)
            self._pool_type = pool
            self.pool = nn.Identity()
            self.conv = nn.Conv1d(n_channels, n_features, kernel_size=3,
                                  stride=head_stride, padding=0 if head_stride == 2 else 1)
            self.activation = nn.GELU()
            self.batch_norm = nn.BatchNorm1d(n_features)

    # Scoped construction hook; all other modules and operations come from LISA.
    with patch.object(nn_modules, 'ConvHead', CompatibleConvHead):
        model, _ = load_from_config(hyper, None, torch.device('cpu'))
    original = model.spatial_module.subject_layer
    if original is None:
        raise ValueError('exp10 requires the subject layer')
    model.spatial_module.subject_layer = SharedSubjectLayer(
        original.A, [s - 1 for s in spec['train_subjects']], spec['p_personal'],
        mean_at_eval=spec['variant'] == 'individual',
    )
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model_state_dict'])
    return model.to(device)
