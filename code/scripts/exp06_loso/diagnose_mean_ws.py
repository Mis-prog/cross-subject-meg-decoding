"""Diagnose why zero-shot mean_Ws gets worse when the holdout scaler sees more data.

For each evaluated holdout (seed-0 shared model), mean_Ws is scored on test windows under:
  - scalers fitted on the whole train pool, on the validation split, and on the test split itself
    (the last one is transductive, diagnostic only);
  - scalers fitted on many random 30-s blocks, with the block's per-channel scale/offset recorded;
  - hybrids: block offset + pool scale and pool offset + block scale;
  - the pool scaler with a global input gain.
"""
import json
import sys

import numpy as np
import pandas as pd

from pathlib import Path

# Support direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.exp06_loso import train_loso as T
from scripts.exp06_loso.prepare_loso import apply_scaler, fit_scaler

GAINS = (0.5, 0.7, 1.0, 1.4, 2.0)
BLOCKS = 20


def affine(params):
    p = {k: np.asarray(v) for k, v in params.items()}
    return p['robust_center'] + p['standard_mean'] * p['robust_scale'], p['robust_scale'] * p['standard_scale']


def as_params(offset, scale):
    zeros = np.zeros_like(offset)
    return {'robust_center': offset, 'robust_scale': scale, 'standard_mean': zeros, 'standard_scale': zeros + 1}


def main(holdouts):
    protocol = T.DATA.protocol
    records = []
    for h in holdouts:
        cal = T.Calibrator('shared', h, 0)
        A = T.torch.load(T.model_path('shared', h, 0)[0], map_location=T.DEVICE,
                         weights_only=False)['model_state_dict']['spatial_module.subject_layer.A'][cal.sources]
        source_norm = float(A.flatten(1).norm(dim=1).mean())
        mean_norm = float(cal.mean_ws.norm())
        cosines = T.torch.nn.functional.cosine_similarity(A.flatten(1)[:, None], A.flatten(1)[None], dim=2)
        off_diagonal = float(cosines[~T.torch.eye(len(A), dtype=bool, device=A.device)].mean())
        norm_info = {'source_norm': source_norm, 'mean_ws_norm': mean_norm, 'source_cosine': off_diagonal}
        cal.reset(cal.mean_ws)
        test_ids = np.sort(T.DATA.ids('test', [h]))
        raw_test = T.DATA.unscaled[test_ids]

        def score(params, gain=1.0, **extra):
            x = T.torch.from_numpy(apply_scaler(raw_test, params) * gain).to(T.DEVICE)
            s = cal.score(x, test_ids)
            records.append({'holdout': h, **norm_info, **extra, 'gain': gain, 'top10': s['top10'], 'mean_rank': s['mean_rank'],
                            'loss': s['loss'], 'input_rms': float(x.pow(2).mean().sqrt())})

        pool = T.DATA.pool(h)
        pool_params = fit_scaler(T.DATA.unscaled[np.sort(pool)])
        p_off, p_scale = affine(pool_params)
        for gain in GAINS:
            score(pool_params, gain, scaler='pool')
        score(fit_scaler(T.DATA.unscaled[np.sort(T.DATA.ids('validation', [h]))]), scaler='validation')
        score(fit_scaler(raw_test), scaler='test_transductive')
        score(T.population_scaler(h), scaler='population')
        for name, init in (('mean_ws_norm_matched', cal.mean_ws * source_norm / mean_norm),):
            cal.reset(init)
            score(pool_params, scaler='pool', variant=name)
            score(T.population_scaler(h), scaler='population', variant=name)
            for gain in GAINS:
                score(pool_params, gain, scaler='pool_gain', variant=name)
        cal.reset(cal.mean_ws)
        rng = np.random.default_rng(h)
        n = 30 // protocol['window_seconds']
        for b, start in enumerate(rng.integers(0, len(pool) - n + 1, size=BLOCKS)):
            block = pool[start:start + n]
            params = fit_scaler(T.DATA.unscaled[np.sort(block)])
            b_off, b_scale = affine(params)
            info = {'block': b, 'block_start': int(start), 'block_task': int(T.DATA.windows.task[block[0]]),
                    'scale_ratio': float(np.median(b_scale / p_scale)),
                    'offset_shift': float(np.median(np.abs(b_off - p_off) / p_scale))}
            score(params, scaler='block', **info)
            score(as_params(b_off, p_scale), scaler='block_offset_pool_scale', **info)
            score(as_params(p_off, b_scale), scaler='pool_offset_block_scale', **info)
        print(f'h{h:02d} done', flush=True)
    out = pd.DataFrame(records)
    path = T.RES / 'diagnose_mean_ws.csv'
    out.to_csv(path, index=False)
    out['variant'] = out.get('variant', pd.Series(dtype=object)).fillna('mean_ws')
    print(out.groupby('holdout')[['source_norm', 'mean_ws_norm', 'source_cosine']].first().mean().round(3).to_string())
    print(out[out.scaler == 'pool_gain'].groupby('gain')[['top10', 'mean_rank']].mean().round(3).to_string())
    s = out[out.gain == 1].groupby(['variant', 'scaler'])[['top10', 'mean_rank', 'input_rms']].mean()
    print(s.round(3).to_string())
    print(out[(out.scaler == 'pool') & (out.variant == 'mean_ws')].groupby('gain')[['top10', 'mean_rank']].mean().round(3).to_string())
    blocks = out[out.scaler == 'block']
    print('block: corr(top10, scale_ratio) =', round(blocks.top10.corr(blocks.scale_ratio), 3),
          'corr(top10, offset_shift) =', round(blocks.top10.corr(blocks.offset_shift), 3),
          'median scale_ratio =', round(blocks.scale_ratio.median(), 3))
    print(json.dumps({'within_subject_corr_scale_ratio': blocks.groupby('holdout')[['top10', 'scale_ratio']].apply(
        lambda g: g.top10.corr(g.scale_ratio)).mean()}))


if __name__ == '__main__':
    T.DATA = T.Data()
    main([int(a) for a in sys.argv[1:]] or [s for s in T.DATA.protocol['subjects'] if s not in T.DATA.protocol['dev_subjects']])
