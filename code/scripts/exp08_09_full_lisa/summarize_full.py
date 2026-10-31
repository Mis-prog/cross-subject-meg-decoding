"""Tables for exp08/exp09 from the per-window evaluation files written by full_lisa.py eval.

Per window -> per subject (mean over that subject's rows) -> mean over seeds within subject ->
mean and 95% t-interval across subjects; paired Wilcoxon across subjects with Holm correction.

  python summarize_full.py            # writes experiments/exp08_09_full_lisa/results/full/*.csv and prints the tables
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Support direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.exp08_09_full_lisa.full_lisa import CANON, OUT, ROOT, STRIDE_SAMPLES, load_protocol

RESULTS = ROOT / 'experiments/exp08_09_full_lisa/results/full'
METRICS = ('top1', 'top10', 'top1_tol1', 'rank')


def neighbour_map(canon):
    """wav_index -> set of wav_index of its 1-s neighbours in the same sound file."""
    chunks = pd.read_csv(Path(canon) / 'dataframe' / 'chunks_info_test.csv')
    by_pos = {(r.story_id, r.sound_id, r.wav_start): r.wav_index for r in chunks.itertuples(index=False)}
    return {r.wav_index: {by_pos.get((r.story_id, r.sound_id, r.wav_start + d)) for d in (-STRIDE_SAMPLES, STRIDE_SAMPLES)} - {None}
            for r in chunks.itertuples(index=False)}


def per_subject(df, neighbours):
    df = df.assign(top1=df['rank'] == 0, top10=df['rank'] < 10)
    df['top1_tol1'] = df.top1 | [t in neighbours.get(w, ()) for w, t in zip(df.wav_index, df.top1_wav_index)]
    return df.groupby('subject_id')[list(METRICS)].mean()


def load_condition(out, task, condition, neighbours, session=None, subjects=None, by=None):
    path = Path(out) / 'eval' / task / f'{condition}.csv.gz'
    df = pd.read_csv(path)
    if session is not None:
        df = df[df.session_id == session]
    if subjects is not None:
        df = df[df.subject_id.isin(subjects)]
    if by:  # e.g. other_cross: score each donor separately, then average over donors
        return pd.concat([per_subject(g, neighbours) for _, g in df.groupby(by)]).groupby(level=0).mean()
    return per_subject(df, neighbours)


def over_seeds(frames):
    return pd.concat(frames).groupby(level=0).mean()


def describe(table):
    n = len(table)
    half = stats.t.ppf(0.975, n - 1) * table.std(ddof=1) / np.sqrt(n)
    return pd.DataFrame({'mean': table.mean(), 'ci95': half, 'n_subjects': n})


def compare(name, a, b):
    common = a.index.intersection(b.index)
    rows = []
    for metric in METRICS:
        diff = a.loc[common, metric] - b.loc[common, metric]
        p = stats.wilcoxon(diff).pvalue if len(common) > 1 and diff.abs().sum() > 0 else np.nan
        n = len(diff)
        half = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)
        better = int((diff > 0).sum() if metric != 'rank' else (diff < 0).sum())
        rows.append({'comparison': name, 'metric': metric, 'diff': diff.mean(), 'ci95': half,
                     'better': f'{better}/{n}', 'p': p})
    return rows


def holm(df):
    p = df.p.to_numpy()
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    m = int(np.isfinite(p).sum())
    adjusted, running = np.full(len(p), np.nan), 0.0
    for i, idx in enumerate(order[:m]):
        running = max(running, min(1.0, (m - i) * p[idx]))
        adjusted[idx] = running
    return df.assign(p_holm=adjusted)


def summarize(protocol, out=OUT, canon=CANON, results=RESULTS):
    neighbours = neighbour_map(canon)
    exps = protocol['experiments']
    tables, comparisons = {}, []

    def have(task, condition):
        return (Path(out) / 'eval' / task / f'{condition}.csv.gz').exists()

    # exp08: cross-session, per direction
    xs = exps['exp08_cross_session']
    for direction, spec in xs['directions'].items():
        tasks = [f'xs{direction}_s{s}' for s in xs['seeds'] if have(f'xs{direction}_s{s}', 'own_cross')]
        if not tasks:
            continue
        both = sorted(set.intersection(*[set(pd.read_csv(Path(out) / 'eval' / t / 'mean_cross.csv.gz').subject_id)
                                         for t in tasks]))
        cond = {
            'own_same': over_seeds([load_condition(out, t, 'own_same', neighbours, subjects=both) for t in tasks]),
            'own_cross': over_seeds([load_condition(out, t, 'own_cross', neighbours, subjects=both) for t in tasks]),
            'mean_cross': over_seeds([load_condition(out, t, 'mean_cross', neighbours) for t in tasks]),
            'other_cross': over_seeds([load_condition(out, t, 'other_cross', neighbours, by='slot') for t in tasks]),
        }
        repro = [f'repro_s{s}' for s in exps['exp08_reproduction']['seeds'] if have(f'repro_s{s}', 'own')]
        if repro:
            cond['ceiling'] = over_seeds([load_condition(out, t, 'own', neighbours, session=spec['target_session'],
                                                         subjects=both) for t in repro])
        for name, table in cond.items():
            tables[f'exp08{direction}_{name}'] = table
        comparisons += compare(f'exp08{direction}: own_cross - own_same', cond['own_cross'], cond['own_same'])
        comparisons += compare(f'exp08{direction}: own_cross - other_cross', cond['own_cross'], cond['other_cross'])
        comparisons += compare(f'exp08{direction}: own_cross - mean_cross', cond['own_cross'], cond['mean_cross'])
    if 'exp08A_own_cross' in tables and 'exp08B_own_cross' in tables:
        drop = {d: tables[f'exp08{d}_own_cross'] - tables[f'exp08{d}_own_same'] for d in 'AB'}
        comparisons += compare('exp08: drop(A) - drop(B)', drop['A'], drop['B'])

    # exp09: new subjects, folds pooled into one table of 27 subjects
    ns = exps['exp09_new_subjects']
    parts = {c: [] for c in ns['conditions'] if c != 'own_seen'}
    for k, fold in enumerate(ns['folds']):
        tasks = [f'ns_f{k}_s{s}' for s in ns['seeds'] if have(f'ns_f{k}_s{s}', 'mean')]
        for c in parts:
            if tasks:
                parts[c].append(over_seeds([load_condition(out, t, c, neighbours) for t in tasks]))
    if parts['mean']:
        cond = {c: pd.concat(v) for c, v in parts.items()}
        held = cond['mean'].index
        seen_seed = ns['seeds'][0]
        if have(f'repro_s{seen_seed}', 'own'):
            cond['own_seen'] = load_condition(out, f'repro_s{seen_seed}', 'own', neighbours, subjects=held)
            comparisons += compare('exp09: mean - own_seen', cond['mean'], cond['own_seen'])
            comparisons += compare('exp09: mean_norm - own_seen', cond['mean_norm'], cond['own_seen'])
        comparisons += compare('exp09: mean_norm - mean', cond['mean_norm'], cond['mean'])
        for name, table in cond.items():
            tables[f'exp09_{name}'] = table

    if not tables:
        print('No evaluation results yet.')
        return None
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    per_subj = pd.concat({k: v for k, v in tables.items()}, names=['condition', 'subject_id']).reset_index()
    summary = pd.concat({k: describe(v) for k, v in tables.items()}, names=['condition', 'metric']).reset_index()
    comp = pd.DataFrame(comparisons)
    if not comp.empty:
        comp['experiment'] = comp.comparison.str.split(':').str[0].str[:5]
        comp = pd.concat([holm(g) for _, g in comp[comp.metric != 'rank'].groupby('experiment')] +
                         [comp[comp.metric == 'rank'].assign(p_holm=np.nan)])
    per_subj.to_csv(results / 'per_subject.csv', index=False)
    summary.to_csv(results / 'summary.csv', index=False)
    comp.to_csv(results / 'comparisons.csv', index=False)
    agreement = {p.parent.name: json.loads(p.read_text()).get('trainer_agreement')
                 for p in sorted((Path(out) / 'eval').glob('*/done.json'))}
    (results / 'trainer_agreement.json').write_text(json.dumps(agreement, indent=2))

    pct = summary[summary.metric != 'rank'].assign(mean=lambda d: 100 * d['mean'], ci95=lambda d: 100 * d.ci95)
    print(pct.pivot(index='condition', columns='metric', values='mean').round(1).to_string())
    if not comp.empty:
        show = comp[comp.metric == 'top10'].assign(diff=lambda d: 100 * d['diff'], ci95=lambda d: 100 * d.ci95)
        print(show[['comparison', 'diff', 'ci95', 'better', 'p_holm']].round(4).to_string(index=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--protocol')
    parser.add_argument('--out', default=str(OUT))
    parser.add_argument('--canon', default=str(CANON))
    parser.add_argument('--results', default=str(RESULTS))
    args = parser.parse_args()
    protocol = load_protocol(args.protocol) if args.protocol else load_protocol()
    summarize(protocol, args.out, args.canon, args.results)


if __name__ == '__main__':
    main()
