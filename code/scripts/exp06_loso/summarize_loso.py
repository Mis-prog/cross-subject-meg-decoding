"""Summarise the LOSO series: per-method curves over budgets, paired tests, model tables, figure.

Unit of analysis is the holdout subject: repeats are averaged within (holdout, seed), then seeds
within holdout. Intervals are 95% t-intervals across holdouts; comparisons are paired Wilcoxon tests.
"""
import json
import os

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fqw-mpl')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / 'experiments/exp06_loso/results'

RES = RESULTS
METRICS = ['top1', 'top10', 'mean_rank', 'loss']
# Fixed categorical order (reference palette, light mode); colour follows the method.
STYLE = {
    'mean_Ws': ('#2a78d6', 'Средняя $W_s$'),
    'ft_Ws': ('#eb6834', 'Дообучение $W_s$'),
    'ft_lowrank': ('#1baf7a', 'Низкоранговая поправка'),
    'ft_all': ('#eda100', 'Дообучение всей сети'),
    'ea_ft_Ws': ('#e87ba4', 'EA + дообучение $W_s$'),
    'ea_mean_Ws': ('#008300', 'EA + средняя $W_s$'),
}
COMPARISONS = [('ft_Ws', 'mean_Ws'), ('ft_lowrank', 'ft_Ws'), ('ft_all', 'ft_Ws'),
               ('ea_ft_Ws', 'ft_Ws'), ('ea_mean_Ws', 'mean_Ws'), ('pool_ft_Ws', 'ft_Ws')]


def interval(values):
    values = np.asarray(values, dtype=float)
    half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
    return values.mean(), half


def main():
    raw = pd.concat([pd.read_csv(p) for p in sorted((RES / 'calibration').glob('*.csv'))], ignore_index=True)
    per_subject = (raw.groupby(['holdout', 'seed', 'method', 'budget_seconds'])[METRICS].mean()
                   .groupby(['holdout', 'method', 'budget_seconds']).mean().reset_index())
    per_subject.to_csv(RES / 'per_subject.csv', index=False)
    rows = []
    for (method, budget), g in per_subject.groupby(['method', 'budget_seconds']):
        row = {'method': method, 'budget_seconds': budget, 'subjects': len(g)}
        for m in METRICS:
            row[m], row[f'{m}_ci95'] = interval(g[m])
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(RES / 'summary.csv', index=False)

    tests = []
    wide = per_subject.pivot_table(index=['holdout', 'budget_seconds'], columns='method', values='top10')
    for a, b in COMPARISONS:
        for budget, g in wide[[a, b]].dropna().groupby(level='budget_seconds'):
            diff = g[a] - g[b]
            p = stats.wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
            mean, half = interval(diff)
            tests.append({'a': a, 'b': b, 'budget_seconds': budget, 'subjects': len(diff),
                          'top10_diff': mean, 'top10_diff_ci95': half, 'a_better_subjects': int((diff > 0).sum()),
                          'wilcoxon_p': p})
    tests = pd.DataFrame(tests)
    tests['holm_p'] = np.nan
    if len(tests):
        order = tests.wilcoxon_p.sort_values().index
        m = len(order)
        adjusted = np.maximum.accumulate([min(1.0, (m - i) * tests.wilcoxon_p[j]) for i, j in enumerate(order)])
        tests.loc[order, 'holm_p'] = adjusted
    tests.to_csv(RES / 'comparisons.csv', index=False)

    models = []
    for path in sorted((RES / 'models').glob('*.json')):
        r = json.loads(path.read_text())
        test = pd.DataFrame(r['test_by_subject']).T
        models.append({'kind': r['kind'], 'holdout': r['holdout'], 'seed': r['seed'], 'best_epoch': r['best_epoch'],
                       'epochs_run': r['epochs_run'], 'seconds': r['seconds'],
                       'val_top10': r['selected_validation']['top10'], 'source_test_top10': test.top10.mean()})
        if r['kind'] == 'ceiling':
            for subject, score in r['test_by_subject'].items():
                models[-1][f'{subject}_top10'] = score['top10']
    models = pd.DataFrame(models)
    models.to_csv(RES / 'models.csv', index=False)
    evaluated = sorted(per_subject.holdout.unique())
    ceiling = models[models.kind == 'ceiling']
    ceiling_top10 = [ceiling[f'sub-{h:02d}_top10'].mean() for h in evaluated] if len(ceiling) else []

    fig, ax = plt.subplots(figsize=(8, 5.6), dpi=150)
    budgets = sorted(summary.budget_seconds.unique())
    positions = {b: i for i, b in enumerate(budgets)}
    for method, (color, label) in STYLE.items():
        s = summary[summary.method == method].sort_values('budget_seconds')
        if s.empty:
            continue
        x = [positions[b] for b in s.budget_seconds]
        ax.plot(x, s.top10 * 100, color=color, linewidth=2, marker='o', markersize=5, label=label)
        ax.fill_between(x, (s.top10 - s.top10_ci95) * 100, (s.top10 + s.top10_ci95) * 100, color=color, alpha=0.12, linewidth=0)
    untrained = summary[summary.method == 'untrained_slot']
    if len(untrained):
        ax.scatter([0], untrained.top10 * 100, color='#52514e', marker='x', s=40, zorder=3, label='Необученный слот')
    ax.axhline(100 * 10 / 99, color='#8a8984', linewidth=1, linestyle=':')
    ax.text(len(budgets) - 1, 100 * 10 / 99 + 1, 'случайный уровень', color='#52514e', ha='right', fontsize=8)
    if ceiling_top10:
        c = np.mean(ceiling_top10) * 100
        ax.axhline(c, color='#52514e', linewidth=1, linestyle='--')
        ax.text(0, c + 1, f'потолок (человек в обучении): {c:.0f}%', color='#52514e', fontsize=8)
    ax.set_xticks(range(len(budgets)), [str(b) for b in budgets])
    ax.set_xlabel('Размеченные данные нового человека, с')
    ax.set_ylabel('Top-10 на test, % (99 кандидатов)')
    ax.set_title(f'Калибровка на новом человеке: LOSO, {len(evaluated)} человек, среднее и 95% ДИ', fontsize=10)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(axis='y', color='#e6e5e0', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=4)
    fig.tight_layout()
    fig.savefig(RESULTS / 'loso_calibration_curves.png')

    table = summary.pivot(index='method', columns='budget_seconds', values='top10').mul(100).round(1)
    print(table.to_string())
    print(tests.round(4).to_string())
    print(models.groupby('kind')[['best_epoch', 'seconds', 'val_top10', 'source_test_top10']].mean().round(3).to_string())
    if ceiling_top10:
        print('ceiling top10 on evaluated subjects:', round(float(np.mean(ceiling_top10)), 4))


if __name__ == '__main__':
    main()
