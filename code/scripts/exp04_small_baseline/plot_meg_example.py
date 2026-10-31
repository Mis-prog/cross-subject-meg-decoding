"""Visualize one actual prepared MEG/audio pair from the small baseline."""
import json
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fqw-mpl')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'data/pilot/baseline'


def main():
    row = pd.read_csv(DATA / 'windows.csv').iloc[0]
    meg = np.load(DATA / 'meg_windows.npy', mmap_mode='r')[row.window_id]
    audio = np.load(DATA / 'audio_windows_16k.npy', mmap_mode='r')[row.window_id]
    channels = json.loads((DATA / 'channels.json').read_text())
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), layout='constrained',
                             gridspec_kw={'height_ratios': [1, 2, 2.2]})
    fig.suptitle('Реальные данные: 3 секунды речи и соответствующая МЭГ\n'
                 'sub-01 · lw1_0.wav · аудио 1–4 с · обучающее окно № 0', fontsize=15)
    axes[0].plot(np.arange(len(audio)) / 16000, audio, color='#64748b', lw=0.6)
    axes[0].set(title='1. Аудио, которое слушал испытуемый', ylabel='Амплитуда\nаудиосигнала')
    chosen = [0, 40, 80, 120, 160, 200]
    offsets = np.arange(len(chosen))[::-1] * 9
    t = np.arange(meg.shape[1]) / 100
    for channel, offset, color in zip(chosen, offsets, plt.cm.viridis(np.linspace(.12, .85, len(chosen)))):
        axes[1].plot(t, meg[channel] + offset, color=color, lw=1)
    axes[1].set_yticks(offsets, [channels[c] for c in chosen])
    axes[1].set(title='2. Сигналы шести из 208 датчиков после обработки',
                ylabel='Датчики\n(кривые разнесены по вертикали)')
    heat = axes[2].imshow(meg, aspect='auto', origin='lower', cmap='RdBu_r',
                          vmin=-3, vmax=3, extent=(0, 3, .5, 208.5), interpolation='nearest')
    axes[2].set(title='3. То же окно целиком: матрица 208 датчиков × 300 отсчётов',
                ylabel='Номер датчика', xlabel='Время внутри окна, с', yticks=[1, 50, 100, 150, 208])
    colorbar = fig.colorbar(heat, ax=axes[2], pad=.015, fraction=.035, extend='both')
    colorbar.set_label('Нормализованный сигнал')
    for ax in axes:
        ax.set_xlim(0, 3)
        ax.spines[['top', 'right']].set_visible(False)
    for ax in axes[:2]:
        ax.grid(axis='x', alpha=.15)
    fig.supxlabel('МЭГ очищена ICA и нормализована по train; начало МЭГ-окна сдвинуто на +150 мс к аудио.\n'
                  'Цветовая шкала ограничена ±3; красный/синий показывают знак сигнала, а не слова.', fontsize=10)
    output = ROOT / 'experiments/exp04_small_baseline/results/meg_example.png'
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
