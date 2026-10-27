from pathlib import Path
import json
import statistics
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
scenes = {name: json.loads((root / 'results' / f'{name}.json').read_text()) for name in ['bonsai', 'bicycle']}
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
for col, (scene, data) in enumerate(scenes.items()):
    for row, metric in enumerate(['preprocess_ms', 'gpu_ms']):
        ax = axes[row, col]
        values = [statistics.median(s[metric] for s in r['samples']) for r in data['results']]
        ax.bar(['f32', 'f16'], values, color=['#366b9c', '#df8c32'], width=0.55)
        for i, result in enumerate(data['results']):
            medians = [statistics.median(s[metric] for s in result['samples'] if s['round'] == r) for r in range(data['rounds'])]
            ax.vlines(i, min(medians), max(medians), color='#222222', lw=2)
            ax.scatter([i] * 3, medians, color='#222222', s=14, zorder=3)
            ax.text(i, max(medians) * 1.04, f'{values[i]:.3f}', ha='center')
        ax.set_ylim(0, max(values) * 1.25)
        ax.set_ylabel('GPU time (ms)')
        ax.set_title(scene.capitalize() + (' preprocessing' if row == 0 else ' pipeline'))
fig.suptitle('f32 vs. mixed f16: 90-frame medians\nBlack marks show three per-round medians', fontsize=14)
fig.savefig(root / 'timing.png', dpi=160)
plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout='constrained')
for row, (scene, data) in enumerate(scenes.items()):
    a = np.array(Image.open(root / 'images' / f'{scene}-f32.png').convert('RGB'))
    b = np.array(Image.open(root / 'images' / f'{scene}-f16.png').convert('RGB'))
    delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
    mae = float(delta.mean())
    psnr = 10 * np.log10(255 ** 2 / np.mean(delta.astype(float) ** 2))
    result = data['results'][1]
    assert abs(mae - result['image_mae_255']) < 1e-10
    assert int(delta.max()) == result['image_max_error_255']
    assert abs(psnr - result['image_psnr_db']) < 1e-8
    for col, (pixels, title) in enumerate([(a, 'f32'), (b, 'mixed f16'), (np.minimum(delta * 64, 255).astype(np.uint8), 'absolute RGB difference x64')]):
        axes[row, col].imshow(pixels)
        axes[row, col].set_title(f'{scene.capitalize()}: {title}')
        axes[row, col].axis('off')
fig.savefig(root / 'comparison.png', dpi=150)
plt.close(fig)
print('Figures generated; captured image metrics match recorded results.')
