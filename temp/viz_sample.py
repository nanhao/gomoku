import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

def get_cjk_font():
    for name in ['Microsoft YaHei', 'SimHei', 'SimSun']:
        try:
            fp = fm.findfont(fm.FontProperties(family=name))
            if 'DejaVu' not in fp:
                return name
        except:
            pass
    return None

cjk = get_cjk_font()
if cjk:
    plt.rcParams['font.family'] = cjk
plt.rcParams['axes.unicode_minus'] = False

with open('train_data/fc_v1_cycle_1_3/1/sample.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

BOARD_SIZE = 15
N = len(data)
steps = [s for s in [0, 10, 30, 60, 90, N-1] if s < N]
ncols = len(steps)

fig, axes = plt.subplots(2, ncols, figsize=(ncols * 3.5, 9))
fig.patch.set_facecolor('#1a1a2e')

for col, idx in enumerate(steps):
    sample = data[idx]
    inp    = np.array(sample['input'][:225]).reshape(BOARD_SIZE, BOARD_SIZE)
    output = np.array(sample['output'], dtype=float).reshape(BOARD_SIZE, BOARD_SIZE)
    player = sample['input'][225]

    # --- 上方：棋盘 ---
    ax_b = axes[0][col]
    ax_b.set_facecolor('#DEB887')
    ax_b.set_xlim(-0.5, BOARD_SIZE - 0.5)
    ax_b.set_ylim(-0.5, BOARD_SIZE - 0.5)
    for i in range(BOARD_SIZE):
        ax_b.axhline(i, color='#555', lw=0.5)
        ax_b.axvline(i, color='#555', lw=0.5)
    for sr, sc in [(3,3),(3,7),(3,11),(7,3),(7,7),(7,11),(11,3),(11,7),(11,11)]:
        ax_b.plot(sc, BOARD_SIZE-1-sr, 'o', color='#333', ms=3, zorder=4)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            v = inp[r, c]
            y = BOARD_SIZE - 1 - r
            if v == 1:
                ax_b.add_patch(plt.Circle((c, y), 0.42, color='#111', zorder=5))
            elif v == 2:
                ax_b.add_patch(plt.Circle((c, y), 0.42, color='#eee', ec='#333', lw=1.2, zorder=5))
    stones = int(np.sum(inp > 0))
    p_str = 'Black' if player == 1 else 'White'
    ax_b.set_title(f'Step {idx}  |  {stones} stones\nNext: {p_str}', fontsize=8, color='white')
    ax_b.set_xticks(range(BOARD_SIZE))
    ax_b.set_yticks(range(BOARD_SIZE))
    ax_b.tick_params(labelsize=5, colors='gray')
    ax_b.set_aspect('equal')

    # --- 下方：reward 热力图 ---
    ax_r = axes[1][col]
    reward = output.copy()
    occupied = reward <= -999999
    reward[occupied] = np.nan
    valid = reward[~np.isnan(reward)]

    if len(valid) == 0 or (valid == 0).all():
        ax_r.imshow(np.zeros((BOARD_SIZE, BOARD_SIZE)), cmap='RdYlGn',
                    vmin=-1, vmax=1, origin='upper', aspect='equal')
        ax_r.set_title('Reward (all zero)', fontsize=8, color='white')
    else:
        vmax = np.nanmax(np.abs(reward))
        if vmax == 0:
            vmax = 1
        im = ax_r.imshow(reward, cmap='RdYlGn', vmin=-vmax, vmax=vmax,
                         origin='upper', aspect='equal', interpolation='nearest')
        best_flat = int(np.nanargmax(reward))
        br, bc = divmod(best_flat, BOARD_SIZE)
        ax_r.scatter([bc], [br], marker='*', color='blue', s=120, zorder=5)
        cbar = plt.colorbar(im, ax=ax_r, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6, colors='gray')
        ax_r.set_title(f'Reward Heatmap\nbest=({bc},{BOARD_SIZE-1-br})', fontsize=8, color='white')

    # 已占位置灰色遮罩
    gray = np.zeros((BOARD_SIZE, BOARD_SIZE, 4))
    gray[occupied] = [0.3, 0.3, 0.3, 0.7]
    ax_r.imshow(gray, origin='upper', aspect='equal', interpolation='nearest')
    ax_r.set_xticks(range(BOARD_SIZE))
    ax_r.set_yticks(range(BOARD_SIZE))
    ax_r.tick_params(labelsize=5, colors='gray')
    ax_r.set_facecolor('#222')

fig.suptitle('sample.json - Board State & Reward Distribution\n(fc_v1_cycle_1_3 / game 1)',
             fontsize=12, color='white', fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('temp/sample_viz.png', dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
print('Saved: temp/sample_viz.png')
