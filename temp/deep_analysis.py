"""深度分析：v9.1 随机性、v9.2 value/policy 质量"""
import json, sys, math, random
sys.path.insert(0, 'd:/projects/gomoku/src')

from game.game import check_winner
from ai.sample_visualizer import SampleVisualizer

with open('d:/projects/gomoku/debug/sample.json', 'r', encoding='utf-8') as f:
    samples = json.load(f)


# ========== 辅助函数 ==========

def compute_entropy(policy):
    """策略熵，随机均匀分布的理论最大值 ≈ log(225) = 5.416"""
    return -sum(p * math.log(max(p, 1e-12)) for p in policy)

def find_winning_moves(state):
    """找出当前方可以一步赢的所有位置"""
    board = state[:225]
    player = state[225]
    moves = []
    for pos in range(225):
        if board[pos] == 0:
            bc = board[:]
            bc[pos] = player
            if check_winner(bc, pos) == player:
                moves.append(pos)
    return moves

def policy_mass_on(policy, positions):
    """policy 中指定位置的概率之和"""
    return sum(policy[p] for p in positions)

UNIFORM_ENTROPY = math.log(225)  # ≈ 5.416


# ========== 1) v9.1 随机性检查 ==========

print("=" * 60)
print("1. v9.1 随机性检查")
print("=" * 60)

v9_1_entropies = [compute_entropy(s['v9_1_policy']) for s in samples]
v9_1_values = [s['v9_1_value'] for s in samples]
pi_entropies = [compute_entropy(s['pi']) for s in samples]

print(f"  v9.1 policy 熵: 均值={sum(v9_1_entropies)/len(v9_1_entropies):.4f}, "
      f"最小={min(v9_1_entropies):.4f}, 最大={max(v9_1_entropies):.4f}")
print(f"  参考: 随机均匀熵 = {UNIFORM_ENTROPY:.4f}")
print(f"  偏离均匀的熵差: 均值={UNIFORM_ENTROPY - sum(v9_1_entropies)/len(v9_1_entropies):.4f}")

print(f"  v9.1 value 分布: 均值={sum(v9_1_values)/len(v9_1_values):.4f}, "
      f"std={math.sqrt(sum((v-sum(v9_1_values)/len(v9_1_values))**2 for v in v9_1_values)/len(v9_1_values)):.4f}, "
      f"min={min(v9_1_values):.4f}, max={max(v9_1_values):.4f}")

# 检查 v9.1 policy 在空位上的均匀性
for s in samples[:5]:
    board = s['state'][:225]
    empty_positions = [i for i in range(225) if board[i] == 0]
    empty_probs = [s['v9_1_policy'][i] for i in empty_positions]
    if empty_probs:
        cv = (math.sqrt(sum((p-sum(empty_probs)/len(empty_probs))**2 for p in empty_probs)/len(empty_probs))
              / (sum(empty_probs)/len(empty_probs)))
        print(f"  sample: 空位概率 CV={cv:.3f} (接近0≈均匀)")


# ========== 2) v9.2 value 分析 ==========

print()
print("=" * 60)
print("2. v9.2 value 预测即将胜利的能力")
print("=" * 60)

v9_2_values = [s['v9_2_value'] for s in samples]
z_values = [s['z'] for s in samples]

print(f"  v9.2 value: 均值={sum(v9_2_values)/len(v9_2_values):.4f}, "
      f"min={min(v9_2_values):.4f}, max={max(v9_2_values):.4f}")
print(f"  预期: 所有局面当前方差一步赢, value 应接近 1.0")

pos_count = sum(1 for v in v9_2_values if v > 0)
high_count = sum(1 for v in v9_2_values if v > 0.9)
mid_count = sum(1 for v in v9_2_values if 0.5 < v <= 0.9)
low_count = sum(1 for v in v9_2_values if v <= 0.5)
print(f"  value > 0:   {pos_count}/{len(samples)} ({100*pos_count/len(samples):.1f}%)")
print(f"  value > 0.9: {high_count}/{len(samples)} ({100*high_count/len(samples):.1f}%)")
print(f"  value 0.5~0.9: {mid_count}/{len(samples)} ({100*mid_count/len(samples):.1f}%)")
print(f"  value ≤ 0.5:  {low_count}/{len(samples)} ({100*low_count/len(samples):.1f}%)")

# 对比 v9.1 vs v9.2
print(f"\n  v9.1 value: 均值={sum(v9_1_values)/len(v9_1_values):.4f}")
print(f"  v9.2 value: 均值={sum(v9_2_values)/len(v9_2_values):.4f}")

# z 值一致性：z=1 表示当前方最终赢了（我们这里是差一步赢的局面，z 应大多=1）
z_pos = sum(1 for z in z_values if z > 0)
print(f"  z=1 (当前方最终胜): {z_pos}/{len(samples)}")


# ========== 3) v9.2 policy vs winning moves ==========

print()
print("=" * 60)
print("3. v9.2 policy 对比 MCTS pi — 赢棋位概率集中度")
print("=" * 60)

v9_2_entropies = [compute_entropy(s['v9_2_policy']) for s in samples]

print(f"  v9.2 policy 熵: 均值={sum(v9_2_entropies)/len(v9_2_entropies):.4f}, "
      f"最小={min(v9_2_entropies):.4f}, 最大={max(v9_2_entropies):.4f}")
print(f"  MCTS pi 熵:   均值={sum(pi_entropies)/len(pi_entropies):.4f}, "
      f"最小={min(pi_entropies):.4f}, 最大={max(pi_entropies):.4f}")
print(f"  参考: 随机均匀熵 = {UNIFORM_ENTROPY:.4f}")

# 赢棋位概率
winning_mass_pi = []
winning_mass_v9 = []
for s in samples:
    wm = find_winning_moves(s['state'])
    if wm:
        winning_mass_pi.append(policy_mass_on(s['pi'], wm))
        winning_mass_v9.append(policy_mass_on(s['v9_2_policy'], wm))

print(f"\n  赢棋位的概率集中度 (共 {len(winning_mass_pi)} 样本有赢棋位):")
print(f"  MCTS pi:  均值={sum(winning_mass_pi)/len(winning_mass_pi):.4f}, "
      f"min={min(winning_mass_pi):.4f}, max={max(winning_mass_pi):.4f}")
print(f"  v9.2 policy: 均值={sum(winning_mass_v9)/len(winning_mass_v9):.4f}, "
      f"min={min(winning_mass_v9):.4f}, max={max(winning_mass_v9):.4f}")

# top-1 一致性
top1_match = 0
for s in samples:
    pi_best = max(range(225), key=lambda i: s['pi'][i])
    v9_best = max(range(225), key=lambda i: s['v9_2_policy'][i])
    if pi_best == v9_best:
        top1_match += 1
print(f"\n  pi 和 v9.2 policy 的 argmax 一致: {top1_match}/{len(samples)} ({100*top1_match/len(samples):.1f}%)")

# top-1 是否命中赢棋位
pi_top1_win = sum(1 for s in samples
                  if max(range(225), key=lambda i: s['pi'][i]) in find_winning_moves(s['state']))
v9_top1_win = sum(1 for s in samples
                  if max(range(225), key=lambda i: s['v9_2_policy'][i]) in find_winning_moves(s['state']))
print(f"  pi  argmax 命中赢棋位:    {pi_top1_win}/{len(samples)} ({100*pi_top1_win/len(samples):.1f}%)")
print(f"  v9.2 argmax 命中赢棋位:   {v9_top1_win}/{len(samples)} ({100*v9_top1_win/len(samples):.1f}%)")


# ========== 4) 画 12×2 热力图对比 ==========

print()
print("=" * 60)
print("4. 生成对比热力图")
print("=" * 60)

random.seed(123)
picked = random.sample(samples, 12)

# 为每个样本生成两个条目：pi 和 v9.2 policy
viz_samples = []
for s in picked:
    state = s['state']
    viz_samples.append({"input": state, "output": s['pi'], "_label": "MCTS pi"})
    viz_samples.append({"input": state, "output": s['v9_2_policy'], "_label": "v9.2 policy"})

viz = SampleVisualizer()
out = viz.visualize(
    viz_samples,
    count=24,
    output_path='d:/projects/gomoku/debug/heatmap_pi_vs_v9_2.png',
    show_heatmap=True,
    heatmap_mode="log",
    heatmap_alpha=0.75,
)
print(f'Done: {out}')