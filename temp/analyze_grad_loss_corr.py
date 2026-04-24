import json
from collections import Counter

log_path = 'ai/ai_data/loss_logs/cnn_v1_20260423_215453_aa18b77a/round_1_epoch_level.jsonl'

# 收集所有 grad_norm > 0.1 的样本，分析它们的 loss 分布
high_grad_samples = []
all_samples = []

with open(log_path, 'r') as f:
    for line in f:
        d = json.loads(line)
        all_samples.append(d)
        if d['grad_norm'] > 0.1:
            high_grad_samples.append(d)

print(f"总样本: {len(all_samples)}")
print(f"高梯度样本 (grad>0.1): {len(high_grad_samples)}")
print()

# 高梯度样本的 loss 分布
loss_dist = Counter()
for d in high_grad_samples:
    loss = d['loss']
    if loss < 0.001:
        loss_dist['<0.001'] += 1
    elif loss < 0.002:
        loss_dist['0.001~0.002'] += 1
    elif loss < 0.003:
        loss_dist['0.002~0.003'] += 1
    elif loss < 0.005:
        loss_dist['0.003~0.005'] += 1
    elif loss < 0.008:
        loss_dist['0.005~0.008'] += 1
    else:
        loss_dist['>=0.008'] += 1

print("=== 高梯度样本的 loss 分布 ===")
for bucket in ['<0.001', '0.001~0.002', '0.002~0.003', '0.003~0.005', '0.005~0.008', '>=0.008']:
    count = loss_dist[bucket]
    pct = count / len(high_grad_samples) * 100 if high_grad_samples else 0
    print(f"  {bucket:>12}: {count:>4} ({pct:>5.1f}%)")

# 计算相关系数
import math
n = len(all_samples)
if n > 10000:
    # 样本太多，随机抽 10000 条计算
    import random
    sample = random.sample(all_samples, 10000)
else:
    sample = all_samples

sum_x = sum_y = sum_xy = sum_x2 = sum_y2 = 0
for d in sample:
    x = d['loss']
    y = d['grad_norm']
    sum_x += x
    sum_y += y
    sum_xy += x * y
    sum_x2 += x * x
    sum_y2 += y * y

n = len(sample)
r = (n * sum_xy - sum_x * sum_y) / math.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))

print()
print(f"=== Loss 与 Grad Norm 的相关系数 (随机{n}条样本) ===")
print(f"r = {r:.4f}")
if abs(r) < 0.1:
    print("结论: 几乎无关")
elif abs(r) < 0.3:
    print("结论: 弱相关")
elif abs(r) < 0.5:
    print("结论: 中等相关")
else:
    print("结论: 强相关")

# 看看 loss spike 是否集中在 epoch 边界
print()
print("=== 高梯度样本出现的位置 ===")
steps_per_epoch = len(all_samples) // 2000
print(f"每个 epoch 约 {steps_per_epoch} steps")

# 统计每个 epoch 内的高梯度样本数
epoch_hard = Counter()
for d in high_grad_samples:
    epoch = d['step'] // steps_per_epoch
    epoch_hard[epoch] += 1

# 看前 20 个 epoch 和后 20 个 epoch
print(f"前 20 epoch 高梯度样本: {sum(epoch_hard[e] for e in range(20))}")
print(f"后 20 epoch 高梯度样本: {sum(epoch_hard[e] for e in range(1980, 2000))}")
