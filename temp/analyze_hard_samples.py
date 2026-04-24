import json
from collections import Counter

log_path = 'ai/ai_data/loss_logs/cnn_v1_20260423_215453_aa18b77a/round_1_epoch_level.jsonl'

loss_buckets = Counter()
total = 0
hard_samples = []  # loss > 0.008
high_grad_samples = []  # grad_norm > 0.1

with open(log_path, 'r') as f:
    for line in f:
        d = json.loads(line)
        loss = d['loss']
        grad = d['grad_norm']
        total += 1

        # 分桶统计
        if loss < 0.001:
            loss_buckets['<0.001'] += 1
        elif loss < 0.002:
            loss_buckets['0.001~0.002'] += 1
        elif loss < 0.003:
            loss_buckets['0.002~0.003'] += 1
        elif loss < 0.005:
            loss_buckets['0.003~0.005'] += 1
        elif loss < 0.008:
            loss_buckets['0.005~0.008'] += 1
        else:
            loss_buckets['>=0.008'] += 1
            hard_samples.append(d)

        if grad > 0.1:
            high_grad_samples.append(d)

print(f"总记录数: {total}")
print()
print("=== Loss 分布 ===")
for bucket in ['<0.001', '0.001~0.002', '0.002~0.003', '0.003~0.005', '0.005~0.008', '>=0.008']:
    count = loss_buckets[bucket]
    pct = count / total * 100
    bar = '█' * int(pct)
    print(f"{bucket:>12}: {count:>7} ({pct:>5.2f}%) {bar}")

print()
print(f"=== 困难样本 (loss >= 0.008) ===")
print(f"数量: {len(hard_samples)} / {total} ({len(hard_samples)/total*100:.3f}%)")
if hard_samples:
    avg_loss = sum(d['loss'] for d in hard_samples) / len(hard_samples)
    avg_grad = sum(d['grad_norm'] for d in hard_samples) / len(hard_samples)
    print(f"平均 loss: {avg_loss:.6f}")
    print(f"平均 grad_norm: {avg_grad:.6f}")
    print()
    print("前 10 条困难样本:")
    for d in hard_samples[:10]:
        print(f"  step={d['step']:>6}, loss={d['loss']:.6f}, grad_norm={d['grad_norm']:.6f}")

print()
print(f"=== 高梯度样本 (grad_norm > 0.1) ===")
print(f"数量: {len(high_grad_samples)} / {total} ({len(high_grad_samples)/total*100:.3f}%)")
if high_grad_samples:
    avg_loss = sum(d['loss'] for d in high_grad_samples) / len(high_grad_samples)
    avg_grad = sum(d['grad_norm'] for d in high_grad_samples) / len(high_grad_samples)
    print(f"平均 loss: {avg_loss:.6f}")
    print(f"平均 grad_norm: {avg_grad:.6f}")
