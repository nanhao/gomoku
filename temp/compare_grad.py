import json

# 两个日志路径
log_32 = 'ai/ai_data/loss_logs/cnn_v1_20260423_215453_aa18b77a/round_1_epoch_level.jsonl'
log_128 = 'ai/ai_data/loss_logs/cnn_v1_20260423_232621_5e50c59d/round_1_epoch_level.jsonl'

def analyze(path, name):
    losses = []
    grads = []
    with open(path, 'r') as f:
        for line in f:
            d = json.loads(line)
            losses.append(d['loss'])
            grads.append(d['grad_norm'])

    # 分阶段统计：早期、中期、晚期
    n = len(losses)
    early = slice(0, n // 10)
    mid = slice(n // 2 - n // 20, n // 2 + n // 20)
    late = slice(-n // 10, n)

    def stats(data, s):
        subset = data[s]
        return f"avg={sum(subset)/len(subset):.6f}, max={max(subset):.6f}"

    print(f"=== {name} (总记录: {n}) ===")
    print(f"Loss 早期: {stats(losses, early)}")
    print(f"Loss 中期: {stats(losses, mid)}")
    print(f"Loss 晚期: {stats(losses, late)}")
    print()
    print(f"Grad 早期: {stats(grads, early)}")
    print(f"Grad 中期: {stats(grads, mid)}")
    print(f"Grad 晚期: {stats(grads, late)}")
    print()
    print(f"Grad > 0.1 的比例: {sum(1 for g in grads if g > 0.1) / len(grads) * 100:.3f}%")
    print(f"Grad > 0.05 的比例: {sum(1 for g in grads if g > 0.05) / len(grads) * 100:.3f}%")
    print()

analyze(log_32, "Batch Size 32")
analyze(log_128, "Batch Size 128")
