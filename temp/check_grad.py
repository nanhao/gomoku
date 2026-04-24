import json

with open('ai/ai_data/loss_logs/cnn_v1_20260423_215453_aa18b77a/round_1_epoch_level.jsonl', 'r') as f:
    lines = f.readlines()
    total = len(lines)
    print(f'总记录数: {total}')
    print()
    print('=== 最后 15 条记录（训练末期） ===')
    for line in lines[-15:]:
        d = json.loads(line)
        print(f"step={d['step']:>6}, loss={d['loss']:.6f}, grad_norm={d['grad_norm']:.6f}")
    print()
    print('=== 第 1000~1015 条（中期，lr=0.001） ===')
    for line in lines[999:1015]:
        d = json.loads(line)
        print(f"step={d['step']:>6}, loss={d['loss']:.6f}, grad_norm={d['grad_norm']:.6f}")
    print()
    print('=== 第 100~115 条（早期，lr=0.001） ===')
    for line in lines[99:115]:
        d = json.loads(line)
        print(f"step={d['step']:>6}, loss={d['loss']:.6f}, grad_norm={d['grad_norm']:.6f}")
