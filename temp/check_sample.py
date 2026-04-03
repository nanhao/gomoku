import json, os, numpy as np

data_dir = 'train_data/fc_v1_cycle_1_4'
subdirs = sorted(os.listdir(data_dir), key=lambda x: int(x))

# 全量统计 99 局
total_steps = 0
all_ok = True
problem_steps = []

for d in subdirs:
    path = os.path.join(data_dir, d, 'sample.json')
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    for step_i, s in enumerate(data):
        inp = np.array(s['input'][:225])
        out = np.array(s['output'])
        total_steps += 1
        is_last = (step_i == len(data) - 1)

        occupied_mask = inp > 0
        empty_mask = inp == 0

        # 检查1: 已占位的reward必须是-1000000
        if occupied_mask.sum() > 0:
            occ_rewards = out[occupied_mask]
            if not (occ_rewards == -1000000).all():
                bad = np.unique(occ_rewards[occ_rewards != -1000000])
                problem_steps.append(f'Game {d} step {step_i}: 已占位reward非-1e6: {bad}')
                all_ok = False

        # 检查2: 最后一步空位中必须有一个+1000000
        if is_last:
            empty_rewards = out[empty_mask]
            if not (empty_rewards == 1000000).any():
                problem_steps.append(f'Game {d} step {step_i}(最后): 空位中无+1e6胜利信号')
                all_ok = False

        # 检查3: 非最后一步的空位reward
        if not is_last:
            empty_rewards = out[empty_mask]
            non_zero_empty = empty_rewards[empty_rewards != 0]
            if len(non_zero_empty) > 0:
                # 有非零值，看是什么
                pass  # 先不报错，统计一下

print(f'=== fc_v1_cycle_1_4 全量检查（{len(subdirs)}局）===')
print(f'总步数: {total_steps}')

# 统计空位reward分布
print()
print('=== 抽样分析（第1局）===')
path = os.path.join(data_dir, '1', 'sample.json')
with open(path, encoding='utf-8') as f:
    data = json.load(f)

print(f'第1局总步数: {len(data)}')
for label, idx in [('第1步', 0), ('中间步', len(data)//2), ('最后一步', len(data)-1)]:
    s = data[idx]
    inp = np.array(s['input'][:225])
    out = np.array(s['output'])
    occupied_cnt = int((inp > 0).sum())
    empty_cnt = int((inp == 0).sum())
    occupied_rewards = out[inp > 0]
    empty_rewards = out[inp == 0]
    print(f'  {label}(idx={idx}):')
    print(f'    棋盘: {occupied_cnt}子 / {empty_cnt}空')
    print(f'    已占位reward unique: {np.unique(occupied_rewards).tolist()}')
    print(f'    空位reward unique: {np.unique(empty_rewards).tolist()}')
    print(f'    空位reward为0的数量: {int((empty_rewards == 0).sum())} / {empty_cnt}')
    win_pos = np.where(out == 1000000)[0]
    if len(win_pos):
        print(f'    +1e6位置数量: {len(win_pos)}')

print()
if all_ok and not problem_steps:
    print('结论: 数据符合预期（已占位=-1e6，最后一步有+1e6胜利信号）')
else:
    print(f'发现问题 {len(problem_steps)} 处:')
    for p in problem_steps[:10]:
        print(f'  {p}')

# 额外：检查非最后步的空位是否都是0
print()
print('=== 检查非最后步空位reward ===')
non_last_empty_vals = set()
for d in subdirs[:20]:
    path = os.path.join(data_dir, d, 'sample.json')
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    for step_i, s in enumerate(data[:-1]):  # 排除最后一步
        inp = np.array(s['input'][:225])
        out = np.array(s['output'])
        empty_rewards = out[inp == 0]
        non_last_empty_vals.update(empty_rewards.tolist())

print(f'非最后步空位reward的所有unique值: {sorted(non_last_empty_vals)}')
