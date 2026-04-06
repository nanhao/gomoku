"""深入分析训练过程"""
import sys
sys.path.insert(0, 'd:/projects/gomoku')

import json
import torch
from ai.models.cnn_v1.model import CNNModel

# 加载样本
with open('d:/projects/gomoku/ai/train_data/cnn_v1_cycle_1_1/1/sample.json') as f:
    data = json.load(f)

# 找一个有高概率位置的样本
sample = data[22]  # 之前看到 (7,6) 有 0.48 的概率

board = sample['input'][:225]
target = sample['output']
mask = sample['mask']
player = sample['input'][225]

print('Sample 22 分析:')
print('棋盘上棋子数:', sum(1 for x in board if x != 0))
print('Target (7,6):', target[7*15+6])
print('Target (7,11):', target[7*15+11])
print('Target max:', max(target), 'at', target.index(max(target)))
print('Target sum:', sum(target))
print('Mask sum:', sum(mask))

# 检查棋盘状态
print('\n棋盘位置 (7,6)-(7,11):')
for i in range(6, 12):
    pos = 7*15 + i
    print(f'  (7,{i}): board={board[pos]}, target={target[pos]:.4f}')

# 手动模拟 Evaluator 的 reward 计算
print('\n模拟 reward 计算:')
# 假设 (7,6) 是活四的一端，应该有高 reward
# 但 sample 中的 output 已经是 Softmax 后的概率

# 检查模型训练过程
print('\n=== 手动训练步骤 ===')
model = CNNModel(fresh=True)

# 转换输入
input_tensor = model._vector_to_tensor(sample['input'])
target_tensor = torch.tensor([target], dtype=torch.float32)
mask_tensor = torch.tensor([mask], dtype=torch.float32)

print('Input shape:', input_tensor.shape)
print('Target shape:', target_tensor.shape)

# 前向传播
model.network.train()
output = model.network(input_tensor)
print('Model output shape:', output.shape)
print('Model output (7,6):', output[0, 7*15+6].item())

# 计算 loss
criterion = torch.nn.L1Loss(reduction='none')
loss = criterion(output, target_tensor)
loss = loss * mask_tensor
print('Per-position loss at (7,6):', loss[0, 7*15+6].item())
print('Total loss:', loss.sum().item() / mask_tensor.sum().item())

# 反向传播
optimizer = torch.optim.Adam(model.network.parameters(), lr=0.001)
optimizer.zero_grad()
loss.sum().backward()
optimizer.step()

# 检查更新后的输出
model.network.eval()
with torch.no_grad():
    new_output = model.network(input_tensor)
print('After 1 step, output (7,6):', new_output[0, 7*15+6].item())
