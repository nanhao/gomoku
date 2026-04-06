"""对比 CNN 和 FC 模型的训练"""
import sys
sys.path.insert(0, 'd:/projects/gomoku')

import torch
from ai.models.cnn_v1.trainer import CNNTrainer
from ai.models.fc_v1.trainer import FCv1Trainer

# 准备测试棋盘（有活四威胁）
board = [0] * 225
board[7*15+7] = 1
board[7*15+8] = 1
board[7*15+9] = 1
board[7*15+10] = 1
board_vector = board + [1]

print('=== CNN 模型训练测试 ===')
cnn_trainer = CNNTrainer(fresh=True)
print('CNN 参数量:', sum(p.numel() for p in cnn_trainer.model.network.parameters()))

# 训练前
before = cnn_trainer.model.predict(board_vector)
print(f'训练前 (7,6): {before[7*15+6]:.6f}, (7,11): {before[7*15+11]:.6f}')

# 训练
sample_dirs = ['d:/projects/gomoku/ai/train_data/cnn_v1_cycle_1_1/1']
loss = cnn_trainer.train_batch(sample_dirs, epochs=500, verbose=False)
print(f'训练 loss: {loss:.6f}')

after = cnn_trainer.model.predict(board_vector)
print(f'训练后 (7,6): {after[7*15+6]:.6f}, (7,11): {after[7*15+11]:.6f}')

print('\n=== FC 模型训练测试 ===')
fc_trainer = FCv1Trainer(fresh=True)
print('FC 参数量:', sum(p.numel() for p in fc_trainer.model.network.parameters()))

# 训练前
before = fc_trainer.model.predict(board_vector)
print(f'训练前 (7,6): {before[7*15+6]:.6f}, (7,11): {before[7*15+11]:.6f}')

# 训练
sample_dirs = ['d:/projects/gomoku/ai/train_data/cnn_v1_cycle_1_1/1']
loss = fc_trainer.train_batch(sample_dirs, epochs=500, verbose=False)
print(f'训练 loss: {loss:.6f}')

after = fc_trainer.model.predict(board_vector)
print(f'训练后 (7,6): {after[7*15+6]:.6f}, (7,11): {after[7*15+11]:.6f}')

# 打印 Top 5
print('\nCNN Top 5:', sorted(enumerate(after), key=lambda x: x[1], reverse=True)[:5])
fc_top5 = sorted(enumerate(fc_trainer.model.predict(board_vector)), key=lambda x: x[1], reverse=True)[:5]
print('FC Top 5:', fc_top5)
