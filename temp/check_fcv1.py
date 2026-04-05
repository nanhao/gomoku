"""检查 FC v1 模型大小和参数"""
import sys
sys.path.insert(0, 'd:/projects/gomoku')
import torch
from ai.models.fc_v1.model import FCv1Network

net = FCv1Network()
total = sum(p.numel() for p in net.parameters())

print(f'总参数量: {total:,}')
print(f'约 {total/1e6:.2f}M 参数')
print(f'模型大小约: {total*4/1024/1024:.2f} MB (float32)')
print()
print('各层参数量:')
for name, p in net.named_parameters():
    print(f'  {name}: {p.numel():,}')
