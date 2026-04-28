"""验证 chong4.json 文件中的冲四棋型"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai.shape_detector import detect_shapes

data_path = os.path.join(os.path.dirname(__file__), '..', 'ai', 'merge_samples', 'chong4.json')
data_path = os.path.normpath(data_path)

with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"样本总数: {len(data)}")

chong4_count = 0
multi_chong4_count = 0

for i, s in enumerate(data):
    board = [int(x) for x in s['input'][:225]]
    player = int(s['input'][225])
    
    found_chong4 = False
    found_multi = False
    
    for j in range(225):
        if board[j] == 0:
            x, y = j // 15, j % 15
            shapes = detect_shapes(board, (x, y), player)
            for sh in shapes:
                if sh['shape'] == '冲四':
                    found_chong4 = True
                elif sh['shape'] == '多冲四':
                    found_multi = True
    
    if found_chong4:
        chong4_count += 1
    if found_multi:
        multi_chong4_count += 1
    
    if (i + 1) % 500 == 0:
        print(f"  已验证 {i+1}/{len(data)}")

print(f"\n验证结果:")
print(f"  包含冲四的样本: {chong4_count}/{len(data)}")
print(f"  包含多冲四的样本: {multi_chong4_count}/{len(data)}")
print(f"  冲四覆盖率: {100*(chong4_count+multi_chong4_count)/len(data):.1f}%")
