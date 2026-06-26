import json

with open(r'd:\projects\gomoku\samples\7\v2\sample.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

s = data[0]
print(f'样本数: {len(data)}')
print(f'第1个样本 key: {list(s.keys())}')
print(f'state 长度: {len(s["state"])}')
print(f'state 类型: {type(s["state"][0]).__name__}')
print(f'pi 长度: {len(s["pi"])}')
print(f'pi 类型: {type(s["pi"][0]).__name__}')
print(f'z 值: {s["z"]}, z 类型: {type(s["z"]).__name__}')
print(f'state 最后5位: {s["state"][-5:]}')
print(f'state 前15位(第1行): {s["state"][:15]}')
print(f'state 第226位(落子方): {s["state"][225]}')
