"""模拟 AI 自我对弈测试"""
import sys
sys.path.insert(0, '.')

from ai.models.fc_v1.model import FCv1Model
from ai.engine import AIEngine
from game import GomokuController

wins = {'black': 0, 'white': 0, 'draw': 0}

print("模拟 10 局 FCv1 自我对弈（fresh=True, 每局新模型）...")
for game_num in range(10):
    model = FCv1Model(fresh=True)
    engine = AIEngine(model)
    controller = GomokuController()
    controller.new_game(f'test_{game_num}')
    
    while not controller.is_over():
        state = controller.get_state()
        board = state.get("board", [[0]*15 for _ in range(15)])
        player = state.get("current_player", 1)
        x, y = engine.get_move(board, player)
        controller.make_move(x, y)
    
    winner = controller.get_winner()
    if winner == 1:
        wins['black'] += 1
    elif winner == 2:
        wins['white'] += 1
    else:
        wins['draw'] += 1
    print(f"  第 {game_num+1} 局: {'黑胜' if winner == 1 else '白胜' if winner == 2 else '平局'}")

print(f"\n结果: 黑胜 {wins['black']}, 白胜 {wins['white']}, 平局 {wins['draw']}")

# 再测试：同一个模型多局对弈
print("\n" + "="*50)
print("模拟 10 局 FCv1 自我对弈（同一个模型实例）...")
model = FCv1Model(fresh=True)
engine = AIEngine(model)

wins2 = {'black': 0, 'white': 0, 'draw': 0}
for game_num in range(10):
    controller = GomokuController()
    controller.new_game(f'test_same_{game_num}')
    
    while not controller.is_over():
        state = controller.get_state()
        board = state.get("board", [[0]*15 for _ in range(15)])
        player = state.get("current_player", 1)
        x, y = engine.get_move(board, player)
        controller.make_move(x, y)
    
    winner = controller.get_winner()
    if winner == 1:
        wins2['black'] += 1
    elif winner == 2:
        wins2['white'] += 1
    else:
        wins2['draw'] += 1
    print(f"  第 {game_num+1} 局: {'黑胜' if winner == 1 else '白胜' if winner == 2 else '平局'}")

print(f"\n结果: 黑胜 {wins2['black']}, 白胜 {wins2['white']}, 平局 {wins2['draw']}")
