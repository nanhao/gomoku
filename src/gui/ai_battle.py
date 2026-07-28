"""AI内战运行器

封装 run_ai_vs_ai_mode 函数，解耦 main.py 与 AIBattleProgressWindow 的实例化细节。
"""

import os

from gui.components import AIBattleProgressWindow, ReplayWindow
from ai.move_sampler import TempConfig


def run_ai_vs_ai_mode(num_games, ai1_name="random", ai2_name="random"):
    """运行 AI 内战模式

    第四十一次追加：默认启用温度采样（temperature=1.0, temp_moves=2, temp_side="both"）。

    Args:
        num_games: 对局数量
        ai1_name: AI1 模型名称
        ai2_name: AI2 模型名称
    """
    # 构造默认温度采样配置
    temp_config = TempConfig(temperature=1.0, temp_moves=2, temp_side="both")

    progress = AIBattleProgressWindow(
        num_games, ai1_name, ai2_name, temp_config=temp_config,
    )
    
    def on_complete(recorder, should_replay):
        if should_replay and recorder:
            # 显示回放
            round_dir = recorder.get_round_dir()
            # 获取第一局
            game_dirs = sorted([d for d in os.listdir(round_dir) 
                               if os.path.isdir(os.path.join(round_dir, d))])
            if game_dirs:
                record_file = os.path.join(round_dir, game_dirs[0], "record.json")
                sample_file = record_file.replace("ai_data", "train_data").replace("record.json", "sample.json")
                if not os.path.exists(sample_file):
                    sample_file = None
                ReplayWindow.show(record_file, sample_file)
    
    progress.start_battle(on_complete)
