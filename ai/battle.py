"""AI 对战脚本

独立运行的命令行脚本，用于启动两个 AI 对战。
支持通过参数指定双方 AI 及对战局数。每一局执方随机（黑白随机分配）。
对战结束后输出每局结果及双方胜率。

用法:
    python ai/battle.py --ai1 random --ai2 fc_v1 --games 10
    python ai/battle.py --ai1 random --ai2 random --games 50
    python ai/battle.py --ai1 cnn_v1 --ai2 fc_v1 --games 10
    python ai/battle.py --ai1 fc_v1 --ai2 cnn_v2 --games 50
"""

import sys
import os
import argparse

# 添加项目根目录到路径，并移除 ai/ 目录（防止遮蔽标准库 random 模块）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ai_dir = os.path.dirname(os.path.abspath(__file__))
# 优先移除 ai/ 目录，再插入项目根目录
for _p in list(sys.path):
    if os.path.normcase(os.path.abspath(_p)) == os.path.normcase(_ai_dir):
        sys.path.remove(_p)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ai.engine import AIEngine
from ai.data_recorder import DataRecorder
from ai.game_manager import AIGameManager
from ai.constants import AI_DATA_DIR


# 已知 AI 名称到模型类的映射
_AI_MODEL_MAP = {
    "random": "ai.models.random.model.RandomModel",
    "fc_v1": "ai.models.fc_v1.model.FCv1Model",
    "cnn_v1": "ai.models.cnn_v1.model.CNNV1Model",
    "cnn_v2": "ai.models.cnn_v2.model.CNNV2Model",
}


def load_model(ai_name: str):
    """
    根据 AI 名称动态加载对应模型实例
    
    Args:
        ai_name: AI 名称（对应 ai/ 目录下的子目录名称）
        
    Returns:
        模型实例
        
    Raises:
        ValueError: 未知的 AI 名称
        ImportError: 加载模型失败
    """
    if ai_name not in _AI_MODEL_MAP:
        available = ", ".join(_AI_MODEL_MAP.keys())
        raise ValueError(f"未知的 AI 名称: '{ai_name}'，可用的 AI: {available}")
    
    # 动态导入模型类
    module_path, class_name = _AI_MODEL_MAP[ai_name].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    return model_class()


def get_color_name(color: int) -> str:
    """
    将棋子颜色数字转换为名称
    
    Args:
        color: 1=黑, 2=白
        
    Returns:
        颜色名称字符串
    """
    return "黑" if color == 1 else "白"


def get_winner_name(winner: int) -> str:
    """
    将获胜方数字转换为描述字符串
    
    Args:
        winner: 1=黑胜, 2=白胜, 3=平局
        
    Returns:
        描述字符串
    """
    if winner == 1:
        return "黑方胜"
    elif winner == 2:
        return "白方胜"
    elif winner == 3:
        return "平局"
    else:
        return "未结束"


def print_battle_results(ai1_name: str, ai2_name: str, results: list):
    """
    打印对战结果统计
    
    Args:
        ai1_name: AI1 名称
        ai2_name: AI2 名称
        results: start_ai_vs_ai_random 返回的结果列表
    """
    total = len(results)
    ai1_wins = sum(1 for r in results if r["ai1_won"])
    ai2_wins = sum(1 for r in results if r["ai2_won"])
    draws = sum(1 for r in results if r["winner"] == 3)
    
    print(f"\n=== 对战结果 ===")
    print(f"总局数: {total}")
    
    ai1_rate = ai1_wins / total * 100 if total > 0 else 0.0
    ai2_rate = ai2_wins / total * 100 if total > 0 else 0.0
    draw_rate = draws / total * 100 if total > 0 else 0.0
    
    print(f"{ai1_name} 胜: {ai1_wins} 局 ({ai1_rate:.1f}%)")
    print(f"{ai2_name} 胜: {ai2_wins} 局 ({ai2_rate:.1f}%)")
    print(f"平局: {draws} 局 ({draw_rate:.1f}%)")


def run_battle(ai1_name: str, ai2_name: str, num_games: int):
    """
    执行 AI 对战主流程
    
    Args:
        ai1_name: AI1 名称
        ai2_name: AI2 名称
        num_games: 对战局数
    """
    print(f"开始 AI 对战: {ai1_name} vs {ai2_name}, 共 {num_games} 局")
    print()
    
    # 加载模型
    model1 = load_model(ai1_name)
    model2 = load_model(ai2_name)
    
    # 创建引擎
    engine1 = AIEngine(model1)
    engine2 = AIEngine(model2)
    
    # 创建数据记录器，使用组合 AI 名称作为轮次目录前缀
    round_name = f"{ai1_name}_vs_{ai2_name}"
    # 使用常量定义的 AI 数据目录
    recorder = DataRecorder(round_name, AI_DATA_DIR)
    
    # 执行对战（每局执方随机）
    results = AIGameManager.start_ai_vs_ai_random(engine1, engine2, num_games, recorder)
    
    # 逐局输出结果
    for r in results:
        game_num = r["game"]
        ai1_color = get_color_name(r["ai1_color"])
        ai2_color = get_color_name(r["ai2_color"])
        winner_desc = get_winner_name(r["winner"])
        print(f"第 {game_num} 局: {ai1_name}({ai1_color}) vs {ai2_name}({ai2_color}) -> {winner_desc}")
    
    # 输出统计结果
    print_battle_results(ai1_name, ai2_name, results)
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="AI 对战脚本 - 启动两个 AI 内战，每局执方随机",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # random vs random，对战 10 局
  python ai/battle.py --ai1 random --ai2 random --games 10

  # random vs fc_v1，对战 50 局
  python ai/battle.py --ai1 random --ai2 fc_v1 --games 50

  # cnn_v1 vs fc_v1，对战 10 局
  python ai/battle.py --ai1 cnn_v1 --ai2 fc_v1 --games 10

  # fc_v1 vs cnn_v2，对战 50 局
  python ai/battle.py --ai1 fc_v1 --ai2 cnn_v2 --games 50
        """
    )
    
    parser.add_argument(
        "--ai1",
        type=str,
        required=True,
        help="第一方 AI 名称（如 random, fc_v1, cnn_v1, cnn_v2)"
    )

    parser.add_argument(
        "--ai2",
        type=str,
        required=True,
        help="第二方 AI 名称（如 random, fc_v1, cnn_v1, cnn_v2)"
    )
    
    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="对战局数，默认 10"
    )
    
    args = parser.parse_args()
    
    # 校验局数
    if args.games <= 0:
        parser.error(f"对战局数必须大于 0，当前值: {args.games}")
    
    try:
        run_battle(args.ai1, args.ai2, args.games)
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"对战失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
