"""AI 对战脚本

独立运行的命令行脚本，用于启动两个 AI 对战。
支持通过参数指定双方 AI、对战局数及温度采样行为。
每一局执方随机（黑白随机分配）。对战结束后输出每局结果及双方胜率。

用法:
    python ai/battle.py --ai1 random --ai2 fc_v1 --games 10
    python ai/battle.py --ai1 cnn_v3.1 --ai2 cnn_v3.2 --games 50 --no-temp-sampling
    python ai/battle.py --ai1 fc_v1 --ai2 cnn_v2 --games 20 --temperature 2.0 --temp-moves 6
    python ai/battle.py --ai1 cnn_v3.1 --ai2 cnn_v3.2 --games 10 --temp-seed 42
"""

import sys
import os
import math
import argparse
import warnings

# 添加项目根目录到路径，并移除 ai/ 目录（防止遮蔽标准库 random 模块）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ai_dir = os.path.dirname(os.path.abspath(__file__))
# 优先移除 ai/ 目录，再插入项目根目录
for _p in list(sys.path):
    if os.path.normcase(os.path.abspath(_p)) == os.path.normcase(_ai_dir):
        sys.path.remove(_p)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from typing import Optional

from ai.engine import AIEngine, SinglePolicyAiEngine
from ai.data_recorder import DataRecorder
from ai.game_manager import AIGameManager
from ai.constants import AI_DATA_DIR
from ai.registry import get_model
from ai.move_sampler import TempConfig, SUPPORTED_TEMP_SIDES, TEMP_SIDE_BOTH


def finite_float(value: str) -> float:
    """argparse 自定义 type 函数：将字符串转为有限浮点数。

    argparse 内置 ``type=float`` 会接受 ``inf`` / ``nan`` 等特殊浮点值，
    但温度采样要求 temperature 必须是有限正实数。本函数在 ``float(value)``
    成功后额外校验 ``math.isfinite``，拒绝 inf / -inf / nan。

    Args:
        value: 命令行参数字符串（如 "1.0"、"inf"）

    Returns:
        解析后的有限浮点数

    Raises:
        argparse.ArgumentTypeError: 值为 inf / -inf / nan 时
    """
    f = float(value)
    if not math.isfinite(f):
        raise argparse.ArgumentTypeError(
            f"temperature 必须是有限浮点数，收到: {value}"
        )
    return f


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


def run_battle(ai1_name: str, ai2_name: str, num_games: int,
               temp_config: Optional[TempConfig] = None):
    """
    执行 AI 对战主流程

    Args:
        ai1_name: AI1 名称
        ai2_name: AI2 名称
        num_games: 对战局数
        temp_config: 可选的温度采样配置；None 时透传给 AIGameManager，
            底层解释为全程 argmax。默认启用温度采样由 main() 保证。
    """
    print(f"开始 AI 对战: {ai1_name} vs {ai2_name}, 共 {num_games} 局")
    print()
    
    # 加载模型（直接调用 registry.get_model，根据 AI 名称获取对应模型实例）
    model1 = get_model(ai1_name)
    model2 = get_model(ai2_name)

    # 将模型网络移到 GPU（若可用），加速对弈推理
    if model1.network is not None:
        model1.move_to_device()
    if model2.network is not None:
        model2.move_to_device()
    
    # 创建引擎
    engine1 = SinglePolicyAiEngine(model1)
    engine2 = SinglePolicyAiEngine(model2)
    
    # 创建数据记录器，使用组合 AI 名称作为轮次目录前缀
    round_name = f"{ai1_name}_vs_{ai2_name}"
    # 使用常量定义的 AI 数据目录
    recorder = DataRecorder(round_name, AI_DATA_DIR)
    
    # 执行对战（每局执方随机），透传 temp_config（不做语义转换）
    results = AIGameManager.start_ai_vs_ai_random(
        engine1, engine2, num_games, recorder, temp_config=temp_config)
    
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


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    提取为独立函数便于单元测试直接测试参数解析行为。

    Returns:
        配置完成的 ArgumentParser 实例
    """
    parser = argparse.ArgumentParser(
        description="AI 对战脚本 - 启动两个 AI 内战，每局执方随机",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # random vs random，对战 10 局（默认启用温度采样）
  python ai/battle.py --ai1 random --ai2 random --games 10

  # random vs fc_v1，对战 50 局
  python ai/battle.py --ai1 random --ai2 fc_v1 --games 50

  # cnn_v1 vs fc_v1，对战 10 局
  python ai/battle.py --ai1 cnn_v1 --ai2 fc_v1 --games 10

  # fc_v1 vs cnn_v2，对战 50 局
  python ai/battle.py --ai1 fc_v1 --ai2 cnn_v2 --games 50

  # cnn_v3.1 vs cnn_v3.2（多代模型），对战 50 局
  python ai/battle.py --ai1 cnn_v3.1 --ai2 cnn_v3.2 --games 50

  # 禁用温度采样，全程 argmax
  python ai/battle.py --ai1 cnn_v3.1 --ai2 cnn_v3.2 --games 50 --no-temp-sampling

  # 自定义温度值
  python ai/battle.py --ai1 fc_v1 --ai2 cnn_v2 --games 20 --temperature 2.0

  # 固定随机种子复现对战
  python ai/battle.py --ai1 cnn_v3.1 --ai2 cnn_v3.2 --games 10 --temp-seed 42

  # 组合使用
  python ai/battle.py --ai1 cnn_v3.1 --ai2 cnn_v3.2 --games 100 --temperature 0.8 --temp-moves 8 --temp-side random --temp-seed 2026
        """
    )

    parser.add_argument(
        "--ai1",
        type=str,
        required=True,
        help="第一方 AI 指定（如 random, fc_v1, cnn_v3.1)"
    )

    parser.add_argument(
        "--ai2",
        type=str,
        required=True,
        help="第二方 AI 指定（如 random, fc_v1, cnn_v3.1)"
    )

    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="对战局数，默认 10"
    )

    # ---- 温度采样参数（默认启用，与 create_samples.py 对齐）----
    parser.add_argument(
        "--no-temp-sampling",
        action="store_true",
        default=False,
        help="禁用开局阶段温度采样，全程走 argmax"
    )
    parser.add_argument(
        "--temperature",
        type=finite_float,
        default=1.0,
        help="温度值 T（>0），越大越随机，默认 1.0"
    )
    parser.add_argument(
        "--temp-moves",
        type=int,
        default=12,
        help="前 N 个全局步启用温度采样，之后走 argmax，默认 12"
    )
    parser.add_argument(
        "--temp-side",
        type=str,
        default=TEMP_SIDE_BOTH,
        choices=list(SUPPORTED_TEMP_SIDES),
        help="启用温度采样的一方，默认 both"
    )
    parser.add_argument(
        "--temp-seed",
        type=int,
        default=None,
        help="温度采样随机种子（用于复现对战结果）"
    )

    return parser


def _build_temp_config(args: argparse.Namespace) -> Optional[TempConfig]:
    """从 argparse 解析结果构造 TempConfig 或 None。

    提取为独立函数便于单元测试。

    Args:
        args: argparse 解析后的 Namespace

    Returns:
        TempConfig 实例（启用温度采样时）或 None（禁用时）

    Raises:
        ValueError: temperature 或 temp_moves 非法时（由 TempConfig.__post_init__ 抛出）
    """
    if args.no_temp_sampling:
        # 若用户显式传了温度相关参数，通过 warning 提示被忽略
        # （行为与 create_samples.py 保持一致）
        user_set_temp = (
            args.temperature != 1.0
            or args.temp_moves != 12
            or args.temp_side != TEMP_SIDE_BOTH
            or args.temp_seed is not None
        )
        if user_set_temp:
            warnings.warn(
                "已启用 --no-temp-sampling，以下参数被忽略："
                "--temperature / --temp-moves / --temp-side / --temp-seed"
            )
        return None
    else:
        return TempConfig(
            temperature=args.temperature,
            temp_moves=args.temp_moves,
            temp_side=args.temp_side,
            top_k=None,
            seed=args.temp_seed,
        )


def main():
    """主函数"""
    parser = build_parser()
    args = parser.parse_args()

    # 校验局数
    if args.games <= 0:
        parser.error(f"对战局数必须大于 0，当前值: {args.games}")

    # TempConfig 构造必须在 try 块之内，确保 __post_init__ 的 ValueError 被捕获
    try:
        temp_config = _build_temp_config(args)
        run_battle(args.ai1, args.ai2, args.games, temp_config=temp_config)
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"对战失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
