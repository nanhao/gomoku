"""自我对弈数据生成器模块

仅执行自我对弈 + 评价，生成训练数据，不训练模型。
从 SelfPlayCycle 中提取数据生成部分（步骤1-3），去掉训练相关逻辑。
提供公共函数 run_self_play_and_evaluate，供 SelfPlayGenerator 和 SelfPlayCycle 复用。
"""

import os
from typing import List, Optional, TYPE_CHECKING

from ai.constants import AI_DATA_DIR, TRAIN_DATA_DIR
from ai.engine import AIEngine
from ai.game_manager import AIGameManager
from ai.data_recorder import DataRecorder
from ai.evaluator import Evaluator

if TYPE_CHECKING:
    from ai.move_sampler import TempConfig


def run_self_play_and_evaluate(
    model,
    games: int,
    rate: float = 0.9,
    ai_data_dir: str = None,
    train_data_dir: str = None,
    max_workers: int = None,
    name_prefix: str = "self_play",
    project_root: str = None,
    temp_config: Optional["TempConfig"] = None,
) -> dict:
    """执行自我对弈 + 评价，生成训练样本

    公共逻辑提取，供 SelfPlayGenerator 和 SelfPlayCycle 复用。
    封装单轮对弈 + 评价的完整流程，包含 KeyboardInterrupt 优雅处理。

    Args:
        model: BaseModel 实例，用于创建 AIEngine，需要有 name 属性
        games: 对局数
        rate: 奖励衰减率，传给 Evaluator，默认 0.9
        ai_data_dir: AI 数据目录，None 时使用常量 AI_DATA_DIR
        train_data_dir: 训练数据目录，None 时使用常量 TRAIN_DATA_DIR
        max_workers: 评价阶段并行进程数，None 表示使用默认值
        name_prefix: DataRecorder 命名前缀，最终 ai_name 为 {model.name}_{name_prefix}
        project_root: 项目根目录，传给 AIGameManager.start_ai_vs_ai

    Returns:
        dict: {
            "round_dir": str,           # ai_data 中的轮次目录
            "train_data_dir": str,      # train_data 中对应目录（映射后）
            "sample_dirs": List[str],   # 所有 sample.json 所在目录
            "sample_count": int,        # 样本文件数量
            "results": List[dict],      # 对局结果列表
            "recorder": DataRecorder,   # 数据记录器实例
            "interrupted": bool         # 是否因 KeyboardInterrupt 中断
        }
    """
    # 默认数据目录：未指定时使用模块常量
    if ai_data_dir is None:
        ai_data_dir = AI_DATA_DIR
    if train_data_dir is None:
        train_data_dir = TRAIN_DATA_DIR

    # 确保目录存在
    os.makedirs(ai_data_dir, exist_ok=True)
    os.makedirs(train_data_dir, exist_ok=True)

    # 初始化结果变量（用于 KeyboardInterrupt 时返回部分结果）
    results = []
    recorder = None

    try:
        # 1. 创建 AI 引擎
        engine = AIEngine(model)

        # 2. 创建数据记录器，ai_name 为 {model_name}_{name_prefix}
        model_name = model.name
        recorder = DataRecorder(
            ai_name=f"{model_name}_{name_prefix}",
            ai_data_dir=ai_data_dir
        )

        # 3. 执行 AI 内战（模型自我对弈）
        print(f"执行 {games} 局自我对弈...")
        results = AIGameManager.start_ai_vs_ai(
            engine1=engine,
            engine2=engine,
            num_games=games,
            recorder=recorder,
            project_root=project_root,
            temp_config=temp_config,
        )

        # 统计对局结果
        black_wins = sum(1 for r in results if r["winner"] == 1)
        white_wins = sum(1 for r in results if r["winner"] == 2)
        draws = sum(1 for r in results if r["winner"] == 3)
        print(f"对局结果: 黑胜 {black_wins}, 白胜 {white_wins}, 平局 {draws}")

        # 4. 调用 Evaluator 对每局生成 sample.json
        print("生成训练样本...")
        evaluator = Evaluator(
            ai_data_dir=ai_data_dir,
            train_data_dir=train_data_dir,
            decay_rate=rate,
            max_workers=max_workers
        )

        round_dir = recorder.get_round_dir()
        eval_result = evaluator.evaluate_round(round_dir)

        # 收集 sample 目录（sample.json 所在目录）
        sample_dirs = [os.path.dirname(f) for f in eval_result["sample_files"]]

        # 打印错误信息
        for record_file, err in eval_result["errors"]:
            print(f"  评价失败 {record_file}: {err}")

        print(f"生成 {len(sample_dirs)} 个样本文件")

        # 5. 计算训练数据目录路径（将 ai_data 路径映射为 train_data 路径）
        #    round_dir 如 /path/ai/ai_data/cnn_v1_self_play_1
        #    train_data_dir 如 /path/ai/train_data
        #    映射后为 /path/ai/train_data/cnn_v1_self_play_1
        round_dir_name = os.path.basename(round_dir)
        mapped_train_data_dir = os.path.join(train_data_dir, round_dir_name)

        return {
            "round_dir": round_dir,
            "train_data_dir": mapped_train_data_dir,
            "sample_dirs": sample_dirs,
            "sample_count": len(sample_dirs),
            "results": results,
            "recorder": recorder,
            "interrupted": False
        }

    except KeyboardInterrupt:
        # Ctrl+C 时优雅退出，打印已有结果
        print("\n[中断] 检测到 Ctrl+C，正在整理已有数据...")

        # 尝试获取 round_dir（recorder 可能尚未创建）
        if recorder is not None:
            round_dir = recorder.get_round_dir()
            round_dir_name = os.path.basename(round_dir)
            mapped_train_data_dir = os.path.join(train_data_dir, round_dir_name)
        else:
            round_dir = ""
            mapped_train_data_dir = ""

        black_wins = sum(1 for r in results if r["winner"] == 1)
        white_wins = sum(1 for r in results if r["winner"] == 2)
        draws = sum(1 for r in results if r["winner"] == 3)
        if results:
            print(f"已完成的 {len(results)} 局: 黑胜 {black_wins}, 白胜 {white_wins}, 平局 {draws}")

        print("[中断] 已保存当前进度，可以安全退出")

        return {
            "round_dir": round_dir,
            "train_data_dir": mapped_train_data_dir,
            "sample_dirs": [],
            "sample_count": 0,
            "results": results,
            "recorder": recorder,
            "interrupted": True
        }


class SelfPlayGenerator:
    """自我对弈数据生成器（仅生成数据，不训练）

    执行流程委托给 run_self_play_and_evaluate：
    1. 创建 AIEngine(model)
    2. 创建 DataRecorder(ai_name=f"{model_name}_self_play")
    3. 执行 AIGameManager.start_ai_vs_ai 生成对局数据
    4. 调用 Evaluator.evaluate_round 生成训练样本
    5. 返回结果字典（包含 round_dir、sample_dirs 等）

    与 SelfPlayCycle 的区别：
    - 不依赖 BaseTrainer，只需要 model
    - 不执行训练步骤
    - 不保存模型
    - 不记录 loss 日志
    """

    def __init__(self, model, max_workers: int = None):
        """初始化自我对弈数据生成器

        Args:
            model: BaseModel 实例，用于创建 AIEngine
                   需要具有 name 属性（如 "cnn_v1"）
            max_workers: 评价阶段并行进程数，None 表示使用默认值
        """
        self.model = model
        self.max_workers = max_workers

    def generate(self, games: int = 100, rate: float = 0.9,
                 project_root: str = None,
                 temp_config: Optional["TempConfig"] = None) -> dict:
        """执行自我对弈并生成训练样本

        让模型自己跟自己对战 games 局，然后调用 Evaluator 生成训练样本。
        不执行训练，仅产出数据。委托给 run_self_play_and_evaluate 实现。

        Args:
            games: 对局数，默认 100
            rate: 奖励衰减率，传给 Evaluator，默认 0.9
            project_root: 项目根目录，用于确定数据保存路径
            temp_config: 可选温度采样配置；None 保持原有 argmax 行为

        Returns:
            dict: {
                "round_dir": str,           # ai_data 中的轮次目录
                "train_data_dir": str,      # train_data 中对应目录
                "sample_dirs": List[str],   # 所有 sample.json 所在目录
                "sample_count": int,        # 样本文件数量
                "results": List[dict]       # 对局结果列表
            }
        """
        # 确定项目根目录
        if project_root is None:
            project_root = os.path.dirname(os.path.dirname(__file__))

        # 委托给公共函数，name_prefix 为 "self_play"
        result = run_self_play_and_evaluate(
            model=self.model,
            games=games,
            rate=rate,
            max_workers=self.max_workers,
            name_prefix="self_play",
            project_root=project_root,
            temp_config=temp_config,
        )

        # 返回结果（去掉 interrupted 和 recorder 字段，保持原有返回结构兼容）
        return {
            "round_dir": result["round_dir"],
            "train_data_dir": result["train_data_dir"],
            "sample_dirs": result["sample_dirs"],
            "sample_count": result["sample_count"],
            "results": result["results"]
        }
