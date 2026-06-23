"""评价器模块

从 AI 对战记录生成训练样本。
输入：ai/ai_data/.../record.json
输出：ai/train_data/.../sample.json

使用棋型评估法计算每个位置的 reward 值，
并通过 masked_softmax 转换为概率分布。

支持多进程并行评价（ProcessPoolExecutor），
在 Windows spawn 模式下使用顶层函数 _evaluate_record_worker。
"""

import os
import json
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

from ai.constants import AI_DATA_DIR, TRAIN_DATA_DIR
from ai.reward_calculator import RewardCalculator, RewardConfig
from ai.probability_utils import masked_softmax


def _evaluate_record_worker(record_file: str, ai_data_dir: str,
                          train_data_dir: str) -> str:
    """顶层辅助函数，供 ProcessPoolExecutor 调用

    Windows 使用 spawn 模式创建子进程，子进程会重新导入模块。
    实例方法无法被 pickle，因此需要一个顶层函数作为 worker。
    每个子进程创建独立的 Evaluator 实例，避免进程间共享状态。

    Args:
        record_file: record.json 文件路径
        ai_data_dir: AI 数据目录路径
        train_data_dir: 训练数据目录路径

    Returns:
        生成的 sample.json 文件路径
    """
    evaluator = Evaluator(
        ai_data_dir=ai_data_dir,
        train_data_dir=train_data_dir
    )
    return evaluator.evaluate_record(record_file)


class Evaluator:
    """评价器 - 生成训练样本（棋型评估法）"""

    def __init__(self, ai_data_dir: str = None, train_data_dir: str = None,
                 max_workers: int = None):
        """
        初始化评价器

        创建 RewardConfig 作为唯一的奖励配置来源，
        RewardCalculator 从该配置获取 max_reward / min_reward。

        Args:
            ai_data_dir: AI 数据目录路径，默认为 AI_DATA_DIR
            train_data_dir: 训练数据目录路径，默认为 TRAIN_DATA_DIR
            max_workers: 并行进程数，1 表示串行，默认 min(os.cpu_count(), 4)
        """
        if ai_data_dir is None:
            ai_data_dir = AI_DATA_DIR
        if train_data_dir is None:
            train_data_dir = TRAIN_DATA_DIR
        self.ai_data_dir = ai_data_dir
        self.train_data_dir = train_data_dir

        # 奖励配置：单一来源，RewardCalculator 从此获取
        self._reward_config = RewardConfig()

        # 并行进程数：默认 min(cpu_count, 4)，1 表示串行
        if max_workers is not None:
            self.max_workers = max_workers
        else:
            self.max_workers = min(os.cpu_count() or 4, 4)

        self._reward_calculator = RewardCalculator(self._reward_config)
    
    def evaluate_record(self, record_file: str) -> str:
        """
        评价单个记录文件，生成 sample.json
        
        Args:
            record_file: record.json 文件路径
            
        Returns:
            生成的 sample.json 文件路径
        """
        with open(record_file, 'r', encoding='utf-8') as f:
            record = json.load(f)
        
        history = record.get("history", [])
        winner = record.get("winner", 0)

        if not history:
            raise ValueError("记录文件无效：history 为空")
        if winner == 0:
            raise ValueError("游戏未结束或平局：winner 为 0")

        # 计算每个 step 的 reward（棋型评估）
        sample_dict = self._compute_all_rewards(history)
        
        # 生成输出文件路径
        sample_file = self._get_sample_path(record_file)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(sample_file), exist_ok=True)
        
        # 性能优化：去掉 indent=2，紧凑格式减少文件大小约50%
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_dict, f, ensure_ascii=False)
        
        return sample_file
    
    def _compute_all_rewards(self, history: List[Dict]) -> List[Dict]:
        """计算所有步骤的 reward（棋型评估）

        对每个 step 的每个空位，计算其 reward 值，
        然后通过 masked_softmax 转换为概率分布。

        Args:
            history: 历史记录

        Returns:
            包含 input, output, mask 的样本列表
            output 为概率分布，值域 (0, 1)，总和为 1（由 masked_softmax 生成）
        """
        sample_dict = []

        for step_idx, step in enumerate(history):
            input_vector = step.get("input", [])
            if len(input_vector) < 226:
                continue
            
            # 获取棋盘状态（前 225 位）和当前落子方
            board = input_vector[:225]
            current_player = input_vector[225]
            
            # 初始化 mask 和 raw_rewards
            mask = [0] * 225
            raw_rewards = [0.0] * 225
            
            # 对每个位置计算 reward
            for i in range(225):
                if board[i] != 0:
                    # 已落子位置：mask = 0, reward = 0
                    mask[i] = 0
                    raw_rewards[i] = 0.0
                else:
                    # 空位：mask = 1，计算 reward
                    mask[i] = 1
                    # 使用 RewardCalculator 计算位置 reward
                    x, y = i // 15, i % 15
                    raw_rewards[i] = float(self._reward_calculator.compute_position_reward(board, x, y, current_player))

            # 对 raw_rewards 应用 masked_softmax 转换为概率分布
            output = masked_softmax(raw_rewards, mask)
            
            sample_dict.append({
                "input": input_vector,
                "output": output,
                "mask": mask
            })
        
        return sample_dict
    
    def _get_sample_path(self, record_file: str) -> str:
        """
        根据 record.json 路径生成对应的 sample.json 路径
        
        将 ai_data 替换为 train_data
        """
        rel_path = os.path.relpath(record_file, self.ai_data_dir)
        sample_file = os.path.join(self.train_data_dir, rel_path)
        sample_file = sample_file.replace("record.json", "sample.json")
        
        return sample_file
    
    def evaluate_round(self, round_dir: str, max_workers: int = None) -> dict:
        """
        评价整个轮次的所有对局，支持多进程并行

        当 max_workers=1 时退化为串行执行，与原始行为完全一致。
        当 max_workers>1 时使用 ProcessPoolExecutor 并行评价。

        Args:
            round_dir: 轮次目录路径（如 ai_data/random_1）
            max_workers: 本次调用的进程数，覆盖实例的 max_workers

        Returns:
            dict: {"sample_files": [成功生成的 sample 路径],
                   "errors": [(record_file, exception), ...]}
        """
        if max_workers is None:
            max_workers = self.max_workers

        # 收集所有有效的 record 文件路径
        record_files = []
        if not os.path.exists(round_dir):
            return {"sample_files": [], "errors": []}

        for game_dir in sorted(os.listdir(round_dir)):
            game_path = os.path.join(round_dir, game_dir)
            record_file = os.path.join(game_path, "record.json")
            if os.path.isdir(game_path) and os.path.exists(record_file):
                record_files.append(record_file)

        # 委托给 _evaluate_files 执行
        return self._evaluate_files(record_files, max_workers)

    def _evaluate_serial(self, record_files: List[str],
                         sample_files: List[str], errors: List) -> None:
        """串行评价多个记录文件，逐个调用 evaluate_record

        评价成功的 sample 路径追加到 sample_files，
        评价失败的 (record_file, exception) 追加到 errors。
        单个文件失败不影响后续文件的评价。

        Args:
            record_files: record.json 文件路径列表
            sample_files: 成功结果收集列表，会被原地修改
            errors: 失败结果收集列表，会被原地修改
        """
        for record_file in record_files:
            try:
                sample_file = self.evaluate_record(record_file)
                sample_files.append(sample_file)
            except Exception as e:
                errors.append((record_file, e))
                print(f"评价失败 {record_file}: {e}")

    def _evaluate_files(self, record_files: List[str],
                         max_workers: int = None) -> dict:
        """评价多个记录文件，支持串行和并行模式

        串行模式（max_workers=1）：逐个调用 evaluate_record，简单可靠。
        并行模式（max_workers>1）：使用 ProcessPoolExecutor 并行评价，
            每个子进程通过 _evaluate_record_worker 创建独立 Evaluator。

        Args:
            record_files: record.json 文件路径列表
            max_workers: 进程数，默认使用实例的 max_workers

        Returns:
            dict: {"sample_files": [成功生成的 sample 路径],
                   "errors": [(record_file, exception), ...]}
        """
        if max_workers is None:
            max_workers = self.max_workers

        sample_files = []
        errors = []

        # 串行模式：max_workers=1 或只有一个文件时直接串行
        if max_workers <= 1 or len(record_files) <= 1:
            self._evaluate_serial(record_files, sample_files, errors)
            return {"sample_files": sample_files, "errors": errors}

        # 并行模式：使用 ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务，每个任务调用顶层 worker 函数
            future_to_file = {
                executor.submit(
                    _evaluate_record_worker,
                    record_file,
                    self.ai_data_dir,
                    self.train_data_dir
                ): record_file
                for record_file in record_files
            }

            # 收集结果，个别失败不影响其他任务
            broken_pool = False
            for future in as_completed(future_to_file):
                record_file = future_to_file[future]
                try:
                    sample_file = future.result()
                    sample_files.append(sample_file)
                except BrokenProcessPool as e:
                    # Worker 进程被杀死（如 OOM），回退到串行模式
                    errors.append((record_file, e))
                    print(f"进程池崩溃（可能是内存不足），回退到串行模式: {e}")
                    broken_pool = True
                    break
                except Exception as e:
                    errors.append((record_file, e))
                    print(f"评价失败 {record_file}: {e}")

        # 如果进程池崩溃，回退到串行模式重试剩余文件
        if broken_pool:
            # 用未成功的 record_files 减去已成功的
            successful_record_files = set()
            for sf in sample_files:
                # sample_file 路径推导出 record_file
                rel = os.path.relpath(sf, self.train_data_dir)
                rf = os.path.join(self.ai_data_dir, rel.replace("sample.json", "record.json"))
                successful_record_files.add(rf)
            remaining_files = [f for f in record_files if f not in successful_record_files]

            print(f"串行模式处理剩余 {len(remaining_files)} 个文件...")
            self._evaluate_serial(remaining_files, sample_files, errors)

        return {"sample_files": sample_files, "errors": errors}
