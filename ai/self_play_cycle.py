"""自我对弈训练周期模块

将 BaseTrainer.train_cycle 提取为独立类，消除循环依赖。
"""

import os
import uuid
from datetime import datetime

from ai.constants import AI_DATA_DIR, TRAIN_DATA_DIR
from ai.loss_logger import LossLogger
from ai.engine import AIEngine
from ai.game_manager import AIGameManager
from ai.data_recorder import DataRecorder
from ai.evaluator import Evaluator


class SelfPlayCycle:
    """自我对弈训练周期
    
    让模型自己跟自己对战 n 个轮次，每个轮次 m 场对局。
    每个轮次结束后，将本轮所有 sample 数据合并，调用 train_batch 训练模型。
    """
    
    def __init__(self, trainer):
        """初始化
        
        Args:
            trainer: BaseTrainer 实例，提供 model、model_name、loss_logger、train_batch 等方法
        """
        self.trainer = trainer
    
    def run(
        self,
        rounds: int = 10,
        games: int = 100,
        epochs: int = 1000,
        rate: float = 0.9,
        batch_size: int = None,
        shuffle: bool = False,
        project_root: str = None
    ) -> None:
        """执行训练周期

        让模型自己跟自己对战 n 个轮次，每个轮次 m 场对局。
        每个轮次结束后，将本轮所有 sample 数据合并训练模型。
        提供 batch_size 时启用 mini-batch 训练，否则使用全量更新。

        Args:
            rounds: 轮次数 n，默认 10
            games: 每轮对局数 m，默认 100
            epochs: 每轮训练的轮数，默认 1000
            rate: 奖励衰减率 r，默认 0.9，传给 Evaluator 用于生成 sample
            batch_size: 每批次样本数，提供时启用 mini-batch 训练，默认 None（全量更新）
            shuffle: 每个 epoch 是否打乱数据顺序，仅 mini-batch 模式有效，默认 False
            project_root: 项目根目录，用于确定数据保存路径
        """
        # 确定项目根目录
        if project_root is None:
            project_root = os.path.dirname(os.path.dirname(__file__))
        
        # 创建数据目录（位于 ai/ 子目录下）
        ai_data_dir = AI_DATA_DIR
        train_data_dir = TRAIN_DATA_DIR
        os.makedirs(ai_data_dir, exist_ok=True)
        os.makedirs(train_data_dir, exist_ok=True)
        
        # 创建 AI 引擎（使用当前模型）
        engine = AIEngine(self.trainer.model)

        # 生成训练会话唯一子目录名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:8]
        session_dir = f"{self.trainer.model_name}_{timestamp}_{short_uuid}"

        # 使用注入的 logger 或自动创建（带 session_dir 子目录隔离）
        if self.trainer.loss_logger is not None:
            loss_logger = self.trainer.loss_logger
        else:
            loss_logger = LossLogger(
                model_name=self.trainer.model_name,
                session_dir=session_dir,
                log_type="round_level",
                total_rounds=rounds,
                games=games
            )

        print(f"开始训练周期: {rounds} 轮，每轮 {games} 局，奖励衰减率 rate={rate}")
        print(f"Loss 日志目录: {os.path.join(AI_DATA_DIR, 'loss_logs', session_dir)}")
        print(f"数据保存目录: {ai_data_dir}")
        print(f"训练数据目录: {train_data_dir}")
        print("-" * 50)
        
        for round_num in range(1, rounds + 1):
            print(f"\n===== 第 {round_num}/{rounds} 轮 =====")
            
            # 1. 创建数据记录器
            recorder = DataRecorder(
                ai_name=f"{self.trainer.model_name}_cycle_{round_num}",
                ai_data_dir=ai_data_dir
            )
            
            # 2. 执行 AI 内战（模型自我对弈）
            print(f"执行 {games} 局自我对弈...")
            results = AIGameManager.start_ai_vs_ai(
                engine1=engine,
                engine2=engine,
                num_games=games,
                recorder=recorder,
                project_root=project_root
            )
            
            # 统计结果
            black_wins = sum(1 for r in results if r["winner"] == 1)
            white_wins = sum(1 for r in results if r["winner"] == 2)
            draws = sum(1 for r in results if r["winner"] == 3)
            print(f"对局结果: 黑胜 {black_wins}, 白胜 {white_wins}, 平局 {draws}")
            
            # 3. 调用 Evaluator 对每局生成 sample.json，收集所有 sample 目录
            print("生成训练样本...")
            evaluator = Evaluator(
                ai_data_dir=ai_data_dir,
                train_data_dir=train_data_dir,
                decay_rate=rate
            )
            
            round_dir = recorder.get_round_dir()
            sample_dirs = []
            for game_dir in os.listdir(round_dir):
                game_path = os.path.join(round_dir, game_dir)
                record_file = os.path.join(game_path, "record.json")
                if os.path.isdir(game_path) and os.path.exists(record_file):
                    try:
                        sample_file = evaluator.evaluate_record(record_file)
                        # 收集 sample 所在目录
                        sample_dirs.append(os.path.dirname(sample_file))
                    except Exception as e:
                        print(f"  评价失败 {record_file}: {e}")
            
            print(f"生成 {len(sample_dirs)} 个样本文件")
            
            # 4. 调用 train_batch 将本轮所有 sample 合并训练
            #    train_batch 内部会创建自己的 LossLogger（内层 epoch 级日志）
            if sample_dirs:
                print(f"训练模型 ({epochs} epochs，共 {len(sample_dirs)} 个样本合并训练)...")
                # 根据 batch_size 选择训练方法
                if batch_size is not None and batch_size > 0:
                    avg_loss = self.trainer.train_mini_batch(
                        sample_dirs,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        verbose=True,
                        session_dir=session_dir,
                        log_type="epoch_level",
                        round_num=round_num,
                        total_rounds=rounds,
                        dense_log=True
                    )
                else:
                    avg_loss = self.trainer.train_batch(
                        sample_dirs,
                        epochs=epochs,
                        verbose=True,
                        session_dir=session_dir,
                        log_type="epoch_level",
                        round_num=round_num,
                        total_rounds=rounds
                    )
                print(f"本轮平均损失: {avg_loss:.4f}")

                # 记录每轮的 avg_loss（外层 round 级日志）
                if loss_logger:
                    loss_logger.log(round_num, avg_loss)
            
            # 5. 保存更新后的模型
            model_path = self.trainer.model.save()
            print(f"模型已保存: {model_path}")
            print("-" * 50)
        
        # 生成 loss 曲线图（外层轮次粒度）
        if loss_logger:
            loss_logger.plot()

        # 生成合并对比图（所有内层 epoch 级数据合并到一张图）
        if session_dir:
            comparison_path = LossLogger.plot_comparison(
                session_dir=session_dir,
                model_name=self.trainer.model_name
            )
            if comparison_path:
                print(f"合并对比图已生成: {comparison_path}")
        
        print(f"\n训练周期完成！共 {rounds} 轮，每轮 {games} 局")
