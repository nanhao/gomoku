"""自我对弈训练周期模块

将 BaseTrainer.train_cycle 提取为独立类，消除循环依赖。
对弈 + 评价逻辑委托给 run_self_play_and_evaluate，本模块只负责训练编排。
"""

import os
import uuid
from datetime import datetime

from ai.constants import AI_DATA_DIR, TRAIN_DATA_DIR, LOSS_LOG_DIR
from ai.loss_logger import LossLogger
from ai.self_play_generator import run_self_play_and_evaluate


class SelfPlayCycle:
    """自我对弈训练周期
    
    让模型自己跟自己对战 n 个轮次，每个轮次 m 场对局。
    每个轮次结束后，将本轮所有 sample 数据合并，调用 train_batch 训练模型。
    对弈 + 评价步骤委托给 run_self_play_and_evaluate。
    """
    
    def __init__(self, trainer, max_workers: int = None):
        """初始化

        Args:
            trainer: BaseTrainer 实例，提供 model、model_name、loss_logger、train_batch 等方法
            max_workers: 评价阶段并行进程数，None 表示使用默认值
        """
        self.trainer = trainer
        self.max_workers = max_workers
    
    def run(
        self,
        rounds: int = 10,
        games: int = 100,
        epochs: int = 1000,
        rate: float = 0.9,
        batch_size: int = None,
        shuffle: bool = False,
        project_root: str = None,
        val_ratio: float = 0.0
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
            val_ratio: 验证集比例，0 表示不分割，默认 0.0
        """
        # 确定数据目录：默认使用常量，测试隔离时基于 project_root 拼接
        if project_root is None:
            # 正常运行：直接使用常量
            ai_data_dir = AI_DATA_DIR
            train_data_dir = TRAIN_DATA_DIR
        else:
            # 测试隔离：基于 project_root 拼接路径
            ai_data_dir = os.path.join(project_root, "ai", "ai_data")
            train_data_dir = os.path.join(project_root, "ai", "train_data")

        # 确保目录存在
        os.makedirs(ai_data_dir, exist_ok=True)
        os.makedirs(train_data_dir, exist_ok=True)

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
        print(f"Loss 日志目录: {os.path.join(LOSS_LOG_DIR, session_dir)}")
        print(f"数据保存目录: {ai_data_dir}")
        print(f"训练数据目录: {train_data_dir}")
        print("-" * 50)
        
        interrupted = False

        try:
            for round_num in range(1, rounds + 1):
                print(f"\n===== 第 {round_num}/{rounds} 轮 =====")

                # 1-3. 自我对弈 + 评价（委托给公共函数）
                result = run_self_play_and_evaluate(
                    model=self.trainer.model,
                    games=games,
                    rate=rate,
                    ai_data_dir=ai_data_dir,
                    train_data_dir=train_data_dir,
                    max_workers=self.max_workers,
                    name_prefix=f"cycle_{round_num}",
                    project_root=project_root
                )

                # 检测对弈阶段是否被中断
                if result["interrupted"]:
                    interrupted = True
                    break

                sample_dirs = result["sample_dirs"]

                # 4. 自动选择训练方法（提供 batch_size 时使用 mini-batch，否则全量更新）
                #    train_auto 内部会创建自己的 LossLogger（内层 epoch 级日志）
                if sample_dirs:
                    print(f"训练模型 ({epochs} epochs，共 {len(sample_dirs)} 个样本合并训练)...")
                    avg_loss = self.trainer.train_auto(
                        sample_dirs,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        verbose=True,
                        session_dir=session_dir,
                        log_type="epoch_level",
                        round_num=round_num,
                        total_rounds=rounds,
                        dense_log=True,
                        val_ratio=val_ratio
                    )
                    print(f"本轮平均损失: {avg_loss:.4f}")

                    # 记录每轮的 avg_loss（外层 round 级日志）
                    if loss_logger:
                        loss_logger.log(round_num, avg_loss)

                # 5. 保存更新后的模型
                model_path = self.trainer.model.save()
                print(f"模型已保存: {model_path}")
                print("-" * 50)
        except KeyboardInterrupt:
            interrupted = True
            print("\n[中断] 检测到 Ctrl+C，正在保存当前模型和图表...")

            # 保存模型
            try:
                model_path = self.trainer.model.save()
                print(f"模型已保存: {model_path}")
            except Exception as e:
                print(f"模型保存失败: {e}")

            # flush 并生成图表
            if loss_logger:
                try:
                    loss_logger.flush()
                    plot_path = loss_logger.plot()
                    if plot_path:
                        print(f"Loss 曲线图已生成: {plot_path}")
                except Exception as e:
                    print(f"图表生成失败: {e}")

        # 生成 loss 曲线图（外层轮次粒度）
        if not interrupted and loss_logger:
            loss_logger.plot()

        # 生成合并对比图（所有内层 epoch 级数据合并到一张图）
        if session_dir:
            comparison_path = LossLogger.plot_comparison(
                session_dir=session_dir,
                model_name=self.trainer.model_name
            )
            if comparison_path:
                print(f"合并对比图已生成: {comparison_path}")

        if interrupted:
            print("[中断] 已保存当前进度，可以安全退出")
        else:
            print(f"\n训练周期完成！共 {rounds} 轮，每轮 {games} 局")
