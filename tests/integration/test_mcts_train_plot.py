"""MCTS 训练损失曲线图集成测试

测试训练完成后自动生成双 Y 轴损失曲线图。
"""

import sys
import os
import unittest
import tempfile
import shutil
import json

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch


class TestMCTSTrainingPlot(unittest.TestCase):
    """训练损失曲线图集成测试"""

    @classmethod
    def setUpClass(cls):
        """导入模块"""
        try:
            from ai.dual_head_trainer import DualHeadTrainer
            cls.DualHeadTrainer = DualHeadTrainer
        except ImportError:
            cls.DualHeadTrainer = None
        try:
            from ai.models.cnn_v7.trainer import CNNV7Trainer
            cls.CNNV7Trainer = CNNV7Trainer
        except ImportError:
            cls.CNNV7Trainer = None

    def setUp(self):
        if self.DualHeadTrainer is None:
            self.skipTest("DualHeadTrainer 尚未实现")
        self.temp_dir = tempfile.mkdtemp(prefix="mcts_plot_test_")
        self.loss_log_dir = tempfile.mkdtemp(prefix="mcts_loss_logs_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutil.rmtree(self.loss_log_dir, ignore_errors=True)

    @staticmethod
    def _create_sample_data(output_dir, n_samples=20):
        """创建测试用 sample.json"""
        samples = []
        for i in range(n_samples):
            board = [0] * 225
            board[i % 225] = 1 if i % 3 != 0 else 2
            board[(i * 7 + 11) % 225] = 2 if i % 3 == 0 else 1
            state = board + [1 if i % 2 == 0 else 2]
            pi = [0.0] * 225
            empty_positions = [j for j, v in enumerate(board) if v == 0]
            if empty_positions:
                prob = 1.0 / len(empty_positions)
                for pos in empty_positions:
                    pi[pos] = prob
            else:
                pi = [1.0 / 225] * 225
            z = 1 if i % 3 == 0 else (-1 if i % 3 == 1 else 0)
            samples.append({"state": state, "pi": pi, "z": z})

        os.makedirs(output_dir, exist_ok=True)
        sample_path = os.path.join(output_dir, "sample.json")
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f)
        return sample_path

    def test_plot_generated_after_training(self):
        """训练完成后应生成损失曲线图 PNG 文件"""
        self._create_sample_data(self.temp_dir, n_samples=16)

        # 从单元测试获取具体训练器
        from tests.unit.test_dual_head_trainer import (
            MockDualHeadModel, _make_concrete_trainer,
        )
        ConcreteTrainer = _make_concrete_trainer(self.DualHeadTrainer)
        model = MockDualHeadModel()
        trainer = ConcreteTrainer(model=model)

        # 修改输出目录配置指向临时目录
        import src.config.mcts as mcts_config
        original_dir = mcts_config.MCTS_LOSS_LOG_DIR
        try:
            mcts_config.MCTS_LOSS_LOG_DIR = self.loss_log_dir

            history = trainer.train_epochs(
                sample_dirs=[self.temp_dir],
                epochs=2,
                batch_size=8,
                val_split=0.0,
                shuffle=False,
                verbose=False,
            )

            # 验证训练完成
            self.assertGreater(len(history), 0)
            self.assertIn("grad_norm", history[0])

            # 验证图表文件已生成
            png_files = [f for f in os.listdir(self.loss_log_dir)
                         if f.endswith('.png')]
            self.assertEqual(len(png_files), 1,
                             f"应在 {self.loss_log_dir} 下生成 1 个 PNG 文件，"
                             f"实际发现: {png_files}")
        finally:
            mcts_config.MCTS_LOSS_LOG_DIR = original_dir

    def test_plot_generated_with_val_split(self):
        """有验证集时训练完成后应生成图表（包含 val_total_loss 数据）"""
        self._create_sample_data(self.temp_dir, n_samples=24)

        from tests.unit.test_dual_head_trainer import (
            MockDualHeadModel, _make_concrete_trainer,
        )
        ConcreteTrainer = _make_concrete_trainer(self.DualHeadTrainer)
        model = MockDualHeadModel()
        trainer = ConcreteTrainer(model=model)

        import src.config.mcts as mcts_config
        original_dir = mcts_config.MCTS_LOSS_LOG_DIR
        try:
            mcts_config.MCTS_LOSS_LOG_DIR = self.loss_log_dir

            history = trainer.train_epochs(
                sample_dirs=[self.temp_dir],
                epochs=2,
                batch_size=8,
                val_split=0.25,
                shuffle=False,
                verbose=False,
            )

            self.assertGreater(len(history), 0)
            # 验证 history 包含验证集数据
            self.assertIn("val_total_loss", history[0])

            # 验证图表文件已生成
            png_files = [f for f in os.listdir(self.loss_log_dir)
                         if f.endswith('.png')]
            self.assertEqual(len(png_files), 1)
        finally:
            mcts_config.MCTS_LOSS_LOG_DIR = original_dir

    def test_plot_not_generated_when_matplotlib_unavailable(self):
        """matplotlib 不可用时训练正常完成，不生成图表"""
        from unittest.mock import patch

        self._create_sample_data(self.temp_dir, n_samples=16)

        from tests.unit.test_dual_head_trainer import (
            MockDualHeadModel, _make_concrete_trainer,
        )
        ConcreteTrainer = _make_concrete_trainer(self.DualHeadTrainer)
        model = MockDualHeadModel()
        trainer = ConcreteTrainer(model=model)

        # 模拟 matplotlib 不可用：mock 模块级标志
        # 切换 MCTS_LOSS_LOG_DIR 到临时目录，避免污染项目目录
        import src.config.mcts as mcts_config
        original_dir = mcts_config.MCTS_LOSS_LOG_DIR
        mcts_config.MCTS_LOSS_LOG_DIR = self.loss_log_dir
        try:
            with patch('ai.dual_head_trainer._MATPLOTLIB_AVAILABLE', False):
                # 训练仍应正常完成
                history = trainer.train_epochs(
                    sample_dirs=[self.temp_dir],
                    epochs=2,
                    batch_size=8,
                    val_split=0.0,
                    shuffle=False,
                    verbose=False,
                )
                self.assertGreater(len(history), 0)

            # 无 matplotlib 时不应生成 PNG
            png_files = [f for f in os.listdir(self.loss_log_dir)
                         if f.endswith('.png')]
            self.assertEqual(len(png_files), 0,
                             "matplotlib 不可用时应无 PNG 生成")
        finally:
            mcts_config.MCTS_LOSS_LOG_DIR = original_dir

    def test_plot_output_directory_auto_created(self):
        """输出目录不存在时自动创建"""
        self._create_sample_data(self.temp_dir, n_samples=16)

        from tests.unit.test_dual_head_trainer import (
            MockDualHeadModel, _make_concrete_trainer,
        )
        ConcreteTrainer = _make_concrete_trainer(self.DualHeadTrainer)
        model = MockDualHeadModel()
        trainer = ConcreteTrainer(model=model)

        nested_dir = os.path.join(self.temp_dir, "nested", "loss_logs")

        import src.config.mcts as mcts_config
        original_dir = mcts_config.MCTS_LOSS_LOG_DIR
        try:
            mcts_config.MCTS_LOSS_LOG_DIR = nested_dir

            history = trainer.train_epochs(
                sample_dirs=[self.temp_dir],
                epochs=1,
                batch_size=8,
                val_split=0.0,
                shuffle=False,
                verbose=False,
            )

            self.assertGreater(len(history), 0)
            # 验证嵌套目录已创建
            self.assertTrue(os.path.isdir(nested_dir),
                            f"嵌套目录应在训练后自动创建: {nested_dir}")
        finally:
            mcts_config.MCTS_LOSS_LOG_DIR = original_dir

    def test_cnn_v7_training_produces_plot(self):
        """CNNV7Trainer 使用真实模型训练后应生成图表"""
        if self.CNNV7Trainer is None:
            self.skipTest("CNNV7Trainer 尚未实现")

        self._create_sample_data(self.temp_dir, n_samples=16)

        trainer = self.CNNV7Trainer(fresh=True)

        import src.config.mcts as mcts_config
        original_dir = mcts_config.MCTS_LOSS_LOG_DIR
        try:
            mcts_config.MCTS_LOSS_LOG_DIR = self.loss_log_dir

            history = trainer.train_epochs(
                sample_dirs=[self.temp_dir],
                epochs=2,
                batch_size=8,
                val_split=0.0,
                shuffle=False,
                verbose=False,
            )

            self.assertGreater(len(history), 0)
            self.assertIn("grad_norm", history[0])

            # 验证图表文件已生成
            png_files = [f for f in os.listdir(self.loss_log_dir)
                         if f.endswith('.png')]
            self.assertEqual(len(png_files), 1,
                             f"CNNV7Trainer 训练后应生成 1 个 PNG")
        finally:
            mcts_config.MCTS_LOSS_LOG_DIR = original_dir


if __name__ == "__main__":
    unittest.main()
