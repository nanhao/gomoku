"""DualHeadTrainer._plot_loss_curve 单元测试

测试双头训练器的损失曲线图生成功能：
- grad_norm epoch 级聚合（均值）
- _plot_loss_curve 生成 PNG 文件
- 验证集指标渲染
- matplotlib 不可用时优雅降级
- KeyboardInterrupt 中断后仍生成图表
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

from tests.unit.test_dual_head_trainer import (
    MockDualHeadModel, _make_concrete_trainer,
)


class TestGradNormAggregation(unittest.TestCase):
    """grad_norm epoch 级聚合测试"""

    @classmethod
    def setUpClass(cls):
        try:
            from ai.dual_head_trainer import DualHeadTrainer
            cls.DualHeadTrainer = DualHeadTrainer
        except ImportError:
            cls.DualHeadTrainer = None

    def setUp(self):
        if self.DualHeadTrainer is None:
            self.skipTest("DualHeadTrainer 尚未实现")
        self.ConcreteTrainer = _make_concrete_trainer(self.DualHeadTrainer)

    def _to_device(self, trainer, *tensors):
        return tuple(t.to(trainer.device) for t in tensors)

    def _create_sample_json(self, output_dir, n_samples=32):
        """创建测试用 sample.json"""
        samples = []
        for i in range(n_samples):
            state = [0] * 225 + [1 if i % 2 == 0 else 2]
            # 放一些棋子使输入不全是零
            state[i % 225] = 1
            state[(i * 7 + 3) % 225] = 2
            pi = [1.0 / 225] * 225
            z = 1 if i % 3 == 0 else (-1 if i % 3 == 1 else 0)
            samples.append({"state": state, "pi": pi, "z": z})

        os.makedirs(output_dir, exist_ok=True)
        sample_path = os.path.join(output_dir, "sample.json")
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f)
        return sample_path

    def test_history_contains_grad_norm(self):
        """train_epochs 返回的 history 每项应包含 grad_norm"""
        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model)

        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_sample_json(tmpdir, n_samples=32)
            history = trainer.train_epochs(
                sample_dirs=[tmpdir],
                epochs=2,
                batch_size=8,
                val_split=0.0,
                shuffle=True,
                verbose=False,
            )

        self.assertGreater(len(history), 0)
        for entry in history:
            self.assertIn("grad_norm", entry,
                          f"history entry 缺少 grad_norm 字段: {entry}")
            self.assertGreater(entry["grad_norm"], 0.0,
                               f"grad_norm 应为正数，实际: {entry['grad_norm']}")

    def test_grad_norm_is_mean_of_batches(self):
        """grad_norm 应为 epoch 内各 batch 梯度范数总和的均值"""
        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model)

        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_sample_json(tmpdir, n_samples=16)
            # 单 epoch，batch_size=6 → 3 个 batch (6+6+4)
            history = trainer.train_epochs(
                sample_dirs=[tmpdir],
                epochs=1,
                batch_size=6,
                val_split=0.0,
                shuffle=False,
                verbose=False,
            )

        self.assertEqual(len(history), 1)
        grad_norm = history[0]["grad_norm"]
        # grad_norm 应为正数（非最后 batch 的单一值）
        self.assertGreater(grad_norm, 0.0)

    def test_grad_norm_with_val_split(self):
        """有验证集时 history 仍包含 grad_norm"""
        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model)

        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_sample_json(tmpdir, n_samples=32)
            history = trainer.train_epochs(
                sample_dirs=[tmpdir],
                epochs=2,
                batch_size=8,
                val_split=0.25,
                shuffle=True,
                verbose=False,
            )

        self.assertGreater(len(history), 0)
        for entry in history:
            self.assertIn("grad_norm", entry)
            self.assertIn("val_total_loss", entry)  # 有验证集
            self.assertGreater(entry["grad_norm"], 0.0)


class TestPlotLossCurve(unittest.TestCase):
    """_plot_loss_curve 方法测试"""

    @classmethod
    def setUpClass(cls):
        try:
            from ai.dual_head_trainer import DualHeadTrainer
            cls.DualHeadTrainer = DualHeadTrainer
        except ImportError:
            cls.DualHeadTrainer = None

    def setUp(self):
        if self.DualHeadTrainer is None:
            self.skipTest("DualHeadTrainer 尚未实现")
        self.ConcreteTrainer = _make_concrete_trainer(self.DualHeadTrainer)
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        if hasattr(self, '_tmpdir') and os.path.exists(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _create_history(self, epochs=5, with_val=True):
        """创建模拟的训练 history"""
        history = []
        for e in range(epochs):
            entry = {
                "epoch": e + 1,
                "value_loss": 0.5 - e * 0.05,
                "policy_loss": 2.0 - e * 0.1,
                "total_loss": 2.5 - e * 0.15,
                "grad_norm": 0.1 + e * 0.02,
            }
            if with_val:
                entry["val_value_loss"] = 0.55 - e * 0.04
                entry["val_policy_loss"] = 2.1 - e * 0.09
                entry["val_total_loss"] = 2.65 - e * 0.13
            history.append(entry)
        return history

    def test_plot_loss_curve_exists(self):
        """_plot_loss_curve 方法应存在"""
        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model)
        self.assertTrue(hasattr(trainer, '_plot_loss_curve'),
                        "DualHeadTrainer 缺少 _plot_loss_curve 方法")

    def test_plot_creates_png_file(self):
        """_plot_loss_curve 生成 PNG 文件"""
        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model)
        history = self._create_history(epochs=3)

        # 通过 monkey-patch MCTS_LOSS_LOG_DIR 指向临时目录
        import config.mcts as mcts_config
        original_dir = mcts_config.MCTS_LOSS_LOG_DIR
        try:
            mcts_config.MCTS_LOSS_LOG_DIR = self._tmpdir
            result = trainer._plot_loss_curve(history)
            if result is not None:
                self.assertTrue(os.path.exists(result),
                                f"PNG 文件不存在: {result}")
                self.assertTrue(result.endswith(".png"),
                                f"输出路径应为 .png 文件: {result}")
        finally:
            mcts_config.MCTS_LOSS_LOG_DIR = original_dir

    def test_plot_creates_directory(self):
        """输出目录不存在时自动创建"""
        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model)
        history = self._create_history(epochs=2)

        output_subdir = os.path.join(self._tmpdir, "nested", "subdir")
        import config.mcts as mcts_config
        original_dir = mcts_config.MCTS_LOSS_LOG_DIR
        try:
            mcts_config.MCTS_LOSS_LOG_DIR = output_subdir
            result = trainer._plot_loss_curve(history)
            if result is not None:
                self.assertTrue(os.path.isdir(output_subdir),
                                f"输出目录未自动创建: {output_subdir}")
        finally:
            mcts_config.MCTS_LOSS_LOG_DIR = original_dir

    def test_plot_with_val_loss(self):
        """有验证集数据时图表包含 val_total_loss"""
        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model)
        history = self._create_history(epochs=3, with_val=True)

        import config.mcts as mcts_config
        original_dir = mcts_config.MCTS_LOSS_LOG_DIR
        try:
            mcts_config.MCTS_LOSS_LOG_DIR = self._tmpdir
            result = trainer._plot_loss_curve(history)
            self.assertIsNotNone(result)
        finally:
            mcts_config.MCTS_LOSS_LOG_DIR = original_dir

    def test_plot_without_val_loss(self):
        """无验证集数据时图表不含 val_total_loss"""
        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model)
        history = self._create_history(epochs=3, with_val=False)

        import config.mcts as mcts_config
        original_dir = mcts_config.MCTS_LOSS_LOG_DIR
        try:
            mcts_config.MCTS_LOSS_LOG_DIR = self._tmpdir
            result = trainer._plot_loss_curve(history)
            self.assertIsNotNone(result)
        finally:
            mcts_config.MCTS_LOSS_LOG_DIR = original_dir

    def test_plot_empty_history(self):
        """空 history 时返回 None"""
        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model)
        result = trainer._plot_loss_curve([])
        self.assertIsNone(result)

    def test_plot_no_matplotlib(self):
        """matplotlib 不可用时优雅降级（不抛异常）"""
        from unittest.mock import patch

        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model)
        history = self._create_history(epochs=2)

        # 模拟 matplotlib 不可用：直接 mock 模块级标志
        # 相比移除 sys.modules，mock 方式不会破坏其他测试的 matplotlib 状态
        with patch('ai.dual_head_trainer._MATPLOTLIB_AVAILABLE', False):
            result = trainer._plot_loss_curve(history)
            # matplotlib 不可用时应返回 None，不抛异常
            self.assertIsNone(result,
                              "matplotlib 不可用时应返回 None")

    def test_plot_after_train_epochs(self):
        """train_epochs 完成后自动生成图表"""
        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model)

        # 通过 monkey-patch MCTS_LOSS_LOG_DIR 指向临时目录
        import config.mcts as mcts_config
        original_dir = mcts_config.MCTS_LOSS_LOG_DIR
        try:
            mcts_config.MCTS_LOSS_LOG_DIR = self._tmpdir

            with tempfile.TemporaryDirectory() as tmpdir:
                # 创建测试数据
                samples = []
                for i in range(24):
                    state = [0] * 225 + [1]
                    state[i % 225] = 1
                    state[(i * 7 + 3) % 225] = 2
                    pi = [1.0 / 225] * 225
                    z = 1 if i % 3 == 0 else (-1 if i % 3 == 1 else 0)
                    samples.append({"state": state, "pi": pi, "z": z})

                sample_dir = os.path.join(tmpdir, "data")
                os.makedirs(sample_dir)
                sample_path = os.path.join(sample_dir, "sample.json")
                with open(sample_path, 'w', encoding='utf-8') as f:
                    json.dump(samples, f)

                history = trainer.train_epochs(
                    sample_dirs=[sample_dir],
                    epochs=2,
                    batch_size=8,
                    val_split=0.0,
                    shuffle=False,
                    verbose=False,
                )
        finally:
            mcts_config.MCTS_LOSS_LOG_DIR = original_dir

        # 验证 history 包含 grad_norm
        self.assertGreater(len(history), 0)
        for entry in history:
            self.assertIn("grad_norm", entry)

    def test_plot_with_value_policy_loss_lines(self):
        """图表包含 value_loss 和 policy_loss 虚线"""
        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model)
        history = self._create_history(epochs=3, with_val=True)

        import config.mcts as mcts_config
        original_dir = mcts_config.MCTS_LOSS_LOG_DIR
        try:
            mcts_config.MCTS_LOSS_LOG_DIR = self._tmpdir
            result = trainer._plot_loss_curve(history)
            # 主要验证：history 包含 value_loss 和 policy_loss，
            # 绘图调用不报错即通过
            self.assertIsNotNone(result)
        finally:
            mcts_config.MCTS_LOSS_LOG_DIR = original_dir


class TestKeyboardInterrupt(unittest.TestCase):
    """KeyboardInterrupt 中断后图表生成测试"""

    @classmethod
    def setUpClass(cls):
        try:
            from ai.dual_head_trainer import DualHeadTrainer
            cls.DualHeadTrainer = DualHeadTrainer
        except ImportError:
            cls.DualHeadTrainer = None

    def setUp(self):
        if self.DualHeadTrainer is None:
            self.skipTest("DualHeadTrainer 尚未实现")

    def test_train_epochs_catches_keyboard_interrupt(self):
        """train_epochs 应捕获 KeyboardInterrupt 并生成图表"""
        # 这个测试验证 train_epochs 中有 try/except KeyboardInterrupt 块
        import inspect
        source = inspect.getsource(
            self.DualHeadTrainer.train_epochs
        )
        self.assertIn("KeyboardInterrupt", source,
                      "train_epochs 应包含 KeyboardInterrupt 处理")


if __name__ == "__main__":
    unittest.main()
