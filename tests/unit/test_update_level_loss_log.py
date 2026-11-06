"""update 级 loss 记录与曲线密度测试

覆盖 DualHeadTrainer.train_epochs 的 log_every_updates 参数：
- 默认 0：行为完全不变（不产生 update 级记录）
- >0：每 N 次 update 记录一次窗口均值，epoch 结尾 flush 残留窗口
- epoch 级 history 结构保持兼容（仅新增 end_step 字段）
- _plot_loss_curve 支持 update 级数据绘制（X 轴为累计 update 序号）
"""

import sys
import os
import json
import tempfile
import shutil
import unittest

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tests.unit.test_dual_head_trainer import (
    MockDualHeadModel, _make_concrete_trainer,
)


def _create_sample_json(output_dir, n_samples=16):
    """创建测试用 sample.json（与既有训练器测试保持一致的构造方式）"""
    samples = []
    for i in range(n_samples):
        state = [0] * 225 + [1 if i % 2 == 0 else 2]
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


class _PlotCapture:
    """捕获 _plot_loss_curve 调用参数的替身"""

    def __init__(self):
        self.calls = []

    def __call__(self, history, update_history=None):
        self.calls.append({"history": history, "update_history": update_history})
        return None


class TestUpdateLevelLogging(unittest.TestCase):
    """train_epochs 的 log_every_updates 参数测试"""

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

    def _run_train(self, tmpdir, n_samples=16, epochs=2, batch_size=4,
                   log_every_updates=None, val_split=0.0):
        """运行 train_epochs 并捕获 _plot_loss_curve 参数

        n_samples=16, batch_size=4, epochs=2 → 每 epoch 4 次 update，共 8 次
        """
        _create_sample_json(tmpdir, n_samples=n_samples)
        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model)
        capture = _PlotCapture()
        trainer._plot_loss_curve = capture

        kwargs = dict(
            sample_dirs=[tmpdir],
            epochs=epochs,
            batch_size=batch_size,
            val_split=val_split,
            shuffle=False,
            verbose=False,
        )
        if log_every_updates is not None:
            kwargs["log_every_updates"] = log_every_updates
        history = trainer.train_epochs(**kwargs)
        return history, capture

    def test_default_off_no_update_records(self):
        """默认（不传 log_every_updates）不产生 update 级记录，行为不变"""
        with tempfile.TemporaryDirectory() as tmpdir:
            history, capture = self._run_train(tmpdir)
        self.assertEqual(len(capture.calls), 1)
        self.assertIsNone(capture.calls[0]["update_history"])
        self.assertEqual(len(history), 2)

    def test_log_every_zero_disables_update_records(self):
        """显式传 log_every_updates=0 同样关闭 update 级记录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            history, capture = self._run_train(tmpdir, log_every_updates=0)
        self.assertIsNone(capture.calls[0]["update_history"])
        self.assertEqual(len(history), 2)

    def test_update_records_window_and_flush(self):
        """log_every_updates=3 时：完整窗口 + epoch 结尾 flush 残留窗口

        8 次 update（2 epochs × 4 updates），N=3 →
        epoch1: 窗口[1-3] 记 step=3，残留[4] flush 记 step=4
        epoch2: 窗口[5-7] 记 step=7，残留[8] flush 记 step=8
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            _, capture = self._run_train(tmpdir, log_every_updates=3)
        update_history = capture.calls[0]["update_history"]
        self.assertIsNotNone(update_history)
        self.assertEqual(len(update_history), 4)
        self.assertEqual([e["step"] for e in update_history], [3, 4, 7, 8])
        for entry in update_history:
            self.assertIn("value_loss", entry)
            self.assertIn("policy_loss", entry)
            self.assertIn("total_loss", entry)
            self.assertIn("grad_norm", entry)
            self.assertGreater(entry["total_loss"], 0.0)
            self.assertGreater(entry["grad_norm"], 0.0)

    def test_epoch_history_backward_compatible(self):
        """epoch 级 history 结构不变，仅新增 end_step 字段"""
        with tempfile.TemporaryDirectory() as tmpdir:
            history, _ = self._run_train(tmpdir, log_every_updates=3)
        self.assertEqual(len(history), 2)
        for idx, entry in enumerate(history):
            self.assertEqual(entry["epoch"], idx + 1)
            self.assertIn("value_loss", entry)
            self.assertIn("policy_loss", entry)
            self.assertIn("total_loss", entry)
            self.assertIn("grad_norm", entry)
            self.assertIn("end_step", entry)
        # 每 epoch 4 次 update → end_step 分别为 4、8
        self.assertEqual(history[0]["end_step"], 4)
        self.assertEqual(history[1]["end_step"], 8)

    def test_residual_window_mean_matches_epoch_mean(self):
        """N 大于单 epoch update 数时，整个 epoch 落入一个残留窗口，
        窗口均值应等于 epoch 级均值"""
        with tempfile.TemporaryDirectory() as tmpdir:
            history, capture = self._run_train(
                tmpdir, epochs=1, batch_size=4, log_every_updates=10)
        update_history = capture.calls[0]["update_history"]
        self.assertIsNotNone(update_history)
        self.assertEqual(len(update_history), 1)
        self.assertEqual(update_history[0]["step"], 4)
        self.assertAlmostEqual(
            update_history[0]["total_loss"], history[0]["total_loss"], places=10)
        self.assertAlmostEqual(
            update_history[0]["value_loss"], history[0]["value_loss"], places=10)
        self.assertAlmostEqual(
            update_history[0]["policy_loss"], history[0]["policy_loss"], places=10)

    def test_update_records_with_val_split(self):
        """有验证集时 update 级记录与 end_step 均正常"""
        with tempfile.TemporaryDirectory() as tmpdir:
            history, capture = self._run_train(
                tmpdir, n_samples=32, batch_size=8, epochs=2,
                log_every_updates=2, val_split=0.25)
        update_history = capture.calls[0]["update_history"]
        # 每 epoch 3 次 update（24 训练样本 / 8）→ 窗口[1-2] + flush[3]，
        # 窗口[4-5] + flush[6]
        self.assertEqual([e["step"] for e in update_history], [2, 3, 5, 6])
        for entry in history:
            self.assertIn("end_step", entry)
            self.assertIn("val_total_loss", entry)
        self.assertEqual(history[-1]["end_step"], 6)


class TestPlotWithUpdateHistory(unittest.TestCase):
    """_plot_loss_curve 的 update 级数据绘制测试"""

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

    def _create_history(self, epochs=3, with_val=True):
        """创建带 end_step 的模拟 epoch 级 history"""
        history = []
        for e in range(epochs):
            entry = {
                "epoch": e + 1,
                "end_step": (e + 1) * 4,
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

    def _create_update_history(self, n=12):
        """创建模拟 update 级 history"""
        return [
            {
                "step": (i + 1) * 1,
                "value_loss": 0.5 - i * 0.01,
                "policy_loss": 2.0 - i * 0.02,
                "total_loss": 2.5 - i * 0.03,
                "grad_norm": 0.1 + i * 0.005,
            }
            for i in range(n)
        ]

    def _plot(self, history, update_history):
        """在临时输出目录下调用 _plot_loss_curve，返回结果路径"""
        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model)

        import config.mcts as mcts_config
        original_dir = mcts_config.MCTS_LOSS_LOG_DIR
        try:
            mcts_config.MCTS_LOSS_LOG_DIR = self._tmpdir
            return trainer._plot_loss_curve(history, update_history)
        finally:
            mcts_config.MCTS_LOSS_LOG_DIR = original_dir

    def test_plot_with_update_history_creates_png(self):
        """传入 update_history 时生成 PNG 文件"""
        history = self._create_history(epochs=3, with_val=True)
        update_history = self._create_update_history(n=12)
        result = self._plot(history, update_history)
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))
        self.assertTrue(result.endswith(".png"))

    def test_plot_update_history_none_keeps_epoch_level(self):
        """update_history=None 时保持 epoch 级绘制（向后兼容）"""
        history = self._create_history(epochs=3, with_val=True)
        result = self._plot(history, None)
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))

    def test_plot_update_history_without_val(self):
        """update 级数据 + 无验证集 history 时正常绘制"""
        history = self._create_history(epochs=2, with_val=False)
        update_history = self._create_update_history(n=8)
        result = self._plot(history, update_history)
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))


if __name__ == "__main__":
    unittest.main()
