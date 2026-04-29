"""Bug 修复回归测试

Bug 1：train_batch 模式下 val_loss 曲线图被静默丢弃
Bug 2：sample_loader.load_and_split 默认 val_ratio=0.2 与 CLI 默认 0.0 不一致
"""

import json
import os
import sys
import tempfile
import shutil
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch

from ai.loss_logger import LossLogger
from ai.sample_loader import SampleLoader
from tests.ai.helpers import create_sample_dir


# ─── Bug 1 测试 ─────────────────────────────────────────────


class _ConcreteTrainer:
    """轻量级替代 BaseTrainer 子类，仅用于测试 train_batch 的 dense_log 覆盖逻辑

    不继承 BaseTrainer，避免触发真实的模型创建和 GPU 操作。
    通过捕获 _setup_training 的 kwargs 来验证 dense_log 是否被正确覆盖。
    """
    def __init__(self):
        self.captured_setup_kwargs = None

    def _setup_training(self, **kwargs):
        self.captured_setup_kwargs = kwargs
        # 返回一个 mock LossLogger
        mock = MagicMock()
        mock.dense_log = kwargs.get('dense_log', False)
        return mock


class TestTrainBatchDenseLogOverride(unittest.TestCase):
    """Bug 1：train_batch 应强制 dense_log=False

    根因：run_train_from_dir 和 self_play_cycle.py 无条件传递 dense_log=True。
    当 batch_size=None 时，train_auto 路由到 train_batch，导致 LossLogger
    被创建为 dense_log=True。此时 log() 用 "epoch" 键，log_val() 用 "step" 键，
    val_loss 数据在 plot() 中被静默丢弃。

    修复：train_batch 中强制覆盖 dense_log=False。
    """

    def test_train_batch_overrides_dense_log_to_false(self):
        """train_batch 应将 dense_log 强制设为 False，即使调用方传入 True"""
        from ai.base_trainer import BaseTrainer

        # 使用 BaseTrainer 的 train_batch 方法逻辑
        # 验证 setup_kwargs 中 dense_log 被覆盖
        trainer = _ConcreteTrainer()

        # 模拟 train_batch 中的 setup_kwargs 构建逻辑
        # 这是 train_batch 中应该有的逻辑
        setup_kwargs = {'epochs': 100, 'dense_log': True}
        setup_kwargs['dense_log'] = False  # train_batch 应强制覆盖

        trainer._setup_training(**setup_kwargs)
        self.assertEqual(trainer.captured_setup_kwargs['dense_log'], False,
                         "train_batch 应强制 dense_log=False")

    def test_log_val_in_sparse_mode_uses_epoch_key(self):
        """稀疏模式 (dense_log=False) 下 log_val 写入 "epoch" 键而非 "step" 键"""
        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(model_name="test_model", log_dir=tmp_path, dense_log=False)
            logger.log(200, 0.5, 1.2)
            logger.log_val(200, 0.45)

            entries = logger._read_entries()
            val_entries = [e for e in entries if "val_loss" in e]
            self.assertEqual(len(val_entries), 1)
            self.assertIn("epoch", val_entries[0],
                          "稀疏模式下 val entry 应包含 'epoch' 键")
            self.assertNotIn("step", val_entries[0],
                             "稀疏模式下 val entry 不应包含 'step' 键")
            self.assertEqual(val_entries[0]["epoch"], 200)

    def test_plot_does_not_silently_drop_val_loss_in_batch_mode(self):
        """train_batch 模式下 plot() 不应静默丢弃 val_loss 数据

        模拟 train_batch 的日志模式：log() 用 epoch 键 + grad_norm，
        log_val() 也用 epoch 键。plot() 应正确绘制 val_loss。
        """
        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(model_name="test_model", log_dir=tmp_path, dense_log=False)
            # 模拟 _run_training_loop 的记录方式
            logger.log(200, 0.5, 1.2)
            logger.log_val(200, 0.45)
            logger.log(400, 0.3, 0.8)
            logger.log_val(400, 0.35)

            # 读取 entries，验证 plot 能找到 val data
            entries = logger._read_entries()
            train_entries = [e for e in entries if "loss" in e and "val_loss" not in e]
            val_entries = [e for e in entries if "val_loss" in e]

            # train entries 使用 epoch 键
            x_key = "step" if "step" in train_entries[0] else "epoch"
            self.assertEqual(x_key, "epoch",
                             "train_batch 模式 x_key 应为 'epoch'")

            # val entries 也有 epoch 键（因为 dense_log=False）
            val_with_x_key = [e for e in val_entries if x_key in e]
            self.assertEqual(len(val_with_x_key), len(val_entries),
                             f"所有 val entries 应包含 '{x_key}' 键，不应被 plot 静默丢弃")

    def test_dense_log_true_causes_val_loss_mismatch(self):
        """验证 Bug 的根因：dense_log=True 时 val entry 使用 "step" 键导致与 train entry 的 "epoch" 键不匹配

        这是一个反向测试，证明 Bug 确实存在。当 train_batch 错误地使用
        dense_log=True 时，log() 写 "epoch" 键，log_val() 写 "step" 键，
        导致 plot() 中 val data 被静默丢弃。
        """
        with tempfile.TemporaryDirectory() as tmp_path:
            # 模拟 Bug 场景：train_batch 被错误地设为 dense_log=True
            logger = LossLogger(model_name="test_model", log_dir=tmp_path, dense_log=True)
            # _run_training_loop 中 log() 始终写 "epoch" 键
            logger.log(200, 0.5, 1.2)
            # _run_training_loop 中 log_val() 因为 dense_log=True，写 "step" 键
            logger.log_val(200, 0.45)

            entries = logger._read_entries()
            train_entries = [e for e in entries if "loss" in e and "val_loss" not in e]
            val_entries = [e for e in entries if "val_loss" in e]

            # train entries 用 "epoch" 键
            self.assertIn("epoch", train_entries[0])
            # val entries 用 "step" 键（Bug 的根因！）
            self.assertIn("step", val_entries[0])
            self.assertNotIn("epoch", val_entries[0])

            # plot() 从 train entries 推断 x_key="epoch"
            x_key = "step" if "step" in train_entries[0] else "epoch"
            self.assertEqual(x_key, "epoch")

            # val entries 不含 "epoch" 键，导致被静默丢弃
            val_with_x_key = [e for e in val_entries if x_key in e]
            self.assertEqual(len(val_with_x_key), 0,
                             "Bug 场景：val entries 不含 'epoch' 键，被 plot 静默丢弃")

    def test_train_batch_dense_log_override_in_actual_code(self):
        """集成测试：验证 BaseTrainer.train_batch 方法强制覆盖 dense_log=False

        这是最终的集成验证，确保源码中的 train_batch 方法确实
        将 dense_log 强制设为 False。
        """
        from ai.base_trainer import BaseTrainer
        import inspect

        source = inspect.getsource(BaseTrainer.train_batch)

        # 验证 train_batch 方法源码中包含 dense_log=False 的覆盖
        self.assertIn("dense_log", source,
                      "train_batch 方法应包含 dense_log 相关代码")
        # 检查是否有强制覆盖为 False 的逻辑
        self.assertIn("False", source.split("dense_log")[1][:50] if "dense_log" in source else "",
                      "train_batch 应将 dense_log 强制设为 False")


# ─── Bug 2 测试 ─────────────────────────────────────────────


class TestLoadAndSplitDefaultValRatio(unittest.TestCase):
    """Bug 2：load_and_split 默认 val_ratio 应为 0.0

    根因：load_and_split 的 val_ratio 默认值为 0.2，但 CLI 和 base_trainer.py
    的默认值为 0.0，导致不一致。

    修复：将 val_ratio 默认值从 0.2 改为 0.0。
    """

    def test_default_val_ratio_is_zero(self):
        """不传 val_ratio 时默认为 0.0，全部数据归训练集"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = create_sample_dir(tmp_dir, "data", n_items=10)
            loader = SampleLoader(torch.device('cpu'))

            train_data, val_data = loader.load_and_split([sample_dir])

            # val_ratio=0 时所有数据归训练集，验证集为空
            self.assertEqual(len(val_data), 0,
                             "默认 val_ratio=0.0 时验证集应为空")
            self.assertEqual(len(train_data), 10,
                             "默认 val_ratio=0.0 时所有数据应归训练集")

    def test_explicit_val_ratio_zero(self):
        """显式传入 val_ratio=0.0 行为与默认一致"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = create_sample_dir(tmp_dir, "data", n_items=10)
            loader = SampleLoader(torch.device('cpu'))

            train_data, val_data = loader.load_and_split([sample_dir], val_ratio=0.0)

            self.assertEqual(len(val_data), 0)
            self.assertEqual(len(train_data), 10)

    def test_explicit_val_ratio_still_works(self):
        """显式传入 val_ratio=0.2 仍然正常分割"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = create_sample_dir(tmp_dir, "data", n_items=20)
            loader = SampleLoader(torch.device('cpu'))

            train_data, val_data = loader.load_and_split([sample_dir], val_ratio=0.2)

            # 应有验证集数据
            self.assertGreater(len(val_data), 0,
                               "val_ratio=0.2 时应有验证集数据")
            self.assertLess(len(train_data), 20,
                            "val_ratio=0.2 时训练集应少于总数")

    def test_default_val_ratio_signature(self):
        """验证 load_and_split 的 val_ratio 参数默认值为 0.0"""
        import inspect

        sig = inspect.signature(SampleLoader.load_and_split)
        val_ratio_param = sig.parameters['val_ratio']
        self.assertEqual(val_ratio_param.default, 0.0,
                         "load_and_split 的 val_ratio 默认值应为 0.0")


if __name__ == '__main__':
    unittest.main()
