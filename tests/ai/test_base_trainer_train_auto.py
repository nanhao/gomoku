"""BaseTrainer.train_auto 方法测试

测试 train_auto 的分发逻辑：
- 提供 batch_size 时调用 train_mini_batch
- 不提供 batch_size 时调用 train_batch
- 额外参数（kwargs）正确透传
"""

import json
import os
import sys
import tempfile
import shutil
import unittest
from unittest.mock import MagicMock, patch, call

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.base_trainer import BaseTrainer


def _create_sample_dir(tmp_dir, name="sample_0"):
    """辅助函数：在临时目录下创建包含 sample.json 的子目录"""
    sample_dir = os.path.join(tmp_dir, name)
    os.makedirs(sample_dir, exist_ok=True)
    sample_data = [{
        "input": [0.0] * 226,
        "output": [1.0 / 225.0] * 225,
        "mask": [1] * 225
    }]
    with open(os.path.join(sample_dir, "sample.json"), 'w', encoding='utf-8') as f:
        json.dump(sample_data, f)
    return sample_dir


class _ConcreteTrainer(BaseTrainer):
    """用于测试的具体训练器子类"""

    def _create_model(self, fresh):
        from ai.models.fc_v1.model import FCv1Model
        return FCv1Model(fresh=fresh)

    @property
    def model_name(self):
        return "test_train_auto"


class TestTrainAutoDispatch(unittest.TestCase):
    """测试 train_auto 的分发逻辑（使用 mock 避免真实训练）"""

    def setUp(self):
        """创建训练器实例，mock 掉底层训练方法"""
        from ai.models.fc_v1.model import FCv1Model

        # 重定向模型路径，避免测试污染真实模型文件
        self.temp_dir = tempfile.mkdtemp()
        self._original_path = FCv1Model.DEFAULT_MODEL_PATH
        FCv1Model.DEFAULT_MODEL_PATH = os.path.join(self.temp_dir, "fc_v1.pth")

        self.trainer = _ConcreteTrainer(fresh=True)

        # mock 掉 train_batch 和 train_mini_batch，避免真正训练
        self.trainer.train_batch = MagicMock(return_value=0.123)
        self.trainer.train_mini_batch = MagicMock(return_value=0.456)

    def tearDown(self):
        """恢复模型路径并清理临时目录"""
        from ai.models.fc_v1.model import FCv1Model
        FCv1Model.DEFAULT_MODEL_PATH = self._original_path
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ---------- 主流程：batch_size 决定分发路径 ----------

    def test_batch_size_none_calls_train_batch(self):
        """batch_size=None 时应调用 train_batch"""
        result = self.trainer.train_auto(
            ["dir1", "dir2"], epochs=10, batch_size=None
        )
        self.trainer.train_batch.assert_called_once()
        self.trainer.train_mini_batch.assert_not_called()
        self.assertAlmostEqual(result, 0.123)

    def test_batch_size_zero_calls_train_batch(self):
        """batch_size=0 时应调用 train_batch（视为无效值）"""
        result = self.trainer.train_auto(
            ["dir1", "dir2"], epochs=10, batch_size=0
        )
        self.trainer.train_batch.assert_called_once()
        self.trainer.train_mini_batch.assert_not_called()

    def test_batch_size_positive_calls_train_mini_batch(self):
        """batch_size>0 时应调用 train_mini_batch"""
        result = self.trainer.train_auto(
            ["dir1", "dir2"], epochs=10, batch_size=32
        )
        self.trainer.train_mini_batch.assert_called_once()
        self.trainer.train_batch.assert_not_called()
        self.assertAlmostEqual(result, 0.456)

    def test_batch_size_one_calls_train_mini_batch(self):
        """batch_size=1（边界值）时应调用 train_mini_batch"""
        result = self.trainer.train_auto(
            ["dir1"], epochs=5, batch_size=1
        )
        self.trainer.train_mini_batch.assert_called_once()
        self.trainer.train_batch.assert_not_called()

    def test_negative_batch_size_calls_train_batch(self):
        """batch_size<0 时应调用 train_batch（视为无效值）"""
        result = self.trainer.train_auto(
            ["dir1"], epochs=5, batch_size=-1
        )
        self.trainer.train_batch.assert_called_once()
        self.trainer.train_mini_batch.assert_not_called()

    # ---------- 参数透传测试 ----------

    def test_kwargs_passed_to_train_batch(self):
        """额外参数应透传给 train_batch"""
        self.trainer.train_auto(
            ["dir1"], epochs=50, batch_size=None,
            verbose=False, session_dir="/tmp/s", dense_log=True
        )
        self.trainer.train_batch.assert_called_once_with(
            ["dir1"], epochs=50, verbose=False,
            session_dir="/tmp/s", dense_log=True, val_ratio=0.0
        )

    def test_kwargs_passed_to_train_mini_batch(self):
        """额外参数应透传给 train_mini_batch"""
        self.trainer.train_auto(
            ["dir1"], epochs=50, batch_size=16, shuffle=False,
            verbose=False, session_dir="/tmp/s", log_type="epoch_level",
            round_num=1, total_rounds=10, dense_log=True
        )
        self.trainer.train_mini_batch.assert_called_once_with(
            ["dir1"], epochs=50, batch_size=16, shuffle=False,
            verbose=False, session_dir="/tmp/s", log_type="epoch_level",
            round_num=1, total_rounds=10, dense_log=True, val_ratio=0.0
        )

    def test_default_parameter_values(self):
        """未传参数时使用默认值"""
        self.trainer.train_auto(["dir1"])
        # 默认 batch_size=None → 走 train_batch
        self.trainer.train_batch.assert_called_once_with(
            ["dir1"], epochs=1000, verbose=True, val_ratio=0.0
        )

    # ---------- 返回值测试 ----------

    def test_returns_train_batch_result(self):
        """train_batch 路径应返回其结果"""
        self.trainer.train_batch.return_value = 0.789
        result = self.trainer.train_auto(["dir1"], batch_size=None)
        self.assertAlmostEqual(result, 0.789)

    def test_returns_train_mini_batch_result(self):
        """train_mini_batch 路径应返回其结果"""
        self.trainer.train_mini_batch.return_value = 0.012
        result = self.trainer.train_auto(["dir1"], batch_size=64)
        self.assertAlmostEqual(result, 0.012)


class TestTrainAutoIntegration(unittest.TestCase):
    """集成测试：train_auto 端到端执行真实训练流程"""

    def setUp(self):
        """创建训练器实例和临时样本目录"""
        from ai.models.fc_v1.model import FCv1Model

        # 重定向模型路径，避免测试污染真实模型文件
        self.temp_dir = tempfile.mkdtemp()
        self._original_path = FCv1Model.DEFAULT_MODEL_PATH
        FCv1Model.DEFAULT_MODEL_PATH = os.path.join(self.temp_dir, "fc_v1.pth")

        self.trainer = _ConcreteTrainer(fresh=True)
        # 注入 mock logger 避免污染真实 loss_logs 目录
        self.trainer.loss_logger = MagicMock()
        self.trainer.loss_logger.dense_log = False

        # 创建两个样本目录
        self.sample_dirs = []
        for i in range(2):
            self.sample_dirs.append(
                _create_sample_dir(self.temp_dir, f"sample_{i}")
            )

    def tearDown(self):
        """恢复模型路径并清理临时目录"""
        from ai.models.fc_v1.model import FCv1Model
        FCv1Model.DEFAULT_MODEL_PATH = self._original_path
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_auto_with_batch_none_runs_train_batch(self):
        """batch_size=None 时端到端执行 train_batch 并返回有效 loss"""
        avg_loss = self.trainer.train_auto(
            self.sample_dirs, epochs=1000, batch_size=None, verbose=False
        )
        # train_batch 应返回非零 loss
        self.assertGreater(avg_loss, 0.0)
        # mock logger 的 plot 方法应被调用（_finalize_training 内部）
        self.assertTrue(self.trainer.loss_logger.plot.called)

    def test_auto_with_batch_size_runs_train_mini_batch(self):
        """batch_size>0 时端到端执行 train_mini_batch 并返回有效 loss"""
        avg_loss = self.trainer.train_auto(
            self.sample_dirs, epochs=10, batch_size=1, shuffle=False, verbose=False
        )
        # train_mini_batch 应返回非零 loss
        self.assertGreater(avg_loss, 0.0)


if __name__ == '__main__':
    unittest.main()
