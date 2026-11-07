"""CNN v10 训练器单元测试

验证 CNNV10Trainer 的构造、样本加载、训练流程、模型保存。
遵循 TDD 流程，先写测试再实现。

setUp 会把 config.mcts.MCTS_LOSS_LOG_DIR 重定向到临时目录：
train_epochs 结束时必定调用 _plot_loss_curve 落盘 PNG，若不重定向
会写进仓库真实的 mcts_loss_logs/ 目录（违反测试不得污染真实文件的规范）。
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch

from ai.base_model import ModelPredict


class TestCNNV10Trainer(unittest.TestCase):
    """CNNV10Trainer 单元测试（设计 TC-TRN-01 ~ TC-TRN-08）"""

    @classmethod
    def setUpClass(cls):
        from ai.models.cnn_v10.model import CNNV10Model
        cls.ModelCls = CNNV10Model

    def setUp(self):
        """每个测试前创建 fresh 模型，并把 loss 曲线图输出重定向到临时目录

        为何需要重定向：train_epochs 在 finally 中无条件调用 _plot_loss_curve，
        会把损失曲线图写进 config.mcts.MCTS_LOSS_LOG_DIR（默认为仓库下的
        mcts_loss_logs/）。测试不得向真实目录写文件，故统一指向临时目录。

        为何放在 setUp：本文件有 5 个用例会调 train_epochs，统一在 setUp 处理
        可以避免在每个用例里重复写重定向代码。
        """
        self.model = self.ModelCls(fresh=True)

        import shutil

        import config.mcts as mcts_config

        self._plot_dir = tempfile.mkdtemp()
        original_plot_dir = mcts_config.MCTS_LOSS_LOG_DIR
        mcts_config.MCTS_LOSS_LOG_DIR = self._plot_dir
        # addCleanup 按 LIFO 执行：先恢复模块常量，再删除临时目录
        self.addCleanup(shutil.rmtree, self._plot_dir, ignore_errors=True)
        self.addCleanup(setattr, mcts_config, "MCTS_LOSS_LOG_DIR", original_plot_dir)

    def _make_sample_json(self, tmp_dir: str, n_samples: int = 10):
        """在临时目录中创建 sample.json（state + z 格式）

        Args:
            tmp_dir: 临时目录路径
            n_samples: 样本数量
        """
        samples = []
        for i in range(n_samples):
            samples.append({
                "state": [0] * 226,  # 226 维 board_vector
                "z": 1 if i % 2 == 0 else -1,
            })
        path = os.path.join(tmp_dir, "sample.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(samples, f)
        return path

    def _make_learnable_sample_json(self, tmp_dir: str, n_samples: int = 10):
        """在临时目录中创建「可学习」的 sample.json（仅供 loss 下降类用例使用）

        与 _make_sample_json 的区别：每条样本的 state 互不相同，且 z 与该 state
        的符号一一对应，价值头因此能够真正拟合 ±1 目标，loss 才会稳定下降。

        为何不能复用 _make_sample_json：它生成的 state 全为 [0]*226（输入完全
        相同），z 交替 +1/-1。价值头输出经 tanh 后，对同一输入只能给出同一常数
        v，此时 MSE = (v-1)^2/2 + (v+1)^2/2 = v^2 + 1 恒 >= 1，不存在可学信号，
        “后 3 epoch 均值 < 前 3 epoch 均值” 会完全由随机初始化的 |v| 大小决定，
        实测约 1/5 概率失败。

        Args:
            tmp_dir: 临时目录路径
            n_samples: 样本数量
        """
        samples = []
        for i in range(n_samples):
            # 每条样本在棋盘向量上占据一个互不相同的位置，符号即目标 z
            sign = 1 if i % 2 == 0 else -1
            state = [0] * 226
            state[i % 225] = sign
            samples.append({"state": state, "z": sign})
        path = os.path.join(tmp_dir, "sample.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(samples, f)
        return path

    def test_constructor_with_model(self):
        """TC-TRN-01: 构造器接受 model 实例"""
        from ai.models.cnn_v10.trainer import CNNV10Trainer
        trainer = CNNV10Trainer(model=self.model)
        self.assertIsNotNone(trainer.optimizer)
        self.assertIs(trainer.model, self.model)

    def test_constructor_fresh(self):
        """TC-TRN-07: fresh=True 创建新模型"""
        from ai.models.cnn_v10.trainer import CNNV10Trainer
        trainer = CNNV10Trainer(fresh=True, gen=1)
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.model.network)

    def test_load_samples_single_dir(self):
        """TC-TRN-02: _load_samples 加载单目录"""
        from ai.models.cnn_v10.trainer import CNNV10Trainer

        tmp_dir = tempfile.mkdtemp()
        try:
            self._make_sample_json(tmp_dir, n_samples=5)
            states, zs = CNNV10Trainer._load_samples([tmp_dir])
            self.assertEqual(len(states), 5)
            self.assertEqual(len(zs), 5)
            self.assertEqual(len(states[0]), 226)
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_load_samples_no_file_raises(self):
        """TC-TRN-08: 无 sample.json 抛异常"""
        from ai.models.cnn_v10.trainer import CNNV10Trainer

        tmp_dir = tempfile.mkdtemp()
        try:
            with self.assertRaises(FileNotFoundError):
                CNNV10Trainer._load_samples([tmp_dir])
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_convert_inputs(self):
        """_convert_inputs 将 226 维 state 转为 (N, 3, 15, 15)"""
        from ai.models.cnn_v10.trainer import CNNV10Trainer

        states = [[0] * 226 for _ in range(3)]
        tensor = CNNV10Trainer._convert_inputs(states)
        self.assertEqual(tensor.shape, (3, 3, 15, 15))

    def test_train_epochs_single_epoch(self):
        """TC-TRN-03: train_epochs 单 epoch 返回 history"""
        from ai.models.cnn_v10.trainer import CNNV10Trainer

        tmp_dir = tempfile.mkdtemp()
        try:
            self._make_sample_json(tmp_dir, n_samples=20)
            trainer = CNNV10Trainer(model=self.model)
            history = trainer.train_epochs(
                [tmp_dir], epochs=1, batch_size=8, verbose=False,
            )
            self.assertEqual(len(history), 1)
            self.assertIn("value_loss", history[0])
            self.assertIn("total_loss", history[0])
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_train_epochs_val_split(self):
        """TC-TRN-04: val_split 产生验证集 loss"""
        from ai.models.cnn_v10.trainer import CNNV10Trainer

        tmp_dir = tempfile.mkdtemp()
        try:
            self._make_sample_json(tmp_dir, n_samples=50)
            trainer = CNNV10Trainer(model=self.model)
            history = trainer.train_epochs(
                [tmp_dir], epochs=2, batch_size=8,
                val_split=0.2, verbose=False,
            )
            # 至少最后一个 epoch 含 val_value_loss
            self.assertIn("val_value_loss", history[-1])
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_save_model(self):
        """TC-TRN-05: save_model 保存权重到指定路径"""
        from ai.models.cnn_v10.trainer import CNNV10Trainer

        tmp_dir = tempfile.mkdtemp()
        try:
            out_path = os.path.join(tmp_dir, "trained.pth")
            trainer = CNNV10Trainer(model=self.model)
            result = trainer.save_model(out_path)
            self.assertEqual(result, out_path)
            self.assertTrue(os.path.exists(out_path))
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_train_epochs_shuffle(self):
        """shuffle=True 时不抛异常"""
        from ai.models.cnn_v10.trainer import CNNV10Trainer

        tmp_dir = tempfile.mkdtemp()
        try:
            self._make_sample_json(tmp_dir, n_samples=20)
            trainer = CNNV10Trainer(model=self.model)
            history = trainer.train_epochs(
                [tmp_dir], epochs=1, batch_size=8,
                shuffle=True, verbose=False,
            )
            self.assertEqual(len(history), 1)
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_train_epochs_no_shuffle(self):
        """shuffle=False 时不抛异常"""
        from ai.models.cnn_v10.trainer import CNNV10Trainer

        tmp_dir = tempfile.mkdtemp()
        try:
            self._make_sample_json(tmp_dir, n_samples=20)
            trainer = CNNV10Trainer(model=self.model)
            history = trainer.train_epochs(
                [tmp_dir], epochs=1, batch_size=8,
                shuffle=False, verbose=False,
            )
            self.assertEqual(len(history), 1)
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_train_epochs_loss_decreases(self):
        """多次 epoch 训练 loss 整体呈下降趋势

        使用 _make_learnable_sample_json 而非 _make_sample_json：后者所有样本的
        state 完全相同，价值头对同一输入只能给出同一常数预测，MSE 下界恒为 1，
        loss 无下降空间，断言会随随机初始化偶发失败。
        """
        from ai.models.cnn_v10.trainer import CNNV10Trainer

        tmp_dir = tempfile.mkdtemp()
        try:
            self._make_learnable_sample_json(tmp_dir, n_samples=50)
            trainer = CNNV10Trainer(model=self.model)
            history = trainer.train_epochs(
                [tmp_dir], epochs=10, batch_size=8,
                val_split=0.2, verbose=False,
            )
            self.assertGreater(len(history), 0)
            # 检查后期 loss 是否低于前期（粗粒度：后 3 epoch 均值 < 前 3 epoch 均值）
            first_3_avg = sum(h["total_loss"] for h in history[:3]) / 3
            last_3_avg = sum(h["total_loss"] for h in history[-3:]) / 3
            self.assertLess(last_3_avg, first_3_avg,
                            f"期望 training loss 下降，前3均值={first_3_avg:.4f}，后3均值={last_3_avg:.4f}")
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_load_samples_multiple_dirs(self):
        """_load_samples 合并多个目录"""
        from ai.models.cnn_v10.trainer import CNNV10Trainer

        tmp_dir1 = tempfile.mkdtemp()
        tmp_dir2 = tempfile.mkdtemp()
        try:
            self._make_sample_json(tmp_dir1, n_samples=3)
            self._make_sample_json(tmp_dir2, n_samples=4)
            states, zs = CNNV10Trainer._load_samples([tmp_dir1, tmp_dir2])
            self.assertEqual(len(states), 7)
        finally:
            import shutil
            shutil.rmtree(tmp_dir1, ignore_errors=True)
            shutil.rmtree(tmp_dir2, ignore_errors=True)

    def test_constructor_with_lr_and_weight_decay(self):
        """构造器支持自定义 lr 和 weight_decay"""
        from ai.models.cnn_v10.trainer import CNNV10Trainer
        trainer = CNNV10Trainer(model=self.model, lr=0.01, weight_decay=0.001)
        actual_lr = trainer.optimizer.param_groups[0]['lr']
        self.assertEqual(actual_lr, 0.01)

    def test_constructor_with_device(self):
        """构造器支持自定义 device"""
        from ai.models.cnn_v10.trainer import CNNV10Trainer
        device = torch.device("cpu")
        trainer = CNNV10Trainer(model=self.model, device=device)
        self.assertEqual(trainer.device, device)


if __name__ == '__main__':
    unittest.main()
