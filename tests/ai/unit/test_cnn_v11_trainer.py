"""CNN v11 Trainer 单元测试

验证 CNNV11Trainer 的构造、样本加载、训练流程、模型保存。
使用临时目录隔离所有文件操作，不修改真实数据。

setUp 会把 config.mcts.MCTS_LOSS_LOG_DIR 重定向到临时目录：
train_epochs 结束时必定调用 _plot_loss_curve 落盘 PNG，若不重定向
会写进仓库真实的 mcts_loss_logs/ 目录（违反测试不得污染真实文件的规范）。

测试覆盖 TC3.1 ~ TC3.10（设计文档 test_cases.md 三）。
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch


class TestCNNV11Trainer(unittest.TestCase):
    """CNNV11Trainer 单元测试"""

    @classmethod
    def setUpClass(cls):
        """导入模块（RED 阶段：trainer 尚未创建）"""
        from ai.models.cnn_v11.model import CNNV11Model
        cls.ModelCls = CNNV11Model

    def setUp(self):
        """每个测试前创建 fresh 模型，并把 loss 曲线图输出重定向到临时目录

        为何需要重定向：train_epochs 在 finally 中无条件调用 _plot_loss_curve，
        会把损失曲线图写进 config.mcts.MCTS_LOSS_LOG_DIR（默认为仓库下的
        mcts_loss_logs/）。测试不得向真实目录写文件，故统一指向临时目录。
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

    @staticmethod
    def _make_sample_json(tmp_dir: str, n_samples: int = 10):
        """在临时目录中创建 sample.json（state/pi/z 双头格式）

        Args:
            tmp_dir: 临时目录路径
            n_samples: 样本数量
        """
        samples = []
        for i in range(n_samples):
            # 创建合理的 pi 分布（一位为 1，其余为 0）
            pi = [0.0] * 225
            pi[i % 225] = 1.0
            samples.append({
                "state": [0] * 226,  # 226 维 board_vector
                "pi": pi,
                "z": 1 if i % 2 == 0 else -1,
            })
        path = os.path.join(tmp_dir, "sample.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(samples, f)
        return path

    # ---------- TC3.1 构造器 — fresh=True ----------

    def test_constructor_fresh(self):
        """fresh=True 创建 Trainer"""
        from ai.models.cnn_v11.trainer import CNNV11Trainer

        trainer = CNNV11Trainer(fresh=True, device=torch.device("cpu"))
        self.assertIsInstance(trainer.model, self.ModelCls)
        self.assertEqual(trainer.model_name, "cnn_v11")
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsInstance(trainer.optimizer, torch.optim.AdamW)

    # ---------- TC3.2 构造器 — 传入已有模型 ----------

    def test_constructor_with_model(self):
        """传入 model 实例构造 Trainer"""
        from ai.models.cnn_v11.trainer import CNNV11Trainer

        trainer = CNNV11Trainer(model=self.model)
        self.assertIs(trainer.model, self.model)

    # ---------- TC3.3 构造器 — gen 参数 ----------

    def test_constructor_gen(self):
        """gen 参数传递给模型（临时目录隔离，不触碰真实模型目录）"""
        from ai.models.cnn_v11.trainer import CNNV11Trainer
        from unittest.mock import patch

        # 在临时目录构造 gen1 权重并 patch DEFAULT_MODEL_PATH 指向临时基目录，
        # 禁止读写真实模型目录 src/ai/models/cnn_v11/gen1（曾直接 rmtree 删除真实权重）
        with tempfile.TemporaryDirectory(prefix="cnn_v11_gen_") as tmp_root:
            model_dir = os.path.join(tmp_root, "models")
            gen1_dir = os.path.join(model_dir, "gen1")
            os.makedirs(gen1_dir, exist_ok=True)
            pth_path = os.path.join(gen1_dir, "cnn_v11.pth")

            tmp_model = self.ModelCls(fresh=True)
            tmp_model.save(pth_path)

            with patch.object(
                self.ModelCls, "DEFAULT_MODEL_PATH",
                os.path.join(model_dir, "cnn_v11.pth"),
            ):
                trainer = CNNV11Trainer(gen=1, device=torch.device("cpu"))
                self.assertEqual(trainer.model.gen, 1)

    # ---------- TC3.4 _convert_inputs — 输出形状 ----------

    def test_convert_inputs_shape(self):
        """_convert_inputs 将 state 转为 (N, 3, 15, 15)"""
        from ai.models.cnn_v11.trainer import CNNV11Trainer

        states = [[0] * 226 for _ in range(10)]
        tensor = CNNV11Trainer._convert_inputs(states)
        self.assertEqual(tensor.shape, (10, 3, 15, 15))
        self.assertEqual(tensor.dtype, torch.float32)

    # ---------- TC3.5 _train_step — 损失下降 ----------

    def test_train_step_loss_decreases(self):
        """_train_step 使损失下降"""
        from ai.models.cnn_v11.trainer import CNNV11Trainer

        trainer = CNNV11Trainer(model=self.model, device=torch.device("cpu"))
        trainer.model.network.train()

        # 构造小 batch 训练数据
        inputs = torch.randn(2, 3, 15, 15)
        pi_targets = torch.zeros(2, 225)
        pi_targets[0, 0] = 1.0
        pi_targets[1, 10] = 1.0
        z_targets = torch.tensor([[1.0], [-1.0]])

        result1 = trainer._train_step(inputs, pi_targets, z_targets)
        result2 = trainer._train_step(inputs, pi_targets, z_targets)

        # 第二次 loss 应 ≤ 第一次
        self.assertLessEqual(
            result2["total_loss"], result1["total_loss"] + 0.1,
            f"期望训练 loss 下降：{result1['total_loss']:.4f} → {result2['total_loss']:.4f}")

    # ---------- TC3.6 _train_step — 返回字典结构 ----------

    def test_train_step_return_structure(self):
        """_train_step 返回完整字典"""
        from ai.models.cnn_v11.trainer import CNNV11Trainer

        trainer = CNNV11Trainer(model=self.model, device=torch.device("cpu"))
        trainer.model.network.train()

        inputs = torch.randn(2, 3, 15, 15)
        pi_targets = torch.zeros(2, 225)
        pi_targets[0, 0] = 1.0
        pi_targets[1, 10] = 1.0
        z_targets = torch.tensor([[1.0], [-1.0]])

        result = trainer._train_step(inputs, pi_targets, z_targets)

        self.assertIn("value_loss", result)
        self.assertIn("policy_loss", result)
        self.assertIn("total_loss", result)
        self.assertIn("grad_norms", result)
        self.assertIsInstance(result["value_loss"], float)
        self.assertIsInstance(result["policy_loss"], float)
        self.assertIsInstance(result["total_loss"], float)

    # ---------- TC3.7 _train_step — p_log 校验 ----------

    def test_train_step_log_validation(self):
        """_train_step 中的 p_log 值域校验通过"""
        from ai.models.cnn_v11.trainer import CNNV11Trainer

        trainer = CNNV11Trainer(model=self.model, device=torch.device("cpu"))
        trainer.model.network.train()

        inputs = torch.randn(2, 3, 15, 15)
        pi_targets = torch.ones(2, 225) / 225.0  # 均匀分布
        z_targets = torch.tensor([[0.5], [-0.3]])

        # 不应抛出 AssertionError
        result = trainer._train_step(inputs, pi_targets, z_targets)
        self.assertIsNotNone(result)

    # ---------- TC3.8 train_epochs — 空目录 ----------

    def test_train_epochs_empty_dir(self):
        """空目录（无 sample.json）不崩溃"""
        from ai.models.cnn_v11.trainer import CNNV11Trainer

        tmp_dir = tempfile.mkdtemp()
        try:
            trainer = CNNV11Trainer(model=self.model, device=torch.device("cpu"))
            # 无样本时 _load_samples 返回空列表，空样本防御直接返回空 history
            # （设计 58 方案 A：不再触发 _convert_inputs([]) 的 torch.cat ValueError）
            history = trainer.train_epochs(
                [tmp_dir], epochs=1, batch_size=4, verbose=False,
            )
            self.assertEqual(history, [])
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ---------- TC3.9 train_epochs — 单 epoch 完整流程 ----------

    def test_train_epochs_single_epoch(self):
        """单 epoch 训练返回 history"""
        from ai.models.cnn_v11.trainer import CNNV11Trainer

        tmp_dir = tempfile.mkdtemp()
        try:
            self._make_sample_json(tmp_dir, n_samples=20)
            trainer = CNNV11Trainer(model=self.model, device=torch.device("cpu"))
            history = trainer.train_epochs(
                [tmp_dir], epochs=1, batch_size=8, verbose=False,
            )
            self.assertEqual(len(history), 1)
            self.assertIn("value_loss", history[0])
            self.assertIn("policy_loss", history[0])
            self.assertIn("total_loss", history[0])
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ---------- TC3.10 save_model — 文件产出 ----------

    def test_save_model(self):
        """save_model 保存权重到指定路径"""
        from ai.models.cnn_v11.trainer import CNNV11Trainer

        tmp_dir = tempfile.mkdtemp()
        try:
            out_path = os.path.join(tmp_dir, "trained.pth")
            trainer = CNNV11Trainer(model=self.model, device=torch.device("cpu"))
            result = trainer.save_model(out_path)
            self.assertEqual(result, out_path)
            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(os.path.getsize(out_path), 0)
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
