"""OffPolicyTrainLoop 训练编排单元测试（设计 58 TC-OPL-01~13）

验证完整训练流程：加载模型 → 训练 → 保存 gen{N+1}，以及
中断、设备回退、参数校验、批次切分等分支与边界条件。

所有测试使用最小数据规模（100 条样本、batch 32），并通过
monkeypatch 将 CNNV11Model.DEFAULT_MODEL_PATH 与 MCTS_LOSS_LOG_DIR
重定向到临时目录，禁止触碰真实模型与样本文件。
"""

import contextlib
import glob
import io
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch


class OffPolicyTrainLoopTestBase(unittest.TestCase):
    """OffPolicyTrainLoop 测试公共基类

    setUp 准备：
    - 临时模型目录（gen1/cnn_v11.pth，从共享 seed 复制）
    - 临时样本目录（100 条 sample.json）
    - 临时绘图目录（重定向 MCTS_LOSS_LOG_DIR）
    - patch CNNV11Model.DEFAULT_MODEL_PATH 指向临时模型目录
    """

    # 共享 seed 模型权重目录（setUpClass 创建一次，避免每个用例重建 843 万参数模型）
    _seed_dir = None

    @classmethod
    def setUpClass(cls):
        from ai.models.cnn_v11.model import CNNV11Model
        cls._seed_dir = tempfile.mkdtemp(prefix="off_policy_seed_")
        seed_model = CNNV11Model(fresh=True, gen=1)
        seed_model.save(os.path.join(cls._seed_dir, "seed.pth"))

    @classmethod
    def tearDownClass(cls):
        if cls._seed_dir and os.path.exists(cls._seed_dir):
            shutil.rmtree(cls._seed_dir, ignore_errors=True)

    def setUp(self):
        try:
            from ai.off_policy_train_loop import OffPolicyTrainLoop
            self.OffPolicyTrainLoop = OffPolicyTrainLoop
        except ImportError:
            self.OffPolicyTrainLoop = None
        if self.OffPolicyTrainLoop is None:
            self.skipTest("OffPolicyTrainLoop 尚未实现")

        from ai.models.cnn_v11.model import CNNV11Model

        self._tmp_root = tempfile.mkdtemp(prefix="off_policy_opl_")

        # 临时模型目录：复制 seed 权重到 gen1/
        self._model_dir = os.path.join(self._tmp_root, "models")
        gen1_dir = os.path.join(self._model_dir, "gen1")
        os.makedirs(gen1_dir, exist_ok=True)
        shutil.copyfile(
            os.path.join(self._seed_dir, "seed.pth"),
            os.path.join(gen1_dir, "cnn_v11.pth"),
        )

        # 临时样本目录：100 条 sample.json
        self._data_dir = os.path.join(self._tmp_root, "data")
        os.makedirs(self._data_dir, exist_ok=True)
        self._create_sample_json(self._data_dir, n_samples=100)

        # 临时绘图目录：重定向 MCTS_LOSS_LOG_DIR
        import config.mcts as mcts_config
        self._plot_dir = os.path.join(self._tmp_root, "loss_logs")
        self._orig_plot_dir = mcts_config.MCTS_LOSS_LOG_DIR
        mcts_config.MCTS_LOSS_LOG_DIR = self._plot_dir
        self.addCleanup(setattr, mcts_config, "MCTS_LOSS_LOG_DIR", self._orig_plot_dir)

        # patch 模型基目录到临时目录
        self._patcher = patch.object(
            CNNV11Model, "DEFAULT_MODEL_PATH",
            os.path.join(self._model_dir, "cnn_v11.pth"),
        )
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

    def tearDown(self):
        if hasattr(self, "_tmp_root") and os.path.exists(self._tmp_root):
            shutil.rmtree(self._tmp_root, ignore_errors=True)

    # ---------- 样本构造辅助 ----------

    @classmethod
    def _create_sample_json(cls, output_dir, n_samples=100):
        """创建测试用 sample.json（state 226 维 / pi one-hot / z ∈ {±1}）

        pi 采用 one-hot 集中分布（pi[i%225]=0.9），使策略目标有可学习模式，
        保证 TC-OPL-12 的 loss 下降断言稳定（均匀分布 pi 无可学习模式）。
        """
        samples = []
        for i in range(n_samples):
            state = [0] * 225
            state[i % 225] = 1 if i % 2 == 0 else 2
            state[(i * 7 + 3) % 225] = 2 if i % 2 == 0 else 1
            state = state + [1 if i % 2 == 0 else 2]
            pi = [0.1 / 224.0] * 225
            pi[i % 225] = 0.9
            z = 1.0 if i % 3 == 0 else -1.0
            samples.append({"state": state, "pi": pi, "z": z})
        with open(os.path.join(output_dir, "sample.json"), "w", encoding="utf-8") as f:
            json.dump(samples, f)

    # ---------- 断言辅助 ----------

    def assert_gen2_saved(self, result=None):
        """断言 gen2/ 下已保存可加载的 pth 与 torchscript"""
        pth_path = os.path.join(self._model_dir, "gen2", "cnn_v11.pth")
        ts_path = os.path.join(self._model_dir, "gen2", "cnn_v11_torchscript.pt")
        self.assertTrue(os.path.isfile(pth_path), f"cnn_v11.pth 未生成: {pth_path}")
        self.assertTrue(os.path.isfile(ts_path), f"cnn_v11_torchscript.pt 未生成: {ts_path}")
        self.assertGreater(os.path.getsize(pth_path), 0)
        self.assertGreater(os.path.getsize(ts_path), 0)
        # pth 可加载
        state_dict = torch.load(pth_path, weights_only=True, map_location="cpu")
        self.assertGreater(len(state_dict), 0)
        # torchscript 可加载
        torch.jit.load(ts_path, map_location="cpu")

    def _make_loop(self, **kwargs):
        """构造 OffPolicyTrainLoop（默认 CPU + 最小参数）"""
        params = {
            "model_spec": "cnn_v11.1",
            "data_dir": self._data_dir,
            "batch_size": 32,
            "epochs": 2,
            "val_split": 0.1,
            "lr": 0.001,
            "weight_decay": 1e-4,
            "device": "cpu",
        }
        params.update(kwargs)
        return self.OffPolicyTrainLoop(**params)


class TestOffPolicyTrainLoopBasic(OffPolicyTrainLoopTestBase):
    """TC-OPL-01~02：正常训练与覆盖保存"""

    def test_normal_training_saves_new_gen(self):
        """TC-OPL-01：正常训练保存 gen2，结果字段正确"""
        loop = self._make_loop(epochs=2)
        result = loop.run()

        self.assertEqual(result.saved_gen, 2)
        self.assertEqual(result.saved_spec, "cnn_v11.2")
        self.assertEqual(result.requested_epochs, 2)
        self.assertEqual(result.total_epochs, 2)
        self.assertFalse(result.interrupted)
        self.assertGreater(result.final_total_loss, 0.0)
        self.assertTrue(result.final_val_loss is not None)
        self.assertEqual(len(result.history), 2)
        self.assert_gen2_saved(result)

    def test_existing_gen2_overwritten(self):
        """TC-OPL-02：gen2 已存在时覆盖并打印提示"""
        # 预置 gen2 旧权重（文本文件，加载必然失败 → 覆盖成功即可区分）
        gen2_dir = os.path.join(self._model_dir, "gen2")
        os.makedirs(gen2_dir, exist_ok=True)
        old_pth = os.path.join(gen2_dir, "cnn_v11.pth")
        old_ts = os.path.join(gen2_dir, "cnn_v11_torchscript.pt")
        with open(old_pth, "w", encoding="utf-8") as f:
            f.write("OLD_WEIGHTS")
        with open(old_ts, "w", encoding="utf-8") as f:
            f.write("OLD_TS")

        loop = self._make_loop(epochs=1)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = loop.run()

        # 打印"已存在，将覆盖"提示
        self.assertIn("已存在", buf.getvalue())
        self.assertIn("覆盖", buf.getvalue())
        # 旧文件已被真实权重覆盖
        state_dict = torch.load(old_pth, weights_only=True, map_location="cpu")
        self.assertGreater(len(state_dict), 0)
        torch.jit.load(old_ts, map_location="cpu")
        self.assertEqual(result.saved_gen, 2)


class TestOffPolicyTrainLoopErrors(OffPolicyTrainLoopTestBase):
    """TC-OPL-03~05：构造与运行错误路径"""

    def test_model_not_exist_raises_value_error(self):
        """TC-OPL-03：模型不存在时抛 ValueError 且信息含 spec"""
        with self.assertRaises(ValueError) as ctx:
            self._make_loop(model_spec="cnn_v11.99")
        self.assertIn("cnn_v11.99", str(ctx.exception))

    def test_non_cnn_v11_spec_raises_value_error(self):
        """TC-OPL-04：非 cnn_v11 spec 抛 ValueError，信息明确仅支持 cnn_v11"""
        with self.assertRaises(ValueError) as ctx:
            self._make_loop(model_spec="cnn_v7.1")
        self.assertIn("cnn_v11", str(ctx.exception))

    def test_empty_samples_raises_value_error(self):
        """TC-OPL-05：样本为 0 时 run() 抛 ValueError，不产生 gen 目录"""
        empty_dir = os.path.join(self._tmp_root, "empty_data")
        os.makedirs(empty_dir, exist_ok=True)
        loop = self._make_loop(data_dir=empty_dir, epochs=1)

        with self.assertRaises(ValueError) as ctx:
            loop.run()
        self.assertIn("无有效样本", str(ctx.exception))
        # 不产生任何 gen 目录
        self.assertFalse(os.path.exists(os.path.join(self._model_dir, "gen2")))


class TestOffPolicyTrainLoopModes(OffPolicyTrainLoopTestBase):
    """TC-OPL-06~09：验证集、打印、曲线图"""

    def test_val_split_zero_no_val_fields(self):
        """TC-OPL-06：val_split=0 时 history 无 val_* 字段，final_val_loss 为 None"""
        loop = self._make_loop(epochs=2, val_split=0.0)
        result = loop.run()

        self.assertEqual(result.total_epochs, 2)
        self.assertIsNone(result.final_val_loss)
        for entry in result.history:
            self.assertNotIn("val_total_loss", entry)
            self.assertNotIn("val_value_loss", entry)
            self.assertNotIn("val_policy_loss", entry)

    def test_val_split_positive_computes_val(self):
        """TC-OPL-07：val_split=0.1 时 history 含 val_*，final_val_loss 为有限值"""
        loop = self._make_loop(epochs=2, val_split=0.1)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = loop.run()

        self.assertTrue(result.final_val_loss is not None)
        self.assertTrue(math.isfinite(result.final_val_loss))
        # 每 epoch 打印含 val_total_loss
        self.assertIn("val_total_loss:", buf.getvalue())
        for entry in result.history:
            self.assertIn("val_value_loss", entry)
            self.assertIn("val_policy_loss", entry)
            self.assertIn("val_total_loss", entry)

    def test_epoch_print_contains_grad_norm(self):
        """TC-OPL-08：每个 epoch 打印行含 grad_norm 字段且数值 >= 0"""
        loop = self._make_loop(epochs=1)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = loop.run()

        output = buf.getvalue()
        self.assertIn("grad_norm:", output)
        for entry in result.history:
            self.assertIn("grad_norm", entry)
            self.assertGreaterEqual(entry["grad_norm"], 0.0)

    def test_loss_curve_png_generated(self):
        """TC-OPL-09：正常训练后在临时绘图目录生成非空 PNG"""
        loop = self._make_loop(epochs=2)
        loop.run()

        png_files = glob.glob(os.path.join(self._plot_dir, "cnn_v11_*.png"))
        self.assertEqual(len(png_files), 1, f"应生成 1 个曲线图: {png_files}")
        self.assertGreater(os.path.getsize(png_files[0]), 0)


class TestOffPolicyTrainLoopInterruptAndFallback(OffPolicyTrainLoopTestBase):
    """TC-OPL-10~11：中断与设备回退"""

    def test_interrupt_during_first_epoch_still_saves(self):
        """TC-OPL-10：第 1 epoch 中断后仍保存模型，标记中断，曲线图不生成"""
        from ai.off_policy_train_loop import OffPolicyTrainLoop

        loop = self._make_loop(epochs=100)

        def raiser(*args, **kwargs):
            raise KeyboardInterrupt()

        with patch.object(loop._trainer, "_train_step", side_effect=raiser):
            result = loop.run()

        self.assertTrue(result.interrupted)
        self.assertEqual(result.total_epochs, 0)
        self.assertLess(result.total_epochs, result.requested_epochs)
        self.assertEqual(result.requested_epochs, 100)
        # 模型仍保存且可加载
        self.assert_gen2_saved(result)
        # history 为空 → 曲线图不生成
        self.assertEqual(result.history, [])
        self.assertEqual(glob.glob(os.path.join(self._plot_dir, "*.png")), [])

    def test_device_fallback_to_cpu(self):
        """TC-OPL-11：cuda 不可用时回退 CPU 并打印警告"""
        with patch("torch.cuda.is_available", return_value=False):
            # 警告在构造（设备回退检查）时打印，捕获范围需覆盖构造与训练
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                loop = self._make_loop(device="cuda", epochs=1)
                self.assertEqual(str(loop._trainer.device), "cpu")
                result = loop.run()

        self.assertIn("回退", buf.getvalue())
        self.assertIn("CPU", buf.getvalue())
        self.assertEqual(result.total_epochs, 1)
        self.assert_gen2_saved(result)


class TestOffPolicyTrainLoopDataFlow(OffPolicyTrainLoopTestBase):
    """TC-OPL-12~13：loss 趋势与批次切分"""

    def test_training_loss_decreases(self):
        """TC-OPL-12：100 条样本训练 5 epochs，total_loss 末尾 < 首项"""
        loop = self._make_loop(epochs=5, val_split=0.0)
        result = loop.run()

        self.assertEqual(len(result.history), 5)
        losses = [entry["total_loss"] for entry in result.history]
        self.assertLess(losses[-1], losses[0],
                        f"末尾 loss 应小于首项: {losses}")

    def test_batch_split_correct(self):
        """TC-OPL-13：100 条样本 batch 32 → 每 epoch 恰好 4 个 batch"""
        loop = self._make_loop(epochs=1, val_split=0.0)
        real_step = loop._trainer._train_step
        call_count = {"n": 0}

        def counting_step(*args, **kwargs):
            call_count["n"] += 1
            return real_step(*args, **kwargs)

        with patch.object(loop._trainer, "_train_step", side_effect=counting_step):
            result = loop.run()

        # 100 / 32 = 4 batch（32+32+32+4）
        self.assertEqual(call_count["n"], 4)
        self.assertEqual(len(result.history), 1)
        # history 各指标为有限数值（batch 均值）
        for entry in result.history:
            self.assertTrue(math.isfinite(entry["total_loss"]))
            self.assertTrue(math.isfinite(entry["value_loss"]))
            self.assertTrue(math.isfinite(entry["policy_loss"]))


if __name__ == "__main__":
    unittest.main()
