"""CNN v12 Model 单元测试

验证 CNNV12Model 的构造、推理、批量推理、存取往返、相对视角编码继承。
测试覆盖 TC2.1 ~ TC2.11（设计 61 test_cases.md 二）。

约束：需要落盘的用例一律写入 tempfile 临时目录并清理，
禁止读写 src/ai/models/cnn_v12 真实目录。
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch

from ai.base_model import ModelPredict


class TestCNNV12Model(unittest.TestCase):
    """CNNV12Model 单元测试"""

    @classmethod
    def setUpClass(cls):
        """导入被测与对照模块"""
        from ai.models.cnn_v11.model import CNNV11Model
        from ai.models.cnn_v12.model import CNNV12Model, CNNV12Network
        cls.ModelCls = CNNV12Model
        cls.NetworkCls = CNNV12Network
        cls.V11ModelCls = CNNV11Model

    @staticmethod
    def _make_board_vector() -> list:
        """构造一条合法的 226 维 board_vector（黑棋 (0,0)、白棋 (1,0)、黑方落子）"""
        bv = [0] * 226
        bv[0] = 1     # (0,0) 黑棋
        bv[15] = 2    # (1,0) 白棋
        bv[225] = 1   # 当前黑方落子
        return bv

    # ---------- TC2.1 模型名 ----------

    def test_model_name(self):
        """name 恒为 cnn_v12"""
        model = self.ModelCls(fresh=True)
        self.assertEqual(model.name, "cnn_v12")

    # ---------- TC2.2 默认权重路径 ----------

    def test_default_model_path(self):
        """DEFAULT_MODEL_PATH 指向 cnn_v12 包目录下的 cnn_v12.pth"""
        path = self.ModelCls.DEFAULT_MODEL_PATH
        self.assertTrue(path.endswith("cnn_v12.pth"),
                        f"路径应以 cnn_v12.pth 结尾，实际 {path}")
        self.assertEqual(os.path.basename(os.path.dirname(path)), "cnn_v12",
                         f"父目录名应为 cnn_v12，实际 {path}")

    # ---------- TC2.3 预测结果类型 ----------

    def test_predict_result_type(self):
        """predict 返回 TYPE_POLICY_VALUE"""
        model = self.ModelCls(fresh=True)
        result = model.predict(self._make_board_vector())
        self.assertEqual(result.type, ModelPredict.TYPE_POLICY_VALUE)

    # ---------- TC2.4 预测结果语义 ----------

    def test_predict_result_semantics(self):
        """policy 长度 225 且为概率分布，value 落在 [-1, 1]"""
        model = self.ModelCls(fresh=True)
        result = model.predict(self._make_board_vector())

        self.assertEqual(len(result.policy), 225)
        for prob in result.policy:
            self.assertGreater(prob, 0.0, f"概率应 > 0，实际 {prob}")
            self.assertLessEqual(prob, 1.0, f"概率应 ≤ 1，实际 {prob}")
        self.assertAlmostEqual(sum(result.policy), 1.0, places=4)
        self.assertGreaterEqual(result.value, -1.0)
        self.assertLessEqual(result.value, 1.0)

    # ---------- TC2.5 代数路径改写 ----------

    def test_gen_path_rewrite(self):
        """gen=3 时 DEFAULT_MODEL_PATH 改写为 gen3 目录（纯字符串，无文件 IO）"""
        model = self.ModelCls(fresh=True, gen=3)
        self.assertIn("gen3", model.DEFAULT_MODEL_PATH.replace("\\", "/"),
                      f"DEFAULT_MODEL_PATH 应包含 gen3，实际 {model.DEFAULT_MODEL_PATH}")
        self.assertEqual(model.gen, 3)

    # ---------- TC2.6 缺权重报错 ----------

    def test_gen_missing_weight_raises(self):
        """gen 指向的权重缺失时抛 FileNotFoundError

        用临时目录替换 DEFAULT_MODEL_PATH，确保该场景不受仓库现有
        gen 目录状态影响，且不触碰真实模型目录。
        """
        with tempfile.TemporaryDirectory(prefix="cnn_v12_missing_") as tmp:
            fake_default = os.path.join(tmp, "cnn_v12.pth")
            with patch.object(self.ModelCls, "DEFAULT_MODEL_PATH", fake_default):
                with self.assertRaises(FileNotFoundError):
                    self.ModelCls(gen=99)

    # ---------- TC2.7 完整名 ----------

    def test_full_name(self):
        """gen=2 时 full_name 为 cnn_v12.2"""
        model = self.ModelCls(fresh=True, gen=2)
        self.assertEqual(model.full_name, "cnn_v12.2")

    # ---------- TC2.8 批量与单条一致 ----------

    def test_predict_batch_consistency(self):
        """predict_batch 与逐条 predict 逐元素差 < 1e-5"""
        model = self.ModelCls(fresh=True)

        board_vectors = []
        for offset in (0, 15, 30):
            bv = [0] * 226
            bv[offset] = 1
            bv[225] = 1
            board_vectors.append(bv)

        batch_results = model.predict_batch(board_vectors)
        single_results = [model.predict(bv) for bv in board_vectors]

        self.assertEqual(len(batch_results), 3)
        for idx in range(3):
            for pos in range(225):
                self.assertAlmostEqual(
                    batch_results[idx].policy[pos],
                    single_results[idx].policy[pos],
                    places=5,
                    msg=f"样本 {idx} 位置 {pos} policy 不一致")
            self.assertAlmostEqual(
                batch_results[idx].value,
                single_results[idx].value,
                places=5,
                msg=f"样本 {idx} value 不一致")

    # ---------- TC2.9 空批量 ----------

    def test_predict_batch_empty(self):
        """空列表输入返回空列表，不抛异常"""
        model = self.ModelCls(fresh=True)
        self.assertEqual(model.predict_batch([]), [])

    # ---------- TC2.10 存取往返 ----------

    def test_save_load_roundtrip(self):
        """save 到临时目录 → load → 预测结果完全一致"""
        model = self.ModelCls(fresh=True)
        board_vector = self._make_board_vector()
        before = model.predict(board_vector)

        with tempfile.TemporaryDirectory(prefix="cnn_v12_roundtrip_") as tmp:
            pth_path = os.path.join(tmp, "cnn_v12.pth")
            saved_path = model.save(pth_path)
            self.assertEqual(saved_path, pth_path)
            self.assertTrue(os.path.exists(pth_path))

            reloaded = self.ModelCls(model_path=pth_path)
            after = reloaded.predict(board_vector)

        for pos in range(225):
            self.assertAlmostEqual(
                before.policy[pos], after.policy[pos], places=5,
                msg=f"位置 {pos} policy 不一致")
        self.assertAlmostEqual(before.value, after.value, places=5)

    # ---------- TC2.11 相对视角编码与 cnn_v11 逐位一致 ----------

    def test_prepare_input_matches_v11(self):
        """同一 board_vector 在 v11 与 v12 上得到完全相同的输入张量"""
        board_vector = self._make_board_vector()
        device = torch.device("cpu")

        v11_model = self.V11ModelCls(fresh=True)
        v12_model = self.ModelCls(fresh=True)

        v11_tensor = v11_model._prepare_input(board_vector, device)
        v12_tensor = v12_model._prepare_input(board_vector, device)

        self.assertEqual(tuple(v11_tensor.shape), (1, 3, 15, 15))
        self.assertEqual(tuple(v12_tensor.shape), (1, 3, 15, 15))
        self.assertTrue(torch.equal(v11_tensor, v12_tensor),
                        "v12 相对视角编码应与 v11 逐位一致（未重写编码）")


if __name__ == '__main__':
    unittest.main()
