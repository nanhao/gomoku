"""CNN v11 Model 单元测试

验证 CNNV11Model 的构造、推理、保存/加载、批量推理。
遵循 TDD 流程，先写测试再实现（当前为 RED 阶段）。

测试覆盖 TC2.1 ~ TC2.12（设计文档 test_cases.md 二）。
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch

from ai.base_model import ModelPredict


class TestCNNV11Model(unittest.TestCase):
    """CNNV11Model 单元测试"""

    @classmethod
    def setUpClass(cls):
        """导入模块"""
        from ai.models.cnn_v11.model import CNNV11Model, CNNV11Network
        cls.ModelCls = CNNV11Model
        cls.NetworkCls = CNNV11Network

    # ---------- TC2.1 构造器 — 默认创建新模型 ----------

    def test_constructor_fresh(self):
        """fresh=True 创建模型，name 正确，network 为 CNNV11Network"""
        model = self.ModelCls(fresh=True)
        self.assertEqual(model.name, "cnn_v11")
        self.assertIsInstance(model.network, self.NetworkCls)

    # ---------- TC2.2 构造器 — 加载已保存权重 ----------

    def test_constructor_load(self):
        """指定 model_path 加载已保存权重"""
        # 先保存一个模型
        tmp_dir = tempfile.mkdtemp()
        try:
            pth_path = os.path.join(tmp_dir, "cnn_v11.pth")
            model = self.ModelCls(fresh=True)
            model.save(pth_path)
            # 加载
            model2 = self.ModelCls(model_path=pth_path)
            self.assertIsInstance(model2.network, self.NetworkCls)
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ---------- TC2.3 构造器 — 权重文件不存在 ----------

    def test_constructor_file_not_found(self):
        """不存在 model_path 抛出 FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            self.ModelCls(model_path="nonexistent_file.pth")

    # ---------- TC2.4 构造器 — gen 参数 ----------

    def test_constructor_gen(self):
        """gen=1 加载 gen1 目录下的权重"""
        # 使用临时目录避免污染真实 model 目录
        import shutil
        temp_dir = tempfile.mkdtemp()
        gen1_dir = os.path.join(temp_dir, "gen1")
        os.makedirs(gen1_dir, exist_ok=True)
        pth_path = os.path.join(gen1_dir, "cnn_v11.pth")
        # DEFAULT_MODEL_PATH 设置为 temp_dir 下的路径（不含 gen1），
        # NeuralNetworkModel.__init__ 在 gen=1 时会自动追加 gen1/ 子目录
        fake_default_path = os.path.join(temp_dir, "cnn_v11.pth")

        try:
            model = self.ModelCls(fresh=True)
            original_path = self.ModelCls.DEFAULT_MODEL_PATH
            self.ModelCls.DEFAULT_MODEL_PATH = fake_default_path
            try:
                model.save(pth_path)
                # gen=1 加载（DEFAULT_MODEL_PATH/gen1/cnn_v11.pth → 临时文件）
                model2 = self.ModelCls(gen=1)
                self.assertEqual(model2.gen, 1)
                self.assertEqual(model2.full_name, "cnn_v11.1")
            finally:
                self.ModelCls.DEFAULT_MODEL_PATH = original_path
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ---------- TC2.5 predict — 空棋盘输出 ----------

    def test_predict_empty_board(self):
        """空棋盘 predict 返回 TYPE_POLICY_VALUE"""
        model = self.ModelCls(fresh=True)
        result = model.predict([0] * 226)
        self.assertEqual(result.type, ModelPredict.TYPE_POLICY_VALUE)
        self.assertEqual(len(result.policy), 225)
        self.assertTrue(-1.0 <= result.value <= 1.0,
                        f"value {result.value} 应在 [-1, 1]")

    # ---------- TC2.6 predict — 策略概率和为 1 ----------

    def test_predict_probability_sum(self):
        """predict 返回的策略概率和为 1"""
        model = self.ModelCls(fresh=True)
        # 一些非空棋盘
        bv = [0] * 226
        bv[0] = 1   # 黑棋
        bv[15] = 2  # 白棋
        bv[225] = 1  # 黑方落子
        result = model.predict(bv)
        prob_sum = sum(result.policy)
        self.assertAlmostEqual(prob_sum, 1.0, places=4,
                               msg=f"概率和 {prob_sum} ≠ 1.0")
        for p in result.policy:
            self.assertGreater(p, 0, f"概率应 > 0，实际 {p}")

    # ---------- TC2.7 predict_batch — 批量推理 ----------

    def test_predict_batch(self):
        """predict_batch 返回等长结果列表"""
        model = self.ModelCls(fresh=True)
        bvs = [[0] * 226 for _ in range(3)]
        bvs[0][225] = 1  # 黑方
        bvs[1][225] = 2  # 白方
        results = model.predict_batch(bvs)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertEqual(r.type, ModelPredict.TYPE_POLICY_VALUE)
            self.assertEqual(len(r.policy), 225)

    # ---------- TC2.8 predict_batch — 空输入 ----------

    def test_predict_batch_empty(self):
        """空列表输入返回空列表"""
        model = self.ModelCls(fresh=True)
        results = model.predict_batch([])
        self.assertEqual(results, [])

    # ---------- TC2.9 predict_batch — 与逐条 predict 一致 ----------

    def test_predict_batch_consistency(self):
        """predict_batch 与逐条 predict 结果一致"""
        model = self.ModelCls(fresh=True)
        bvs = [[0] * 226 for _ in range(3)]
        bvs[0][0] = 1
        bvs[1][15] = 2
        bvs[2][30] = 1
        for i in range(3):
            bvs[i][225] = 1

        batch_results = model.predict_batch(bvs)
        single_results = [model.predict(bv) for bv in bvs]

        for i in range(3):
            for j in range(225):
                self.assertAlmostEqual(
                    batch_results[i].policy[j],
                    single_results[i].policy[j],
                    places=5,
                    msg=f"样本 {i} 位置 {j} 不一致")
            self.assertAlmostEqual(
                batch_results[i].value,
                single_results[i].value,
                places=5,
                msg=f"样本 {i} value 不一致")

    # ---------- TC2.10 save & load — 往返一致性 ----------

    def test_save_load_roundtrip(self):
        """save 后 load 的输出一致"""
        tmp_dir = tempfile.mkdtemp()
        try:
            pth_path = os.path.join(tmp_dir, "roundtrip.pth")
            model = self.ModelCls(fresh=True)
            bv = [0] * 226
            bv[0] = 1
            bv[225] = 1

            result_before = model.predict(bv)

            # 保存
            saved = model.save(pth_path)
            self.assertEqual(saved, pth_path)
            self.assertTrue(os.path.exists(pth_path))

            # 加载
            model2 = self.ModelCls(model_path=pth_path)
            result_after = model2.predict(bv)

            # 比较
            for j in range(225):
                self.assertAlmostEqual(
                    result_before.policy[j],
                    result_after.policy[j],
                    places=5,
                    msg=f"位置 {j} policy 不一致")
            self.assertAlmostEqual(
                result_before.value, result_after.value, places=5)
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ---------- TC2.11 move_to_device — GPU 可用时迁移 ----------

    def test_move_to_device(self):
        """move_to_device 迁移到 CPU（GPU 不可随时默认 auto→CPU）"""
        model = self.ModelCls(fresh=True)
        device = model.move_to_device()
        # 所有参数应在同一设备上
        for p in model.network.parameters():
            self.assertEqual(p.device.type, device.type)

    # ---------- TC2.12 _prepare_input — 相对视角编码 ----------

    def test_prepare_input_relative_encoding(self):
        """_prepare_input 使用相对视角编码"""
        model = self.ModelCls(fresh=True)
        # 黑方视角：位置 0 有黑棋(1)
        bv = [0] * 226
        bv[0] = 1    # 黑棋在 (0,0)
        bv[225] = 1  # 黑方落子
        tensor = model._prepare_input(bv, torch.device("cpu"))
        self.assertEqual(tensor.shape, (1, 3, 15, 15))
        # 通道 0（我方=黑棋）：位置 (0,0,0,0) = 1.0
        self.assertAlmostEqual(tensor[0, 0, 0, 0].item(), 1.0, places=5)
        # 通道 1（对方=白棋）：全 0
        self.assertEqual(tensor[0, 1, 0, 0].item(), 0.0)
        # 通道 2（颜色指示器：黑方=1）：全 1
        self.assertAlmostEqual(tensor[0, 2, 0, 0].item(), 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
