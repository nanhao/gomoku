"""FC v1 批量训练与张量准备测试

测试 train_batch 方法和 _prepare_tensors 的 mask 逻辑：
- TestFCv1TrainerBatch: 批量训练主流程与边缘测试
- TestFCv1TrainerPrepareTensors: 张量准备与 mask 一致性
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
import copy

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.models.fc_v1.trainer import FCv1Trainer
from ai.models.fc_v1.model import FCv1Model
from tests.ai.helpers import create_output


class TestFCv1TrainerBatch(unittest.TestCase):
    """测试 FCv1Trainer.train_batch 方法
    
    主流程：多个 sample 文件合并训练，每 epoch 只做一次参数更新
    """
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        # 隔离默认模型路径，避免加载真实模型
        self.original_default_path = FCv1Model.DEFAULT_MODEL_PATH
        FCv1Model.DEFAULT_MODEL_PATH = os.path.join(self.temp_dir, "fc_v1.pth")
    
    def tearDown(self):
        """测试后清理"""
        FCv1Model.DEFAULT_MODEL_PATH = self.original_default_path
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_sample_dir(self, name: str, n_items: int = 2) -> str:
        """创建一个包含 sample.json 的目录"""
        sample_dir = os.path.join(self.temp_dir, name)
        os.makedirs(sample_dir, exist_ok=True)
            
        sample_data = []
        for i in range(n_items):
            # 使用非零输入和非零目标，确保梯度不为零
            output = [0.0] * 225
            output[0] = 1000.0  # 索引 0 对应位置 (0,0) 的 reward 非零
            sample_data.append({
                "input": [float(i + 1)] * 226,
                "output": output,
                "mask": [1] + [0] * 224
            })
            
        sample_path = os.path.join(sample_dir, "sample.json")
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f)
            
        return sample_dir
    
    # ------- 主流程测试 -------
    
    def test_train_batch_returns_float(self):
        """测试 train_batch 返回 float 损失值"""
        dir1 = self._create_sample_dir("s1")
        dir2 = self._create_sample_dir("s2")
        trainer = FCv1Trainer(fresh=True)
        
        result = trainer.train_batch([dir1, dir2], epochs=3)
        
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
    
    def test_train_batch_single_sample_dir(self):
        """测试 train_batch 处理单个 sample 目录"""
        dir1 = self._create_sample_dir("s1")
        trainer = FCv1Trainer(fresh=True)
        
        result = trainer.train_batch([dir1], epochs=2)
        
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
    
    def test_train_batch_multiple_sample_dirs_merged(self):
        """测试 train_batch 将多个 sample 目录数据合并训练
        
        验证：每 epoch 只更新一次参数（而不是对每个 sample 文件分别训练）
        通过检查 3 个 sample 目录 x 2 epochs 只输出 0 次 loss（<200 不输出）
        """
        dir1 = self._create_sample_dir("s1")
        dir2 = self._create_sample_dir("s2")
        dir3 = self._create_sample_dir("s3")
        trainer = FCv1Trainer(fresh=True)
        
        import io
        captured = io.StringIO()
        sys.stdout = captured
        try:
            trainer.train_batch([dir1, dir2, dir3], epochs=2, verbose=True)
            output = captured.getvalue()
        finally:
            sys.stdout = sys.__stdout__
        
        # 2 个 epoch 均小于 200，所以不应有任何 Epoch 输出
        lines = [l for l in output.strip().split('\n') if 'Epoch' in l]
        self.assertEqual(len(lines), 0)
    
    def test_train_batch_verbose_outputs_epoch_loss(self):
        """测试 verbose=True 时每 200 个 epoch 输出一次 loss"""
        dir1 = self._create_sample_dir("s1")
        dir2 = self._create_sample_dir("s2")
        trainer = FCv1Trainer(fresh=True)
        
        import io
        captured = io.StringIO()
        sys.stdout = captured
        try:
            # 使用 400 个 epoch，应在第 200 和第 400 个 epoch 输出
            trainer.train_batch([dir1, dir2], epochs=400, verbose=True)
            output = captured.getvalue()
        finally:
            sys.stdout = sys.__stdout__
        
        self.assertIn("Epoch 200/400", output)
        self.assertIn("Epoch 400/400", output)
        self.assertIn("loss", output.lower())
        # 确认只输出了 2 行
        lines = [l for l in output.strip().split('\n') if 'Epoch' in l]
        self.assertEqual(len(lines), 2)
    
    def test_train_batch_verbose_false_no_epoch_output(self):
        """测试 verbose=False 时不输出 epoch 信息"""
        dir1 = self._create_sample_dir("s1")
        trainer = FCv1Trainer(fresh=True)
        
        import io
        captured = io.StringIO()
        sys.stdout = captured
        try:
            trainer.train_batch([dir1], epochs=3, verbose=False)
            output = captured.getvalue()
        finally:
            sys.stdout = sys.__stdout__
        
        self.assertNotIn("Epoch", output)
    
    # ------- 边缘测试 -------
    
    def test_train_batch_empty_list_raises(self):
        """测试 sample_dirs 为空列表时抛出异常"""
        trainer = FCv1Trainer(fresh=True)
        with self.assertRaises(ValueError):
            trainer.train_batch([], epochs=1)
    
    def test_train_batch_missing_sample_file_raises(self):
        """测试 sample 目录不存在时抛出异常"""
        trainer = FCv1Trainer(fresh=True)
        fake_dir = os.path.join(self.temp_dir, "nonexistent")
        with self.assertRaises(FileNotFoundError):
            trainer.train_batch([fake_dir], epochs=1)
    
    def test_train_batch_updates_model_params(self):
        """测试训练后模型参数确实更新"""
        import torch
        dir1 = self._create_sample_dir("s1")
        dir2 = self._create_sample_dir("s2")
        trainer = FCv1Trainer(fresh=True)
        
        # 记录训练前的参数
        params_before = copy.deepcopy(
            list(trainer.model.network.parameters())[0].data.clone()
        )
        
        trainer.train_batch([dir1, dir2], epochs=5)
        
        # 训练后参数应已更新
        params_after = list(trainer.model.network.parameters())[0].data
        self.assertFalse(torch.equal(params_before, params_after))


class TestFCv1TrainerPrepareTensors(unittest.TestCase):
    """测试 _prepare_tensors 的 mask 逻辑
    
    output 格式为 225 维，索引 i 对应位置 (i//15, i%15)的 reward。
    mask 也是 225 维，直接对应到 output。
    """
    
    def setUp(self):
        self.trainer = FCv1Trainer(fresh=True)
    
    def _make_sample(self, mask_idx: int) -> dict:
        """创建只有第 mask_idx 个位置 mask=1 的样本"""
        output = [0.0] * 225
        output[mask_idx] = 999.0  # reward 非零
        mask = [0] * 225
        mask[mask_idx] = 1
        return {
            "input": [0.0] * 226,
            "output": output,
            "mask": mask
        }
    
    def test_prepare_tensors_output_shapes(self):
        """_prepare_tensors 返回的张量形状正确"""
        sample = self._make_sample(0)
        inputs_t, targets_t, masks_t = self.trainer._prepare_tensors([sample])
        
        # inputs: (1, 226), targets: (1, 225), masks: (1, 225)
        self.assertEqual(inputs_t.shape, (1, 226))
        self.assertEqual(targets_t.shape, (1, 225))
        self.assertEqual(masks_t.shape, (1, 225))
    
    def test_mask_tensor_matches_input_mask(self):
        """mask 张量应直接对应输入 mask"""
        for i in [0, 1, 7, 100, 223]:
            with self.subTest(mask_idx=i):
                sample = self._make_sample(i)
                _, _, masks_t = self.trainer._prepare_tensors([sample])
                
                # 只有第 i 个位置是 1
                self.assertEqual(masks_t[0, i].item(), 1.0,
                    f"masks_t[0, {i}] 应为 1")
                
                # 其他位置应为 0
                nonzero_count = (masks_t[0] != 0).sum().item()
                self.assertEqual(nonzero_count, 1)
    
    def test_masked_loss_consistent_with_compute_loss(self):
        """_masked_loss 与 _compute_loss 语义说明
        
        _compute_loss 是 Python 纯实现，不走 _prepare_tensors，接收原始值。
        _masked_loss 接收的 targets 是 Softmax 概率（由 Evaluator 生成）。
        两者在各自的语义下结果应正确。
        
        注意：现在 sample.json 中的 output 已经是 Softmax 概率，
        Trainer 不再进行归一化。
        """
        import torch
        
        # _compute_loss 使用原始值
        output_estimated = [0.0] * 225
        output_estimated[0] = 0.7  # 模拟 Softmax 概率
        output_target = [0.0] * 225
        output_target[0] = 0.5
        mask = [0] * 225
        mask[0] = 1
        
        # _compute_loss：|0.7 - 0.5| = 0.2
        py_loss = self.trainer._compute_loss(output_estimated, output_target, mask)
        self.assertAlmostEqual(py_loss, 0.2, places=4)
        
        # _masked_loss 接收的 targets 已经是 Softmax 概率
        # 直接读取，不做转换
        sample = {
            "input": [0.0] * 226,
            "output": output_target,  # 传入 Softmax 概率
            "mask": mask
        }
        inputs_t, targets_t, masks_t = self.trainer._prepare_tensors([sample])
        
        outputs_t = torch.tensor([output_estimated], dtype=torch.float32)
        # 将张量移到与训练器相同的设备
        outputs_t = outputs_t.to(self.trainer.device)
        tensor_loss = self.trainer._masked_loss(outputs_t, targets_t, masks_t)
        
        # targets 已经是概率：|0.7 - 0.5| = 0.2
        self.assertAlmostEqual(tensor_loss.item(), 0.2, places=4,
                               msg="targets 是 Softmax 概率，_masked_loss 应直接计算")


if __name__ == '__main__':
    unittest.main()
