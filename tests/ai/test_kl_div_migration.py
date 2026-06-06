"""KL 散度迁移验证测试

验证训练损失函数从 L1 Loss 迁移到 KL 散度的完整性：
- T3: SampleInspector 删除验证
- T6: 网络输出层验证（LogSoftmax、nn.Softmax 无残留、旧函数名清理）
- T7: LR Range Test 验证
- T9: 数值稳定性边界验证
"""

import math
import os
import sys
import unittest
import tempfile
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn

from ai.models.fc_v1.trainer import FCv1Trainer
from ai.models.fc_v1.model import FCv1Model
from ai.models.cnn_v3.model import CNNV3Model, CNNV3Network
from ai.sample_loader import compute_masked_kl_loss
from ai.lr_range_test import run_lr_range_test


class TestSampleInspectorDeletion(unittest.TestCase):
    """T3: SampleInspector 删除验证"""

    def test_sample_inspector_py_deleted(self):
        """T3-01: sample_inspector.py 已删除"""
        path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'ai', 'models', 'fc_v1', 'sample_inspector.py'
        )
        self.assertFalse(os.path.exists(path),
                         f"sample_inspector.py 应已删除，但仍存在于 {path}")

    def test_inspect_main_py_deleted(self):
        """T3-02: inspect_main.py 已删除"""
        path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'ai', 'models', 'fc_v1', 'inspect_main.py'
        )
        self.assertFalse(os.path.exists(path),
                         f"inspect_main.py 应已删除，但仍存在于 {path}")

    def test_test_fc_v1_inspect_py_deleted(self):
        """T3-03: test_fc_v1_inspect.py 已删除"""
        path = os.path.join(os.path.dirname(__file__), 'test_fc_v1_inspect.py')
        self.assertFalse(os.path.exists(path),
                         f"test_fc_v1_inspect.py 应已删除，但仍存在于 {path}")

    def test_no_sample_inspector_references(self):
        """T3-04: SampleInspector 无残留引用（搜索源码目录）"""
        # 搜索 ai/ 和 tests/ 目录下所有 .py 文件
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        found = []
        # 排除本测试文件自身（本文件包含 "SampleInspector" 是为了检查残留，不是残留引用）
        this_file = os.path.abspath(__file__)
        for root, _dirs, files in os.walk(os.path.join(project_root, 'ai')):
            for f in files:
                if f.endswith('.py'):
                    filepath = os.path.join(root, f)
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
                        content = fh.read()
                        if 'SampleInspector' in content:
                            found.append(filepath)
        for root, _dirs, files in os.walk(os.path.join(project_root, 'tests')):
            for f in files:
                if f.endswith('.py'):
                    filepath = os.path.join(root, f)
                    if os.path.abspath(filepath) == this_file:
                        continue  # 跳过本测试文件自身
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
                        content = fh.read()
                        if 'SampleInspector' in content:
                            found.append(filepath)
        self.assertEqual(len(found), 0,
                         f"SampleInspector 应无残留引用，但在以下文件中发现: {found}")


class TestNetworkOutputLayer(unittest.TestCase):
    """T6: 网络输出层验证"""

    def test_fcv1_forward_output_is_log_probability(self):
        """T6-01: FCv1Network forward 输出为 log-probability（值 <= 0，exp 后总和 ≈ 1）"""
        network = FCv1Model(fresh=True).network
        network.eval()
        input_tensor = torch.randn(1, 226)
        with torch.no_grad():
            output = network(input_tensor)
        # 所有输出值应 <= 0（log-probability 特征）
        self.assertTrue(torch.all(output <= 0),
                        f"LogSoftmax 输出应全部 <= 0，但存在 > 0 值")
        # exp 后总和应 ≈ 1
        prob_sum = output.exp().sum(dim=1).item()
        self.assertAlmostEqual(prob_sum, 1.0, places=5,
                               msg=f"exp(log-probability) 总和应为 1.0，实际为 {prob_sum}")

    def test_cnnv3_forward_output_is_log_probability(self):
        """T6-02: CNNV3Network forward 输出为 log-probability"""
        network = CNNV3Model(fresh=True).network
        network.eval()
        # CNN 输入格式: (1, 3, 15, 15)
        input_tensor = torch.randn(1, 3, 15, 15)
        with torch.no_grad():
            output = network(input_tensor)
        self.assertTrue(torch.all(output <= 0),
                        f"CNNV3 LogSoftmax 输出应全部 <= 0，但存在 > 0 值")
        prob_sum = output.exp().sum(dim=1).item()
        self.assertAlmostEqual(prob_sum, 1.0, places=5,
                               msg=f"CNNV3 exp(log-probability) 总和应为 1.0，实际为 {prob_sum}")

    def test_no_softmax_in_models(self):
        """T6-03: 无 nn.Softmax 残留（模型代码中）"""
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        models_dir = os.path.join(project_root, 'ai', 'models')
        found = []
        for root, _dirs, files in os.walk(models_dir):
            for f in files:
                if f.endswith('.py'):
                    filepath = os.path.join(root, f)
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
                        content = fh.read()
                        if 'nn.Softmax' in content:
                            found.append(filepath)
        self.assertEqual(len(found), 0,
                         f"模型代码中不应存在 nn.Softmax，但在以下文件中发现: {found}")

    def test_no_old_function_name(self):
        """T6-04: 旧函数名 compute_masked_l1_loss 无残留（源码和注释）"""
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        found = []
        this_file = os.path.abspath(__file__)  # 排除本测试文件自身
        # 搜索 ai/ 和 tests/ 下的 .py 文件
        for subdir in ['ai', 'tests']:
            search_dir = os.path.join(project_root, subdir)
            for root, _dirs, files in os.walk(search_dir):
                for f in files:
                    if f.endswith('.py'):
                        filepath = os.path.join(root, f)
                        if os.path.abspath(filepath) == this_file:
                            continue  # 跳过本测试文件自身
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
                            content = fh.read()
                            if 'compute_masked_l1_loss' in content:
                                found.append(filepath)
        self.assertEqual(len(found), 0,
                         f"compute_masked_l1_loss 应无残留引用，但在以下文件中发现: {found}")


class TestLRRangeTestVerification(unittest.TestCase):
    """T7: LR Range Test 验证"""

    def test_all_losses_are_finite(self):
        """T7-01: 所有 loss 为有限数（小规模 range test）"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 创建简单训练数据
            sample_data = []
            for i in range(2):
                output = [0.0] * 225
                output[0] = 0.5
                output[1] = 0.5
                sample_data.append({
                    "input": [float(i + 1)] * 226,
                    "output": output,
                    "mask": [1] * 225
                })
            merge_path = os.path.join(tmp_dir, "test_data.json")
            with open(merge_path, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f)

            lrs, losses = run_lr_range_test(
                model_name="cnn_v3",
                data_path=merge_path,
                rounds=3,
                epochs=2,
                min_lr=1e-7,
                max_lr=1e-3,
                device=torch.device('cpu')
            )

            for loss in losses:
                self.assertTrue(torch.isfinite(torch.tensor(loss)).item(),
                                f"loss 值 {loss} 不是有限数")

    def test_lr_range_test_criterion_is_kl_div(self):
        """T7-02: LR Range Test 使用 KLDivLoss"""
        # 验证 lr_range_test.py 中的 criterion 定义
        lr_range_test_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'ai', 'lr_range_test.py'
        )
        with open(lr_range_test_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 检查 KLDivLoss 存在
        self.assertIn('nn.KLDivLoss', content,
                       "lr_range_test.py 应使用 nn.KLDivLoss")
        # 检查 L1Loss 不存在
        self.assertNotIn('nn.L1Loss', content,
                          "lr_range_test.py 不应包含 nn.L1Loss")


class TestNumericalStability(unittest.TestCase):
    """T9: 数值稳定性边界验证"""

    def test_extreme_logits_loss_is_finite(self):
        """T9-01: 模型输出含极端值时 loss 仍有限"""
        trainer = FCv1Trainer(fresh=True)
        # 构造极端 logits：某个位置 logit 远大于其他位置
        extreme_logits = torch.zeros(1, 225)
        extreme_logits[0, 0] = 100.0  # 极端大值
        for i in range(1, 225):
            extreme_logits[0, i] = -100.0  # 极端小值

        # LogSoftmax 应正常处理极端 logits
        log_softmax = nn.LogSoftmax(dim=1)
        log_prob_output = log_softmax(extreme_logits)

        # 构造简单 target 和 mask
        target = torch.zeros(1, 225)
        target[0, 0] = 1.0  # 全部概率集中在位置 0
        mask = torch.ones(1, 225)

        loss = trainer._masked_loss(log_prob_output, target, mask)
        self.assertTrue(torch.isfinite(loss),
                        f"极端 logits 时 loss 应为有限数，实际为 {loss.item()}")

    def test_all_zero_target_with_mask(self):
        """T9-02: mask=1 位置 target 全为 0 时 _masked_loss 返回与 autograd 图连接的零损失"""
        trainer = FCv1Trainer(fresh=True)
        # 全零 target，mask=1
        outputs = torch.tensor([[math.log(0.5)] + [-10.0] * 224], dtype=torch.float32, requires_grad=True)
        targets = torch.zeros(1, 225)  # 全零 target
        masks = torch.ones(1, 225)  # 全部 mask=1

        loss = trainer._masked_loss(outputs, targets, masks)
        self.assertAlmostEqual(loss.item(), 0.0, places=5,
                               msg="mask=1 位置 target 全为 0 时，_masked_loss 应返回 0.0")
        # 验证返回的零损失仍连接 autograd 图，backward() 不会报错
        self.assertTrue(loss.requires_grad,
                       "防御返回的零损失应连接 autograd 图（requires_grad=True）")
        # 补充 backward() 可行性验证（Review T1）：确保 backward() 不报 "no grad_fn" 错误
        loss.backward()

    def test_compute_masked_kl_loss_target_zero_no_exception(self):
        """T9-03: compute_masked_kl_loss 中 target[i]=0 不抛异常"""
        log_estimated = [math.log(0.5)] + [0.0] * 224
        target = [0.0] * 225  # 全零 target
        mask = [1] * 225

        loss = compute_masked_kl_loss(log_estimated, target, mask)
        # 全零 target 时所有位置贡献为 0
        self.assertAlmostEqual(loss, 0.0, places=5,
                               msg="全零 target 时 compute_masked_kl_loss 应返回 0.0")


if __name__ == '__main__':
    unittest.main()