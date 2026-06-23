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


if __name__ == '__main__':
    unittest.main()