"""FC v1 Softmax 概率输入测试

测试 sample.json 中 output 为 Softmax 概率分布时的训练行为：
- _prepare_tensors 直接读取概率，不做转换
- 概率总和为 1
- loss 量级合理
- 部分 mask 时处理正确
- 训练可收敛
"""

import os
import sys
import unittest
import tempfile
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.models.fc_v1.trainer import FCv1Trainer
from unittest.mock import MagicMock


class TestSoftmaxProbabilityInput(unittest.TestCase):
    """Softmax 概率输入测试
    
    主流程：sample.json 中的 output 已经是 Softmax 概率分布（由 Evaluator 生成），
    Trainer 直接读取使用，不再进行归一化。
    """

    def setUp(self):
        """tests 前准备：创建 fresh 训练器"""
        self.trainer = FCv1Trainer(fresh=True)

    # ---------- primary tests ----------

    def test_targets_are_softmax_probabilities(self):
        """[primary] _prepare_tensors 应直接读取 Softmax 概率，不做转换"""
        # 模拟 Evaluator 生成的 Softmax 概率分布
        # 3个空位，概率分别为 0.7, 0.2, 0.1
        output = [0.0] * 225
        output[0] = 0.7
        output[1] = 0.2
        output[2] = 0.1
        
        sample_data = [
            {
                "input": [0] * 225 + [1],
                "output": output,
                "mask": [1] * 225
            }
        ]
        _, targets_tensor, _ = self.trainer._sample_loader.prepare_tensors(sample_data)

        # 应直接读取，不做转换
        self.assertAlmostEqual(targets_tensor[0][0].item(), 0.7, places=5,
                               msg="Softmax 概率应直接读取")
        self.assertAlmostEqual(targets_tensor[0][1].item(), 0.2, places=5,
                               msg="Softmax 概率应直接读取")
        self.assertAlmostEqual(targets_tensor[0][2].item(), 0.1, places=5,
                               msg="Softmax 概率应直接读取")

    def test_softmax_probabilities_sum_to_one(self):
        """[primary] Softmax 概率总和应为 1（mask=1 的位置）"""
        # 模拟均匀分布的 Softmax 概率
        prob = 1.0 / 225.0
        output = [prob] * 225
        
        sample_data = [
            {
                "input": [0] * 225 + [1],
                "output": output,
                "mask": [1] * 225
            }
        ]
        _, targets_tensor, _ = self.trainer._sample_loader.prepare_tensors(sample_data)

        # 验证概率总和为 1
        prob_sum = targets_tensor[0].sum().item()
        self.assertAlmostEqual(prob_sum, 1.0, places=5,
                               msg="Softmax 概率总和应为 1")

    def test_loss_scale_with_softmax_input(self):
        """[primary] LogSoftmax 输出 + Softmax target 时 KL 散度 loss 的量级应在合理范围内"""
        # 使用 Softmax 概率作为输入
        output = [0.0] * 225
        output[0] = 0.9  # 高概率位置
        for i in range(1, 225):
            output[i] = 0.1 / 224  # 其余位置均分剩余概率

        sample_data = [
            {
                "input": [0] * 225 + [1],
                "output": output,
                "mask": [1] * 225
            }
        ]
        inputs_tensor, targets_tensor, masks_tensor = self.trainer._sample_loader.prepare_tensors(sample_data)

        # 模型输出是 LogSoftmax（log-probability），target 是 Softmax 概率
        # KLDivLoss 计算 KL 散度，loss 应在合理范围内
        self.trainer.model.network.train()
        outputs = self.trainer.model.network(inputs_tensor)
        loss = self.trainer._masked_loss(outputs, targets_tensor, masks_tensor)

        loss_value = loss.item()
        self.assertGreater(loss_value, 0, "KL 散度 loss 应大于 0")
        self.assertLess(loss_value, 5,
                        f"KL 散度 loss 应小于 5，实际为 {loss_value:.4f}")

    def test_original_sample_data_unchanged(self):
        """[primary] _prepare_tensors 不应修改原始 sample_data 列表"""
        output = [1.0 / 225.0] * 225
        sample_data = [
            {
                "input": [0] * 225 + [1],
                "output": output,
                "mask": [1] * 225
            }
        ]
        original_output_val = sample_data[0]["output"][0]

        self.trainer._sample_loader.prepare_tensors(sample_data)

        # 原始数据不变
        self.assertEqual(sample_data[0]["output"][0], original_output_val,
                         "_prepare_tensors 不应修改原始 sample_data")

    # ---------- edge tests ----------

    def test_partial_mask_softmax(self):
        """[edge] 部分 mask=0 时，Softmax 概率处理正确"""
        # 只有前3个位置是空位（mask=1）
        output = [0.0] * 225
        output[0] = 0.7
        output[1] = 0.2
        output[2] = 0.1
        mask = [1, 1, 1] + [0] * 222
        
        sample_data = [
            {
                "input": [0] * 225 + [1],
                "output": output,
                "mask": mask
            }
        ]
        _, targets_tensor, masks_tensor = self.trainer._sample_loader.prepare_tensors(sample_data)

        # mask=1 的位置保持原值
        self.assertAlmostEqual(targets_tensor[0][0].item(), 0.7, places=5)
        self.assertAlmostEqual(targets_tensor[0][1].item(), 0.2, places=5)
        self.assertAlmostEqual(targets_tensor[0][2].item(), 0.1, places=5)
        
        # mask=0 的位置应为 0（但 mask 会处理）
        self.assertEqual(masks_tensor[0][3].item(), 0.0)

    def test_training_converges_with_softmax(self):
        """[edge] 使用 Softmax 概率 + KLDivLoss 训练，loss 应能收敛"""
        # 创建一个有明确信号的 sample（Softmax 概率）
        output = [0.0] * 225
        output[0] = 0.9  # 位置 0 高概率
        for i in range(1, 225):
            output[i] = 0.1 / 224

        sample_data = [
            {
                "input": [0] * 225 + [1],
                "output": output,
                "mask": [1] * 225
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_path = os.path.join(tmpdir, "sample.json")
            with open(sample_path, 'w') as f:
                json.dump(sample_data, f)

            trainer = FCv1Trainer(fresh=True)
            # 注入 mock logger 避免污染真实 loss_logs 目录
            trainer.loss_logger = MagicMock()
            trainer.loss_logger.dense_log = False
            loss_epoch1 = trainer.train_single(tmpdir, epochs=1)

            trainer2 = FCv1Trainer(fresh=True)
            # 注入 mock logger 避免污染真实 loss_logs 目录
            trainer2.loss_logger = MagicMock()
            trainer2.loss_logger.dense_log = False
            loss_epoch50 = trainer2.train_single(tmpdir, epochs=50)

        # KL 散度 loss 应随训练递减
        self.assertLess(loss_epoch50, loss_epoch1,
                        f"50 epoch 平均 KL散度 loss({loss_epoch50:.6f}) 应小于 1 epoch loss({loss_epoch1:.6f})")


if __name__ == '__main__':
    unittest.main()
