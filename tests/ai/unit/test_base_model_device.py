"""NeuralNetworkModel.move_to_device() 单元测试

测试 NeuralNetworkModel 新增的 GPU 自动移动方法。
遵循 TDD：测试优先于实现。

测试范围：
- move_to_device() 方法存在且可调用
- 显式指定 CPU 设备可正常工作
- 移动后 predict 仍可正常推理
- CUDA 可用时自动检测并移到 GPU
- CUDA 不可用时留在 CPU
- RandomModel（无 network）不受影响
"""

import unittest
from unittest.mock import patch

import torch

from ai.base_model import NeuralNetworkModel
from ai.models.cnn_v3.model import CNNV3Model
from ai.models.random.model import RandomModel


class TestMoveToDeviceExistence(unittest.TestCase):
    """move_to_device 方法存在性测试"""

    def test_method_exists_on_neural_network_model(self):
        """NeuralNetworkModel 应具有 move_to_device 方法"""
        model = CNNV3Model(fresh=True)
        self.assertTrue(hasattr(model, 'move_to_device'))
        self.assertTrue(callable(model.move_to_device))

    def test_method_not_on_random_model(self):
        """RandomModel（无 network）不应受 move_to_device 影响
        —— base_model 基类不要求此方法"""
        model = RandomModel()
        # RandomModel 不需要此方法
        self.assertIsNone(model.network)


class TestMoveToDeviceExplicitCPU(unittest.TestCase):
    """显式指定 CPU 设备的测试"""

    def test_move_to_cpu_returns_cpu_device(self):
        """move_to_device(torch.device('cpu')) 返回 cpu 设备"""
        model = CNNV3Model(fresh=True)
        device = model.move_to_device(torch.device('cpu'))
        self.assertIsInstance(device, torch.device)
        self.assertEqual(device.type, 'cpu')

    def test_move_to_cpu_network_stays_on_cpu(self):
        """移到 CPU 后网络参数在 CPU 上"""
        model = CNNV3Model(fresh=True)
        model.move_to_device(torch.device('cpu'))
        param_device = next(model.network.parameters()).device
        self.assertEqual(param_device.type, 'cpu')

    def test_predict_works_after_move_to_cpu(self):
        """move_to_device('cpu') 后 predict 仍正常返回 225 维"""
        model = CNNV3Model(fresh=True)
        model.move_to_device(torch.device('cpu'))
        probs = model.predict([0] * 225 + [1])
        self.assertEqual(len(probs), 225)
        for p in probs:
            self.assertGreaterEqual(p, 0.0)

    def test_move_to_device_returns_device(self):
        """move_to_device() 返回移动后的设备"""
        model = CNNV3Model(fresh=True)
        cpu_device = torch.device('cpu')
        result = model.move_to_device(cpu_device)
        self.assertEqual(result, cpu_device)


class TestMoveToDeviceAutoDetect(unittest.TestCase):
    """自动检测设备的测试（mock torch.cuda.is_available）"""

    @patch('torch.cuda.is_available', return_value=True)
    def test_auto_gpu_when_cuda_available(self, mock_is_available):
        """CUDA 可用时，move_to_device() 应自动移到 GPU

        通过 mock network.to 避免实际调用 .to('cuda')，只验证调用参数。
        """
        model = CNNV3Model(fresh=True)
        with patch.object(model.network, 'to', wraps=model.network.to) as mock_to:
            device = model.move_to_device()
            # 验证 torch.cuda.is_available 被调用（可能被模型初始化多次调用）
            mock_is_available.assert_called()
            # 验证 network.to 被调用，且参数是 cuda 设备
            mock_to.assert_called_once()
            called_device = mock_to.call_args[0][0]
            self.assertEqual(str(called_device), 'cuda')

    @patch('torch.cuda.is_available', return_value=False)
    def test_auto_cpu_when_cuda_unavailable(self, mock_is_available):
        """CUDA 不可用时，move_to_device() 应留在 CPU"""
        model = CNNV3Model(fresh=True)
        device = model.move_to_device()
        mock_is_available.assert_called_once()
        self.assertEqual(device.type, 'cpu')
        param_device = next(model.network.parameters()).device
        self.assertEqual(param_device.type, 'cpu')


class TestMoveToDeviceEdgeCases(unittest.TestCase):
    """边界条件测试"""

    def test_move_to_device_idempotent_cpu(self):
        """重复调用 move_to_device('cpu') 不出错"""
        model = CNNV3Model(fresh=True)
        model.move_to_device(torch.device('cpu'))
        model.move_to_device(torch.device('cpu'))
        param_device = next(model.network.parameters()).device
        self.assertEqual(param_device.type, 'cpu')

    def test_move_to_device_with_fresh_model(self):
        """fresh=True 创建的模型同样支持 move_to_device"""
        model = CNNV3Model(fresh=True)
        device = model.move_to_device(torch.device('cpu'))
        self.assertEqual(device.type, 'cpu')

    def test_move_to_device_after_multiple_predicts(self):
        """多次 predict 后 move_to_device 仍正常"""
        model = CNNV3Model(fresh=True)
        for _ in range(3):
            model.predict([0] * 225 + [1])
        model.move_to_device(torch.device('cpu'))
        probs = model.predict([0] * 225 + [1])
        self.assertEqual(len(probs), 225)


if __name__ == '__main__':
    unittest.main()
