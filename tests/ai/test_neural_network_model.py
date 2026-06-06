"""NeuralNetworkModel 抽象基类测试

测试 NeuralNetworkModel 提取的通用逻辑：
- __init__ 的 fresh / model_path / DEFAULT_MODEL_PATH 加载行为
- predict 的前向传播
- save / load 的模型持久化
- _create_network / _prepare_input 抽象方法约束
- board_vector_to_cnn_tensor 工具函数
"""

import os
import sys
import unittest
import unittest.mock
import tempfile
import shutil

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
from ai.base_model import NeuralNetworkModel, board_vector_to_cnn_tensor, BaseModel


# ========== 测试用具体子类 ==========

class DummyNetwork(nn.Module):
    """测试用简单网络，1 层线性变换 + LogSoftmax"""
    
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(226, 225)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        return self.softmax(self.fc(x))


class DummyModel(NeuralNetworkModel):
    """测试用具体模型，用于验证 NeuralNetworkModel 的通用逻辑"""
    
    DEFAULT_MODEL_PATH = None  # 测试中动态设置
    
    @property
    def name(self) -> str:
        return "dummy"
    
    def _create_network(self) -> nn.Module:
        return DummyNetwork()
    
    def _prepare_input(self, board_vector, device):
        return torch.tensor(board_vector, dtype=torch.float32, device=device).unsqueeze(0)


# ========== NeuralNetworkModel 测试 ==========

class TestNeuralNetworkModelAbstract(unittest.TestCase):
    """测试 NeuralNetworkModel 的抽象约束"""
    
    def test_cannot_instantiate_directly(self):
        """NeuralNetworkModel 是抽象类，不能直接实例化"""
        with self.assertRaises(TypeError):
            NeuralNetworkModel()
    
    def test_missing_create_network_cannot_instantiate(self):
        """缺少 _create_network 实现的子类不能实例化"""
        class IncompleteModel(NeuralNetworkModel):
            DEFAULT_MODEL_PATH = None
            
            @property
            def name(self):
                return "incomplete"
            
            def _prepare_input(self, board_vector, device):
                pass
        
        with self.assertRaises(TypeError):
            IncompleteModel()
    
    def test_missing_prepare_input_cannot_instantiate(self):
        """缺少 _prepare_input 实现的子类不能实例化"""
        class IncompleteModel(NeuralNetworkModel):
            DEFAULT_MODEL_PATH = None
            
            @property
            def name(self):
                return "incomplete"
            
            def _create_network(self):
                return DummyNetwork()
        
        with self.assertRaises(TypeError):
            IncompleteModel()
    
    def test_inherits_from_base_model(self):
        """NeuralNetworkModel 应继承自 BaseModel"""
        self.assertTrue(issubclass(NeuralNetworkModel, BaseModel))


class TestNeuralNetworkModelInit(unittest.TestCase):
    """测试 NeuralNetworkModel 的 __init__ 行为"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_fresh_creates_new_model(self):
        """fresh=True 时创建新模型，不加载任何权重"""
        model = DummyModel(fresh=True)
        self.assertIsNotNone(model.network)
        self.assertEqual(model.name, "dummy")
    
    def test_default_init_without_path(self):
        """不传参数且 DEFAULT_MODEL_PATH=None 时，创建新模型"""
        model = DummyModel()
        self.assertIsNotNone(model.network)
    
    def test_load_from_model_path(self):
        """指定 model_path 且文件存在时，加载该模型"""
        # 先创建并保存一个模型
        model1 = DummyModel(fresh=True)
        model_path = os.path.join(self.temp_dir, "dummy.pth")
        model1.save(model_path)
        
        # 加载模型
        model2 = DummyModel(model_path=model_path)
        self.assertEqual(model2.name, "dummy")
        
        # 验证加载成功：两次预测结果应相同
        board_vector = [0] * 225 + [1]
        result1 = model1.predict(board_vector)
        result2 = model2.predict(board_vector)
        self.assertEqual(result1, result2)
    
    def test_nonexistent_model_path_creates_new(self):
        """model_path 指向不存在的文件时，创建新模型"""
        nonexistent = os.path.join(self.temp_dir, "nonexistent.pth")
        model = DummyModel(model_path=nonexistent)
        self.assertEqual(model.name, "dummy")
    
    def test_default_model_path_auto_load(self):
        """DEFAULT_MODEL_PATH 存在时自动加载"""
        # 设置 DEFAULT_MODEL_PATH
        default_path = os.path.join(self.temp_dir, "default.pth")
        DummyModel.DEFAULT_MODEL_PATH = default_path
        
        try:
            # 创建并保存模型
            model1 = DummyModel(fresh=True)
            board_vector = [0] * 225 + [1]
            result1 = model1.predict(board_vector)
            model1.save()
            
            # 不传参数，应自动加载默认路径
            model2 = DummyModel()
            result2 = model2.predict(board_vector)
            
            self.assertEqual(result1, result2)
        finally:
            DummyModel.DEFAULT_MODEL_PATH = None
    
    def test_model_path_priority_over_default(self):
        """model_path 优先于 DEFAULT_MODEL_PATH"""
        default_path = os.path.join(self.temp_dir, "default.pth")
        custom_path = os.path.join(self.temp_dir, "custom.pth")
        DummyModel.DEFAULT_MODEL_PATH = default_path
        
        try:
            # 创建并保存两个不同的模型
            model1 = DummyModel(fresh=True)
            model1.save()  # 保存到默认路径
            
            model2 = DummyModel(fresh=True)
            board_vector = [0] * 225 + [1]
            result2 = model2.predict(board_vector)
            model2.save(custom_path)
            
            # 加载自定义路径
            model3 = DummyModel(model_path=custom_path)
            result3 = model3.predict(board_vector)
            
            # 应与模型2的结果相同，而非模型1
            self.assertEqual(result2, result3)
        finally:
            DummyModel.DEFAULT_MODEL_PATH = None
    
    def test_fresh_skips_default_load(self):
        """fresh=True 时跳过 DEFAULT_MODEL_PATH 加载"""
        default_path = os.path.join(self.temp_dir, "default.pth")
        DummyModel.DEFAULT_MODEL_PATH = default_path
        
        try:
            # 保存模型到默认路径
            model1 = DummyModel(fresh=True)
            model1.save()
            
            # fresh=True 应跳过加载
            model2 = DummyModel(fresh=True)
            self.assertIsNotNone(model2.network)
        finally:
            DummyModel.DEFAULT_MODEL_PATH = None


class TestNeuralNetworkModelPredict(unittest.TestCase):
    """测试 NeuralNetworkModel 的 predict 通用逻辑"""
    
    def test_predict_returns_225_floats(self):
        """predict 返回 225 维浮点列表"""
        model = DummyModel(fresh=True)
        board_vector = [0] * 225 + [1]
        result = model.predict(board_vector)
        
        self.assertEqual(len(result), 225)
        for prob in result:
            self.assertIsInstance(prob, float)
    
    def test_predict_probability_range(self):
        """predict 输出概率在 (0, 1) 范围"""
        model = DummyModel(fresh=True)
        board_vector = [0] * 225 + [1]
        result = model.predict(board_vector)
        
        for prob in result:
            self.assertGreater(prob, 0.0)
            self.assertLess(prob, 1.0)
    
    def test_predict_probability_sum(self):
        """predict 输出概率总和为 1"""
        model = DummyModel(fresh=True)
        board_vector = [0] * 225 + [1]
        result = model.predict(board_vector)
        
        self.assertAlmostEqual(sum(result), 1.0, places=5)
    
    def test_predict_consistency(self):
        """相同输入多次预测结果一致"""
        model = DummyModel(fresh=True)
        board_vector = [0] * 225 + [1]
        
        result1 = model.predict(board_vector)
        result2 = model.predict(board_vector)
        self.assertEqual(result1, result2)

    def test_exp_log_softmax_equals_softmax(self):
        """T5-02: exp(log_softmax) = softmax，predict 返回值与直接 Softmax 一致"""
        model = DummyModel(fresh=True)
        board_vector = [0] * 225 + [1]

        # 获取网络直接输出（log-probability）
        device = next(model.network.parameters()).device
        input_tensor = torch.tensor(board_vector, dtype=torch.float32,
                                    device=device).unsqueeze(0)
        model.network.eval()
        with torch.no_grad():
            log_prob_output = model.network(input_tensor)

        # 用 Softmax 直接计算
        softmax_output = torch.softmax(model.network.fc(input_tensor), dim=1)

        # predict 返回值（exp 还原后的概率分布）
        predict_result = model.predict(board_vector)

        # exp(log_softmax) 应与 softmax 一致（1e-5 内）
        exp_log_prob = log_prob_output.exp()
        for i in range(225):
            self.assertAlmostEqual(exp_log_prob[0, i].item(),
                                   softmax_output[0, i].item(),
                                   places=5,
                                   msg=f"exp(log_softmax) 应等于 softmax，位置 {i} 不一致")

        # predict 返回值应与 softmax 一致
        for i in range(225):
            self.assertAlmostEqual(predict_result[i],
                                   softmax_output[0, i].item(),
                                   places=5,
                                   msg=f"predict 返回值应与 softmax 一致，位置 {i} 不一致")


class TestNeuralNetworkModelSaveLoad(unittest.TestCase):
    """测试 NeuralNetworkModel 的 save / load 通用逻辑"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_creates_file(self):
        """save 创建模型文件"""
        model = DummyModel(fresh=True)
        model_path = os.path.join(self.temp_dir, "test.pth")
        saved_path = model.save(model_path)
        
        self.assertEqual(saved_path, model_path)
        self.assertTrue(os.path.exists(model_path))
    
    def test_save_load_preserves_predictions(self):
        """save + load 保持预测结果一致"""
        model1 = DummyModel(fresh=True)
        board_vector = [0] * 225 + [1]
        result1 = model1.predict(board_vector)
        
        model_path = os.path.join(self.temp_dir, "test.pth")
        model1.save(model_path)
        
        model2 = DummyModel(model_path=model_path)
        result2 = model2.predict(board_vector)
        
        self.assertEqual(result1, result2)
    
    def test_save_default_path(self):
        """save 不指定路径时使用 DEFAULT_MODEL_PATH"""
        default_path = os.path.join(self.temp_dir, "default.pth")
        DummyModel.DEFAULT_MODEL_PATH = default_path
        
        try:
            model = DummyModel(fresh=True)
            saved_path = model.save()
            
            self.assertEqual(saved_path, default_path)
            self.assertTrue(os.path.exists(default_path))
        finally:
            DummyModel.DEFAULT_MODEL_PATH = None
    
    def test_load_sets_eval_mode(self):
        """load 后网络应处于 eval 模式"""
        model1 = DummyModel(fresh=True)
        model_path = os.path.join(self.temp_dir, "test.pth")
        model1.save(model_path)
        
        model2 = DummyModel(model_path=model_path)
        self.assertFalse(model2.network.training)


# ========== board_vector_to_cnn_tensor 测试 ==========

class TestBoardVectorToCnnTensor(unittest.TestCase):
    """测试 board_vector_to_cnn_tensor 工具函数"""
    
    def test_output_shape(self):
        """输出形状应为 (1, 3, 15, 15)"""
        board_vector = [0] * 225 + [1]
        tensor = board_vector_to_cnn_tensor(board_vector, torch.device("cpu"))
        
        self.assertEqual(tensor.shape, (1, 3, 15, 15))
    
    def test_black_channel(self):
        """黑棋通道：值为 1 的位置应为 1.0，其余为 0.0"""
        board_vector = [0] * 225 + [1]
        board_vector[0] = 1        # (0,0) 黑棋
        board_vector[15] = 1       # (1,0) 黑棋
        
        tensor = board_vector_to_cnn_tensor(board_vector, torch.device("cpu"))
        black_channel = tensor[0, 0]
        
        self.assertEqual(black_channel[0, 0].item(), 1.0)
        self.assertEqual(black_channel[1, 0].item(), 1.0)
        self.assertEqual(black_channel[0, 1].item(), 0.0)
    
    def test_white_channel(self):
        """白棋通道：值为 2 的位置应为 1.0，其余为 0.0"""
        board_vector = [0] * 225 + [2]
        board_vector[7 * 15 + 7] = 2  # (7,7) 白棋
        
        tensor = board_vector_to_cnn_tensor(board_vector, torch.device("cpu"))
        white_channel = tensor[0, 1]
        
        self.assertEqual(white_channel[7, 7].item(), 1.0)
        self.assertEqual(white_channel[0, 0].item(), 0.0)
    
    def test_player_channel_black(self):
        """当前方为黑(1)时，player 通道全为 1.0"""
        board_vector = [0] * 225 + [1]
        tensor = board_vector_to_cnn_tensor(board_vector, torch.device("cpu"))
        player_channel = tensor[0, 2]
        
        # 所有点都应为 1.0
        self.assertTrue(torch.all(player_channel == 1.0))
    
    def test_player_channel_white(self):
        """当前方为白(2)时，player 通道全为 0.0"""
        board_vector = [0] * 225 + [2]
        tensor = board_vector_to_cnn_tensor(board_vector, torch.device("cpu"))
        player_channel = tensor[0, 2]
        
        # 所有点都应为 0.0
        self.assertTrue(torch.all(player_channel == 0.0))
    
    def test_empty_board(self):
        """空棋盘时黑棋和白棋通道全为 0"""
        board_vector = [0] * 225 + [1]
        tensor = board_vector_to_cnn_tensor(board_vector, torch.device("cpu"))
        
        self.assertTrue(torch.all(tensor[0, 0] == 0.0))  # 无黑棋
        self.assertTrue(torch.all(tensor[0, 1] == 0.0))  # 无白棋
    
    def test_device_placement(self):
        """张量应在指定设备上创建"""
        board_vector = [0] * 225 + [1]
        tensor = board_vector_to_cnn_tensor(board_vector, torch.device("cpu"))
        
        self.assertEqual(tensor.device.type, "cpu")
    
    def test_short_vector_defaults_to_black(self):
        """向量长度不足 226 时，当前方默认为黑(1)"""
        board_vector = [0] * 225  # 缺少最后一位
        tensor = board_vector_to_cnn_tensor(board_vector, torch.device("cpu"))
        
        # player 通道应全为 1.0（默认黑方）
        self.assertTrue(torch.all(tensor[0, 2] == 1.0))
    
    def test_mixed_board(self):
        """混合棋盘测试"""
        board_vector = [0] * 226
        board_vector[0] = 1               # (0,0) 黑
        board_vector[1] = 2               # (0,1) 白
        board_vector[225] = 1             # 当前黑方
        
        tensor = board_vector_to_cnn_tensor(board_vector, torch.device("cpu"))
        
        # 黑棋通道：(0,0) 为 1，(0,1) 为 0
        self.assertEqual(tensor[0, 0, 0, 0].item(), 1.0)
        self.assertEqual(tensor[0, 0, 0, 1].item(), 0.0)
        
        # 白棋通道：(0,0) 为 0，(0,1) 为 1
        self.assertEqual(tensor[0, 1, 0, 0].item(), 0.0)
        self.assertEqual(tensor[0, 1, 0, 1].item(), 1.0)


class TestSubclassIntegration(unittest.TestCase):
    """验证三个具体模型类都正确继承 NeuralNetworkModel"""
    
    def test_fcv1_is_neural_network_model(self):
        """FCv1Model 应是 NeuralNetworkModel 的子类"""
        from ai.models.fc_v1.model import FCv1Model
        self.assertTrue(issubclass(FCv1Model, NeuralNetworkModel))
    
    def test_cnnv1_is_neural_network_model(self):
        """CNNV1Model 应是 NeuralNetworkModel 的子类"""
        from ai.models.cnn_v1.model import CNNV1Model
        self.assertTrue(issubclass(CNNV1Model, NeuralNetworkModel))
    
    def test_cnnv2_is_neural_network_model(self):
        """CNNV2Model 应是 NeuralNetworkModel 的子类"""
        from ai.models.cnn_v2.model import CNNV2Model
        self.assertTrue(issubclass(CNNV2Model, NeuralNetworkModel))
    
    def test_random_model_is_not_neural_network_model(self):
        """RandomModel 不应是 NeuralNetworkModel 的子类"""
        from ai.models.random.model import RandomModel
        self.assertFalse(issubclass(RandomModel, NeuralNetworkModel))
    
    def test_random_model_still_base_model(self):
        """RandomModel 仍然直接继承 BaseModel"""
        from ai.models.random.model import RandomModel
        self.assertTrue(issubclass(RandomModel, BaseModel))


if __name__ == '__main__':
    unittest.main()
