"""Random 模型测试 - 接口契约验证"""
import unittest
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.models.random.model import RandomModel


class TestRandomModelContract(unittest.TestCase):
    """Random 模型接口契约测试"""

    def setUp(self):
        self.model = RandomModel()

    def test_predict_prob_sum_normal_board(self):
        """正常棋盘：概率之和应为 1"""
        board_vector = [0] * 225 + [1]  # 空棋盘，黑方回合
        result = self.model.predict(board_vector)
        self.assertAlmostEqual(sum(result), 1.0, places=5)

    def test_predict_prob_sum_full_board(self):
        """满棋盘：概率之和应为 1（修复空棋盘返回全0的bug）"""
        board_vector = [1] * 225 + [1]  # 满棋盘（全黑子）
        result = self.model.predict(board_vector)
        # 修复后空棋盘也返回均匀分布
        self.assertAlmostEqual(sum(result), 1.0, places=5,
                               msg="满棋盘时概率之和也应为1")

    def test_predict_full_board_uniform(self):
        """满棋盘：所有位置概率应相等"""
        board_vector = [1] * 225 + [1]
        result = self.model.predict(board_vector)
        expected_prob = 1.0 / 225
        for prob in result:
            self.assertAlmostEqual(prob, expected_prob, places=5)

    def test_predict_empty_positions_zero_prob(self):
        """已落子位置概率应为 0"""
        board = [0] * 225
        board[7 * 15 + 7] = 1  # (7,7) 有黑子
        board_vector = board + [1]
        result = self.model.predict(board_vector)
        
        # (7,7) 位置概率为 0
        self.assertEqual(result[7 * 15 + 7], 0.0)
        # 其他空位概率大于 0
        self.assertGreater(result[0], 0.0)

    def test_name_property(self):
        """模型名称应为 random"""
        self.assertEqual(self.model.name, "random")

    def test_has_save_method(self):
        """Bug M11: RandomModel 应实现 save 方法（BaseModel 接口契约）"""
        self.assertTrue(hasattr(self.model, 'save'),
                        "RandomModel 应实现 save 方法")
        self.assertTrue(callable(self.model.save),
                        "save 应为可调用方法")

    def test_has_load_method(self):
        """Bug M11: RandomModel 应实现 load 方法（BaseModel 接口契约）"""
        self.assertTrue(hasattr(self.model, 'load'),
                        "RandomModel 应实现 load 方法")
        self.assertTrue(callable(self.model.load),
                        "load 应为可调用方法")

    def test_has_network_property(self):
        """Bug M11: RandomModel 应实现 network 属性（BaseModel 接口契约）"""
        # RandomModel 不使用神经网络，但属性必须存在以满足 BaseTrainer 的依赖
        self.assertTrue(hasattr(self.model, 'network'),
                        "RandomModel 应实现 network 属性")
        # RandomModel 的 network 应为 None
        self.assertIsNone(self.model.network,
                         "RandomModel 的 network 应为 None")


class TestBaseModelInterface(unittest.TestCase):
    """Bug M11: BaseModel 抽象基类接口完整性测试"""

    def test_base_model_defines_save(self):
        """BaseModel 应声明 save 为抽象方法"""
        from ai.base_model import BaseModel
        self.assertTrue(hasattr(BaseModel, 'save'),
                        "BaseModel 应声明 save 方法")

    def test_base_model_defines_load(self):
        """BaseModel 应声明 load 为抽象方法"""
        from ai.base_model import BaseModel
        self.assertTrue(hasattr(BaseModel, 'load'),
                        "BaseModel 应声明 load 方法")

    def test_base_model_documents_network(self):
        """BaseModel 文档中应声明 network 为必要属性"""
        from ai.base_model import BaseModel
        # network 不是抽象方法（因为是实例属性），但文档中明确声明
        self.assertIn('network', BaseModel.__doc__)

    def test_cannot_instantiate_without_save_load(self):
        """缺少 save/load 实现的类不能被实例化"""
        from ai.base_model import BaseModel
        
        class IncompleteModel(BaseModel):
            @property
            def name(self):
                return "incomplete"
            def predict(self, board_vector):
                return [0.0] * 225
            # 缺少 save 和 load
        
        with self.assertRaises(TypeError):
            IncompleteModel()


if __name__ == '__main__':
    unittest.main()
