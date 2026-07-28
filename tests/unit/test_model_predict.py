"""ModelPredict 单元测试

测试 BaseModel.predict() 返回值的统一封装类。
遵循 TDD：先写测试，再实现。
"""

import os
import sys
import unittest

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestModelPredictConstruction(unittest.TestCase):
    """测试 ModelPredict 构造"""

    @classmethod
    def setUpClass(cls):
        from ai.base_model import ModelPredict
        cls.ModelPredict = ModelPredict

        # 准备测试用的 225 维概率分布
        cls._sample_policy = [1.0 / 225] * 225

    def test_construct_policy_only(self):
        """构造 TYPE_POLICY_ONLY：只传 policy，value 为 None"""
        result = self.ModelPredict(policy=self._sample_policy)
        self.assertEqual(result.type, self.ModelPredict.TYPE_POLICY_ONLY)
        self.assertIsNone(result.value)
        self.assertEqual(result.policy, self._sample_policy)

    def test_construct_policy_value(self):
        """构造 TYPE_POLICY_VALUE：同时传 policy 和 value"""
        result = self.ModelPredict(policy=self._sample_policy, value=0.5)
        self.assertEqual(result.type, self.ModelPredict.TYPE_POLICY_VALUE)
        self.assertEqual(result.value, 0.5)
        self.assertEqual(result.policy, self._sample_policy)

    def test_construct_with_explicit_none_value(self):
        """显式传 value=None 时 type 为 policy_only"""
        result = self.ModelPredict(policy=self._sample_policy, value=None)
        self.assertEqual(result.type, self.ModelPredict.TYPE_POLICY_ONLY)
        self.assertIsNone(result.value)

    def test_type_constant_values(self):
        """类型常量值正确"""
        self.assertEqual(self.ModelPredict.TYPE_POLICY_ONLY, "policy_only")
        self.assertEqual(self.ModelPredict.TYPE_POLICY_VALUE, "policy_value")


class TestModelPredictPolicyValidation(unittest.TestCase):
    """测试 policy 长度校验"""

    @classmethod
    def setUpClass(cls):
        from ai.base_model import ModelPredict
        cls.ModelPredict = ModelPredict

    def test_policy_length_225_valid(self):
        """225 维 policy 构造成功"""
        policy = [1.0 / 225] * 225
        result = self.ModelPredict(policy=policy)
        self.assertEqual(len(result.policy), 225)

    def test_policy_length_not_225_raises(self):
        """非 225 维 policy 应抛出 ValueError"""
        with self.assertRaises(ValueError):
            self.ModelPredict(policy=[0.5, 0.5])  # 只有 2 维

    def test_policy_length_0_raises(self):
        """空 policy 应抛出 ValueError"""
        with self.assertRaises(ValueError):
            self.ModelPredict(policy=[])


class TestModelPredictUnpacking(unittest.TestCase):
    """测试元组解包支持"""

    @classmethod
    def setUpClass(cls):
        from ai.base_model import ModelPredict
        cls.ModelPredict = ModelPredict

        cls._sample_policy = [1.0 / 225] * 225

    def test_unpack_policy_only(self):
        """TYPE_POLICY_ONLY 解包为 (policy, None)"""
        result = self.ModelPredict(policy=self._sample_policy)
        p, v = result
        self.assertEqual(p, self._sample_policy)
        self.assertIsNone(v)

    def test_unpack_policy_value(self):
        """TYPE_POLICY_VALUE 解包为 (policy, value)"""
        result = self.ModelPredict(policy=self._sample_policy, value=-0.8)
        p, v = result
        self.assertEqual(p, self._sample_policy)
        self.assertEqual(v, -0.8)

    def test_unpack_to_list(self):
        """list(ModelPredict) 返回 [policy, value]"""
        result = self.ModelPredict(policy=self._sample_policy, value=0.3)
        lst = list(result)
        self.assertEqual(len(lst), 2)
        self.assertEqual(lst[0], self._sample_policy)
        self.assertEqual(lst[1], 0.3)

    def test_index_access(self):
        """支持下标访问 result[0] 和 result[1]"""
        result = self.ModelPredict(policy=self._sample_policy, value=0.3)
        self.assertEqual(result[0], self._sample_policy)
        self.assertEqual(result[1], 0.3)

    def test_index_access_policy_only(self):
        """TYPE_POLICY_ONLY 时 result[1] 为 None"""
        result = self.ModelPredict(policy=self._sample_policy)
        self.assertEqual(result[0], self._sample_policy)
        self.assertIsNone(result[1])


class TestModelPredictValueOnly(unittest.TestCase):
    """测试 TYPE_VALUE_ONLY 类型构造与解包（设计 TC-MP-01, TC-MP-02）"""

    @classmethod
    def setUpClass(cls):
        from ai.base_model import ModelPredict
        cls.ModelPredict = ModelPredict

    def test_construct_value_only_type(self):
        """TC-MP-01: 仅传 value 时 type == TYPE_VALUE_ONLY"""
        result = self.ModelPredict(value=0.5)
        self.assertEqual(result.type, self.ModelPredict.TYPE_VALUE_ONLY)
        self.assertIsNone(result.policy)
        self.assertEqual(result.value, 0.5)

    def test_construct_value_only_negative(self):
        """TYPE_VALUE_ONLY 支持负值"""
        result = self.ModelPredict(value=-0.8)
        self.assertEqual(result.type, self.ModelPredict.TYPE_VALUE_ONLY)
        self.assertEqual(result.value, -0.8)
        self.assertIsNone(result.policy)

    def test_unpack_value_only(self):
        """TC-MP-02: 元组解包 policy 为 None, value 正确"""
        result = self.ModelPredict(value=0.5)
        p, v = result
        self.assertIsNone(p)
        self.assertEqual(v, 0.5)

    def test_index_access_value_only(self):
        """TYPE_VALUE_ONLY 下标访问 result[0] 为 None, result[1] 为 value"""
        result = self.ModelPredict(value=-0.3)
        self.assertIsNone(result[0])
        self.assertEqual(result[1], -0.3)

    def test_repr_value_only(self):
        """TYPE_VALUE_ONLY 的 repr 包含 value_only 和 value"""
        result = self.ModelPredict(value=0.75)
        r = repr(result)
        self.assertIn("value_only", r)
        self.assertIn("0.75", r)

    def test_value_only_constant_exists(self):
        """TYPE_VALUE_ONLY 常量值正确"""
        self.assertEqual(self.ModelPredict.TYPE_VALUE_ONLY, "value_only")

    def test_both_none_raises(self):
        """policy 和 value 都为 None 时应抛异常"""
        with self.assertRaises(ValueError):
            self.ModelPredict()


class TestModelPredictRepr(unittest.TestCase):
    """测试 __repr__ 输出"""

    @classmethod
    def setUpClass(cls):
        from ai.base_model import ModelPredict
        cls.ModelPredict = ModelPredict

        cls._sample_policy = [1.0 / 225] * 225

    def test_repr_policy_only(self):
        """TYPE_POLICY_ONLY 的 repr 包含 type 和 sum(policy)"""
        result = self.ModelPredict(policy=self._sample_policy)
        r = repr(result)
        self.assertIn("policy_only", r)
        self.assertIn("225", r)  # policy 长度

    def test_repr_policy_value(self):
        """TYPE_POLICY_VALUE 的 repr 包含 type 和 value"""
        result = self.ModelPredict(policy=self._sample_policy, value=0.75)
        r = repr(result)
        self.assertIn("policy_value", r)
        self.assertIn("0.75", r)


if __name__ == '__main__':
    unittest.main()
