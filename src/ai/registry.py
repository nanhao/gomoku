"""统一模型注册表（懒加载）—— 多代模型扩展

集中管理所有 AI 模型和训练器的映射关系，支持多代模型指定。
使用 importlib 实现按需懒加载，首次调用时才导入对应模块。

注册表数据结构：
    MODEL_REGISTRY: dict[str, RegistryEntry]
    - key: 模型名称（如 "random"、"fc_v1"），禁止包含 '.'
    - value: RegistryEntry(model_class_path, trainer_class_path, supports_multi_gen)

提供的公共函数：
    - get_model(spec) -> BaseModel: 根据 spec 返回模型实例
    - get_trainer(spec, **kwargs) -> BaseTrainer: 根据 spec 返回训练器实例
    - list_models_with_full_name() -> list[str]: 返回含代数的完整模型列表

模块私有函数：
    - _parse_model_spec(spec) -> ParsedModelSpec: 解析模型指定字符串
    - _is_multi_gen_model(model_name) -> bool: 判断是否为多代模型
    - _list_models() -> list[str]: 返回注册表中的基础模型名
"""

import importlib
import os
from typing import NamedTuple


# ---------------------------------------------------------------------------
# NamedTuple 定义
# ---------------------------------------------------------------------------
class RegistryEntry(NamedTuple):
    """注册表条目

    Attributes:
        model_class_path: 模型类的全限定路径
        trainer_class_path: 训练器类的全限定路径，无训练器时为 None
        supports_multi_gen: 是否为多代模型
    """
    model_class_path: str
    trainer_class_path: str | None
    supports_multi_gen: bool


class ParsedModelSpec(NamedTuple):
    """解析后的模型指定

    Attributes:
        name: 模型名称（不含代数部分）
        gen: 代数，非多代模型或未指定时为 None
    """
    name: str
    gen: int | None


# ---------------------------------------------------------------------------
# 注册表数据：统一维护所有模型和训练器的映射
# 格式: { 名称: RegistryEntry(model_path, trainer_path, supports_multi_gen) }
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, RegistryEntry] = {
    "random": RegistryEntry("ai.models.random.model.RandomModel", None, False),
    "fc_v1": RegistryEntry(
        "ai.models.fc_v1.model.FCv1Model",
        "ai.models.fc_v1.trainer.FCv1Trainer", False),
    "cnn_v1": RegistryEntry(
        "ai.models.cnn_v1.model.CNNV1Model",
        "ai.models.cnn_v1.trainer.CNNV1Trainer", False),
    "cnn_v2": RegistryEntry(
        "ai.models.cnn_v2.model.CNNV2Model",
        "ai.models.cnn_v2.trainer.CNNV2Trainer", False),
    "cnn_v3": RegistryEntry(
        "ai.models.cnn_v3.model.CNNV3Model",
        "ai.models.cnn_v3.trainer.CNNV3Trainer", True),
    "cnn_v4": RegistryEntry(
        "ai.models.cnn_v4.model.CNNV4Model",
        "ai.models.cnn_v4.trainer.CNNV4Trainer", True),
    "cnn_v5": RegistryEntry(
        "ai.models.cnn_v5.model.CNNV5Model",
        "ai.models.cnn_v5.trainer.CNNV5Trainer", True),
    "cnn_v6": RegistryEntry(
        "ai.models.cnn_v6.model.CNNV6Model",
        "ai.models.cnn_v6.trainer.CNNV6Trainer", True),
    "cnn_v7": RegistryEntry(
        "ai.models.cnn_v7.model.CNNV7Model",
        "ai.models.cnn_v7.trainer.CNNV7Trainer", True),
    "cnn_v8": RegistryEntry(
        "ai.models.cnn_v8.model.CNNV8Model",
        "ai.models.cnn_v8.trainer.CNNV8Trainer", True),
    "cnn_v9": RegistryEntry(
        "ai.models.cnn_v9.model.CNNV9Model",
        "ai.models.cnn_v9.trainer.CNNV9Trainer", True),
    "cnn_v10": RegistryEntry(
        "ai.models.cnn_v10.model.CNNV10Model",
        "ai.models.cnn_v10.trainer.CNNV10Trainer", True),
    "cnn_v11": RegistryEntry(
        "ai.models.cnn_v11.model.CNNV11Model",
        "ai.models.cnn_v11.trainer.CNNV11Trainer", True),
    "cnn_v12": RegistryEntry(
        "ai.models.cnn_v12.model.CNNV12Model",
        "ai.models.cnn_v12.trainer.CNNV12Trainer", True),
}

# 注册校验：模型名称禁止包含 '.'
for _name in MODEL_REGISTRY:
    if '.' in _name:
        raise ValueError("模型名称禁止包含 '.'")

# 模型目录基础路径（registry.py 位于 ai/ 目录下，models 在同级）
_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def _import_class(dotted_path: str):
    """根据全限定路径动态导入类

    使用 importlib 按需加载，避免在模块级别导入所有模型。

    Args:
        dotted_path: 类的全限定路径，如 "ai.models.fc_v1.model.FCv1Model"

    Returns:
        导入的类对象

    Raises:
        ImportError: 模块导入失败
        AttributeError: 模块中不存在指定类
    """
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _parse_model_spec(spec: str) -> ParsedModelSpec:
    """解析模型指定字符串

    使用右侧分割第一个 "."，将点号右侧部分作为代数解析。

    Args:
        spec: 模型指定字符串，如 "random"、"cnn_v3.1"

    Returns:
        ParsedModelSpec(name, gen) 解析结果

    Raises:
        ValueError: 代数格式错误（非正整数、为空、小于 1）
    """
    if '.' not in spec:
        return ParsedModelSpec(name=spec, gen=None)

    name, gen_str = spec.rsplit(".", 1)
    if not gen_str:
        raise ValueError("代数必须为正整数")
    try:
        gen = int(gen_str)
    except (ValueError, TypeError):
        raise ValueError("代数必须为正整数")
    if gen < 1:
        raise ValueError("代数必须 ≥ 1")
    return ParsedModelSpec(name=name, gen=gen)


def _is_multi_gen_model(model_name: str) -> bool:
    """判断模型是否为多代模型

    仅依据 MODEL_REGISTRY 中的 supports_multi_gen 声明，
    不涉及文件系统检查。

    Args:
        model_name: 模型名称（不含代数部分）

    Returns:
        True 表示多代模型

    Raises:
        KeyError: 模型名称不在注册表中
    """
    return MODEL_REGISTRY[model_name].supports_multi_gen


def get_model(spec: str):
    """根据 spec 返回模型实例（懒加载）

    核心判定原则：用户显式指定 gen → 优先信任用户意图。

    Args:
        spec: 模型指定字符串，如 "random"、"cnn_v3.1"

    Returns:
        模型实例（BaseModel 子类）

    Raises:
        ValueError: 未知模型、非多代模型指定代数、多代模型未指定代数
    """
    parsed = _parse_model_spec(spec)
    model_name = parsed.name
    gen = parsed.gen

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"未知的模型: '{model_name}'")

    if gen is not None:
        # 用户显式指定了代数
        if not _is_multi_gen_model(model_name):
            raise ValueError("非多代模型不能指定代数")
    else:
        # 未指定代数
        if _is_multi_gen_model(model_name):
            raise ValueError("多代模型必须指定代数，如 cnn_v3.1")

    model_path = MODEL_REGISTRY[model_name].model_class_path
    model_cls = _import_class(model_path)
    if gen is not None:
        return model_cls(gen=gen)
    return model_cls()


def get_trainer(spec: str, **kwargs):
    """根据 spec 返回训练器实例（懒加载）

    与 get_model 相同的 spec 格式和校验逻辑。

    Args:
        spec: 模型指定字符串，如 "fc_v1"、"cnn_v3.2"
        **kwargs: 传递给训练器构造函数的额外参数

    Returns:
        训练器实例（BaseTrainer 子类）

    Raises:
        ValueError: 不支持的模型、非多代模型指定代数、多代模型未指定代数
    """
    parsed = _parse_model_spec(spec)
    model_name = parsed.name
    gen = parsed.gen

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"不支持的模型: '{model_name}'")

    if gen is not None:
        if not _is_multi_gen_model(model_name):
            raise ValueError("非多代模型不能指定代数")
    else:
        if _is_multi_gen_model(model_name):
            raise ValueError("多代模型必须指定代数，如 cnn_v3.1")

    trainer_path = MODEL_REGISTRY[model_name].trainer_class_path
    if trainer_path is None:
        raise ValueError(f"不支持的模型: '{model_name}'")

    trainer_cls = _import_class(trainer_path)
    if gen is not None:
        return trainer_cls(gen=gen, **kwargs)
    return trainer_cls(**kwargs)


def _list_models() -> list[str]:
    """返回注册表中的基础模型名列表

    降级为模块私有函数，仅返回注册表中的 key 列表。

    Returns:
        模型名称列表，如 ["random", "fc_v1", "cnn_v1", "cnn_v2", "cnn_v3"]
    """
    return list(MODEL_REGISTRY.keys())


def list_models_with_full_name() -> list[str]:
    """返回所有可加载的模型指定字符串列表（含代数的完整名称）

    - 非多代模型：返回模型名称本身
    - 多代模型：扫描 gen{i}/ 子目录，仅当该目录下存在至少一个 .pth
      文件时才识别为一个有效代数；无 .pth 的空 gen{i}/ 目录会被跳过
      （例如训练中断或尚未产生权重的残留目录）
    - 若多代模型无已保存代数，则不展示该模型
    - 扫描时若检测到 pth 文件与 gen{i}/ 目录共存，抛出 ValueError

    Returns:
        模型指定字符串列表，如 ["random", "fc_v1", "cnn_v3.1", "cnn_v3.2"]

    Raises:
        ValueError: 模型目录下 pth 文件与 gen 子目录共存
    """
    result = []
    for name, entry in MODEL_REGISTRY.items():
        if not entry.supports_multi_gen:
            # 非多代模型：返回模型名称本身
            result.append(name)
            continue

        # 多代模型：扫描 gen{i}/ 子目录
        model_dir = os.path.join(_MODELS_DIR, name)
        if not os.path.isdir(model_dir):
            continue

        has_pth = False
        gen_nums = []
        for item in os.listdir(model_dir):
            item_path = os.path.join(model_dir, item)
            if os.path.isdir(item_path) and item.startswith("gen"):
                # 检查 gen{i} 格式，i 为正整数
                gen_str = item[3:]
                if gen_str:
                    try:
                        gen_num = int(gen_str)
                        if gen_num >= 1 and _gen_dir_has_pth(item_path):
                            # 仅当 gen 目录下至少存在一个 .pth 文件时
                            # 才认为是有效代数，避免空目录被误展示
                            gen_nums.append(gen_num)
                    except ValueError:
                        pass
            elif os.path.isfile(item_path) and item.endswith(".pth"):
                has_pth = True

        # pth 文件与 gen 目录共存 → 报错
        if has_pth and gen_nums:
            raise ValueError(
                f"模型 {name} 目录下 pth 文件与 gen 目录共存")

        # 无 gen 目录 → 不展示该模型
        if not gen_nums:
            continue

        for gen in sorted(gen_nums):
            result.append(f"{name}.{gen}")

    return result


def get_highest_gen(model_name: str) -> int:
    """获取多代模型的最高可用代数

    扫描 model_name 对应的 gen{i}/ 子目录，返回最大的有效代数。
    仅当 gen 目录下存在 .pth 文件时才视为有效代数。

    Args:
        model_name: 模型名称（不含代数部分），如 "cnn_v11"

    Returns:
        最高可用代数（正整数）

    Raises:
        ValueError: 模型不在注册表中、非多代模型、或没有找到任何可用代数
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"未知的模型: '{model_name}'")

    if not _is_multi_gen_model(model_name):
        raise ValueError(
            f"模型 '{model_name}' 不支持多代，无需查询最高代数"
        )

    model_dir = os.path.join(_MODELS_DIR, model_name)
    if not os.path.isdir(model_dir):
        raise ValueError(
            f"模型 '{model_name}' 没有找到任何可用代数"
        )

    max_gen = 0
    for item in os.listdir(model_dir):
        item_path = os.path.join(model_dir, item)
        if os.path.isdir(item_path) and item.startswith("gen"):
            gen_str = item[3:]
            if gen_str:
                try:
                    gen_num = int(gen_str)
                    if gen_num >= 1 and _gen_dir_has_pth(item_path):
                        if gen_num > max_gen:
                            max_gen = gen_num
                except ValueError:
                    pass

    if max_gen == 0:
        raise ValueError(
            f"模型 '{model_name}' 没有找到任何可用代数"
        )

    return max_gen


def _gen_dir_has_pth(gen_dir: str) -> bool:
    """判断 gen{i}/ 目录下是否至少存在一个 .pth 文件

    用于在 list_models_with_full_name 中过滤掉空 gen 目录，避免将
    训练中断或尚未产生权重的残留目录误识为可用代数，引发下游
    FileNotFoundError 或静默回退。

    Args:
        gen_dir: gen{i}/ 目录的绝对路径（调用方保证目录存在）

    Returns:
        True 表示目录下至少有一个 .pth 文件
    """
    for f in os.listdir(gen_dir):
        if f.endswith(".pth") and os.path.isfile(os.path.join(gen_dir, f)):
            return True
    return False
