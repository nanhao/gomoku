"""统一模型注册表（懒加载）

集中管理所有 AI 模型和训练器的映射关系，消除 engine.py、battle.py、train_main.py
三处的重复注册逻辑。使用 importlib 实现按需懒加载，首次调用时才导入对应模块。

注册表数据结构：
    MODEL_REGISTRY: dict[str, tuple[str, str | None]]
    - key: 模型名称（如 "random"、"fc_v1"）
    - value: (模型类全限定路径, 训练器类全限定路径或 None)

提供的公共函数：
    - get_model(name) -> BaseModel: 根据名称返回模型实例
    - get_trainer(name, **kwargs) -> BaseTrainer: 根据名称返回训练器实例
    - list_models() -> list[str]: 返回所有注册的模型名称列表
"""

import importlib


# ---------------------------------------------------------------------------
# 注册表数据：统一维护所有模型和训练器的映射
# 格式: { 名称: (模型类全限定路径, 训练器类全限定路径 | None) }
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, tuple[str, str | None]] = {
    "random": ("ai.models.random.model.RandomModel", None),
    "fc_v1": ("ai.models.fc_v1.model.FCv1Model", "ai.models.fc_v1.trainer.FCv1Trainer"),
    "cnn_v1": ("ai.models.cnn_v1.model.CNNV1Model", "ai.models.cnn_v1.trainer.CNNV1Trainer"),
    "cnn_v2": ("ai.models.cnn_v2.model.CNNV2Model", "ai.models.cnn_v2.trainer.CNNV2Trainer"),
    "cnn_v3": ("ai.models.cnn_v3.model.CNNV3Model", "ai.models.cnn_v3.trainer.CNNV3Trainer"),
}


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


def get_model(name: str):
    """根据名称返回模型实例（懒加载）

    首次调用时才通过 importlib 导入对应模块，后续调用利用 sys.modules 缓存。

    Args:
        name: 模型名称，如 "random"、"fc_v1"、"cnn_v1"、"cnn_v2"

    Returns:
        模型实例（BaseModel 子类）

    Raises:
        ValueError: 未知的模型名称
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"未知的模型: '{name}'")

    model_path, _ = MODEL_REGISTRY[name]
    model_cls = _import_class(model_path)
    return model_cls()


def get_trainer(name: str, **kwargs):
    """根据名称返回训练器实例（懒加载）

    首次调用时才通过 importlib 导入对应模块，后续调用利用 sys.modules 缓存。

    Args:
        name: 模型名称，如 "fc_v1"、"cnn_v1"、"cnn_v2"
        **kwargs: 传递给训练器构造函数的额外参数（如 fresh=True, lr_scheduler_type="step"）

    Returns:
        训练器实例（BaseTrainer 子类）

    Raises:
        ValueError: 不支持的模型名称，或该模型没有训练器（如 random）
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"不支持的模型: '{name}'")

    _, trainer_path = MODEL_REGISTRY[name]
    if trainer_path is None:
        raise ValueError(f"不支持的模型: '{name}'")

    trainer_cls = _import_class(trainer_path)
    return trainer_cls(**kwargs)


def list_models() -> list[str]:
    """返回所有注册的模型名称列表

    Returns:
        模型名称列表，如 ["random", "fc_v1", "cnn_v1", "cnn_v2"]
    """
    return list(MODEL_REGISTRY.keys())
