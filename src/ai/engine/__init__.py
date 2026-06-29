"""AI 引擎子包

统一对外导出所有引擎相关符号。
"""

from ai.engine.ai_engine import AIEngine, MoveDecision
from ai.engine.single_policy import SinglePolicyAiEngine
from ai.engine.policy_value import PolicyValueAiEngine
from ai.engine.builder import AIEngineBuilder


def create_engine(ai_name: str):
    """根据 AI 名称创建对应的 SinglePolicyAiEngine 实例。

    AI 名称到模型的映射由 ai.registry 统一维护，外部调用者无需了解具体模型类。

    异常策略：
    - 未知模型名称、多代模型未指定代数等场景，直接上抛 ValueError，
      由 app 层捕获并转换为 ModelLoadError，向 GUI 层传达模型加载失败信息。
    - 不再回退到 RandomModel，避免静默降级导致用户误以为使用了指定模型。

    Args:
        ai_name: AI 名称，如 "random"、"fc_v1"、"cnn_v3.1"

    Returns:
        对应的 SinglePolicyAiEngine 实例

    Raises:
        ValueError: 模型名称未知、多代模型未指定代数等
    """
    return AIEngineBuilder.build_single_policy(ai_name)


__all__ = [
    "AIEngine",
    "MoveDecision",
    "SinglePolicyAiEngine",
    "PolicyValueAiEngine",
    "AIEngineBuilder",
    "create_engine",
]
