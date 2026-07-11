"""AI 引擎工厂模块

AIEngineBuilder 提供统一的引擎创建入口，封装两种引擎的构造逻辑。
create_engine 提供智能路由：根据模型预测输出类型自动选择对应引擎。
"""

from typing import Optional


class AIEngineBuilder:
    """AI 引擎工厂类

    提供两个静态工厂方法，统一对外创建引擎实例：

    - build_single_policy: 创建 SinglePolicyAiEngine（通过 registry 加载模型）
    - build_policy_value: 创建 PolicyValueAiEngine（内部创建 MCTSSearcher）
    """

    @staticmethod
    def build_single_policy(
        ai_name: str,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        rng=None,
    ):
        """创建单策略引擎

        通过 registry.get_model(ai_name) 获取模型，再包装为
        SinglePolicyAiEngine。

        Args:
            ai_name: AI 名称，如 "random"、"fc_v1"、"cnn_v3.1"
            temperature: 温度参数，仅在 sampling 策略时生效
            top_k: 采样截断阈值，None 表示不截断
            rng: 可选随机源

        Returns:
            SinglePolicyAiEngine 实例

        Raises:
            ValueError: 模型名称未知、多代模型未指定代数等
        """
        from ai.registry import get_model
        from ai.engine.single_policy import SinglePolicyAiEngine

        model = get_model(ai_name)
        return SinglePolicyAiEngine(
            model,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
        )

    @staticmethod
    def build_policy_value(
        model,
        num_simulations: int = 200,
        c_puct: float = 2.0,
        dirichlet_epsilon: float = 0.25,
        dirichlet_alpha: float = 0.03,
        batch_size: Optional[int] = None,
        virtual_loss: Optional[float] = None,
        temperature: float = 1.0,
        temp_steps: int = 8,
        log_key_prefix: Optional[str] = None,
        log_dir=None,
    ):
        """创建策略-价值引擎

        内部创建 MCTSSearcher + PolicyValueAiEngine。
        每次调用均创建全新实例，满足跨局不复用约束。

        Args:
            model: BaseModel 实例
            num_simulations: 每次搜索的模拟次数，默认 200
            c_puct: PUCT 探索系数，默认 2.0
            dirichlet_epsilon: 狄利克雷噪声比例，默认 0.25
            dirichlet_alpha: 狄利克雷分布参数，默认 0.03
            batch_size: 每批 simulation 数，None 时读配置默认 16
            virtual_loss: virtual loss 值，None 时读配置默认 3.0
            temperature: 温度参数，默认 1.0
            temp_steps: 温度探索步数，默认 8
            log_key_prefix: 日志键前缀
            log_dir: 搜索日志输出目录

        Returns:
            PolicyValueAiEngine 实例
        """
        from ai.mcts.searcher import MCTSSearcher
        from ai.engine.policy_value import PolicyValueAiEngine

        searcher = MCTSSearcher(
            model=model,
            num_simulations=num_simulations,
            c_puct=c_puct,
            dirichlet_epsilon=dirichlet_epsilon,
            dirichlet_alpha=dirichlet_alpha,
            batch_size=batch_size,
            virtual_loss=virtual_loss,
            log_dir=log_dir,
        )
        return PolicyValueAiEngine(
            searcher,
            temperature=temperature,
            temp_steps=temp_steps,
            log_key_prefix=log_key_prefix,
        )


# ---------- create_engine 智能路由 ----------

def create_engine(ai_name: str):
    """根据 AI 名称智能创建对应类型的 AIEngine 实例。

    通过 ``registry.get_model(ai_name)`` 加载模型后，自动执行一次空棋盘测试预测，
    根据 ``ModelPredict.type`` 判断模型输出类型：

    - ``TYPE_POLICY_VALUE`` → 创建 ``PolicyValueAiEngine``（双头模型，含 MCTS）
    - ``TYPE_POLICY_ONLY``  → 创建 ``SinglePolicyAiEngine``（单头模型）

    GPU 迁移：对 ``model.network`` 非 None 的模型自动调用 ``model.move_to_device()``，
    调用方无需手动处理设备迁移。

    异常策略：
    - 模型加载失败（未知名称、多代未指定代数） → 上抛 ``ValueError``
    - 测试预测失败（模型异常、设备错误） → 上抛 ``ValueError``
    - 引擎构造内部异常 → 直接透传

    Args:
        ai_name: AI 名称，如 "random"、"fc_v1"、"cnn_v7.1"

    Returns:
        AIEngine 子类实例（SinglePolicyAiEngine 或 PolicyValueAiEngine）

    Raises:
        ValueError: 模型未知、多代未指定、或测试预测失败
    """
    from ai.registry import get_model
    from ai.base_model import ModelPredict

    # 1. 加载模型
    model = get_model(ai_name)

    # 2. GPU 迁移：network 非 None 且 CUDA 可用时自动迁移
    if model.network is not None:
        model.move_to_device()

    # 3. 测试预测：空棋盘 [0]*226，仅用 result.type 判断引擎类型
    try:
        result = model.predict([0] * 226)
    except Exception as e:
        raise ValueError(
            f"create_engine('{ai_name}'): 测试预测失败，无法判断引擎类型: {e}"
        ) from e

    # 4. 根据类型智能路由
    if result.type == ModelPredict.TYPE_POLICY_VALUE:
        # 双头模型 → 创建 PolicyValueAiEngine（使用默认 MCTS 参数）
        return AIEngineBuilder.build_policy_value(
            model,
            num_simulations=800,
            c_puct=2.0,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=0.03,
            temperature=1.0,
            temp_steps=8,
            log_key_prefix=None,
            log_dir=None,
        )
    else:
        # 单头模型（TYPE_POLICY_ONLY）→ 创建 SinglePolicyAiEngine
        return AIEngineBuilder.build_single_policy(ai_name)
