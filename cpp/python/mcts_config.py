"""C++ MCTS 模块配置参数

所有参数从 mcts_config.json 动态读取，修改 JSON 后无需重新编译。
JSON 文件位于本模块同目录下，与 C++ 端共享同一配置文件。

用法：
    from cpp.python.mcts_config import NUM_SIMULATIONS, NUM_THREADS
    searcher = CppMctsAdapter(model_path, num_simulations=NUM_SIMULATIONS)
"""

import json
import os

# ---- 从 JSON 加载 ----

_config_path = os.path.join(os.path.dirname(__file__), "mcts_config.json")
_config = {}

try:
    with open(_config_path, "r", encoding="utf-8") as f:
        _config = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    pass  # 使用下方硬编码回退值


def _get(key: str, default):
    """从 JSON 配置读取，缺失时回退到默认值"""
    return _config.get(key, default)


# ---- 配置项 ----

# 每次搜索的总模拟次数
NUM_SIMULATIONS: int = _get("num_simulations", 800)

# MCTS 工作线程数（不含 GPU 推理线程）
NUM_THREADS: int = _get("num_threads", 32)

# PUCT 探索系数：控制探索 vs 利用的平衡
C_PUCT: float = _get("c_puct", 2.0)

# 狄利克雷噪声混合比例（ε）：P' = (1-ε)*P + ε*noise
DIRICHLET_EPSILON: float = _get("dirichlet_epsilon", 0.25)

# 狄利克雷分布浓度参数（α）：越小噪声越集中
DIRICHLET_ALPHA: float = _get("dirichlet_alpha", 0.03)

# GPU 推理攒批大小：推理线程攒够此数量后一次前向（向后兼容）
BATCH_SIZE: int = _get("batch_size", 16)

# GPU 推理攒批下限：队列少于此值时推理线程继续等待（配合 1ms 超时）
BATCH_LOWER_BOUND: int = _get("batch_lower_bound", 16)

# GPU 推理攒批上限：单次推理最多取走的请求数
BATCH_UPPER_BOUND: int = _get("batch_upper_bound", 500)

# 虚拟损失值：选择阶段对节点施加的惩罚系数
VIRTUAL_LOSS: float = _get("virtual_loss", 3.0)

# 是否使用 GPU 推理（False 时使用 CPU）
USE_GPU: bool = _get("use_gpu", True)

# ---- 全局线程池 ----

def init_global_thread_pool(num_threads: int = NUM_THREADS) -> None:
    """Initialize the global MCTS thread pool (process-wide, once only).

    Call this before the first CppMctsAdapter creation to control the
    thread count.  Without an explicit call, the pool auto-initialises
    with NUM_THREADS on the first search().

    Args:
        num_threads: Number of persistent worker threads.
    """
    import cpp_mcts
    cpp_mcts.init_thread_pool(num_threads)


def thread_pool_size() -> int:
    """Return the number of threads in the global pool."""
    import cpp_mcts
    return cpp_mcts.thread_pool_size()
