"""MCTS 搜索器配置常量

控制 MCTS 搜索日志的开关和输出目录。
搜索器在构造时读取这些常量，运行时可通过修改模块属性动态调整。
"""

# 是否启用 MCTS 搜索日志（默认关闭，自对弈时可按需开启）
MCTS_LOG_ENABLED: bool = False

# MCTS 搜索日志输出目录（相对于项目根目录或绝对路径）
MCTS_LOG_DIR: str = "mcts_log"
