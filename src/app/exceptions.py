"""app 层自定义异常

定义 app 层向 GUI 层传达的异常类型，
将底层（ai/game 包）的技术性异常转换为面向用户的友好提示。
"""


class ModelLoadError(Exception):
    """模型加载失败异常

    由 app 层在捕获 create_engine 的 ValueError 后抛出，
    向 GUI 层传达模型加载失败的详细信息。
    GUI 层捕获此异常后弹窗提示用户，回退到模式选择界面。

    典型场景：
    - 用户选择了不存在的模型名称
    - 多代模型（如 cnn_v3）未指定代数

    Examples:
        >>> try:
        ...     engine = create_engine("unknown")
        ... except ValueError as e:
        ...     raise ModelLoadError(f"模型加载失败：{e}") from e
    """
