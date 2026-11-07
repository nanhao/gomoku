"""CNN v12 训练器实现

负责训练 CNN v12 轻量模型（MCTS 新体系）。
继承 CNNV11Trainer，使用双头损失（MSE + CrossEntropy），
相对视角编码由父类 _convert_inputs 提供，与 cnn_v11 逐位一致。

依赖声明：本模块依赖 ai.models.cnn_v11.trainer
（继承 CNNV11Trainer）。若删除 cnn_v11 包，cnn_v12 将同时失效。
"""

from ai.models.cnn_v11.trainer import CNNV11Trainer

from .model import CNNV12Model


class CNNV12Trainer(CNNV11Trainer):
    """CNN v12 训练器（MCTS 新体系）

    仅覆盖 _create_model 与 model_name；
    _convert_inputs / train_epochs / _train_step / save_model 全部继承。

    Attributes:
        model: CNNV12Model 实例
        optimizer: AdamW 优化器
        device: 训练设备
    """

    def _create_model(self, fresh: bool) -> CNNV12Model:
        """创建 CNN v12 模型实例

        Args:
            fresh: 是否创建新模型（不加载已保存的模型）

        Returns:
            CNNV12Model 实例
        """
        gen = getattr(self, 'gen', None)
        return CNNV12Model(fresh=fresh, gen=gen) if gen is not None else CNNV12Model(fresh=fresh)

    @property
    def model_name(self) -> str:
        """模型名称

        Returns:
            "cnn_v12"
        """
        return "cnn_v12"
