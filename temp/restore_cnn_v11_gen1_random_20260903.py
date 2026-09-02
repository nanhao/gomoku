"""恢复 cnn_v11.1（gen1）为随机权重

背景：tests/ai/unit/test_cnn_v11_trainer.py::test_constructor_gen 曾直接
shutil.rmtree 删除真实 src/ai/models/cnn_v11/gen1 目录，导致 cnn_v11.1
权重丢失且无法从 git 恢复（*.pth/*.pt 被 .gitignore 忽略）。
本脚本重建 gen1 目录：cnn_v11.pth（随机 state_dict）+
cnn_v11_torchscript.pt（TorchScript 导出），格式与现有 gen2 完全一致。

运行：python temp/restore_cnn_v11_gen1_random_20260903.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from ai.models.cnn_v11.model import CNNV11Model


def main() -> None:
    """生成随机权重并导出 TorchScript，输出文件路径与大小"""
    gen1_dir = os.path.join(
        os.path.dirname(CNNV11Model.DEFAULT_MODEL_PATH), "gen1"
    )
    os.makedirs(gen1_dir, exist_ok=True)
    pth_path = os.path.join(gen1_dir, "cnn_v11.pth")
    ts_path = os.path.join(gen1_dir, "cnn_v11_torchscript.pt")

    # 随机初始化权重（fresh=True 不加载任何已有权重）
    model = CNNV11Model(fresh=True)
    model.network.eval()
    saved = model.save(pth_path)
    print(f"[gen1 pth] {saved} ({os.path.getsize(saved)} bytes)")

    # 导出 TorchScript（与 CnnV11TrainLoop._export_torchscript 相同方式）
    scripted = torch.jit.script(model.network)
    torch.jit.save(scripted, ts_path)
    print(f"[gen1 ts ] {ts_path} ({os.path.getsize(ts_path)} bytes)")

    # 回读验证：非 fresh + gen=1 应能从 gen1 加载
    check = CNNV11Model(fresh=False, gen=1)
    print(f"[verify  ] gen={check.gen} 加载成功: {check.DEFAULT_MODEL_PATH}")


if __name__ == "__main__":
    main()
