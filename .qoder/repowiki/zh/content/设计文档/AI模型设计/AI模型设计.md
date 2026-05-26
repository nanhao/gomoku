# AI模型设计

<cite>
**本文引用的文件**
- [ai/base_model.py](file://ai/base_model.py)
- [ai/base_trainer.py](file://ai/base_trainer.py)
- [ai/models/fc_v1/model.py](file://ai/models/fc_v1/model.py)
- [ai/models/fc_v1/trainer.py](file://ai/models/fc_v1/trainer.py)
- [ai/models/cnn_v1/model.py](file://ai/models/cnn_v1/model.py)
- [ai/models/cnn_v1/trainer.py](file://ai/models/cnn_v1/trainer.py)
- [ai/models/cnn_v2/model.py](file://ai/models/cnn_v2/model.py)
- [ai/models/cnn_v2/trainer.py](file://ai/models/cnn_v2/trainer.py)
- [ai/models/cnn_v3/model.py](file://ai/models/cnn_v3/model.py)
- [ai/models/cnn_v3/trainer.py](file://ai/models/cnn_v3/trainer.py)
- [design/fc_v1_design.md](file://design/fc_v1_design.md)
- [design/cnn_design.md](file://design/cnn_design.md)
- [design/cnn_v2_design.md](file://design/cnn_v2_design.md)
- [design/cnn_v3_design.md](file://design/cnn_v3_design.md)
</cite>

## 目录
1. [简介](#简介)
2. [项目结构](#项目结构)
3. [核心组件](#核心组件)
4. [架构总览](#架构总览)
5. [详细组件分析](#详细组件分析)
6. [依赖分析](#依赖分析)
7. [性能考量](#性能考量)
8. [故障排查指南](#故障排查指南)
9. [结论](#结论)
10. [附录](#附录)

## 简介
本设计文档面向五子棋AI模型，系统梳理并对比三种模型的设计理念与架构选择：FCv1全连接网络、CNNV1/V2/V3卷积神经网络。文档从输入输出设计、网络结构、参数配置、训练与评估方案、性能权衡等方面进行深入分析，并提供架构图与数据流图，帮助读者理解模型的前向传播、掩码损失计算、学习率调度与训练周期等关键流程，为模型选择与优化提供实证依据。

## 项目结构
本项目采用“模型分层 + 通用基类”的组织方式：
- ai/base_model.py 定义统一的模型接口与通用前向推理逻辑，以及board向CNN输入的转换工具。
- ai/base_trainer.py 定义通用训练器抽象与训练流程（单轮次、批量、mini-batch、训练周期），内置AdamW优化器、L1Loss掩码损失、StepLR/ReduceLROnPlateau学习率调度。
- ai/models/fc_v1、cnn_v1、cnn_v2、cnn_v3 分别实现不同架构的模型与对应训练器，均继承自NeuralNetworkModel/BaseTrainer，遵循统一的输入/输出约定与训练接口。

```mermaid
graph TB
subgraph "基础层"
BM["ai/base_model.py<br/>模型基类/通用推理/输入转换"]
BT["ai/base_trainer.py<br/>训练器基类/训练流程/损失/调度"]
end
subgraph "模型层"
FC["ai/models/fc_v1/*<br/>全连接模型"]
C1["ai/models/cnn_v1/*<br/>卷积模型V1"]
C2["ai/models/cnn_v2/*<br/>卷积模型V2(轻量化)"]
C3["ai/models/cnn_v3/*<br/>卷积模型V3(极简)"]
end
BM --> FC
BM --> C1
BM --> C2
BM --> C3
BT --> FC
BT --> C1
BT --> C2
BT --> C3
```

**图示来源**
- [ai/base_model.py:1-323](file://ai/base_model.py#L1-L323)
- [ai/base_trainer.py:1-699](file://ai/base_trainer.py#L1-L699)
- [ai/models/fc_v1/model.py:1-132](file://ai/models/fc_v1/model.py#L1-L132)
- [ai/models/cnn_v1/model.py:1-148](file://ai/models/cnn_v1/model.py#L1-L148)
- [ai/models/cnn_v2/model.py:1-160](file://ai/models/cnn_v2/model.py#L1-L160)
- [ai/models/cnn_v3/model.py:1-139](file://ai/models/cnn_v3/model.py#L1-L139)

**章节来源**
- [ai/base_model.py:1-323](file://ai/base_model.py#L1-L323)
- [ai/base_trainer.py:1-699](file://ai/base_trainer.py#L1-L699)

## 核心组件
- 模型基类与通用推理
  - ai/base_model.NeuralNetworkModel：封装predict/save/load，统一前向推理流程（eval模式、no_grad、Softmax输出）。
  - ai/base_model.board_vector_to_cnn_tensor：将226维向量转换为(1,3,15,15)张量，供CNN模型共享。
- 训练器基类
  - ai/base_trainer.BaseTrainer：统一训练流程（单轮次/批量/mini-batch/训练周期）、设备选择、AdamW优化器、L1Loss掩码损失、StepLR/ReduceLROnPlateau学习率调度。
- FCv1模型
  - ai/models/fc_v1.model.FCv1Network：10层隐藏层（每层1000神经元，ReLU），输出层225维Softmax。
  - ai/models/fc_v1.trainer.FCv1Trainer：继承BaseTrainer，创建FCv1Model。
- CNNV1模型
  - ai/models/cnn_v1.model.CNNV1Network：3层卷积（3→32→64→128）+2层全连接（28800→150→225）。
  - ai/models/cnn_v1.trainer.CNNV1Trainer：继承BaseTrainer，创建CNNV1Model。
- CNNV2模型
  - ai/models/cnn_v2.model.CNNV2Network：统一卷积通道至64，FC1从28800→64，显著降低参数量。
  - ai/models/cnn_v2.trainer.CNNV2Trainer：继承BaseTrainer，创建CNNV2Model。
- CNNV3模型
  - ai/models/cnn_v3.model.CNNV3Network：纯卷积（无FC/BN/Dropout），用1×1卷积替代FC，输出15×15空间概率图再展平Softmax。
  - ai/models/cnn_v3.trainer.CNNV3Trainer：继承BaseTrainer，创建CNNV3Model。

**章节来源**
- [ai/base_model.py:107-323](file://ai/base_model.py#L107-L323)
- [ai/base_trainer.py:32-699](file://ai/base_trainer.py#L32-L699)
- [ai/models/fc_v1/model.py:18-132](file://ai/models/fc_v1/model.py#L18-L132)
- [ai/models/cnn_v1/model.py:26-148](file://ai/models/cnn_v1/model.py#L26-L148)
- [ai/models/cnn_v2/model.py:34-160](file://ai/models/cnn_v2/model.py#L34-L160)
- [ai/models/cnn_v3/model.py:31-139](file://ai/models/cnn_v3/model.py#L31-L139)
- [ai/models/fc_v1/trainer.py:11-44](file://ai/models/fc_v1/trainer.py#L11-L44)
- [ai/models/cnn_v1/trainer.py:13-72](file://ai/models/cnn_v1/trainer.py#L13-L72)
- [ai/models/cnn_v2/trainer.py:13-72](file://ai/models/cnn_v2/trainer.py#L13-L72)
- [ai/models/cnn_v3/trainer.py:13-73](file://ai/models/cnn_v3/trainer.py#L13-L73)

## 架构总览
下图展示模型与训练器的类关系与继承层次，体现统一接口与可替换的训练流程。

```mermaid
classDiagram
class BaseModel {
+predict(board_vector) List[float]
+name str
+full_name str
+save(path) str
+load(path) void
+network nn.Module|None
}
class NeuralNetworkModel {
+DEFAULT_MODEL_PATH str
+predict(board_vector) List[float]
+save(path) str
+load(path) void
+_create_network() nn.Module
+_prepare_input(vector, device) Tensor
}
class FCv1Network {
+forward(x) Tensor
}
class FCv1Model {
+name "fc_v1"
+_create_network() nn.Module
+_prepare_input(vector, device) Tensor
}
class CNNV1Network {
+forward(x) Tensor
}
class CNNV1Model {
+name "cnn_v1"
+_create_network() nn.Module
+_prepare_input(vector, device) Tensor
}
class CNNV2Network {
+forward(x) Tensor
}
class CNNV2Model {
+name "cnn_v2"
+_create_network() nn.Module
+_prepare_input(vector, device) Tensor
}
class CNNV3Network {
+forward(x) Tensor
}
class CNNV3Model {
+name "cnn_v3"
+_create_network() nn.Module
+_prepare_input(vector, device) Tensor
}
class BaseTrainer {
+train_single(dir, epochs, verbose) float
+train_batch(dirs, epochs, verbose, val_ratio) float
+train_mini_batch(d, epochs, bs, ...) float
+train_cycle(rounds, games, epochs, ...) void
+_masked_loss(outputs, targets, masks) Tensor
}
class FCv1Trainer {
+model_name "fc_v1"
+_create_model(fresh) FCv1Model
}
class CNNV1Trainer {
+model_name "cnn_v1"
+_create_model(fresh) CNNV1Model
+_convert_inputs(lists) Tensor
}
class CNNV2Trainer {
+model_name "cnn_v2"
+_create_model(fresh) CNNV2Model
+_convert_inputs(lists) Tensor
}
class CNNV3Trainer {
+model_name "cnn_v3"
+_create_model(fresh) CNNV3Model
+_convert_inputs(lists) Tensor
}
BaseModel <|-- NeuralNetworkModel
NeuralNetworkModel <|-- FCv1Model
NeuralNetworkModel <|-- CNNV1Model
NeuralNetworkModel <|-- CNNV2Model
NeuralNetworkModel <|-- CNNV3Model
FCv1Model --> FCv1Network : "组合"
CNNV1Model --> CNNV1Network : "组合"
CNNV2Model --> CNNV2Network : "组合"
CNNV3Model --> CNNV3Network : "组合"
BaseTrainer <|-- FCv1Trainer
BaseTrainer <|-- CNNV1Trainer
BaseTrainer <|-- CNNV2Trainer
BaseTrainer <|-- CNNV3Trainer
```

**图示来源**
- [ai/base_model.py:24-323](file://ai/base_model.py#L24-L323)
- [ai/base_trainer.py:32-699](file://ai/base_trainer.py#L32-L699)
- [ai/models/fc_v1/model.py:91-132](file://ai/models/fc_v1/model.py#L91-L132)
- [ai/models/cnn_v1/model.py:106-148](file://ai/models/cnn_v1/model.py#L106-L148)
- [ai/models/cnn_v2/model.py:117-160](file://ai/models/cnn_v2/model.py#L117-L160)
- [ai/models/cnn_v3/model.py:96-139](file://ai/models/cnn_v3/model.py#L96-L139)
- [ai/models/fc_v1/trainer.py:11-44](file://ai/models/fc_v1/trainer.py#L11-L44)
- [ai/models/cnn_v1/trainer.py:13-72](file://ai/models/cnn_v1/trainer.py#L13-L72)
- [ai/models/cnn_v2/trainer.py:13-72](file://ai/models/cnn_v2/trainer.py#L13-L72)
- [ai/models/cnn_v3/trainer.py:13-73](file://ai/models/cnn_v3/trainer.py#L13-L73)

## 详细组件分析

### FCv1全连接模型
- 设计理念
  - 采用10层隐藏层（每层1000神经元）的全连接网络，直接对展平的226维向量进行建模，输出225维Softmax概率分布。
  - 该设计强调监督学习与策略预测，但缺乏对棋盘空间结构的建模能力。
- 输入输出设计
  - 输入：226维向量（前225维为棋盘状态按行展开，第226维为当前落子方）。
  - 输出：225维Softmax概率分布，索引i对应位置(i//15, i%15)。
- 网络结构与参数
  - 结构：226→10×ReLU(1000)→225，参数量约946万。
  - 激活：ReLU；输出：Softmax。
- 训练与损失
  - 优化器：AdamW（默认学习率与权重衰减可配置）。
  - 损失：L1Loss（MAE），使用掩码张量对有效位置加权。
  - 目标：Softmax概率分布（由采样器生成）。
- 性能特点
  - 参数量大，训练成本高；对棋盘局部性建模弱，易受噪声影响。
  - 适合探索监督学习策略，但需配合更强的特征工程或结构改进。

```mermaid
flowchart TD
Start(["输入: 226维向量"]) --> Prep["展平/张量化"]
Prep --> F1["线性层1(226→1000)+ReLU"]
F1 --> F2["线性层2(1000→1000)+ReLU"]
F2 --> F3["线性层3(1000→1000)+ReLU"]
F3 --> F4["线性层4(1000→1000)+ReLU"]
F4 --> F5["线性层5(1000→1000)+ReLU"]
F5 --> F6["线性层6(1000→1000)+ReLU"]
F6 --> F7["线性层7(1000→1000)+ReLU"]
F7 --> F8["线性层8(1000→1000)+ReLU"]
F8 --> F9["线性层9(1000→1000)+ReLU"]
F9 --> F10["线性层10(1000→1000)+ReLU"]
F10 --> Out["线性层输出(1000→225)"]
Out --> Softmax["Softmax(225)"]
Softmax --> End(["输出: 225维概率"])
```

**图示来源**
- [ai/models/fc_v1/model.py:18-88](file://ai/models/fc_v1/model.py#L18-L88)

**章节来源**
- [ai/models/fc_v1/model.py:18-132](file://ai/models/fc_v1/model.py#L18-L132)
- [design/fc_v1_design.md:1-712](file://design/fc_v1_design.md#L1-L712)

### CNNV1卷积模型
- 设计理念
  - 采用3层卷积（3→32→64→128）+紧凑全连接（28800→150→225），参数量与FCv1相当，强调空间特征提取。
- 输入输出设计
  - 输入：(1,3,15,15)三通道张量（黑棋、白棋、当前方）。
  - 输出：225维Softmax概率分布。
- 网络结构与参数
  - 结构：Conv→BN→ReLU（3层）→Flatten→Linear(28800→150)→ReLU→Linear(150→225)→Softmax。
  - 参数量：约445万。
- 训练与损失
  - 优化器：AdamW；损失：L1Loss掩码；学习率调度：StepLR或ReduceLROnPlateau。
- 性能特点
  - 利用卷积的空间局部性，特征表达更稳健；参数量适中，适合监督学习策略。

```mermaid
flowchart TD
S(["输入: (1,3,15,15)"]) --> C1["Conv3→32 + BN + ReLU"]
C1 --> C2["Conv32→64 + BN + ReLU"]
C2 --> C3["Conv64→128 + BN + ReLU"]
C3 --> Flat["Flatten(128×15×15)"]
Flat --> FC1["Linear(28800→150)+ReLU"]
FC1 --> FCOut["Linear(150→225)"]
FCOut --> SM["Softmax(225)"]
SM --> E(["输出: 225维概率"])
```

**图示来源**
- [ai/models/cnn_v1/model.py:26-103](file://ai/models/cnn_v1/model.py#L26-L103)

**章节来源**
- [ai/models/cnn_v1/model.py:26-148](file://ai/models/cnn_v1/model.py#L26-L148)
- [design/cnn_design.md:1-285](file://design/cnn_design.md#L1-L285)

### CNNV2卷积模型（轻量化）
- 设计理念
  - 在CNNV1基础上统一卷积通道至64，FC1从28800→64，显著降低参数量与计算开销。
- 输入输出设计
  - 输入/输出与CNNV1一致。
- 网络结构与参数
  - 结构：Conv3→64（3层）→Linear(14400→64)→Linear(64→225)→Softmax。
  - 参数量：约101万（-77%）。
- 训练与损失
  - 优化器：AdamW；损失：L1Loss掩码；学习率调度：StepLR或ReduceLROnPlateau。
- 性能特点
  - 显著提升训练效率与内存占用；在保持预测精度的同时大幅缩短收敛时间。

```mermaid
flowchart TD
S2(["输入: (1,3,15,15)"]) --> C1_2["Conv3→64 + BN + ReLU"]
C1_2 --> C2_2["Conv64→64 + BN + ReLU"]
C2_2 --> C3_2["Conv64→64 + BN + ReLU"]
C3_2 --> Flat2["Flatten(64×15×15=14400)"]
Flat2 --> FC1_2["Linear(14400→64)+ReLU"]
FC1_2 --> FCOut2["Linear(64→225)"]
FCOut2 --> SM2["Softmax(225)"]
SM2 --> E2(["输出: 225维概率"])
```

**图示来源**
- [ai/models/cnn_v2/model.py:34-114](file://ai/models/cnn_v2/model.py#L34-L114)

**章节来源**
- [ai/models/cnn_v2/model.py:34-160](file://ai/models/cnn_v2/model.py#L34-L160)
- [design/cnn_v2_design.md:1-96](file://design/cnn_v2_design.md#L1-L96)

### CNNV3卷积模型（极简）
- 设计理念
  - 纯卷积网络，无FC/BN/Dropout，用1×1卷积替代全连接层，直接输出15×15空间概率图，全程保持空间结构。
- 输入输出设计
  - 输入/输出与CNNV1一致。
- 网络结构与参数
  - 结构：Conv3→32（3层）→Conv1×1(32→1)→ReLU→展平→Softmax。
  - 参数量：约1.9万（-98%）。
- 训练与损失
  - 优化器：AdamW；损失：L1Loss掩码；学习率调度：StepLR或ReduceLROnPlateau。
- 性能特点
  - 参数量极小，推理与训练速度最快；结构最简，便于理解与调试；在监督学习场景下具备良好泛化潜力。

```mermaid
flowchart TD
S3(["输入: (1,3,15,15)"]) --> C1_3["Conv3→32 + ReLU"]
C1_3 --> C2_3["Conv32→32 + ReLU"]
C2_3 --> C3_3["Conv32→32 + ReLU"]
C3_3 --> COut["Conv1×1(32→1) + ReLU"]
COut --> Flat3["展平(1×15×15)"]
Flat3 --> SM3["Softmax(225)"]
SM3 --> E3(["输出: 225维概率"])
```

**图示来源**
- [ai/models/cnn_v3/model.py:31-93](file://ai/models/cnn_v3/model.py#L31-L93)

**章节来源**
- [ai/models/cnn_v3/model.py:31-139](file://ai/models/cnn_v3/model.py#L31-L139)
- [design/cnn_v3_design.md:1-173](file://design/cnn_v3_design.md#L1-L173)

### 训练流程与数据流
- 训练流程（BaseTrainer）
  - 设备选择：优先CUDA，否则CPU。
  - 优化器：AdamW（可配置权重衰减）。
  - 损失：L1Loss（reduction='none'），结合掩码张量进行加权平均。
  - 学习率调度：StepLR或ReduceLROnPlateau。
  - 训练模式：单轮次、批量、mini-batch、训练周期。
- 数据流（以CNN为例）
  - 输入：226维向量→(N,3,15,15)张量。
  - 前向：卷积→ReLU→展平→全连接→Softmax。
  - 损失：masked L1Loss，按mask有效位置求平均。
  - 反向：计算梯度并更新参数。

```mermaid
sequenceDiagram
participant T as "训练器(BaseTrainer)"
participant D as "数据加载(SampleLoader)"
participant M as "模型(具体模型)"
participant O as "优化器(AdamW)"
T->>D : 加载/准备样本(N,226)与(N,225)目标
D-->>T : 输入张量(N,3,15,15)或(N,226)
T->>M : 前向传播
M-->>T : 输出概率分布(N,225)
T->>T : 计算masked L1Loss
T->>O : 反向传播与参数更新
O-->>M : 更新权重
T-->>T : 记录loss/梯度范数/学习率
```

**图示来源**
- [ai/base_trainer.py:240-322](file://ai/base_trainer.py#L240-L322)
- [ai/base_trainer.py:662-699](file://ai/base_trainer.py#L662-L699)
- [ai/base_model.py:278-323](file://ai/base_model.py#L278-L323)

**章节来源**
- [ai/base_trainer.py:32-699](file://ai/base_trainer.py#L32-L699)
- [ai/base_model.py:107-323](file://ai/base_model.py#L107-L323)

## 依赖分析
- 组件耦合
  - 模型与训练器通过抽象接口解耦：模型仅需实现网络创建与输入准备，训练器统一处理优化、损失、调度与日志。
  - CNN模型共享输入转换函数，避免重复实现。
- 外部依赖
  - PyTorch（nn.Module、optim、L1Loss、Softmax等）。
  - 设备自动检测（CUDA优先）。
- 潜在风险
  - 训练器对模型的network属性强依赖，需确保子类正确设置。
  - 掩码机制需与采样器生成的mask保持一致。

```mermaid
graph LR
BT["BaseTrainer"] --> OPT["torch.optim.AdamW"]
BT --> CR["nn.L1Loss(reduction='none')"]
BT --> SM["nn.Softmax(dim=1)"]
BT --> CV["board_vector_to_cnn_tensor"]
CV --> CNNT["CNN输入转换"]
FCN["FCv1Network"] --> SM
CNN1["CNNV1Network"] --> SM
CNN2["CNNV2Network"] --> SM
CNN3["CNNV3Network"] --> SM
```

**图示来源**
- [ai/base_trainer.py:89-101](file://ai/base_trainer.py#L89-L101)
- [ai/base_model.py:278-323](file://ai/base_model.py#L278-L323)
- [ai/models/fc_v1/model.py:18-88](file://ai/models/fc_v1/model.py#L18-L88)
- [ai/models/cnn_v1/model.py:26-103](file://ai/models/cnn_v1/model.py#L26-L103)
- [ai/models/cnn_v2/model.py:34-114](file://ai/models/cnn_v2/model.py#L34-L114)
- [ai/models/cnn_v3/model.py:31-93](file://ai/models/cnn_v3/model.py#L31-L93)

**章节来源**
- [ai/base_trainer.py:32-144](file://ai/base_trainer.py#L32-L144)
- [ai/base_model.py:278-323](file://ai/base_model.py#L278-L323)

## 性能考量
- 计算复杂度
  - FCv1：线性层参数量大，前向/反向传播开销高；适合探索性实验。
  - CNNV1：卷积+全连接，空间特征提取强，计算量中等。
  - CNNV2：统一卷积通道+压缩FC，显著降低计算与内存占用。
  - CNNV3：纯卷积+1×1卷积，参数量最小，计算与内存开销最低。
- 内存占用
  - CNNV3 < CNNV2 < CNNV1 << FCv1。
- 训练效率
  - CNNV3最快，CNNV2次之，CNNV1再次之，FCv1最慢。
- 收敛特性
  - CNN系列利用空间结构，通常具有更好的泛化与稳定性；FCv1需更强的正则与特征工程。

[本节为通用性能讨论，无需列出具体文件来源]

## 故障排查指南
- 模型加载失败
  - 检查DEFAULT_MODEL_PATH是否存在；多代模型需明确gen参数以定位权重文件。
- 设备错误
  - 确认CUDA可用性；训练器会自动选择GPU/CPU。
- 训练中断
  - Ctrl+C中断会保存当前进度与图表；可安全重启。
- 掩码问题
  - 确保targets/masks维度与输出一致；BaseTrainer会按mask有效位置求平均。

**章节来源**
- [ai/base_model.py:126-166](file://ai/base_model.py#L126-L166)
- [ai/base_trainer.py:115-120](file://ai/base_trainer.py#L115-L120)
- [ai/base_trainer.py:314-322](file://ai/base_trainer.py#L314-L322)
- [ai/base_trainer.py:678-699](file://ai/base_trainer.py#L678-L699)

## 结论
- FCv1适合探索监督学习策略与快速原型验证，但受限于全连接结构与高参数量。
- CNNV1在空间特征提取方面优于FCv1，适合稳健的监督学习策略。
- CNNV2在CNNV1基础上显著降低参数量与计算开销，兼顾效率与精度。
- CNNV3以极简结构实现最强的训练/推理效率，适合资源受限场景与快速迭代。
- 建议：中小规模实验优先CNNV2；大规模训练与部署优先CNNV3；需要更强特征表达时采用CNNV1。

[本节为总结性内容，无需列出具体文件来源]

## 附录
- 训练入口与CLI
  - 支持单轮次训练、批量训练、mini-batch训练与训练周期；可配置学习率调度与验证集比例。
- 评估与验证
  - 支持验证集损失记录与曲线生成；支持密集日志与稀疏日志模式。
- 设计文档参考
  - FCv1、CNNV1/V2/V3设计文档提供了详细的参数量估算、接口说明与测试用例。

**章节来源**
- [ai/base_trainer.py:324-660](file://ai/base_trainer.py#L324-L660)
- [design/fc_v1_design.md:1-712](file://design/fc_v1_design.md#L1-L712)
- [design/cnn_design.md:1-285](file://design/cnn_design.md#L1-L285)
- [design/cnn_v2_design.md:1-96](file://design/cnn_v2_design.md#L1-L96)
- [design/cnn_v3_design.md:1-173](file://design/cnn_v3_design.md#L1-L173)