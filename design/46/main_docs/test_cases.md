# 测试用例集

> 用例使用文字和符号描述。**GIVEN** = 前置条件，**WHEN** = 触发操作，**THEN** = 预期结果。

---

## 1. BoardState 单元测试

### TC-BS-01 空棋盘检测
- **GIVEN** 空棋盘（全 0），当前玩家=1
- **WHEN** 调用 has_winner()
- **THEN** 返回 false
- **WHEN** 调用 is_full()
- **THEN** 返回 false
- **WHEN** 调用 is_terminal()
- **THEN** 返回 false
- **WHEN** 调用 get_legal_actions()
- **THEN** 返回长度为 225 的列表

### TC-BS-02 满棋盘平局
- **GIVEN** 满棋盘（无空位），无五连
- **WHEN** 调用 has_winner()
- **THEN** 返回 false
- **WHEN** 调用 is_full()
- **THEN** 返回 true
- **WHEN** 调用 is_terminal()
- **THEN** 返回 true
- **WHEN** 调用 terminal_value()
- **THEN** 返回 0.0

### TC-BS-03 水平五连（黑胜）
- **GIVEN** 棋盘第 0 行位置 [0,1,2,3,4] 均为黑子(1)，其余随机填充但无其他五连
- **WHEN** 调用 has_winner()
- **THEN** 返回 true
- **WHEN** 调用 get_winner()
- **THEN** 返回 1（黑胜）
- **WHEN** 调用 terminal_value()（当前玩家=1）
- **THEN** 返回 1.0

### TC-BS-04 垂直五连（白胜）
- **GIVEN** 棋盘第 0 列位置 [0,15,30,45,60] 均为白子(2)，其余随机
- **WHEN** 调用 get_winner()
- **THEN** 返回 2（白胜）

### TC-BS-05 主对角线五连
- **GIVEN** 位置 [0,16,32,48,64] 均为黑子(1)
- **WHEN** 调用 get_winner()
- **THEN** 返回 1

### TC-BS-06 副对角线五连
- **GIVEN** 位置 [4,18,32,46,60] 均为白子(2)
- **WHEN** 调用 get_winner()
- **THEN** 返回 2

### TC-BS-07 边缘对角线（最后一组）
- **GIVEN** 位置 [210,196,182,168,154] 均为黑子(1)
- **WHEN** 调用 get_winner()
- **THEN** 返回 1

### TC-BS-08 四连无胜者
- **GIVEN** 位置 [0,1,2,3] 为黑子，位置 [4] 为空
- **WHEN** 调用 has_winner()
- **THEN** 返回 false

### TC-BS-09 apply_action 创建新状态
- **GIVEN** BoardState("空棋盘", player=1)
- **WHEN** 调用 apply_action(0)
- **THEN** 返回新 BoardState，其 board[0]=1, current_player=2
- **WHEN** 再次检查原 BoardState
- **THEN** 原状态不变（board[0]=0, current_player=1）

### TC-BS-10 终局状态 terminal_value
- **GIVEN** 黑子五连的棋盘，current_player=2（白方视角）
- **WHEN** 调用 terminal_value()
- **THEN** 返回 -1.0（白方视角下黑胜=白负）

### TC-BS-11 编码一致性
- **GIVEN** 棋盘：位置 0=1(黑)，位置 1=2(白)，current_player=1
- **WHEN** 调用 encode_for_network()
- **THEN** 返回 3×15×15 相对视角编码：
  - 通道0（我方=黑）：位置 0=1.0, 其余 0
  - 通道1（对方=白）：位置 1=1.0, 其余 0
  - 通道2（我方颜色）：全 1.0
- **AND** 编码结果与 Python `board_vector_to_cnn_tensor_relative` 输出逐元素一致（误差 < 1e-6）

---

## 2. MctsNode 单元测试

### TC-MN-01 根节点创建
- **GIVEN** MctsNode(parent=nullptr, prior_prob=1.0)
- **WHEN** 调用 is_leaf()
- **THEN** 返回 true
- **WHEN** 调用 select_child(c_puct=2.0, virtual_loss=3.0)
- **THEN** 返回 -1（无子节点可选）

### TC-MN-02 展开子节点
- **GIVEN** 根节点，legal_actions=[0,1,2], action_priors=[0.5, 0.3, 0.2]
- **WHEN** 调用 expand(action_priors, legal_actions)
- **THEN** is_leaf() 返回 false
- **WHEN** 再次调用 expand 相同参数
- **THEN** 幂等，不创建重复子节点

### TC-MN-03 PUCT 选择最优子节点
- **GIVEN** 根节点已展开，3 个子节点：action 0(N=10,W=5,Q=0.5,P=0.5)，action 1(N=3,W=2,Q=0.67,P=0.3)，action 2(N=0,W=0,Q=0,P=0.2)
- **WHEN** 调用 select_child(c_puct=2.0, virtual_loss=0)
- **THEN** 返回 PUCT 得分最高的 action 索引
- **AND** 被选中的子节点 N_unobserved 和 virtual_loss 递增

### TC-MN-04 Virtual Loss 施加效果
- **GIVEN** 同上节点状态，virtual_loss=3.0
- **WHEN** 连续调用 select_child 3 次
- **THEN** 3 次选中不同 action（因 VL 惩罚使之前选中的分支变差）
- **AND** 3 个子节点的 virtual_loss 均为 3

### TC-MN-05 Backup 回传
- **GIVEN** 二叉树：root → child（action=0），root.get_visit_counts()[0]=3
- **WHEN** 对 child 调用 backup(0.8)
- **THEN** child.Q = (0*Q + 0.8) / 1 = 0.8, child.N = 1 + before
- **AND** root 边 [0] 的 N += 1, W += (-0.8)，Q = W/N（取反传播）
- **AND** child 的 virtual_loss -= 1, n_unobserved -= 1

### TC-MN-06 多层 Backup 取反
- **GIVEN** 三层树：root → A(action=0) → B(action=1)，leaf_value=1.0
- **WHEN** 对 B 调用 backup(1.0)
- **THEN** B.Q = (0 + 1.0) / 1 = 1.0
- **AND** A 边[1] W += -1.0 → Q = -1.0（取反）
- **AND** root 边[0] W += -(-1.0) = 1.0 → Q = 1.0（二次取反）

### TC-MN-07 get_visit_counts 维度
- **GIVEN** 根节点已展开，action=[0,5,10] 各有子节点
- **WHEN** 调用 get_visit_counts()
- **THEN** 返回 225 维向量，非零值仅在 action=[0,5,10] 位置

---

## 3. NeuralNet 单元测试

### TC-NN-01 模型加载成功
- **GIVEN** 有效的 TorchScript 模型文件路径
- **WHEN** 创建 NeuralNet(model_path, batch_size=16, use_gpu=false)
- **THEN** 构造成功，不抛出异常

### TC-NN-02 模型文件不存在
- **GIVEN** 不存在的文件路径
- **WHEN** 创建 NeuralNet
- **THEN** 抛出异常，包含文件路径信息

### TC-NN-03 单次同步推理
- **GIVEN** 已加载的 NeuralNet，一个合法的 encoded_board 输入
- **WHEN** 调用 predict(encoded_board).get()
- **THEN** 返回 NeuralNetResult：policy 长度=225，value ∈ [-1, 1]
- **AND** policy 各元素 ∈ (0, 1)，总和 ≈ 1.0（误差 < 1e-5）

### TC-NN-04 批量异步推理正确性
- **GIVEN** batch_size=4，提交 4 个不同的 encoded_board
- **WHEN** 分别对 4 个 future 调用 .get()
- **THEN** 4 个结果均正确返回
- **AND** 每对 future 的返回顺序与提交顺序无关（各自独立）

### TC-NN-05 大批量压力测试
- **GIVEN** batch_size=32，提交 100 个推理请求
- **WHEN** 等待所有 future.get() 完成
- **THEN** 100 个结果全部正确，无遗漏、无崩溃

### TC-NN-06 非法输入处理
- **GIVEN** encoded_board 长度不等于 3×15×15=675
- **WHEN** 调用 predict
- **THEN** future.get() 抛出异常

### TC-NN-07 输出与 Python 模型一致
- **GIVEN** C++ NeuralNet 加载 TorchScript 模型，Python CNNV11Model 加载相同权重的 .pth
- **WHEN** 对同一棋盘状态分别进行推理
- **THEN** policy 向量逐元素差异 < 1e-5，value 差异 < 1e-5

---

## 4. MctsSearcher 单元测试

### TC-MS-01 空棋盘搜索
- **GIVEN** MctsSearcher(num_simulations=100)，空棋盘 state（全 0，player=1）
- **WHEN** 调用 search(state)
- **THEN** 返回 MctsResult：visit_counts 总和 = 100（无子树复用时）
- **AND** root_policy 长度=225，各元素 > 0

### TC-MS-02 近终局搜索（胜局一步到位）
- **GIVEN** 棋盘黑子四连 + 一个空位可五连，player=1
- **WHEN** 调用 search(state)
- **THEN** 可五连的空位 visit_counts 远大于其他位置

### TC-MS-03 终局状态搜索
- **GIVEN** state 已终局（有人五连）
- **WHEN** 调用 search(state)
- **THEN** root_value 为终局价值（胜方视角 1.0，负方视角 -1.0）
- **AND** root_policy 为均匀分布

### TC-MS-04 advance_root 正常路径
- **GIVEN** search 后根节点存在 action=0 的子节点
- **WHEN** 调用 advance_root(0)
- **THEN** 返回 true
- **WHEN** 对同一 action 再次调用 advance_root
- **THEN** 返回 false（不再是根的直接子节点）

### TC-MS-05 advance_root 不存在路径
- **GIVEN** search 后 action=999（越界）或无此子节点
- **WHEN** 调用 advance_root(999)
- **THEN** 返回 false

### TC-MS-06 子树复用时访问计数增量
- **GIVEN** 两局共用同一 searcher：第一局 search → advance_root → search，第二局继续 search
- **WHEN** 第二局 search
- **THEN** visit_counts 仅反映第二局新增访问数（不含第一局残留）

### TC-MS-07 多线程正确性
- **GIVEN** num_threads=8，num_simulations=800
- **WHEN** 调用 search(state)
- **THEN** visit_counts 总和 = 800（所有 simulation 完成）
- **AND** 无崩溃、无死锁、无数据竞争

### TC-MS-08 狄利克雷噪声影响
- **GIVEN** dirichlet_epsilon=0.25, dirichlet_alpha=0.03
- **WHEN** 调用 search 两次（不同随机种子）
- **THEN** 两次 root_policy 噪声混合后的值存在差异

### TC-MS-09 Virtual Loss=0 退化为串行
- **GIVEN** virtual_loss=0, num_threads=1
- **WHEN** 调用 search 与 virtual_loss=0, num_threads=4
- **THEN** 两次 visit_counts 总和相同，分布可能存在合理差异（随机性）

### TC-MS-10 GPU/CPU 结果一致性
- **GIVEN** 两个 MctsSearcher：use_gpu=true 和 use_gpu=false，其余参数相同
- **WHEN** 对同一 state 各执行 50 次 search，统计平均 visit_counts
- **THEN** 两次平均分布的 Top-3 action 集合相同

---

## 5. pybind11 桥接层测试

### TC-PB-01 模块导入
- **GIVEN** 已编译的 cpp_mcts.pyd 在 Python 路径中
- **WHEN** 执行 `import cpp_mcts`
- **THEN** 导入成功，`cpp_mcts.MctsSearcher` 可访问

### TC-PB-02 构造器参数默认值
- **GIVEN** 有效的 TorchScript 模型路径
- **WHEN** 调用 `cpp_mcts.MctsSearcher(model_path)`
- **THEN** 构造成功，使用全部默认参数

### TC-PB-03 search 返回类型
- **GIVEN** 已创建的 MctsSearcher 实例
- **WHEN** result = searcher.search(state)
- **THEN** result 为 tuple：(visit_counts, root_value, root_policy)
- **AND** isinstance(visit_counts, list) 且 len=225
- **AND** isinstance(root_value, float)
- **AND** isinstance(root_policy, list) 且 len=225

### TC-PB-04 advance_root 返回类型
- **GIVEN** search 后
- **WHEN** result = searcher.advance_root(0)
- **THEN** isinstance(result, bool)

### TC-PB-05 非法 state 输入
- **GIVEN** state 长度不为 226
- **WHEN** 调用 search(state)
- **THEN** 抛出 RuntimeError（pybind11 内建类型检查）

---

## 6. Python 适配层测试

### TC-AD-01 CppMctsAdapter 接口兼容
- **GIVEN** CppMctsAdapter 实例
- **WHEN** 调用 search(state)
- **THEN** 返回格式与 Python MCTSSearcher.search() 完全一致

### TC-AD-02 log_key 参数忽略无异常
- **GIVEN** CppMctsAdapter 实例
- **WHEN** 调用 search(state, log_key="test_key")
- **THEN** 正常返回，无异常（log_key 被忽略）

### TC-AD-03 与 PolicyValueAiEngine 集成
- **GIVEN** CppMctsAdapter 注入 PolicyValueAiEngine
- **WHEN** 调用 engine.get_move(board, current_player, "argmax")
- **THEN** 返回有效 MoveDecision，position 在合法范围内
- **WHEN** 调用 engine.after_move(x, y)
- **THEN** 无异常

---

## 7. 集成测试

### TC-IT-01 Python MCTS vs C++ MCTS 决策一致性
- **GIVEN** 同一棋盘状态，Python MCTSSearcher 与 C++ MctsSearcher 使用相同权重模型，相同参数
- **WHEN** 两者分别执行 search
- **THEN** visit_counts 的 Top-3 action 集合相同
- **AND** root_value 符号一致（正负相同）
- **AND** root_policy 逐元素差异 < 1e-4

### TC-IT-02 完整对局（C++ MCTS 自对弈）
- **GIVEN** 两个 PolicyValueAiEngine（均使用 CppMctsAdapter），num_simulations=50
- **WHEN** 从头开始对弈至终局
- **THEN** 对局正常结束（不崩溃、不超时、不死锁）
- **AND** 终局检测正确（胜/负/平）

### TC-IT-03 完整对局（C++ MCTS vs Random）
- **GIVEN** PolicyValueAiEngine(C++ MCTS) vs Random AI
- **WHEN** 对弈 20 局
- **THEN** C++ MCTS 胜率 > 90%（Random AI 作为 baseline）
- **AND** 无内存泄漏（连续多局后内存稳定）

### TC-IT-04 子树复用贯穿对局
- **GIVEN** PolicyValueAiEngine(C++ MCTS)，一局完整对弈
- **WHEN** 每步调用 advance_root 后 search
- **THEN** advance_root 每次都返回 true（前 20 步内）
- **AND** 最终搜索结果正确

### TC-IT-05 并行多局对弈
- **GIVEN** Python ThreadPoolExecutor(4 workers)，每 worker 独立创建 CppMctsAdapter 实例
- **WHEN** 4 个 worker 同时执行对弈
- **THEN** 4 局均正常完成，无交叉影响
