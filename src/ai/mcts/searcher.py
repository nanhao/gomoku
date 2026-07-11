"""MCTS 搜索器核心模块

实现 AlphaGo Zero 风格的蒙特卡洛树搜索：
- PUCT 选择
- Batch MCTS 循环（collect B leaves → batch eval → revert VL → backup）
- 回传取反（每层切换玩家视角）
- Virtual Loss 临时惩罚（批次内路径多样化）
- 狄利克雷噪声
- 子树复用（advance_root）

日志功能委托给 search_logger.MCTSSearchLogger。
"""

import math
import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np

from game.game import check_winner, is_board_full
from ai.mcts.search_logger import MCTSSearchLogger
from config.mcts import MCTS_LOG_ENABLED, MCTS_LOG_DIR, MCTS_BATCH_SIZE, MCTS_VIRTUAL_LOSS

# 哨兵值：区分"未传参"和"显式传 None"
_LOG_UNSET = object()


class MCTSSearcher:
    """MCTS 搜索器

    对给定棋盘状态执行指定次数的模拟（simulation），
    每次模拟经历 selection → evaluation → backup 三阶段。

    支持 Batch MCTS + Virtual Loss：每批收集 B 个叶节点合并推理，
    选择阶段施加 VL 临时惩罚以实现批次内路径多样化。
    batch_size=1, virtual_loss=0 退化为原逐 simulation 模式。
    """

    def __init__(
        self,
        model,
        num_simulations: int = 200,
        c_puct: float = 2.0,
        dirichlet_epsilon: float = 0.25,
        dirichlet_alpha: float = 0.03,
        batch_size: Optional[int] = None,
        virtual_loss: Optional[float] = None,
        log_dir: Optional[str] = _LOG_UNSET,
    ):
        """初始化 MCTS 搜索器

        Args:
            model: CNN v7 模型实例，需支持 predict_batch(board_vectors) -> List[ModelPredict]
            num_simulations: 每次搜索的总模拟次数，默认 200
            c_puct: PUCT 探索系数，默认 2.0
            dirichlet_epsilon: 狄利克雷噪声比例，默认 0.25
            dirichlet_alpha: 狄利克雷分布参数，默认 0.03
            batch_size: 每批收集 simulation 数。None 时读取 MCTS_BATCH_SIZE 配置（默认 16）
            virtual_loss: 虚拟损失值。None 时读取 MCTS_VIRTUAL_LOSS 配置（默认 3.0）
            log_dir: 搜索日志输出目录。未传参时读取 config/mcts.py 配置；
                    显式传 None 禁用日志
        """
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_epsilon = dirichlet_epsilon
        self.dirichlet_alpha = dirichlet_alpha
        self.batch_size = batch_size if batch_size is not None else MCTS_BATCH_SIZE
        self.virtual_loss = virtual_loss if virtual_loss is not None else MCTS_VIRTUAL_LOSS
        self.log_dir: Optional[str] = None
        if log_dir is _LOG_UNSET:
            # 未传参：读取全局配置
            self.log_dir = MCTS_LOG_DIR if MCTS_LOG_ENABLED else None
        else:
            self.log_dir = log_dir

        # 搜索树：state_hash -> Node
        self.tree: Dict[str, dict] = {}
        self._root_hash: Optional[str] = None

        # 根边 N 快照（子树复用时记录初始值，用于 get_result 中计算增量）
        self._root_edge_N_snapshot: Dict[int, int] = {}

        # 日志记录器
        self._logger = MCTSSearchLogger(self.log_dir, c_puct) if self.log_dir else None

    # ---------- 公共 API ----------

    def search(
        self,
        state: List[int],
        num_simulations: Optional[int] = None,
        log_key: Optional[str] = None,
    ) -> Tuple[List[int], float, List[float]]:
        """执行 MCTS 搜索

        Args:
            state: 226 维 board_vector（前 225=棋盘按行展开，末位=当前落子方）
            num_simulations: 覆盖默认模拟次数，None 时使用构造参数
            log_key: 日志标识，非 None 且 log_dir 已配置时输出日志 JSON

        Returns:
            (visit_counts, root_value, root_policy) 三元组：
            - visit_counts: 225 维各动作原始访问计数
            - root_value: 根节点平均价值（从当前玩家视角）
            - root_policy: 根节点先验策略分布（施加噪声前的模型原始 softmax，225 维）
        """
        sims = num_simulations if num_simulations is not None else self.num_simulations
        self._init_search(state)

        # 初始化日志
        log_data = None
        if self._logger is not None and log_key is not None:
            log_data = self._logger.init_log_data(
                board=state[:225],
                current_player=state[225],
                simulations=sims,
                dirichlet_epsilon=self.dirichlet_epsilon,
                dirichlet_alpha=self.dirichlet_alpha,
                root_noise=self._get_root_noise(),
            )

        # 主搜索循环：batch MCTS（collect → batch eval → revert VL → backup）
        simulation_logs = []
        sim_id = 0
        remaining = sims

        while remaining > 0:
            B = min(self.batch_size, remaining)
            batch_leaves: List[dict] = []   # 非终局叶节点（需模型评估）
            terminal_leaves: List[dict] = []  # 终局叶节点（规则判定）

            # —— 阶段 1：收集 B 个叶节点（带 VL）——
            for _ in range(B):
                leaf = self._select_one()
                if leaf.get("is_terminal", False):
                    self._revert_vl(leaf.get("path", []))
                    terminal_leaves.append(leaf)
                else:
                    batch_leaves.append(leaf)

            # —— 阶段 2：batch 模型评估（仅非终局节点）——
            if batch_leaves:
                bv_list = [leaf["board"] + [leaf["player"]] for leaf in batch_leaves]
                results = self.model.predict_batch(bv_list)
            else:
                results = []

            # —— 阶段 3：撤销 VL + 回传（先终局、后非终局）——
            for leaf in terminal_leaves:
                leaf["value"] = leaf.get("terminal_value", 0.0)
                self._create_terminal_node(leaf)
                self._backup(leaf)
                if log_data is not None:
                    sim_log = self._logger.build_simulation_log(
                        sim_id, leaf.get("path", []), leaf, self.tree
                    )
                    simulation_logs.append(sim_log)
                sim_id += 1

            for leaf, result in zip(batch_leaves, results):
                self._revert_vl(leaf.get("path", []))
                leaf["value"] = result.value
                leaf["policy"] = result.policy
                self._expand_leaf(leaf, result.policy)
                self._backup(leaf)
                if log_data is not None:
                    sim_log = self._logger.build_simulation_log(
                        sim_id, leaf.get("path", []), leaf, self.tree
                    )
                    simulation_logs.append(sim_log)
                sim_id += 1

            remaining -= B

        # 组装结果
        visit_counts, root_value, root_policy = self._get_result()

        # 写入日志
        if log_data is not None and log_key is not None:
            log_data["simulation_logs"] = simulation_logs
            log_data["result"] = {
                "visit_counts": visit_counts,
                "root_value": root_value,
                "pi": self._compute_pi(visit_counts, temperature=1.0),
            }
            self._logger.write_log(log_key, log_data)

        return visit_counts, root_value, root_policy

    def advance_root(self, action: int) -> bool:
        """将 action 对应子节点提升为新根，丢弃兄弟子树。

        Args:
            action: 要推进的动作索引 0-224

        Returns:
            True 如果子节点存在并成功提升
        """
        if self._root_hash is None or self._root_hash not in self.tree:
            return False

        root = self.tree[self._root_hash]
        if action not in root["edges"]:
            return False

        edge = root["edges"][action]
        child_hash = edge.get("child_hash")
        if child_hash is None or child_hash not in self.tree:
            return False

        # 收集子树中所有节点哈希（递归）
        subtree_hashes = self._collect_subtree(child_hash)

        # 清除不在子树中的节点
        old_hashes = list(self.tree.keys())
        for h in old_hashes:
            if h not in subtree_hashes:
                del self.tree[h]

        self._root_hash = child_hash
        return True

    def _collect_subtree(self, root_hash: str) -> set:
        """收集以 root_hash 为根的子树中所有节点哈希（DFS）"""
        result = set()
        stack = [root_hash]
        while stack:
            h = stack.pop()
            if h in result or h not in self.tree:
                continue
            result.add(h)
            for edge in self.tree[h].get("edges", {}).values():
                ch = edge.get("child_hash")
                if ch is not None and ch not in result:
                    stack.append(ch)
        return result

    # ---------- 内部：初始化 ----------

    @staticmethod
    def _state_hash(board: List[int], player: int) -> str:
        """计算状态确定性哈希（MD5 前 8 字符）"""
        data = bytes(board) + bytes([player])
        return hashlib.md5(data).hexdigest()[:8]

    def _init_search(self, state: List[int]):
        """初始化搜索：创建根节点 + 狄利克雷噪声
        
        若根节点已存在于树中（子树复用场景），仍重新施加新鲜噪声。
        若根状态已终局（有人胜或满棋盘），跳过模型预测。
        """
        board = state[:225]
        player = state[225]
        root_hash = self._state_hash(board, player)

        # 记录根边 N 快照（子树复用时，边可能带有上轮搜索的 N）
        self._root_edge_N_snapshot = {}

        if root_hash not in self.tree:
            self.tree.clear()
            self._root_hash = root_hash

            # 检测根状态是否已终局
            root_winner = self._check_board_winner(board)
            root_full = is_board_full(board)
            if root_winner != 0 or root_full:
                # 终局状态：跳过模型预测，使用均匀分布策略
                policy = [1.0 / 225] * 225
                value = -1.0 if root_winner != 0 else 0.0
            else:
                predict_result = self.model.predict(state)
                policy = predict_result.policy
                value = predict_result.value

            # 创建根节点
            self.tree[root_hash] = {
                "board": list(board),
                "player": player,
                "N": 0,
                "W": 0.0,
                "Q": 0.0,
                "edges": {},
                "_root_policy": list(policy),
            }
            root = self.tree[root_hash]

            # 为所有空位创建边（终局时 board 可能有空位但无合法动作）
            for action in range(225):
                if board[action] == 0:
                    root["edges"][action] = {
                        "N": 0, "W": 0.0, "Q": 0.0,
                        "P": policy[action], "child_hash": None,
                    }
            # 全新树：所有边初始 N=0，快照均为 0
            self._root_edge_N_snapshot = {a: 0 for a in range(225) if board[a] == 0}
        else:
            self._root_hash = root_hash
            # 子树复用：记录当前 N 快照，get_result 中只报告增量
            root = self.tree[root_hash]
            for action, edge in root.get("edges", {}).items():
                self._root_edge_N_snapshot[action] = edge["N"]

        # 始终施加狄利克雷噪声（包括子树复用场景）
        self._add_dirichlet_noise(root_hash)

    def _add_dirichlet_noise(self, node_hash: str):
        """对根节点的合法动作施加狄利克雷噪声
        
        以节点存储的 _root_policy（模型原始 softmax 输出）为基底混合噪声。
        支持子树复用后重新施加噪声（_root_policy 在 _expand_leaf 和 _init_search 中均会存储）。
        """
        node = self.tree[node_hash]
        edges = node["edges"]
        if not edges:
            return
        actions = list(edges.keys())
        n_actions = len(actions)
        noise = np.random.dirichlet([self.dirichlet_alpha] * n_actions)
        # 以 _root_policy 为基底（模型原始输出），若非根节点回退到当前 P 值
        root_policy = node.get("_root_policy")
        for i, action in enumerate(actions):
            base_P = root_policy[action] if root_policy else edges[action]["P"]
            edges[action]["P"] = (
                (1 - self.dirichlet_epsilon) * base_P
                + self.dirichlet_epsilon * noise[i]
            )

    def _get_root_noise(self) -> List[float]:
        """获取根节点噪声施加后的 P' 值（日志用）"""
        if self._root_hash is None or self._root_hash not in self.tree:
            return []
        root = self.tree[self._root_hash]
        return [round(root["edges"].get(a, {}).get("P", 0.0), 6) for a in range(225)]

    # ---------- 内部：选择阶段（带 Virtual Loss）----------

    def _select_one(self) -> dict:
        """选择阶段（带 VL）：沿 PUCT 下降直到叶节点，沿途施加 virtual loss。

        对遍历的每条边施加 VL：N += VL, W -= VL, Q = W/N。
        促使批次内后续 selection 走不同路径。

        Returns:
            叶节点信息字典，含 path, board, player, is_terminal 等
        """
        path = []
        current_hash = self._root_hash
        vl = self.virtual_loss

        while True:
            node = self.tree[current_hash]
            board = node["board"]
            player = node["player"]

            # PUCT 选择最优子节点
            best_action = self._puct_select(current_hash)
            if best_action is None:
                # 无合法动作（终局或满棋盘）
                return self._make_terminal_leaf(current_hash, board, player, path)

            edge = node["edges"][best_action]

            # 施加 virtual loss
            if vl != 0:
                edge["N"] += vl
                edge["W"] -= vl
                edge["Q"] = edge["W"] / edge["N"] if edge["N"] > 0 else 0.0
                node["N"] += vl
                node["W"] -= vl
                node["Q"] = node["W"] / node["N"] if node["N"] > 0 else 0.0

            path.append({
                "state_hash": current_hash,
                "action": best_action,
                "player": player,
            })

            child_hash = edge.get("child_hash")
            if child_hash is None:
                # 子节点未展开 → 叶节点
                new_board = list(board)
                new_board[best_action] = player
                new_player = 2 if player == 1 else 1

                # 检查子棋盘是否为终局
                winner = self._check_board_winner(new_board)
                if winner != 0:
                    return {
                        "path": path,
                        "state_hash": self._state_hash(new_board, new_player),
                        "action": best_action,
                        "board": new_board,
                        "player": new_player,
                        "is_terminal": True,
                        "terminal_value": -1.0,
                    }
                if is_board_full(new_board):
                    return {
                        "path": path,
                        "state_hash": self._state_hash(new_board, new_player),
                        "action": best_action,
                        "board": new_board,
                        "player": new_player,
                        "is_terminal": True,
                        "terminal_value": 0.0,
                    }

                return {
                    "path": path,
                    "state_hash": self._state_hash(new_board, new_player),
                    "action": best_action,
                    "board": new_board,
                    "player": new_player,
                    "is_terminal": False,
                    "terminal_value": None,
                }

            current_hash = child_hash

    def _puct_select(self, node_hash: str) -> Optional[int]:
        """PUCT 选择：a = argmax(Q + c_puct * P * sqrt(sum_N) / (1 + N))

        Returns:
            最优动作索引，无合法动作时返回 None
        """
        node = self.tree[node_hash]
        edges = node["edges"]
        board = node["board"]
        sum_N = sum(e["N"] for e in edges.values())
        sqrt_sum_N = math.sqrt(max(sum_N, 1))

        best_action = None
        best_score = float("-inf")

        for action, edge in edges.items():
            if board[action] != 0:
                continue
            Q = edge["Q"]
            U = self.c_puct * edge["P"] * sqrt_sum_N / (1 + edge["N"])
            score = Q + U
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    @staticmethod
    def _make_terminal_leaf(node_hash, board, player, path):
        """构造终局叶节点信息"""
        winner = MCTSSearcher._check_board_winner(board)
        if winner != 0:
            return {
                "path": path, "state_hash": node_hash,
                "board": list(board), "player": player,
                "is_terminal": True, "terminal_value": -1.0,
            }
        if is_board_full(board):
            return {
                "path": path, "state_hash": node_hash,
                "board": list(board), "player": player,
                "is_terminal": True, "terminal_value": 0.0,
            }
        return {
            "path": path, "state_hash": node_hash,
            "board": list(board), "player": player,
            "is_terminal": False, "terminal_value": None,
        }

    # ---------- 内部：VL 撤销 ----------

    def _revert_vl(self, path: List[dict]):
        """撤销路径上所有边的 virtual loss。

        与 _select_one 中的施加对称：N -= VL, W += VL, Q = W/N。
        重复叶节点路径被施加两次则撤销两次，VL 天然对称。

        Args:
            path: 选择路径，每步含 state_hash, action
        """
        vl = self.virtual_loss
        if vl == 0:
            return

        for step in reversed(path):
            node_hash = step["state_hash"]
            action = step["action"]
            if node_hash in self.tree and action in self.tree[node_hash]["edges"]:
                edge = self.tree[node_hash]["edges"][action]
                edge["N"] -= vl
                edge["W"] += vl
                edge["Q"] = edge["W"] / edge["N"] if edge["N"] > 0 else 0.0
                node = self.tree[node_hash]
                node["N"] -= vl
                node["W"] += vl
                node["Q"] = node["W"] / node["N"] if node["N"] > 0 else 0.0

    # ---------- 内部：评估与扩展 ----------

    @staticmethod
    def _check_board_winner(board: List[int]) -> int:
        """检查棋盘上是否有胜者"""
        for a in range(225):
            if board[a] != 0:
                w = check_winner(board, a)
                if w != 0:
                    return w
        return 0

    def _expand_leaf(self, leaf: dict, policy: List[float]):
        """为叶节点创建搜索树节点和边"""
        child_hash = self._state_hash(leaf["board"], leaf["player"])
        if child_hash not in self.tree:
            self.tree[child_hash] = {
                "board": list(leaf["board"]),
                "player": leaf["player"],
                "N": 0, "W": 0.0, "Q": 0.0,
                "edges": {},
                "_root_policy": list(policy),
            }
            child_node = self.tree[child_hash]
            for action in range(225):
                if leaf["board"][action] == 0:
                    child_node["edges"][action] = {
                        "N": 0, "W": 0.0, "Q": 0.0,
                        "P": policy[action], "child_hash": None,
                    }

        leaf["child_hash"] = child_hash

        # 在父节点边中记录 child_hash
        if "action" in leaf and leaf.get("path"):
            parent_info = leaf["path"][-1]
            parent_hash = parent_info["state_hash"]
            parent_action = leaf["action"]
            if parent_hash in self.tree and parent_action in self.tree[parent_hash]["edges"]:
                self.tree[parent_hash]["edges"][parent_action]["child_hash"] = child_hash

    def _create_terminal_node(self, leaf: dict):
        """为终局叶节点创建搜索树节点（无子边）并链接父边 child_hash。

        终局位置是合法游戏状态，应存在于搜索树中，以便 advance_root
        能通过 child_hash 定位并推进根节点。终局节点的 edges 恒为空
        （无合法落子位置），与 _expand_leaf 创建的节点结构一致。

        若节点已存在于树中（如 _make_terminal_leaf 传入的当前节点），
        则跳过创建，仅确保父边 child_hash 已链接（幂等）。

        Args:
            leaf: 叶节点字典，含 board/player/action/path 字段
        """
        child_hash = self._state_hash(leaf["board"], leaf["player"])
        if child_hash not in self.tree:
            # 创建终局节点：与普通节点结构一致，仅 edges 为空
            self.tree[child_hash] = {
                "board": list(leaf["board"]),
                "player": leaf["player"],
                "N": 0, "W": 0.0, "Q": 0.0,
                "edges": {},
            }

        leaf["child_hash"] = child_hash

        # 在父节点边中记录 child_hash
        if "action" in leaf and leaf.get("path"):
            parent_info = leaf["path"][-1]
            parent_hash = parent_info["state_hash"]
            parent_action = leaf["action"]
            if parent_hash in self.tree and parent_action in self.tree[parent_hash]["edges"]:
                self.tree[parent_hash]["edges"][parent_action]["child_hash"] = child_hash

    # ---------- 内部：回传 ----------

    def _backup(self, leaf_info: dict):
        """回传阶段：沿路径向上更新 N/W/Q，每层取反 value。

        对路径上的每条边：
        1. v = -v（翻转为当前节点玩家视角）
        2. N += 1, W += v, Q = W / N
        3. 同步更新节点统计量
        """
        v = leaf_info.get("value", 0.0)

        for step in reversed(leaf_info.get("path", [])):
            node_hash = step["state_hash"]
            action = step["action"]

            # 翻转为当前节点玩家视角（叶节点 value 是对方视角）
            v = -v

            # 更新边统计量
            if node_hash in self.tree and action in self.tree[node_hash]["edges"]:
                edge = self.tree[node_hash]["edges"][action]
                edge["N"] += 1
                edge["W"] += v
                edge["Q"] = edge["W"] / edge["N"] if edge["N"] > 0 else 0.0

                # 更新节点统计量
                node = self.tree[node_hash]
                node["N"] += 1
                node["W"] += v
                node["Q"] = node["W"] / node["N"] if node["N"] > 0 else 0.0

    # ---------- 内部：结果组装 ----------

    def _get_result(self) -> Tuple[List[int], float, List[float]]:
        """从搜索树中提取搜索结果

        使用 N 增量计算 visit_counts，避免子树复用带来的跨步 N 残留。
        """
        root = self.tree.get(self._root_hash, {})
        edges = root.get("edges", {})

        visit_counts = [0] * 225
        for action, edge in edges.items():
            # 只报告本搜索新增的访问计数
            visit_counts[action] = edge["N"] - self._root_edge_N_snapshot.get(action, 0)

        root_value = root.get("Q", 0.0) if root.get("N", 0) > 0 else 0.0
        root_policy = root.get("_root_policy", [1.0 / 225] * 225)

        return visit_counts, root_value, root_policy

    @staticmethod
    def _compute_pi(visit_counts: List[int], temperature: float = 1.0) -> List[float]:
        """从访问计数计算 π 分布：π(a) = N(a)^(1/τ) / Σ N(b)^(1/τ)"""
        if temperature <= 0:
            max_count = max(visit_counts)
            max_indices = [i for i, c in enumerate(visit_counts) if c == max_count]
            pi = [0.0] * 225
            for i in max_indices:
                pi[i] = 1.0 / len(max_indices)
            return pi

        total = sum(c ** (1.0 / temperature) for c in visit_counts)
        if total > 0:
            return [(c ** (1.0 / temperature)) / total for c in visit_counts]
        return [1.0 / 225] * 225
