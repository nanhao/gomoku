"""MCTS 搜索日志记录器

负责将搜索过程序列化为结构化 JSON 日志文件。
日志旨在支持可视化渲染和正确性验证——拿到日志即可逐 simulation 重现并校验 MCTS 全流程。
"""

import json
import os
import math
from typing import Dict, List, Optional


class MCTSSearchLogger:
    """MCTS 搜索日志记录器

    在搜索过程中收集数据，搜索结束后写入 JSON 文件。
    """

    def __init__(self, log_dir: str, c_puct: float):
        """初始化日志记录器

        Args:
            log_dir: 日志输出目录
            c_puct: PUCT 探索系数（用于日志中的 hyperparams）
        """
        self.log_dir = log_dir
        self.c_puct = c_puct

    def init_log_data(
        self,
        board: List[int],
        current_player: int,
        simulations: int,
        dirichlet_epsilon: float,
        dirichlet_alpha: float,
        root_noise: List[float],
        batch_size: int = 16,
        virtual_loss: float = 3.0,
    ) -> dict:
        """初始化日志顶层结构

        Returns:
            日志数据字典
        """
        return {
            "board": list(board),
            "current_player": current_player,
            "simulations": simulations,
            "hyperparams": {
                "c_puct": self.c_puct,
                "dirichlet_epsilon": dirichlet_epsilon,
                "dirichlet_alpha": dirichlet_alpha,
                "batch_size": batch_size,
                "virtual_loss": virtual_loss,
            },
            "root_noise": root_noise,
            "simulation_logs": [],
            "result": {},
        }

    def build_simulation_log(
        self,
        sim_id: int,
        path_info: List[dict],
        leaf_info: dict,
        tree: dict,
    ) -> dict:
        """构建单条 simulation 日志

        Args:
            sim_id: simulation 序号
            path_info: 选择路径，每步含 state_hash, action, player
            leaf_info: 叶节点信息，含 state_hash, is_terminal, value, policy 等
            tree: 搜索树字典

        Returns:
            单条 simulation 日志字典
        """
        log = {"sim_id": sim_id, "selection": [], "leaf": {}, "backup": []}

        # Selection 路径
        for step in path_info:
            node_hash = step["state_hash"]
            if node_hash in tree:
                node = tree[node_hash]
                action = step["action"]
                edge = node["edges"].get(action, {})
                sum_N = sum(e["N"] for e in node["edges"].values())
                N = edge.get("N", 0)
                P = edge.get("P", 0)
                Q = edge.get("Q", 0)
                sqrt_sum_N = math.sqrt(max(sum_N, 1))
                U = self.c_puct * P * sqrt_sum_N / (1 + N)
                log["selection"].append({
                    "state_hash": node_hash,
                    "current_player": step.get("player", node.get("player", 0)),
                    "action": action,
                    "P": round(P, 6),
                    "Q": round(Q, 6),
                    "N": N,
                    "sum_N_parent": sum_N,
                    "U": round(U, 6),
                    "puct_score": round(Q + U, 6),
                })

        # Leaf
        leaf_log = {
            "state_hash": leaf_info.get("state_hash", ""),
            "is_terminal": leaf_info.get("is_terminal", False),
            "value": leaf_info.get("value", 0.0),
        }
        if "policy" in leaf_info:
            leaf_log["policy"] = [round(p, 6) for p in leaf_info["policy"]]
        log["leaf"] = leaf_log

        # Backup（逆序），每层取反 value（与 _backup 中的顺序一致：先取反再加）
        v = leaf_info.get("value", 0.0)
        for step in reversed(path_info):
            v = -v
            node_hash = step["state_hash"]
            action = step["action"]
            if node_hash in tree and action in tree[node_hash]["edges"]:
                edge = tree[node_hash]["edges"][action]
                log["backup"].append({
                    "state_hash": node_hash,
                    "current_player": step.get("player", 0),
                    "action": action,
                    "N_before": max(0, edge["N"] - 1),
                    "N_after": edge["N"],
                    "W_before": round(edge["W"] - v, 4),
                    "W_after": round(edge["W"], 4),
                    "Q_before": round((edge["W"] - v) / max(1, edge["N"] - 1), 4),
                    "Q_after": round(edge["Q"], 4),
                })

        return log

    def write_log(self, log_key: str, log_data: dict):
        """写入搜索日志 JSON 文件

        Args:
            log_key: 日志标识（文件名不含扩展名）
            log_data: 日志数据字典
        """
        if self.log_dir is None:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, f"{log_key}.json")
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
