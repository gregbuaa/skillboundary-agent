import json
from collections import deque
from typing import List, Set

from utils.graph import load_mooc_adj_table

with open("data/mooc_test_data.json", "r", encoding="utf-8") as f:
    gt_data = json.load(f)

with open("outputs/gpt-4o-mooc.json", "r", encoding="utf-8") as f:
    llm_data = json.load(f)

adj_table: List[Set[int]] = load_mooc_adj_table()


def build_predecessors(adj: List[Set[int]]) -> List[Set[int]]:
    n = len(adj)
    preds = [set() for _ in range(n)]
    for u in range(n):
        for v in adj[u]:
            preds[v].add(u)
    return preds


predecessors = build_predecessors(adj_table)


def path_validity(path: List[int], final_subgraph: Set[int], preds: List[Set[int]]) -> bool:
    for node in path:
        if not preds[node].issubset(final_subgraph):
            return False
    return True


def shortest_path_length(init_node: int, targets: Set[int], adj: List[Set[int]]) -> int:
    dist = [-1] * len(adj)
    dist[init_node] = 0
    queue = deque([init_node])
    
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    total_dist = 0
    for t in targets:
        if dist[t] == -1:
            total_dist += 10000
        else:
            total_dist += dist[t]
    return total_dist


def compute_po(init_node: int, targets: Set[int], path: List[int], adj: List[Set[int]]) -> float:
    shortest_len = shortest_path_length(init_node, targets, adj)
    path_len = len(path)
    if path_len == 0:
        return 0.0
    return shortest_len / path_len if path_len > 0 else 0.0


def detour_cost(path: List[int], negative_nodes: Set[int]) -> float:
    if not path:
        return 0.0
    cost_sum = sum(1 if node in negative_nodes else 0 for node in path)
    return cost_sum / len(path)


def success_rate(final_subgraph: Set[int], pos_inner_nodes: Set[int]) -> bool:
    return final_subgraph.issubset(pos_inner_nodes)


violations = 0
po_list = []
dc_list = []
success_num = 0
total_num = 0

for sample_gt in gt_data:
    user_id = sample_gt["user_id"]
    if user_id not in [d["user_id"] for d in llm_data]:
        continue
    sample_pred = next(d for d in llm_data if d["user_id"] == user_id)
    
    total_num += 1
    init_node = sample_gt["init_nodes"]
    pos_inner_nodes = set(sample_gt["gt_subgraph"]["pos_inner_nodes"])
    negative_nodes = set(sample_gt["gt_subgraph"]["neg_front_nodes"])
    
    trace = sample_pred.get("trace", [])
    path = []
    for step in trace:
        if "node" in step:
            path.append(step["node"])
        else:
            path.append(0)
    final_subgraph = set(sample_pred.get("final_subgraph", []))
    
    if not path_validity(path, final_subgraph, predecessors):
        violations += 1
    po = compute_po(init_node, pos_inner_nodes, path, adj_table)
    po_list.append(po)
    dc = detour_cost(path, negative_nodes)
    dc_list.append(dc)
    if success_rate(final_subgraph, pos_inner_nodes):
        success_num += 1

PV = 1 - violations / total_num if total_num > 0 else 0
PO = sum(po_list) / len(po_list) if po_list else 0
DC = sum(dc_list) / len(dc_list) if dc_list else 0
SR = success_num / total_num if total_num > 0 else 0

print(f"Path Validity (PV): {PV:.4f}")
print(f"Path Optimality (PO): {PO:.4f}")
print(f"Detour Cost (DC): {DC:.4f}")
print(f"Success Rate (SR): {SR:.4f}")
