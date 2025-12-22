import json
from collections import deque
from typing import Any, Dict, List, Set

from tqdm import tqdm

import config
from cdm.ncd.predict import get_stu_emb
# change pre-trained CDM model
# from cdm.rcd.predict import get_stu_emb
from utils.graph import calc_metric, load_junyi_adj_table, load_mooc_adj_table

pos_node_threshold = 0.52


def _bfs_k_hop(start: int, k: int) -> Set[int]:
    """start node, return all nodes from k hops"""
    vis = {start}
    queue = deque([(start, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth < k:
            for neighbor in g[node]:
                if neighbor not in vis:
                    vis.add(neighbor)
                    queue.append((neighbor, depth + 1))
    return vis


def psi_with_rule(cnt: int = 10):
    pcs, nbs, up, iou = 0, 0, 0, 0
    cnt = min(cnt, len(test_data))
    for td in tqdm(test_data[:cnt], desc="psi_with_rule"):
        user_id: str = td["user_id"]
        init_nodes: int = td["init_nodes"]
        pos: Set[int] = set(td["gt_subgraph"]["pos_inner_nodes"])
        neg: Set[int] = set(td["gt_subgraph"]["neg_front_nodes"])
        
        w = get_stu_emb(user_id)
        k_hops = _bfs_k_hop(init_nodes, 2)
        cand = set()
        for u in k_hops:
            score = w[u]
            if score >= pos_node_threshold:
                cand.add(u)
        if len(cand) == 0:
            continue
        a, b, c, d = calc_metric(cand, pos, neg, g)
        pcs += a
        nbs += b
        up += c
        iou += d
    
    print(cnt)
    print(f"{pcs / cnt:.4f}, {nbs / cnt:.4f}, {up / cnt:.4f}, {iou / cnt:.4f}")


def weighted_psi(cnt=10):
    pcs, nbs, up, iou = 0, 0, 0, 0
    cnt = min(cnt, len(test_data))
    for td in tqdm(test_data[:cnt], desc="weighted_psi"):
        user_id: str = td["user_id"]
        init_nodes: int = td["init_nodes"]
        pos: Set[int] = set(td["gt_subgraph"]["pos_inner_nodes"])
        neg: Set[int] = set(td["gt_subgraph"]["neg_front_nodes"])
        
        w = get_stu_emb(user_id)
        k_hops = _bfs_k_hop(init_nodes, 2)
        cand = set()
        for u in k_hops:
            score = w[u]
            neighbor_score = sum(w[v] for v in g[u]) / len(g[u])
            score = 0.5 * score + 0.5 * neighbor_score
            if score >= pos_node_threshold:
                cand.add(u)
        
        if len(cand) == 0:
            continue
        a, b, c, d = calc_metric(cand, pos, neg, g)
        pcs += a
        nbs += b
        up += c
        iou += d
    
    print(cnt)
    print(f"{pcs / cnt:.4f}, {nbs / cnt:.4f}, {up / cnt:.4f}, {iou / cnt:.4f}")


def hds_only_search(max_iter: int = 30, p_min: float = 0.1, delta_thr: float = -0.0013, cnt=10):
    pcs, nbs, up, iou = 0, 0, 0, 0
    cnt = min(cnt, len(test_data))
    for td in tqdm(test_data[:cnt], desc="hds_only_search"):
        user_id: str = td["user_id"]
        init_nodes: int = td["init_nodes"]
        pos: Set[int] = set(td["gt_subgraph"]["pos_inner_nodes"])
        neg: Set[int] = set(td["gt_subgraph"]["neg_front_nodes"])
        w = get_stu_emb(user_id)
        
        cand: Set[int] = {init_nodes}
        p_G = w[init_nodes]
        
        for _ in range(max_iter):
            operated = False
            cand_in = set()
            for u in cand:
                for v in g[u]:
                    if v not in cand:
                        cand_in.add(v)
            best_gain = -float('inf')
            best_node = None
            for v in cand_in:
                if w[v] < p_min:
                    continue
                delta = w[v] - p_G
                if delta > best_gain:
                    best_gain = delta
                    best_node = v
            if best_gain > delta_thr:
                cand.add(best_node)
                p_G = 0.1 * p_G + 0.9 * w[best_node]
                operated = True
            
            if len(cand) > 1:
                worst_gain = float('inf')
                worst_node = None
                for v in cand:
                    temp_cand = cand - {v}
                    temp_p_G = sum([w[u] for u in temp_cand]) / len(temp_cand)
                    delta = temp_p_G - p_G
                    if delta < worst_gain:
                        worst_gain = delta
                        worst_node = v
                if worst_gain < delta_thr:
                    cand.remove(worst_node)
                    p_G = sum([w[u] for u in cand]) / len(cand)
                    operated = True
            
            if not operated:
                break
        
        a, b, c, d = calc_metric(cand, pos, neg, g)
        pcs += a
        nbs += b
        up += c
        iou += d
    
    print(cnt)
    print(f"{pcs / cnt:.4f}, {nbs / cnt:.4f}, {up / cnt:.4f}, {iou / cnt:.4f}")


if __name__ == '__main__':
    if config.dataset == "mooc":
        g = load_mooc_adj_table()
        with open("data/mooc_test_data.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)
        with open("data/stu_records_filtered.json", "r", encoding="utf-8") as f:
            records: Dict[str, List[Dict]] = json.load(f)
        with open("data/stu_problems.json", 'r') as f:
            problem_info: Dict[str, Any] = json.load(f)
        with open("data/id2entity.json", 'r') as f:
            id2entity: Dict[str, int] = json.load(f)
    else:
        g = load_junyi_adj_table()
        with open("data/junyi_test_data.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)
        with open("data/id2entity.json", 'r') as f:
            id2entity: Dict[str, int] = json.load(f)
    
    n = len(g)
    
    cnt = 1000
    psi_with_rule(cnt=cnt)
    weighted_psi(cnt=cnt)
    hds_only_search(cnt=cnt)
