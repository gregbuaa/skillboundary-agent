import json
import math
from typing import Dict, List


def load_json(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf8') as f:
        return json.load(f)


def dcg(recommended: List[int], relevant: set, k: int) -> float:
    dcg_val = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg_val += 1.0 / math.log2(i + 2)
    return dcg_val


def ndcg_at_k(recommended: List[int], relevant: set, k: int) -> float:
    ideal_dcg = dcg(list(relevant), relevant, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg(recommended, relevant, k) / ideal_dcg


def hit_at_k(final_subgraph: List[int], neg_front_nodes: List[int], k: int) -> int:
    recommended_k = set(final_subgraph[:k])
    return int(len(recommended_k.intersection(set(neg_front_nodes))) > 0)


def time_to_mastery(trace: List[Dict], boundary_nodes: set) -> int:
    for step in trace:
        if "node" not in step:
            continue
        if step["node"] in boundary_nodes and step["action"] == "expand":
            return step["step"]
    return len(trace) + 1


def evaluate_metrics(gt_path: str, pred_path: str, k: int = 5) -> None:
    gt_data = {d["user_id"]: d for d in load_json(gt_path)}
    pred_data = {d["user_id"]: d for d in load_json(pred_path)}
    
    hit_list = []
    ndcg_list = []
    gain_list = []
    ttm_list = []
    
    for user_id in gt_data:
        if user_id not in pred_data:
            continue
        
        gt = gt_data[user_id]
        pred = pred_data[user_id]
        
        final_subgraph = pred["final_subgraph"]
        trace = pred["trace"]
        metric = pred.get("metric", {})
        iou = metric.get("iou", 0.0)
        
        pos_nodes = set(gt["gt_subgraph"]["pos_inner_nodes"])
        neg_nodes = set(gt["gt_subgraph"]["neg_front_nodes"])
        
        hit_list.append(hit_at_k(final_subgraph, neg_nodes, k))
        ndcg_list.append(ndcg_at_k(final_subgraph, pos_nodes, k))
        ttm_list.append(time_to_mastery(trace, neg_nodes))
    
    total = len(hit_list)
    
    print(f"Hit@{k}: {sum(hit_list) / total:.4f}")
    print(f"nDCG@{k}: {sum(ndcg_list) / total:.4f}")
    print(f"Time-to-Mastery (TTM): {sum(ttm_list) / total:.2f} steps")


if __name__ == "__main__":
    gt_file = "data/junyi_test_data.json"
    pred_file = "outputs/gpt-4o-junyi.json"
    evaluate_metrics(gt_file, pred_file)
