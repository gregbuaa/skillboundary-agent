from collections import deque

import numpy as np
from tqdm import tqdm

import config
from utils.graph import *

data_path = "outputs/metrics_data.npy"


def get_stu_node_state(n: int, u_records: List[Dict[str, Any]],
                       threshold: float = 0.8) -> Tuple[List[Node], Set[int], Set[int]]:
    """
    Calculate the current mastery status of each node of the student on the knowledge graph
    :param n: node number
    :param u_records: user record
    :param threshold: the threshold to judge correct or wrong
    :return: set of mastered and not mastered
    """
    
    w = [Node() for _ in range(n)]
    for r in u_records:
        p = problem_info[str(r["problem_id"])]
        node_ids = ([entity2id[p["course"]], entity2id[p["section"]]] + [entity2id[c] for c in p["concepts"]])
        if r["is_correct"]:
            for node_id in node_ids:
                w[node_id].a += 1
        else:
            for node_id in node_ids:
                w[node_id].b += 1
    
    pos, neg = set(), set()
    for i, x in enumerate(w):
        if x.a + x.b > 0:
            x.label = 1 if x.a / (x.a + x.b) > threshold else -1
        if x.label == 1:
            pos.add(i)
        elif x.label == -1:
            neg.add(i)
    
    return w, pos, neg


def get_cand_vex(cand: set, pos: set, neg: set, iter: int) -> List[int]:
    neg_v: List[int] = []
    unk_v: List[int] = []
    now = cand.copy()
    queue = deque(now)
    
    while queue:
        u = queue.popleft()
        for v in g[u]:
            if len(neg_v) + len(unk_v) == iter:
                break
            if v not in pos and v not in now:
                now.add(v)
                queue.append(v)
                if v in neg:
                    neg_v.append(v)
                else:
                    unk_v.append(v)
    
    return unk_v[:len(unk_v) // 2] + neg_v + unk_v[len(unk_v) // 2:]


def main(g: List[Set[int]], iter: int = 100) -> None:
    n = len(g)
    students_metric: List[List[Tuple]] = []
    for _, u_records in tqdm(records.items(), desc="Processing"):
        
        w, pos, neg = get_stu_node_state(n, u_records)
        if len(pos) == 0:
            continue
        
        student_metric: List[Tuple] = []
        cand = pos.copy()
        cand_v = get_cand_vex(cand, pos, neg, iter)
        for i in range(iter):
            student_metric.append(calc_metric(cand, pos, neg, g)[-1])
            cand.add(cand_v[i])
        
        students_metric.append(student_metric)
    
    np_metric = np.array(students_metric)
    np.save(data_path, np_metric)


if __name__ == '__main__':
    with open("data/stu_records_filtered.json", "r", encoding="utf-8") as f:
        records: Dict[str, List[Dict]] = json.load(f)
    with open("data/stu_problems.json", 'r') as f:
        problem_info: Dict[str, Any] = json.load(f)
    with open("data/entity2id.json", 'r') as f:
        entity2id: Dict[str, int] = json.load(f)
    
    if config.dataset == "mooc":
        g = load_mooc_adj_table()
    else:
        g = load_junyi_adj_table()
    
    main(g)
