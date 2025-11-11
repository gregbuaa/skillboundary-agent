import json
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple


class Node:
    def __init__(self):
        self.a = 0
        self.b = 0
        self.label = 0
        self.problem_id = -1


def calc_metric(cand: set, pos: set, neg: set, g: List[Set[int]]) -> Tuple:
    p = len(cand & pos) / len(cand)
    r = len(cand & pos) / len(pos)
    if p + r:
        pcs = 2 * p * r / (p + r)
    else:
        pcs = 0
    
    e_cnt = 0
    for u in cand:
        if u not in pos:
            continue
        for v in g[u]:
            if v not in cand and v in neg:
                e_cnt += 1
    nbs = e_cnt / len(cand)
    
    up = len(cand - pos - neg) / len(cand)
    
    return pcs, nbs, up, pcs * nbs * (1 - up)


def load_mooc_adj_table() -> List[Set[int]]:
    with open("data/entity2id.json", 'r') as f:
        entity2id: Dict[str, int] = json.load(f)
    with open("data/stu_problems.json", 'r') as f:
        problem_info: Dict[str, Any] = json.load(f)
    
    n = len(entity2id)
    g = [set() for _ in range(n)]
    
    course_sections = defaultdict(set)
    for _, p_info in problem_info.items():
        
        concept_ids: List[int] = [entity2id[c] for c in p_info['concepts']]
        section_id: int = entity2id[p_info['section']]
        course_id: int = entity2id[p_info['course']]
        
        for c in concept_ids:
            g[c].add(section_id)
            g[section_id].add(c)
        g[course_id].add(section_id)
        g[section_id].add(course_id)
        
        for x in concept_ids:
            for y in concept_ids:
                if x != y:
                    g[x].add(y)
        
        course_sections[course_id].add(section_id)
    
    for _, sections in course_sections.items():
        for s in sections:
            for t in sections:
                if s != t:
                    g[s].add(t)
    
    return g


def load_junyi_adj_table():
    g = defaultdict(set)
    with open("data/K_Directed.txt", "r") as f:
        for line in f:
            u, v = line.split()
            g[u].add(v)
    with open("data/K_Undirected.txt", "r") as f:
        for line in f:
            u, v = line.split()
            g[u].add(v)
            g[v].add(u)
    return g
