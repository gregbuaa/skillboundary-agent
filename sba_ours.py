import json
import os
from multiprocessing import Manager, Process
from typing import Dict, List, Set

from tqdm import tqdm

import config
from cdm.ncd.predict import get_stu_emb
# change pre-trained CDM model
# from cdm.rcd.predict import get_stu_emb
from config import gpt_model
from utils.graph import calc_metric, load_junyi_adj_table, load_mooc_adj_table
from utils.llm_api import ask

topk = 3
delta_thr = 0.013

NUM_PROCESSES = 10
PROCESS_LOG_FILE = "data/process.json"
output_file = f"outputs/gpt-4o-mooc.json"
os.makedirs('outputs', exist_ok=True)


def build_prompt(cand, expand_candidates, backtrace_candidates, w, id2entity, topk=topk):
    prompt = "The current subgraph contains the following nodes (Node ID: Mastery Value):\n"
    for v in cand:
        prompt += f"- {v}: {w[v]:.3f}, Content: {id2entity.get(str(v), 'Unknown')}\n"
    
    prompt += "\nCandidate nodes for expansion (Node ID: Gain):\n"
    for v, gain in sorted(expand_candidates, key=lambda x: -x[1])[:topk]:
        prompt += f"- {v}: Gain {gain:.3f}, Mastery {w[v]:.3f}, Content: {id2entity.get(str(v), 'Unknown')}\n"
    
    prompt += "\nCandidate nodes for deletion (Node ID: Mastery Drop):\n"
    for v, gain in sorted(backtrace_candidates, key=lambda x: x[1])[:topk]:
        prompt += f"- {v}: Mastery change after removal {gain:.3f}, Content: {id2entity.get(str(v), 'Unknown')}\n"
    
    prompt += (
        "\nPlease choose one of the following actions and output it in the specified format:\n"
        "- expand <Node ID>\n"
        "- backtrace <Node ID>\n"
        "- stop\n"
    )
    
    return prompt


def run_process(proc_id: int, test_data_slice: List[Dict], g, id2entity, return_list: List, process_log: Dict):
    results = []
    pcs, nbs, up, iou = 0, 0, 0, 0
    
    for idx, td in enumerate(tqdm(test_data_slice, desc=f"Proc-{proc_id}", position=proc_id, leave=True)):
        global_index = proc_id * len(test_data_slice) + idx
        process_log[str(proc_id)] = global_index
        user_id: str = td["user_id"]
        init_node: int = td["init_nodes"]
        pos: Set[int] = set(td["gt_subgraph"]["pos_inner_nodes"])
        neg: Set[int] = set(td["gt_subgraph"]["neg_front_nodes"])
        w = get_stu_emb(user_id)
        trace = []
        step_id = 1
        
        cand: Set[int] = {init_node}
        p_G = w[init_node]
        
        for _ in range(30):
            expand_candidates = []
            for u in cand:
                for v in g[u]:
                    if v not in cand:
                        gain = w[v] - p_G
                        expand_candidates.append((v, gain))
            
            backtrace_candidates = []
            if len(cand) > 1:
                for v in cand:
                    temp_cand = cand - {v}
                    temp_p_G = sum([w[u] for u in temp_cand]) / len(temp_cand)
                    gain = temp_p_G - p_G
                    backtrace_candidates.append((v, gain))
            
            user_prompt = build_prompt(cand, expand_candidates, backtrace_candidates, w, id2entity)
            
            system_prompt = (
                "Role: You are an expert in student cognitive diagnosis, and your task is to assist in mining the student's cognitive subgraph.\n"
                "Background: You will be provided with three types of data:\n"
                "- The student's current cognitive subgraph, including the nodes and their semantic descriptions.\n"
                "- The neighborhood of the current subgraph, including the candidate nodes, their semantics, and the estimated change in the student's cognitive level if each node is added to the subgraph.\n"
                "- The current subgraph again, including the nodes, their semantics, and the estimated change in the student's cognitive level if each node is removed from the subgraph.\n"
                "Task: You are expected to analyze all available information—such as node semantics and mastery scores—and decide whether to expand the subgraph by adding a boundary node, delete an existing node from the subgraph, or terminate the search. The goal is to accurately construct the student's cognitive subgraph.\n"
            )
            
            decision = ask(user_prompt=user_prompt,
                           system_prompt=system_prompt,
                           temperature=0.0,
                           model=gpt_model).strip().lower()
            operated = False
            
            if decision.startswith("expand"):
                v = int(decision.split()[-1])
                cand.add(v)
                p_G = 0.1 * p_G + 0.9 * w[v]
                trace.append({"step": step_id, "action": "expand", "node": v})
                operated = True
            elif decision.startswith("backtrace"):
                v = int(decision.split()[-1])
                cand.remove(v)
                p_G = sum([w[u] for u in cand]) / len(cand)
                trace.append({"step": step_id, "action": "backtrace", "node": v})
                operated = True
            elif "stop" in decision:
                trace.append({"step": step_id, "action": "stop"})
                break
            
            step_id += 1
            if not operated:
                break
        
        a, b, c, d = calc_metric(cand, pos, neg, g)
        pcs += a
        nbs += b
        up += c
        iou += d
        
        results.append({
            "user_id": user_id,
            "trace": trace,
            "final_subgraph": list(cand),
            "metric": {"pcs": a, "nbs": b, "up": c, "iou": d}
        })
    
    return_list.append(results)


def skill_boundary_agent_multi():
    with open("data/id2entity.json", 'r') as f:
        id2entity = json.load(f)
    if config.dataset == "mooc":
        g = load_mooc_adj_table()
        with open("data/mooc_test_data.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)
    else:
        g = load_junyi_adj_table()
        with open("data/junyi_test_data.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)
    
    total = len(test_data)
    size = (total + NUM_PROCESSES - 1) // NUM_PROCESSES
    manager = Manager()
    return_list = manager.list()
    process_log = manager.dict()
    
    processes = []
    for i in range(NUM_PROCESSES):
        sub_data = test_data[i * size:(i + 1) * size]
        p = Process(target=run_process, args=(i, sub_data, g, id2entity, return_list, process_log))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    all_logs = []
    for result in return_list:
        all_logs.extend(result)
    
    os.makedirs("outputs", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_logs, f, ensure_ascii=False, indent=2)
    
    with open(PROCESS_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(dict(process_log), f, ensure_ascii=False, indent=2)


def print_result() -> None:
    with open(output_file, "r", encoding="utf-8") as f:
        all_logs = json.load(f)
    pcs, nbs, up, iou, call_num = 0, 0, 0, 0, 0
    for log in all_logs:
        pcs += log["metric"]["pcs"]
        nbs += log["metric"]["nbs"]
        up += log["metric"]["up"]
        iou += log["metric"]["iou"]
        call_num += len(log["trace"])
    print(f"{pcs / len(all_logs):.3f}", f"{nbs / len(all_logs):.3f}",
          f"{up / len(all_logs):.3f}", f"{iou / len(all_logs):.3f}",
          f"{call_num / len(all_logs):.3f}")


if __name__ == '__main__':
    skill_boundary_agent_multi()
    print_result()
