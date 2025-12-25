import json
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import config
from utils.graph import Node, calc_metric, load_junyi_adj_table, load_mooc_adj_table

data_path = "outputs/metrics_data.npy"
svg_test_path = "./outputs/stu_metrics_test.svg"
svg_path = "./outputs/stu_metrics.svg"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 2.5,
        "figure.facecolor": "white",
    }
)


def _get_stu_node_state(
    n: int, u_records: list[dict[str, Any]], threshold: float = 0.8
) -> tuple[list[Node], set[int], set[int]]:
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
        node_ids = [entity2id[p["course"]], entity2id[p["section"]]] + [
            entity2id[c] for c in p["concepts"]
        ]
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


def _get_cand_vex(cand: set, pos: set, neg: set, iter: int) -> list[int]:
    neg_v: list[int] = []
    unk_v: list[int] = []
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

    return unk_v[: len(unk_v) // 2] + neg_v + unk_v[len(unk_v) // 2 :]


def process_data(g: list[set[int]], iter: int = 100) -> None:
    n = len(g)
    students_metric: list[list[tuple]] = []

    for _, u_records in tqdm(list(records.items()), desc="Processing"):
        w, pos, neg = _get_stu_node_state(n, u_records)
        if len(pos) == 0:
            continue

        student_metric: list[tuple] = []
        cand = pos.copy()
        cand_v = _get_cand_vex(cand, pos, neg, iter)
        for i in range(iter):
            student_metric.append((calc_metric(cand, pos, neg, g)[-1],))
            cand.add(cand_v[i])

        students_metric.append(student_metric)

    np_metric = np.array(students_metric)
    print(np_metric.shape)
    np.save(data_path, np_metric)


def visualize_data() -> None:
    # load data
    data = np.load(data_path)
    print(f"data shape: {data.shape}")

    # choose 9 samples randomly
    sample_indices = np.random.choice(data.shape[0], 9, replace=False)
    samples = data[sample_indices]

    # visualize
    plt.figure(figsize=(20, 20))
    for i, sample in enumerate(samples, 1):
        plt.subplot(3, 3, i)

        plt.plot(sample)
        plt.title(f"Students {sample_indices[i - 1]}")
        plt.xlabel("Step")
        plt.ylabel("Mastery Scores")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(svg_test_path)


def visualize_data_adv(sample_indices: list) -> None:
    # load data
    data = np.load(data_path)  # shape: [N_samples, T_steps]
    print(f"data shape: {data.shape}")

    # sort samples by idx
    sorted_pairs = sorted(zip(sample_indices, range(len(sample_indices))))
    sorted_indices = [pair[0] for pair in sorted_pairs]
    color_order = [pair[1] for pair in sorted_pairs]

    samples = data[sorted_indices]
    steps = np.arange(data.shape[1])

    # config
    transition_points = [0, 41, 59, len(steps)]
    line_colors = ["#65B89E", "#EE9834", "#BE70DF", "#01A4FD", "#F98A8A", "#DEDEDE"]
    bg_colors = ["#B2E2DC", "#C5CCDB", "#CAEAF2"]
    print(transition_points)
    # draw with same X axes
    fig, axes = plt.subplots(1, 1, figsize=(12, 8))

    # draw bg color
    for i in range(len(bg_colors)):
        axes.axvspan(
            transition_points[i],
            transition_points[i + 1],
            color=bg_colors[i],
            alpha=0.3,
            zorder=0,
        )

    # draw plot
    for i, (sample_idx, sample) in enumerate(zip(sorted_indices, samples)):
        original_color_idx = color_order[i]
        axes.plot(
            steps,
            sample,
            label=f"Student #{sample_idx}",
            color=line_colors[original_color_idx],
            linewidth=2.5,
            zorder=2,
        )

    axes.set_xlim(0, 100)
    axes.set_xticks([0, 20, 40, 60, 80, 100])
    axes.set_xticks([10, 30, 50, 70, 90], minor=True)
    axes.tick_params(which="minor", length=4, color="black")

    axes.grid(False)

    for spine in axes.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.5)

    legend = axes.legend(
        loc="upper right",
        frameon=True,
        prop={"family": "Times New Roman", "size": 20, "weight": "normal"},
        fancybox=True,
        shadow=True,
        ncol=2,
        fontsize=17,
        markerscale=1.2,
        borderpad=1,
        columnspacing=1.5,
        handletextpad=0.8,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_linewidth(1)
    legend.get_frame().set_edgecolor("#cccccc")

    plt.tight_layout()
    plt.savefig(svg_path, bbox_inches="tight")


if __name__ == "__main__":
    with open("data/stu_records_filtered.json", encoding="utf-8") as f:
        records: dict[str, list[dict]] = json.load(f)
    with open("data/stu_problems.json") as f:
        problem_info: dict[str, Any] = json.load(f)
    with open("data/entity2id.json") as f:
        entity2id: dict[str, int] = json.load(f)

    if config.dataset == "mooc":
        g = load_mooc_adj_table()
    else:
        g = load_junyi_adj_table()

    if not Path(data_path).exists():
        process_data(g)

    # repeat to choose appropriate sample idxs
    visualize_data()

    # merge add sample into one figure
    visualize_data_adv([643, 4452, 4179, 5085, 1802, 821])
