"""
compare_infosphere_edges.py
-----------------------------
Compares the edge sets of the infospheres defined in the campaign runner.

Parameters extracted from run_campaigns.py:
  INFOSPHERES = [
      {"type": 0, "params": 0},        -> no edges (baseline)
      {"type": 1, "params": 5},        -> noisy infosphere, fold 0-4
      {"type": 2, "params": 10},       -> top-10 papers per author
      {"type": 3, "params": "[5,2]"},  -> top-5 papers per topic, min 2
      {"type": 4, "params": 0},        -> NAIS rec
      {"type": 5, "params": 0},        -> LightGCN rec

Metric: Jaccard similarity on sets of pairs (src_id, dst_id).
  - Invariant to edge ordering
  - Two identical infospheres -> 1.0
  - No common edges   -> 0.0

Output (per relation + global):
  infosphere_similarity_heatmap_<rel>.png
  infosphere_similarity_heatmap_global.png
  infosphere_similarity.csv
  infosphere_similarity_overlap.csv
"""

import os
import ast
from collections import Counter

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from academic_network_project.anp_core.anp_dataset import ANPDataset
from academic_network_project.anp_core.anp_utils import create_infosphere_top_papers_edge_index

ROOT   = "../anp_data"
DEVICE = "cpu"
YEAR   = 2019

INFOSPHERE_1_PARAMS = "5"
INFOSPHERE_1_FOLD   = [0, 1, 2, 3, 4]
INFOSPHERE_3_ARGS   = [5, 2]
INFOSPHERE_2_TOP_K  = 10

CITES, WRITES, ABOUT = 0, 1, 2

INFOSPHERE_LABELS = {
    0: "inf0_none",
    1: "inf1_noisy",
    2: "inf2_top10",
    3: "inf3_top5t",
    4: "inf4_NAIS",
    5: "inf5_LightGCN",
}

ALL_RELATIONS = ("cites", "writes", "about")

OUT_DIR       = "."
CSV_JAC_PATH  = os.path.join(OUT_DIR, "infosphere_similarity.csv")
CSV_OVLP_PATH = os.path.join(OUT_DIR, "infosphere_similarity_overlap.csv")
CSV_DIST_PATH = os.path.join(OUT_DIR, "infosphere_edge_distribution.csv")


_dataset_cache: dict = {}


def _get_data():
    if "data" not in _dataset_cache:
        print("[INFO] Loading ANPDataset...")
        dataset = ANPDataset(root=ROOT)
        _dataset_cache["data"] = dataset[0]
        print("[INFO] Dataset loaded.")
    return _dataset_cache["data"]


def edge_index_to_set(edge_index) -> frozenset:
    if edge_index is None or edge_index.numel() == 0:
        return frozenset()
    return frozenset(zip(edge_index[0].tolist(), edge_index[1].tolist()))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_edges(inf_type: int) -> dict[str, frozenset]:
    """
    Returns { rel_label: edge_frozenset } for the infosphere type.
    Missing relations are returned as empty frozensets so that
    all infospheres have the same keys and are comparable.
    """
    empty = {rel: frozenset() for rel in ALL_RELATIONS}

    if inf_type == 0:
        return empty

    if inf_type == 1:
        fold_str = "_".join(map(str, INFOSPHERE_1_FOLD))
        name = (f"{INFOSPHERE_1_PARAMS}_infosphere_"
                f"{fold_str}_{YEAR}_noisy.pt")
        path = os.path.join(ROOT, "computed_infosphere", str(YEAR), name)
        if not os.path.exists(path):
            print(f"[WARN] inf1 - file not found: {path}")
            return empty
        raw = torch.load(path, map_location=DEVICE)
        result = dict(empty)
        for label, key in [("cites", CITES), ("writes", WRITES), ("about", ABOUT)]:
            try:
                result[label] = edge_index_to_set(raw[key])
            except (KeyError, IndexError, TypeError):
                print(f"[WARN] inf1 - key '{key}' missing in the file.")
        return result

    if inf_type == 2:
        data = _get_data()
        edge = create_infosphere_top_papers_edge_index(data, INFOSPHERE_2_TOP_K, YEAR)
        result = dict(empty)
        result["writes"] = edge_index_to_set(edge)
        return result

    if inf_type == 3:
        a0, a1 = INFOSPHERE_3_ARGS
        path = os.path.join(ROOT, "processed",
                            f"edge_infosphere_3_{a0}_{a1}.pt")
        result = dict(empty)
        if not os.path.exists(path):
            print(f"[WARN] inf3 - file not found: {path}")
            return result
        result["writes"] = edge_index_to_set(
            torch.load(path, map_location=DEVICE))
        return result

    if inf_type == 4:
        path = os.path.join(ROOT, "processed", "rec_edge_10_NAIS.pt")
        result = dict(empty)
        if not os.path.exists(path):
            print(f"[WARN] inf4 - file not found: {path}")
            return result
        result["writes"] = edge_index_to_set(
            torch.load(path, map_location=DEVICE))
        return result

    if inf_type == 5:
        path = os.path.join(ROOT, "processed", "rec_edge_10_LightGCN.pt")
        result = dict(empty)
        if not os.path.exists(path):
            print(f"[WARN] inf5 - file not found: {path}")
            return result
        result["writes"] = edge_index_to_set(
            torch.load(path, map_location=DEVICE))
        return result

    return empty


def global_edge_set(rel_dict: dict[str, frozenset]) -> frozenset:
    """Union of all edge sets of an infosphere (relation-agnostic comparison)."""
    result = frozenset()
    for s in rel_dict.values():
        result = result | s
    return result


def build_matrices(
    edge_dict: dict[str, frozenset]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = list(edge_dict.keys())
    n = len(labels)
    jac  = np.eye(n, dtype=float)
    ovlp = np.zeros((n, n), dtype=int)

    for i in range(n):
        ovlp[i, i] = len(edge_dict[labels[i]])
        for j in range(i + 1, n):
            j_val      = jaccard(edge_dict[labels[i]], edge_dict[labels[j]])
            o_val      = len(edge_dict[labels[i]] & edge_dict[labels[j]])
            jac[i, j]  = jac[j, i]  = j_val
            ovlp[i, j] = ovlp[j, i] = o_val

    df_jac  = pd.DataFrame(jac,  index=labels, columns=labels)
    df_ovlp = pd.DataFrame(ovlp, index=labels, columns=labels)
    return df_jac, df_ovlp


def plot_heatmap(df: pd.DataFrame, title: str, out_path: str) -> None:
    n       = len(df)
    fig_w   = max(6, n * 1.2)
    fig_h   = max(5, n * 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        df,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.6,
        linecolor="#e0e0e0",
        square=True,
        cbar_kws={"label": "Jaccard similarity", "shrink": 0.72, "pad": 0.02},
        annot_kws={"size": 10 if n <= 6 else 8},
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Infosphere", labelpad=8, fontsize=10)
    ax.set_ylabel("Infosphere", labelpad=8, fontsize=10)
    ax.tick_params(axis="x", rotation=35, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Saved -> {out_path}")
    plt.close()


def run_global_comparison(all_edges: dict[int, dict[str, frozenset]]) -> pd.DataFrame:
    """
    6x6 matrix across all infosphere types.
    Each infosphere is represented by the union of all its edge sets
    regardless of relation, so inf0 (empty) can be compared
    with inf1/2/3/4/5.
    """
    print("\n-- Global comparison (union of all relations) ------------")

    global_sets = {
        INFOSPHERE_LABELS[t]: global_edge_set(rel_dict)
        for t, rel_dict in all_edges.items()
    }

    for lbl, s in global_sets.items():
        print(f"  {lbl:20s}  {len(s):>10,} edges (total)")

    df_jac, df_ovlp = build_matrices(global_sets)
    print()
    print(df_jac.round(4).to_string())

    png_path = os.path.join(OUT_DIR, "infosphere_similarity_heatmap_global.png")
    plot_heatmap(
        df_jac,
        title=f"Jaccard similarity - global comparison (year {YEAR})",
        out_path=png_path,
    )

    df_ovlp.to_csv(CSV_OVLP_PATH)
    print(f"[OK] Global overlap CSV -> {CSV_OVLP_PATH}")

    return df_jac


def run_per_relation_comparison(
    all_edges: dict[int, dict[str, frozenset]]
) -> list[pd.DataFrame]:
    """
    One matrix for each relation in ALL_RELATIONS.
    All 6 infospheres are always included; those without edges
    for that relation have an empty frozenset (Jaccard = 0 vs the others).
    """
    all_jac_frames = []

    for rel in ALL_RELATIONS:
        edge_dict = {
            INFOSPHERE_LABELS[t]: rel_dict[rel]
            for t, rel_dict in all_edges.items()
        }

        non_empty = sum(1 for s in edge_dict.values() if s)
        if non_empty < 2:
            print(f"\n[SKIP] Relation '{rel}': less than 2 infospheres with edges.")
            continue

        print(f"\n-- Relation: {rel} -------------------------------------------")
        for lbl, s in edge_dict.items():
            print(f"  {lbl:20s}  {len(s):>10,} edges")

        df_jac, _ = build_matrices(edge_dict)
        print()
        print(df_jac.round(4).to_string())

        png_path = os.path.join(
            OUT_DIR, f"infosphere_similarity_heatmap_{rel}.png")
        plot_heatmap(
            df_jac,
            title=f"Jaccard similarity - relation '{rel}' (year {YEAR})",
            out_path=png_path,
        )

        df_tagged = df_jac.copy()
        df_tagged.index   = [f"{rel}_{l}" for l in df_jac.index]
        df_tagged.columns = [f"{rel}_{c}" for c in df_jac.columns]
        all_jac_frames.append(df_tagged)

    return all_jac_frames


def _get_author_year_map() -> dict[int, set[int]]:
    """
    Returns { author_id -> set of years } using the writes edges of the original graph
    and the year attribute of the papers.
    """
    data = _get_data()
    ei   = data['author', 'writes', 'paper'].edge_index
    years = data["paper"].x[:,0]

    author_years: dict[int, set[int]] = {}
    for auth, paper in zip(ei[0].tolist(), ei[1].tolist()):
        y = int(years[paper].item())
        author_years.setdefault(auth, set()).add(y)
    return author_years


def _gini(values: np.ndarray) -> float:
    """
    Gini coefficient on the values array (including zeros).
    0 = perfectly equal distribution, 1 = maximum concentration.
    """
    if values.sum() == 0:
        return float("nan")
    sorted_v = np.sort(values)
    n = len(sorted_v)
    cumsum = np.cumsum(sorted_v)
    return float((2 * np.sum(cumsum) - (n + 1) * sorted_v.sum()) /
                 (n * sorted_v.sum()))


def _entropy(values: np.ndarray) -> float:
    """Shannon entropy (bits) of the empirical degree distribution (excluding zeros)."""
    counts = values[values > 0]
    if len(counts) == 0:
        return float("nan")
    p = counts