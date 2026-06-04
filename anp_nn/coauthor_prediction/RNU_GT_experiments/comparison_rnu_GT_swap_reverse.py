import os
import json
import torch
import pandas as pd
import numpy as np
import uuid
import random
import gc
import itertools
from tqdm import tqdm
from scipy.stats import pearsonr
from torch_geometric.loader import LinkNeighborLoader
import torch_geometric.transforms as T

from academic_network_project.anp_core.anp_dataset import ANPDataset
import academic_network_project.anp_core.anp_utils as anp_utils
import academic_network_project.anp_nn.models_2025 as models_2025

SINGLE_CAMPAIGN_PATH = "../anp_models_single_infosphere_experiments_7"

ROOT_DATA = "../anp_data"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
YEAR = 2019
BATCH_SIZE = 4096
SESSION_ID = str(uuid.uuid4())[:8]
OUTPUT_DIR = f"consistency_results_{SESSION_ID}"
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")


def ensure_structural_compatibility(data, device):
    possible_infosphere_edges = [
        ('paper', 'infosphere_cites', 'paper'),
        ('paper', 'infosphere_about', 'topic'),
        ('topic', 'rev_infosphere_about', 'paper'),
        ('author', 'infosphere_writes', 'paper')
    ]
    for etype in possible_infosphere_edges:
        if etype not in data.edge_types:
            data[etype].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            if 'edge_label' in data[etype]:
                del data[etype].edge_label
            if 'edge_attr' in data[etype]:
                del data[etype].edge_attr
    return data


def extract_edge_types_from_state_dict(state_dict):
    known_edges = set()
    for key in state_dict.keys():
        if 'p_rel.' in key:
            parts = key.split('p_rel.')
            if len(parts) > 1:
                rel_str = parts[1].split('.')[0]
                if '__' in rel_str:
                    src, rel, dst = rel_str.split('__')
                    known_edges.add((src, rel, dst))
    return list(known_edges)


def create_model_init_data(full_data, allowed_edge_types):
    init_data = full_data.clone()
    for etype in list(init_data.edge_types):
        del init_data[etype]
    for etype in allowed_edge_types:
        if etype in full_data.edge_types:
            init_data[etype].edge_index = full_data[etype].edge_index
        else:
            init_data[etype].edge_index = torch.empty((2, 0), dtype=torch.long)
    return init_data


def get_balanced_test_set(edge_index, num_nodes, ratio=1.0):
    print("Generating balanced test set...")
    pos_edges = edge_index
    num_pos = pos_edges.size(1)
    num_neg = int(num_pos * ratio)
    neg_edges = torch.randint(0, num_nodes, (2, num_neg), dtype=torch.long)
    final_edges = torch.cat([pos_edges, neg_edges], dim=1)
    print(f"Test Set Created: {final_edges.size(1)} total pairs ({num_pos} pos, {num_neg} neg).")
    return final_edges


def get_predictions(model, loader, device, allowed_edge_types):
    model.eval()
    all_preds = []
    allowed_set = set(allowed_edge_types)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            clean_edge_index_dict = {}
            for etype in allowed_set:
                if etype in batch.edge_index_dict:
                    clean_edge_index_dict[etype] = batch.edge_index_dict[etype]
                else:
                    clean_edge_index_dict[etype] = torch.empty((2, 0), dtype=torch.long, device=device)

            if ('author', 'author') in batch.edge_types:
                e_idx = batch['author', 'author'].edge_label_index
            else:
                e_idx = batch['author', 'co_author', 'author'].edge_label_index

            for node_type in batch.node_types:
                if 'x' in batch[node_type]:
                    batch[node_type].x = batch[node_type].x.to(torch.float)

            batch['author'].x = model.embedding_author(batch['author'].n_id)
            batch['topic'].x = model.embedding_topic(batch['topic'].n_id)

            pred = model(batch.x_dict, clean_edge_index_dict, e_idx)
            all_preds.append(pred.cpu())

    return torch.cat(all_preds) if all_preds else torch.tensor([])


def get_model_folders(base_path):
    if not os.path.exists(base_path):
        print(f"Warning: Path not found {base_path}")
        return []
    return [
        os.path.join(base_path, f)
        for f in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, f))
    ]


def is_zero_infosphere(info_type):
    return info_type is None or str(info_type) == '0'


def build_environment_list(mix_folders, single_folders):
    """
    Collects all valid environments from both mix and single campaigns.
    Skips any environment where the primary infosphere type is 0 or None.
    Returns a list of dicts describing each environment.
    """
    environments = [
        {
            "path": None,
            "env_desc": "BaseEnv_NoInfosphere",
            "info_type_1": None,
            "info_params_1": None,
            "info_type_2": None,
            "info_params_2": None,
            "mix_ratio": 1.0,
            "drop_percentage": 0.0,
            "source": "base"
        }
    ]

    for path in mix_folders:
        try:
            with open(os.path.join(path, 'info.json'), 'r') as f:
                cfg = json.load(f)
        except Exception:
            continue

        info_type_1 = cfg.get('infosphere_type')
        if is_zero_infosphere(info_type_1):
            print(f"Skipping zero-infosphere mix environment: {os.path.basename(path)}")
            continue

        info_type_2 = cfg.get('infosphere_type_2')
        mix_ratio = cfg.get('infosphere_mix_ratio', 0.0)
        drop_percentage = cfg.get('drop_percentage', 0.0)

        environments.append({
            "path": path,
            "env_desc": f"MixEnv_{info_type_1}+{info_type_2}_mix{mix_ratio}",
            "info_type_1": info_type_1,
            "info_params_1": cfg.get('infosphere_parameters'),
            "info_type_2": info_type_2,
            "info_params_2": cfg.get('infosphere_parameters_2'),
            "mix_ratio": mix_ratio,
            "drop_percentage": drop_percentage,
            "source": "mix"
        })

    for path in single_folders:
        try:
            with open(os.path.join(path, 'info.json'), 'r') as f:
                cfg = json.load(f)
        except Exception:
            continue

        info_type_1 = cfg.get('infosphere_type')
        if is_zero_infosphere(info_type_1):
            print(f"Skipping zero-infosphere single environment: {os.path.basename(path)}")
            continue

        environments.append({
            "path": path,
            "env_desc": f"SingleEnv_{info_type_1}",
            "info_type_1": info_type_1,
            "info_params_1": cfg.get('infosphere_parameters'),
            "info_type_2": None,
            "info_params_2": None,
            "mix_ratio": 1.0,
            "drop_percentage": cfg.get('drop_percentage', 0.0),
            "source": "single"
        })

    return environments


def build_data_for_environment(base_data, env, device):
    current_data = base_data.clone()
    if env["source"] != "base":
        anp_utils.anp_add_infosphere_mix(
            current_data,
            env["info_type_1"], env["info_params_1"],
            env["info_type_2"], env["info_params_2"],
            env["mix_ratio"], env["drop_percentage"],
            ROOT_DATA, device, YEAR
        )
    current_data = ensure_structural_compatibility(current_data, device)
    current_data = T.ToUndirected()(current_data)
    if ('author', 'co_author', 'author') in current_data.edge_types:
        del current_data['author', 'co_author', 'author']
    return current_data


def run_cross_infosphere_analysis():
    print(f"Loading Dataset from {ROOT_DATA}...")
    dataset = ANPDataset(root=ROOT_DATA)
    base_data = dataset[0]
    base_data['paper'].x = base_data['paper'].x.to(torch.float)

    coauthor_file = os.path.join(ROOT_DATA, "processed", f"author_edge{YEAR + 1}.pt")
    base_data['author', 'co_author', 'author'].edge_index = torch.load(
        coauthor_file, map_location='cpu'
    )["author"]

    print("Filtering data for Group 4 (Fold 4)...")
    fold_4_data = anp_utils.anp_simple_filter_data(base_data, root=ROOT_DATA, folds=[4], max_year=YEAR)
    test_edges = get_balanced_test_set(
        fold_4_data['author', 'co_author', 'author'].edge_index,
        base_data['author'].num_nodes
    )

    mix_folders = get_model_folders(MIX_CAMPAIGN_PATH)
    candidate_folders = get_model_folders(SINGLE_CAMPAIGN_PATH)

    all_environments = build_environment_list(mix_folders, candidate_folders)
    print(f"\nTotal valid environments to test: {len(all_environments)}")
    print(f"Candidate models to evaluate: {len(candidate_folders)}")

    global_predictions = {}
    model_configs = {}

    for env in tqdm(all_environments, desc="Analyzing Environments"):
        env_desc = env["env_desc"]
        print(f"\n--- Building environment: {env_desc} ---")

        try:
            current_data = build_data_for_environment(base_data, env, DEVICE)
        except Exception as e:
            print(f"Error building environment {env_desc}: {e}")
            continue

        val_loader = LinkNeighborLoader(
            data=current_data,
            num_neighbors=[30, 10],
            edge_label_index=(('author', 'co_author', 'author'), test_edges),
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        for model_path in candidate_folders:
            model_name = os.path.basename(model_path)

            try:
                weights, _ = anp_utils.anp_load(model_path + '/', device=DEVICE)
                state_dict = weights.state_dict()

                expected_edges = extract_edge_types_from_state_dict(state_dict)
                if not expected_edges:
                    continue

                model_init_data = create_model_init_data(current_data, expected_edges)
                model = models_2025.Model(hidden_channels=32, data=model_init_data).to(DEVICE)
                model.load_state_dict(state_dict)

                if model_name not in model_configs:
                    with open(os.path.join(model_path, 'info.json'), 'r') as f:
                        m_cfg = json.load(f)
                    model_configs[model_name] = f"Single: {m_cfg.get('infosphere_type')}"

                preds = get_predictions(model, val_loader, DEVICE, allowed_edge_types=expected_edges)

                if model_name not in global_predictions:
                    global_predictions[model_name] = {}
                global_predictions[model_name][env_desc] = preds

                del model, weights, state_dict, model_init_data

            except Exception as e:
                print(f"Error evaluating {model_name} on {env_desc}: {e}")

        del current_data, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    print("\nAll inferences completed. Calculating cross-infosphere consistency...")

    all_results_list = []

    for model_name, env_preds_dict in global_predictions.items():
        env_names = list(env_preds_dict.keys())
        if len(env_names) < 2:
            continue

        pairs = list(itertools.combinations(env_names, 2))
        model_info = model_configs.get(model_name, "Unknown")

        for env_a, env_b in pairs:
            p_a = env_preds_dict[env_a].numpy()
            p_b = env_preds_dict[env_b].numpy()

            if len(p_a) == 0 or len(p_b) == 0:
                continue

            corr, _ = pearsonr(p_a, p_b)
            mse = np.mean((p_a - p_b) ** 2)
            mad = np.mean(np.abs(p_a - p_b))

            all_results_list.append({
                "model_evaluated": model_name,
                "model_info": model_info,
                "environment_a": env_a,
                "environment_b": env_b,
                "pearson_correlation": corr,
                "mse_diff": mse,
                "mae_diff": mad
            })

    if all_results_list:
        df = pd.DataFrame(all_results_list)
        df.sort_values(
            by=["model_evaluated", "pearson_correlation"],
            ascending=[True, False],
            inplace=True
        )
        out_file = os.path.join(OUTPUT_DIR, "consistency_cross_environments.csv")
        df.to_csv(out_file, index=False)
        print(f"\nSaved aggregated cross-environment results to {out_file}")

    print(f"Processing Complete. All results saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    run_cross_infosphere_analysis()