import os
import json
import torch
import pandas as pd
import numpy as np
import uuid
import random
import time
import gc
import itertools
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy.stats import pearsonr
from torch_geometric.loader import LinkNeighborLoader
import torch_geometric.transforms as T

# Project-specific imports
from academic_network_project.anp_core.anp_dataset import ANPDataset
import academic_network_project.anp_core.anp_utils as anp_utils
import academic_network_project.anp_nn.models_2025 as models_2025

# --- CONFIGURATION ---
MIX_CAMPAIGN_PATH = "../anp_models_mix_infosphere_experiments_xmas_2025"
SINGLE_CAMPAIGN_PATH = "../anp_models_dropout_study_hgt2025_d04_30" # Nuova path aggiunta

ROOT_DATA = "../anp_data"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
YEAR = 2019
BATCH_SIZE = 4096 
SESSION_ID = str(uuid.uuid4())[:8]
OUTPUT_CSV = f"gt_pairwise_consistency_combined_{SESSION_ID}_{SINGLE_CAMPAIGN_PATH.split('/')[-1]}.csv"
SEED = 42

# Ensure reproducibility for the test set generation
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

def get_balanced_test_set(edge_index, num_nodes, ratio=1.0):
    """
    Generates a balanced set of positive and negative edges for evaluation.
    This set is generated once and used across all models.
    """
    print("Generating balanced test set...")
    pos_edges = edge_index
    num_pos = pos_edges.size(1)
    
    num_neg = int(num_pos * ratio)
    neg_edges = torch.randint(0, num_nodes, (2, num_neg), dtype=torch.long)
    
    final_edges = torch.cat([pos_edges, neg_edges], dim=1)
    final_labels = torch.cat([torch.ones(num_pos), torch.zeros(num_neg)])
    
    print(f"Test Set Created: {final_edges.size(1)} total pairs ({num_pos} pos, {num_neg} neg).")
    return final_edges, final_labels

def get_predictions(model, loader, device):
    """
    Runs inference on the loader.
    """
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", leave=False):
            batch = batch.to(device)
            
            if ('author', 'author') in batch.edge_types:
                e_idx = batch['author', 'author'].edge_label_index
            else:
                e_idx = batch['author', 'co_author', 'author'].edge_label_index
            
            for node_type in batch.node_types:
                if 'x' in batch[node_type]:
                    batch[node_type].x = batch[node_type].x.to(torch.float)
            
            batch['author'].x = model.embedding_author(batch['author'].n_id)
            batch['topic'].x = model.embedding_topic(batch['topic'].n_id)
            
            pred = model(batch.x_dict, batch.edge_index_dict, e_idx)
            all_preds.append(pred.cpu())
            
    return torch.cat(all_preds) if all_preds else torch.tensor([])

def get_model_folders(base_path):
    """Helper to safely retrieve model subdirectories"""
    if not os.path.exists(base_path):
        print(f"Warning: Path not found {base_path}")
        return []
    return [os.path.join(base_path, f) for f in os.listdir(base_path) 
            if os.path.isdir(os.path.join(base_path, f))]

def run_gt_consistency_check():
    # 1. Load Base Data
    print(f"Loading Dataset from {ROOT_DATA}...")
    dataset = ANPDataset(root=ROOT_DATA)
    base_data = dataset[0]
    base_data['paper'].x = base_data['paper'].x.to(torch.float)
    
    # Load Co-authorship Ground Truth (Target)
    coauthor_file = os.path.join(ROOT_DATA, "processed", f"author_edge{YEAR + 1}.pt")
    base_data['author', 'co_author', 'author'].edge_index = torch.load(coauthor_file, map_location='cpu')["author"]

    # 2. Define Fixed Evaluation Set (Group 4 Only)
    print("Filtering data for Group 4 (Fold 4)...")
    fold_4_data = anp_utils.anp_simple_filter_data(base_data, root=ROOT_DATA, folds=[4], max_year=YEAR)
    
    test_edges, test_labels = get_balanced_test_set(
        fold_4_data['author', 'co_author', 'author'].edge_index, 
        base_data['author'].num_nodes
    )

    # 3. Locate Models (Both Mixed and Single)
    # mix_folders = get_model_folders(MIX_CAMPAIGN_PATH)
    mix_folders = []
    single_folders = get_model_folders(SINGLE_CAMPAIGN_PATH)
    
    all_gt_folders = mix_folders + single_folders
    
    print(f"Found {len(mix_folders)} Mix models and {len(single_folders)} Single models.")
    print(f"Total models to evaluate: {len(all_gt_folders)}")
    
    model_predictions = {}
    model_configs = {}

    # 4. Inference Loop
    for idx, gt_path in enumerate(all_gt_folders):
        model_name = os.path.basename(gt_path)
        print(f"\n--- Processing Model {idx+1}/{len(all_gt_folders)}: {model_name} ---")
        
        try:
            # Load Config
            with open(os.path.join(gt_path, 'info.json'), 'r') as f: 
                cfg = json.load(f)
            
            # --- CONFIG SANITIZATION FOR SINGLE MODELS ---
            # I modelli single hanno 'infosphere_mix_ratio': null e 'infosphere_type_2': null
            # Normalizziamo questi valori per non far crashare anp_add_infosphere_mix
            
            info_type_1 = cfg.get('infosphere_type')
            info_params_1 = cfg.get('infosphere_parameters')
            
            info_type_2 = cfg.get('infosphere_type_2') # Sarà None per i single
            info_params_2 = cfg.get('infosphere_parameters_2') # Sarà None per i single
            
            mix_ratio = cfg.get('infosphere_mix_ratio')
            
            # Se è un modello single (ratio è None), forziamo il ratio a 0.0
            # Assumiamo che anp_add_infosphere_mix usi il ratio per pesare type_2.
            # Se ratio è 0.0, type_2 viene ignorato.
            if mix_ratio is None:
                mix_ratio = 1.0
                
            drop_percentage = cfg.get('drop_percentage', 0.0)

            # Store sanitized config for later reporting
            model_configs[model_name] = {
                "type_1": info_type_1,
                "type_2": info_type_2,
                "ratio": mix_ratio,
                "is_single": (cfg.get('infosphere_type_2') is None)
            }

            # Reconstruct Graph
            current_data = base_data.clone()
            
            anp_utils.anp_add_infosphere_mix(
                current_data, 
                info_type_1, info_params_1,
                info_type_2, info_params_2,
                mix_ratio, drop_percentage, 
                ROOT_DATA, DEVICE, YEAR
            )
            
            current_data = T.ToUndirected()(current_data)
            
            if ('author', 'co_author', 'author') in current_data.edge_types:
                del current_data['author', 'co_author', 'author']

            # Loader
            val_loader = LinkNeighborLoader(
                data=current_data, 
                num_neighbors=[30, 10], 
                edge_label_index=(('author', 'co_author', 'author'), test_edges),
                batch_size=BATCH_SIZE, 
                shuffle=False 
            )

            # Model
            model = models_2025.Model(hidden_channels=32, data=current_data).to(DEVICE)
            weights, _ = anp_utils.anp_load(gt_path + '/', device=DEVICE)
            model.load_state_dict(weights.state_dict())

            # Predict
            preds = get_predictions(model, val_loader, DEVICE)
            model_predictions[model_name] = preds

            del model, weights, current_data, val_loader
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Failed to process {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # 5. Pairwise Comparison
    print("\n--- Running Pairwise Consistency Analysis ---")
    results_list = []
    
    model_names = list(model_predictions.keys())
    combinations = list(itertools.combinations(model_names, 2))
    
    for name_a, name_b in tqdm(combinations, desc="Comparing Models"):
        preds_a = model_predictions[name_a].numpy()
        preds_b = model_predictions[name_b].numpy()
        
        corr, _ = pearsonr(preds_a, preds_b)
        mse = np.mean((preds_a - preds_b)**2)
        mad = np.mean(np.abs(preds_a - preds_b))
        
        c_a = model_configs[name_a]
        c_b = model_configs[name_b]
        
        # Helper string formatter
        def get_info_str(c):
            if c['is_single']:
                return f"Single: {c['type_1']}"
            else:
                return f"{c['type_1']}+{c['type_2']} (mix: {c['ratio']})"

        results_list.append({
            "model_a": name_a,
            "model_b": name_b,
            "pearson_correlation": corr,
            "mse_diff": mse,
            "mae_diff": mad,
            "model_a_info": get_info_str(c_a),
            "model_b_info": get_info_str(c_b),
            "evaluated_on": "Group_4_Fixed"
        })

    # 6. Save Results
    if results_list:
        df = pd.DataFrame(results_list)
        df.sort_values(by="pearson_correlation", ascending=False, inplace=True)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSuccess. Results saved to {OUTPUT_CSV}")
        print("\nTop 5 Most Similar Model Pairs:")
        print(df[['model_a', 'model_b', 'pearson_correlation']].head(5))
        
        print("\nTop 5 Least Similar Model Pairs:")
        print(df[['model_a', 'model_b', 'pearson_correlation']].tail(5))
    else:
        print("No results generated.")

if __name__ == "__main__":
    run_gt_consistency_check()