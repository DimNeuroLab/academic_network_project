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
import re
from datetime import datetime
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
SINGLE_CAMPAIGN_PATH = "../anp_models_single_infosphere_experiments_7"

ROOT_DATA = "../anp_data"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
YEAR = 2019
BATCH_SIZE = 4096
SESSION_ID = str(uuid.uuid4())[:8]
OUTPUT_DIR = f"consistency_results_{SESSION_ID}"
SEED = 42

# Ensure reproducibility
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

def ensure_structural_compatibility(data, device):
    """
    Ensures that the environment graph contains all possible edge types defined in the project.
    This is crucial so that the LinkNeighborLoader doesn't crash when sampling
    for models that expect edges not natively present in the current mix.
    """
    possible_infosphere_edges = [
        ('paper', 'infosphere_cites', 'paper'),
        ('paper', 'infosphere_about', 'topic'),
        ('topic', 'rev_infosphere_about', 'paper'), # Often created by ToUndirected
        ('author', 'infosphere_writes', 'paper')
    ]
    
    for etype in possible_infosphere_edges:
        if etype not in data.edge_types:
            # Create empty edge index
            data[etype].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            # Remove labels/attributes to avoid confusion
            if 'edge_label' in data[etype]: del data[etype].edge_label
            if 'edge_attr' in data[etype]: del data[etype].edge_attr

    return data

def extract_edge_types_from_state_dict(state_dict):
    """
    Parses the weights of a trained model to understand EXACTLY which edge types
    it was trained on. This solves the "Size Mismatch" error definitively.
    """
    known_edges = set()
    # HGT specific: keys look like 'encoder.convs.0.p_rel.paper__infosphere_cites__paper'
    # We look for the 'p_rel' keys which define the relation matrices.
    for key in state_dict.keys():
        if 'p_rel.' in key:
            # Extract the part after p_rel.
            # Example: ...p_rel.paper__infosphere_cites__paper -> paper__infosphere_cites__paper
            parts = key.split('p_rel.')
            if len(parts) > 1:
                rel_str = parts[1].split('.')[0] # Get just the relation string
                # Convert 'src__rel__dst' back to tuple ('src', 'rel', 'dst')
                if '__' in rel_str:
                    src, rel, dst = rel_str.split('__')
                    known_edges.add((src, rel, dst))
    
    return list(known_edges)

def create_model_init_data(full_data, allowed_edge_types):
    """
    Creates a lightweight Data object containing ONLY the edge types expected by the model.
    Used to initialize the Model() class so its weights match the state_dict perfectly.
    """
    # Start with just node features (shared across all models)
    init_data = full_data.clone()
    
    # Remove ALL edges first
    for etype in list(init_data.edge_types):
        del init_data[etype]
        
    # Add back ONLY the allowed edges (copying structure from full_data)
    for etype in allowed_edge_types:
        if etype in full_data.edge_types:
            init_data[etype].edge_index = full_data[etype].edge_index
        else:
            # Should not happen thanks to ensure_structural_compatibility, but safety first
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
    """
    Runs inference ensuring we ONLY pass compatible edge types to the forward pass.
    """
    model.eval()
    all_preds = []
    
    # Convert list of tuples to list of strings for easier checking if needed,
    # but PyG uses tuples as keys in edge_index_dict.
    allowed_set = set(allowed_edge_types)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # --- CRITICAL FIX FOR EDGE_ATTR ERROR ---
            # Instead of filtering the dictionary in-place (which confuses PyG metadata),
            # we construct a CLEAN dictionary to pass to the model.
            
            clean_edge_index_dict = {}
            for etype, edge_index in batch.edge_index_dict.items():
                if etype in allowed_set:
                    clean_edge_index_dict[etype] = edge_index
            
            # Determine target edge index for prediction
            if ('author', 'author') in batch.edge_types:
                e_idx = batch['author', 'author'].edge_label_index
            else:
                e_idx = batch['author', 'co_author', 'author'].edge_label_index
            
            # Float precision fix
            for node_type in batch.node_types:
                if 'x' in batch[node_type]:
                    batch[node_type].x = batch[node_type].x.to(torch.float)
            
            # Embeddings
            batch['author'].x = model.embedding_author(batch['author'].n_id)
            batch['topic'].x = model.embedding_topic(batch['topic'].n_id)
            
            # FORWARD PASS: We pass the CLEAN dictionary.
            # The model was initialized with only these keys, so it won't look for others.
            pred = model(batch.x_dict, clean_edge_index_dict, e_idx)
            all_preds.append(pred.cpu())
            
    return torch.cat(all_preds) if all_preds else torch.tensor([])

def get_model_folders(base_path):
    if not os.path.exists(base_path):
        print(f"Warning: Path not found {base_path}")
        return []
    return [os.path.join(base_path, f) for f in os.listdir(base_path) 
            if os.path.isdir(os.path.join(base_path, f))]

def run_cross_infosphere_analysis():
    # 1. Load Base Data
    print(f"Loading Dataset from {ROOT_DATA}...")
    dataset = ANPDataset(root=ROOT_DATA)
    base_data = dataset[0]
    base_data['paper'].x = base_data['paper'].x.to(torch.float)
    
    coauthor_file = os.path.join(ROOT_DATA, "processed", f"author_edge{YEAR + 1}.pt")
    base_data['author', 'co_author', 'author'].edge_index = torch.load(coauthor_file, map_location='cpu')["author"]

    # 2. Define Fixed Evaluation Set
    print("Filtering data for Group 4 (Fold 4)...")
    fold_4_data = anp_utils.anp_simple_filter_data(base_data, root=ROOT_DATA, folds=[4], max_year=YEAR)
    test_edges = get_balanced_test_set(
        fold_4_data['author', 'co_author', 'author'].edge_index, 
        base_data['author'].num_nodes
    )

    # 3. Identify Models
    mix_folders = get_model_folders(MIX_CAMPAIGN_PATH)
    candidate_folders = get_model_folders(SINGLE_CAMPAIGN_PATH)
    
    print(f"Found {len(mix_folders)} Environments to test.")
    print(f"Found {len(candidate_folders)} Models to evaluate.")

    # --- DIZIONARI PER IL SALVATAGGIO DEI RISULTATI ---
    # Struttura: global_predictions[model_name][env_desc] = predictions_tensor
    global_predictions = {}
    model_configs = {}

    # 4. OUTER LOOP: Manteniamo gli Ambienti all'esterno per evitare bug di PyG
    for i, mix_path in enumerate(tqdm(mix_folders, desc="Analyzing Environments")):
        mix_model_name = os.path.basename(mix_path)
        
        try:
            with open(os.path.join(mix_path, 'info.json'), 'r') as f: cfg = json.load(f)
        except:
            continue

        info_type_1 = cfg.get('infosphere_type')
        info_params_1 = cfg.get('infosphere_parameters')
        info_type_2 = cfg.get('infosphere_type_2')
        info_params_2 = cfg.get('infosphere_parameters_2')
        mix_ratio = cfg.get('infosphere_mix_ratio', 0.0)
        drop_percentage = cfg.get('drop_percentage', 0.0)

        env_desc = f"Env_{info_type_1}+{info_type_2}_mix{mix_ratio}"
        
        # --- A. Costruiamo l'ambiente pulito ---
        current_data = base_data.clone()
        
        try:
            anp_utils.anp_add_infosphere_mix(
                current_data, 
                info_type_1, info_params_1,
                info_type_2, info_params_2,
                mix_ratio, drop_percentage, 
                ROOT_DATA, DEVICE, YEAR
            )
        except Exception as e:
            print(f"Error building infosphere for {mix_model_name}: {e}")
            continue

        current_data = ensure_structural_compatibility(current_data, DEVICE)
        current_data = T.ToUndirected()(current_data)
        
        if ('author', 'co_author', 'author') in current_data.edge_types:
            del current_data['author', 'co_author', 'author']

        val_loader = LinkNeighborLoader(
            data=current_data, 
            num_neighbors=[30, 10], 
            edge_label_index=(('author', 'co_author', 'author'), test_edges),
            batch_size=BATCH_SIZE, 
            shuffle=False 
        )

        # 5. INNER LOOP: Eseguiamo i Modelli in questo Ambiente
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
                
                # Salviamo le info del modello solo una volta
                if model_name not in model_configs:
                    with open(os.path.join(model_path, 'info.json'), 'r') as f: m_cfg = json.load(f)
                    model_configs[model_name] = f"Single: {m_cfg.get('infosphere_type')}"

                # Inferenza
                preds = get_predictions(model, val_loader, DEVICE, allowed_edge_types=expected_edges)
                
                # Registriamo i risultati globalmente
                if model_name not in global_predictions:
                    global_predictions[model_name] = {}
                global_predictions[model_name][env_desc] = preds
                
                del model, weights, state_dict, model_init_data

            except Exception as e:
                print(f"Error evaluating {model_name} on {env_desc}: {e}")

        del current_data, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    # --- 6. FASE DI CONFRONTO (Il tuo "Swap" post-inferenza) ---
    print("\nAll inferences completed. Calculating cross-infosphere consistency...")
    
    all_results_list = []
    
    for model_name, env_preds_dict in global_predictions.items():
        env_names = list(env_preds_dict.keys())
        if len(env_names) < 2:
            continue
            
        # Generiamo le coppie di AMBIENTI (invece che di modelli)
        pairs = list(itertools.combinations(env_names, 2))
        model_info = model_configs.get(model_name, "Unknown")
        
        for env_a, env_b in pairs:
            p_a = env_preds_dict[env_a].numpy()
            p_b = env_preds_dict[env_b].numpy()
            
            if len(p_a) == 0 or len(p_b) == 0: continue
            
            # Calcolo delle metriche per lo STESSO modello tra DUE AMBIENTI DIVERSI
            corr, _ = pearsonr(p_a, p_b)
            mse = np.mean((p_a - p_b)**2)
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

    # --- 7. SALVATAGGIO CSV ---
    if all_results_list:
        df = pd.DataFrame(all_results_list)
        # Ordiniamo prima per modello e poi per la correlazione
        df.sort_values(by=["model_evaluated", "pearson_correlation"], ascending=[True, False], inplace=True)
        
        out_file = os.path.join(OUTPUT_DIR, "consistency_cross_environments.csv")
        df.to_csv(out_file, index=False)
        print(f"\nSaved aggregated cross-environment results to {out_file}")
        
    print(f"Processing Complete. All results saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    run_cross_infosphere_analysis()