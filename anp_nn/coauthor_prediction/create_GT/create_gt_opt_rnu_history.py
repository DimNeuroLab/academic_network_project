import torch
import pandas as pd
import random
import json
import os
import time
import ast
import sys
from datetime import datetime

import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import coalesce
from tqdm import tqdm
from scipy.sparse import coo_matrix

from academic_network_project.anp_core.anp_dataset import ANPDataset
from academic_network_project.anp_core.anp_utils import *
import academic_network_project.anp_nn.models_2025 as models_2025

BATCH_SIZE = 4096 
YEAR = 2019

infosphere_type = int(sys.argv[1])
infosphere_parameters = sys.argv[2]
only_new = str(sys.argv[3]).lower() == 'true'
sorted_connection = str(sys.argv[4]).lower() == 'true'
embedding_hindsight = str(sys.argv[5]).lower() == 'true'
only_new = False

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

ROOT = "../anp_data"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not only_new:
    if infosphere_type == 1 or embedding_hindsight:
        author_embeddings_path_rnu = f'../anp_models_dropout_study_hgt2025_d00_30/anp_link_prediction_co_author_hgt_1_5_False_30_0.0_2026_03_25_21_21_36/author_embeddings_1_5.npy'
    else:
        author_embeddings_path_rnu = f'../anp_models_dropout_study_hgt2025_d00_30/anp_link_prediction_co_author_hgt_1_5_False_30_0.0_2026_03_25_21_21_36/author_embeddings_{infosphere_type}_{infosphere_parameters}.npy'
else:
    exit(0)

author_embeddings_path_history = f'../anp_models_dropout_study_hgt2025_d00_30/anp_link_prediction_co_author_hgt_0_0_False_30_0.0_2026_03_25_16_41_14/author_embeddings_0_0.npy'

print(f"Loading author embeddings from {author_embeddings_path_rnu}...")
author_embeddings_rnu = np.load(author_embeddings_path_rnu)
author_embeddings_tensor_rnu = torch.tensor(author_embeddings_rnu, dtype=torch.float32)
author_embeddings_dict_rnu = { "author": author_embeddings_tensor_rnu.to(DEVICE) }

print(f"Loading author embeddings from {author_embeddings_path_history}...")
author_embeddings_history = np.load(author_embeddings_path_history)
author_embeddings_tensor_history = torch.tensor(author_embeddings_history, dtype=torch.float32)
author_embeddings_dict_history = { "author": author_embeddings_tensor_history.to(DEVICE) }

# Path per conteggi e candidati
author_edge_counts_path = f'../anp_data/processed/author_edge2020_count.pt'
max_author_degrees = torch.load(author_edge_counts_path, map_location=DEVICE)

co_author_infosphere_path = f'../anp_data/processed/co_author_infosphere/co_author_{infosphere_type}_{infosphere_parameters}_{YEAR}.pt'
if infosphere_type == 1:
    co_author_candidates = torch.load(co_author_infosphere_path, map_location='cpu')
else:
    co_author_candidates_top = torch.load(co_author_infosphere_path, map_location='cpu')
    co_author_history_path = f'../anp_data/processed/co_author_infosphere/co_author_history_{YEAR}.pt'
    co_author_candidates_history = torch.load(co_author_history_path, map_location='cpu')


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

def load_anp_model(path, base_data, device):
    loaded = torch.load(path, map_location=device)
    state_dict = loaded.state_dict() if hasattr(loaded, 'state_dict') else loaded
    expected_edges = extract_edge_types_from_state_dict(state_dict)
    init_data = create_model_init_data(base_data, expected_edges)
    model = models_2025.Model(hidden_channels=32, data=init_data).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

print("Loading data and models...")
dataset = ANPDataset(root=ROOT)
data = dataset[0]
data['paper'].x = data['paper'].x.to(torch.float)
data = T.ToUndirected()(data)

model_path_rnu = '../anp_models_dropout_study_hgt2025_d00_30/anp_link_prediction_co_author_hgt_1_5_False_30_0.0_2026_03_25_21_21_36/model.pt'
model_path_history = '../anp_models_dropout_study_hgt2025_d00_30/anp_link_prediction_co_author_hgt_0_0_False_30_0.0_2026_03_25_16_41_14/model.pt'

print("Loading RNU model...")
model_rnu = load_anp_model(model_path_rnu, data, DEVICE)

print("Loading history model...")
model_history = load_anp_model(model_path_history, data, DEVICE)

num_authors = author_embeddings_rnu.shape[0]

def connection_probability(a, b):
    a_tensor = torch.full((b.shape[0],), a, device=DEVICE)
    combined_tensor = torch.stack((a_tensor, b)).to(DEVICE)
    
    with torch.no_grad(): 
        prob_1 = model_rnu.decoder(author_embeddings_dict_rnu, combined_tensor)
        prob_2 = model_history.decoder(author_embeddings_dict_history, combined_tensor)
    
    return (prob_1 + prob_2) / 2.0


start_time = time.time()

if infosphere_type == 1:
    filtered_authors = [
        author_node for author_node in range(num_authors)
        if max_author_degrees[author_node] != 0 and co_author_candidates[author_node] != []
    ]
elif infosphere_type == 2:
    authors_df = pd.read_csv("../anp_data/processed/relevant_authors_2020.csv")
    filtered_authors = authors_df['author_id'].tolist()
else:
    filtered_authors = [
        author_node for author_node in range(num_authors)
        if max_author_degrees[author_node] != 0 
        and co_author_candidates_top[author_node] not in [None, []] 
        and (not isinstance(co_author_candidates_top[author_node], torch.Tensor) 
             or co_author_candidates_top[author_node].numel() > 0) 
    ]

current_author_degrees = torch.zeros(num_authors, device=DEVICE)
edge_index = torch.empty((2, 0), dtype=torch.long, device=DEVICE)
probability_dict = {}

for j, author_node in enumerate(filtered_authors):
    if current_author_degrees[author_node] >= max_author_degrees[author_node]:
        continue

    if infosphere_type == 1:
        coauthor_candidates = co_author_candidates[author_node].to(DEVICE)
    elif infosphere_type == 2:
        combined_tensor = torch.cat((co_author_candidates_top, co_author_candidates_history[author_node]), dim=0)
        coauthor_candidates = torch.unique(combined_tensor, dim=0).to(DEVICE)
    else:
        combined_tensor = torch.cat((co_author_candidates_top[author_node], co_author_candidates_history[author_node]), dim=0)
        coauthor_candidates = torch.unique(combined_tensor, dim=0).to(DEVICE)

    valid_mask = (coauthor_candidates > author_node) & \
                 (current_author_degrees[coauthor_candidates] < max_author_degrees[coauthor_candidates])
    valid_candidates = coauthor_candidates[valid_mask]

    if valid_candidates.size(0) == 0:
        continue

    prob_mask = torch.tensor([probability_dict.get((author_node, candidate.item()), 0) == 0 for candidate in valid_candidates])
    
    if prob_mask.any():
        new_probs = connection_probability(author_node, valid_candidates[prob_mask]).clamp(min=0, max=1)
        for i, candidate in enumerate(valid_candidates[prob_mask]):
            probability_dict[(author_node, candidate.item())] = new_probs[i].item()

    connections = torch.tensor([probability_dict[(author_node, candidate.item())] for candidate in valid_candidates])
    connected_nodes = valid_candidates[connections > 0.5]

    if sorted_connection:
        sorted_indices = torch.argsort(connections[connections > 0.5], descending=True)
        connected_nodes = connected_nodes[sorted_indices]
    else:
        connected_nodes = connected_nodes[torch.randperm(connected_nodes.size(0))]

    for coauthor_node in connected_nodes:
        edge_index = torch.cat((edge_index, torch.tensor([[author_node], [coauthor_node]], device=DEVICE)), dim=1)
        current_author_degrees[author_node] += 1
        current_author_degrees[coauthor_node] += 1

        if current_author_degrees[author_node] >= max_author_degrees[author_node]:
            break

    if (j - 1) % 1000 == 0:
        elapsed_time = time.time() - start_time
        remaining_time = (elapsed_time / (j + 1)) * (len(filtered_authors) - j - 1)
        print(f"Processed {j + 1}/{len(filtered_authors)} authors. Estimated time remaining: {remaining_time / 60:.2f} minutes")

# Salvataggio
string_add = ""
if embedding_hindsight:
    string_add = "_embedding_hindsight"
if sorted_connection:
    string_add += "_sorted"

path = f'../anp_data/processed/rnu_plus_history_gt_edge_index_{infosphere_type}_{infosphere_parameters}_{YEAR}_{string_add}.pt'
torch.save(edge_index, path)
path = f'../anp_data/processed/rnu_plus_history_gt_probability_dict_{infosphere_type}_{infosphere_parameters}_{YEAR}_{string_add}.pt'
torch.save(probability_dict, path)

print("Matrices saved successfully.")