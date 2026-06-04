import json
import os
import sys
import ast
from datetime import datetime

import numpy as np
import torch
import torch_geometric.transforms as T
from tqdm import tqdm

# Project-specific imports (based on the second script)
from academic_network_project.anp_core.anp_dataset import ANPDataset
import academic_network_project.anp_core.anp_utils as anp_utils
import academic_network_project.anp_nn.models_2025 as models_2025

# Constants
BATCH_SIZE = 4096
YEAR = 2019
ROOT = "../anp_data"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Get command line arguments
infosphere_type = int(sys.argv[1])
infosphere_parameters = sys.argv[2]
drop_percentage = 0.0

PATH = f"../anp_models_dropout_study_hgt2025_d00_30/anp_link_prediction_co_author_hgt_1_5_False_30_0.0_2026_03_25_21_21_36/"
if infosphere_type == 0:
    PATH = f"../anp_models_dropout_study_hgt2025_d00_30/anp_link_prediction_co_author_hgt_0_0_False_30_0.0_2026_03_25_16_41_14/"

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
            init_data[etype].edge_index = full_data[etype].edge_index.to(torch.long)
        else:
            init_data[etype].edge_index = torch.empty((2, 0), dtype=torch.long)
    return init_data

dataset = ANPDataset(root=ROOT)
data = dataset[0]

if infosphere_type != 0 and infosphere_type is not None:
    print(f"Adding infosphere {infosphere_type}...")
    anp_utils.anp_add_infosphere(
        data, 
        infosphere_type, 
        infosphere_parameters, 
        drop_percentage, 
        ROOT, 
        DEVICE, 
        YEAR
    )

# Convert paper features to float
data['paper'].x = data['paper'].x.to(torch.float)

print("Sanitizing graph data and checking node constraints...")
for etype in data.edge_types:
    data[etype].edge_index = data[etype].edge_index.to(torch.long)
    
    mask = (data[etype].edge_index[0] >= 0) & (data[etype].edge_index[1] >= 0)
    
    src_type, _, dst_type = etype
    if hasattr(data[src_type], 'num_nodes'):
        mask &= (data[etype].edge_index[0] < data[src_type].num_nodes)
    if hasattr(data[dst_type], 'num_nodes'):
        mask &= (data[etype].edge_index[1] < data[dst_type].num_nodes)
        
    if not mask.all():
        invalid_count = (~mask).sum().item()
        print(f"Warning: Found {invalid_count} invalid indices in edge type {etype}. Filtering them out.")
        data[etype].edge_index = data[etype].edge_index[:, mask]

# Make the graph undirected
data = T.ToUndirected()(data)

DEVICE_INFERENCE = torch.device("cpu")
data = data.to(DEVICE_INFERENCE)

# Load the model weights
model_path = '../anp_models_dropout_study_hgt2025_d00_30/anp_link_prediction_co_author_hgt_1_5_False_30_0.0_2026_03_25_21_21_36/model.pt'
print("Loading the model...")

loaded = torch.load(model_path, map_location=DEVICE_INFERENCE)
if hasattr(loaded, 'state_dict'):
    state_dict = loaded.state_dict()
else:
    state_dict = loaded

expected_edges = extract_edge_types_from_state_dict(state_dict)
model_init_data = create_model_init_data(data, expected_edges)

model = models_2025.Model(hidden_channels=32, data=model_init_data).to(DEVICE_INFERENCE)
model.load_state_dict(state_dict)
model.eval()

print("Generating embeddings...")
with torch.no_grad():
    author_nodes = torch.arange(data['author'].num_nodes, device=DEVICE_INFERENCE)
    data['author'].x = model.embedding_author(author_nodes)
    
    topic_nodes = torch.arange(data['topic'].num_nodes, device=DEVICE_INFERENCE)
    data['topic'].x = model.embedding_topic(topic_nodes)
    
    clean_edge_index_dict = {}
    allowed_set = set(expected_edges)
    for etype in allowed_set:
        if etype in data.edge_index_dict:
            clean_edge_index_dict[etype] = data.edge_index_dict[etype].to(torch.long)
        else:
            clean_edge_index_dict[etype] = torch.empty((2, 0), dtype=torch.long, device=DEVICE_INFERENCE)
    
    z_dict = model.encoder(data.x_dict, clean_edge_index_dict)
    author_embeddings = z_dict['author'].cpu().numpy()
    
os.makedirs(PATH, exist_ok=True)
author_embeddings_path = os.path.join(PATH, f"author_embeddings_{infosphere_type}_{infosphere_parameters}.npy")
np.save(author_embeddings_path, author_embeddings)

print(f"Author embeddings saved to {author_embeddings_path}")