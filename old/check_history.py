import json
import os
import sys
import ast
from datetime import datetime
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn import Linear, Dropout, BatchNorm1d
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.utils import coalesce
from tqdm import tqdm

# Constants
BATCH_SIZE = 4096
YEAR = 2019
ROOT = "academic_network_project/anp_data"
DEVICE = torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu')
from academic_network_project.anp_core.anp_dataset import ANPDataset
from academic_network_project.anp_core.anp_utils import *

# Paths for co-author history and transformed version
print(f"Loading co-author candidates...")
co_author_history_path = f'{ROOT}/processed/co_author_infosphere/co_author_history_{YEAR}.pt'
history_edges_tensor_path = f'{ROOT}/processed/co_author_infosphere/co_author_history_edges_{YEAR}_edges.pt'
if not os.path.exists(history_edges_tensor_path):
    print("Co-author candidates history not found, creating it...")
    co_author_candidates_history = torch.load(co_author_history_path, map_location=lambda storage, loc: storage)

    # Lista dove raccogliere tutti gli edges (src, dst)
    history_edges = []

    print("Creating edges from co-author candidates history...")
    # Per ogni autore, recupera i co-autori e crea tuple (autore, coautore)
    for author_id, coauthors in enumerate(co_author_candidates_history):
        if author_id % 10000 == 0:
            print(f"Processing author {author_id}/{len(co_author_candidates_history)}...")
        if len(coauthors) > 0:
            author_tensor = torch.full((coauthors.size(0),), author_id, dtype=torch.long)
            edge_pairs = torch.stack([author_tensor, coauthors], dim=1)
            history_edges.append(edge_pairs)

    # Concatena tutto in un unico tensor [N, 2]
    history_edges_tensor = torch.cat(history_edges, dim=0)
    torch.save(history_edges_tensor, history_edges_tensor_path)
else:
    print("Loading edges from co-author candidates history...")
    history_edges_tensor = torch.load(history_edges_tensor_path, map_location=DEVICE)
    print(f"History edges shape: {history_edges_tensor.shape}")

# Converti in set di tuple
history_set = set(map(tuple, history_edges_tensor.tolist()))


configuration_list = [(1, 5, 0), (2, 10, 0), (3, '5_2', 0), (4, 1, 0), (1, 5, 2), (2, 10, 2), (3, '5_2', 2), (4, 1, 2), (2, 10, 3), (3, '5_2', 3), (4, 1, 3)]
for (GT_infosphere_type, GT_infosphere_parameters, SUFFIX_VERSION) in configuration_list:
    print(f"Loading GT edges for infosphere type {GT_infosphere_type} and parameters {GT_infosphere_parameters}...")
    suffix_str = f"_{SUFFIX_VERSION}" if SUFFIX_VERSION > 0 else ""
    coauthor_file = f"{ROOT}/processed/gt_edge_index_{GT_infosphere_type}_{GT_infosphere_parameters}_{YEAR}{suffix_str}.pt" 
    edges = torch.load(coauthor_file, map_location=DEVICE)
    print(f"Edges shape: {edges.shape}")

    edges_set = set(map(tuple, edges.t().tolist()))

    # Calcola intersezione e percentuale
    intersection = edges_set & history_set
    percentage = len(intersection) / len(edges_set) * 100

    print(f"Percentage of edges in co_author_candidates_history: {percentage:.2f}% for infosphere type {GT_infosphere_type} and parameters {GT_infosphere_parameters} with suffix {suffix_str}")
