import json
import os
import sys
import ast
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import Linear
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.utils import coalesce
from tqdm import tqdm
import time

# Constants
BATCH_SIZE = 4096
YEAR = 2019
TYPE = "10_NAIS"
ROOT = "../anp_data"
DEVICE = torch.device(f'cuda:{int(sys.argv[2])}' if torch.cuda.is_available() else 'cpu')
from academic_network_project.anp_core.anp_dataset import ANPDataset
from academic_network_project.anp_core.anp_utils import *


# Create ANP dataset
dataset = ANPDataset(root=ROOT)
data = dataset[0]
data = data.to(DEVICE)

import torch
import pandas as pd

path = f'../anp_data/processed/co_author_edge{YEAR}_history.pt'
co_author_edges_history = torch.load(path).to(DEVICE)

# Carica i paper popolari fino al YEAR
author_recbole_path = f"../anp_data/processed/rec_edge_{TYPE}.pt"
author_recbole = torch.load(author_recbole_path).to(DEVICE)

writes_edges = data['author', 'writes', 'paper'].edge_index.to(DEVICE)

# Carica il CSV in un DataFrame
authors_df = pd.read_csv("../anp_data/processed/relevant_authors_2020.csv")

# Converte la colonna 'author_id' in una lista
filtered_authors = authors_df['author_id'].tolist()

list_co_author_edges = []
start_time = time.time()

len_authors = data["author"].num_nodes
for k, author in enumerate(range(data["author"].num_nodes)):
    if author % 1000 == 0:
        elapsed_time = time.time() - start_time  # Tempo trascorso in secondi
        percent_complete = (k / len_authors) * 100
        
        # Calcola il tempo stimato rimanente
        if k > 0:
            time_per_author = elapsed_time / k
            remaining_authors = len_authors - k
            estimated_time_remaining = remaining_authors * time_per_author
            estimated_minutes = estimated_time_remaining / 60
            
            print(f"Progress: {percent_complete:.2f}% completed")
            print(f"Estimated time remaining: {estimated_minutes:.2f} minutes")
        else:
            print(f"Progress: {percent_complete:.2f}% completed")
            
    if author not in filtered_authors:
        list_co_author_edges.append([])
        continue
    
    mask_papers_recommended = torch.isin(author_recbole[0], author)
    recommended_papers = author_recbole[:, mask_papers_recommended][1]
    
    if recommended_papers.size() == torch.Size([0]):
        list_co_author_edges.append([])
        continue
    
    writes_edges = data['author', 'writes', 'paper'].edge_index.to(DEVICE)
    mask_recommended_papers = torch.isin(writes_edges[1], recommended_papers)
    author_recommended_papers = writes_edges[:, mask_recommended_papers][0]
    # print(author_recommended_papers.size())

    # # Ottieni co-autori già presenti nella sua history
    # mask_co_authors = torch.isin(co_author_edges_history[0], author)
    # author_history_tensor = co_author_edges_history[:, mask_co_authors][1]
    
    # combined_tensor = torch.cat((author_recommended_papers, author_history_tensor), dim=0)
    unique_authors_tensor = torch.unique(author_recommended_papers, dim=0)
    
    # Ottieni i co-autori dei co-autori
    mask_co_authors = torch.isin(co_author_edges_history[0], unique_authors_tensor)
    co_author_authors = co_author_edges_history[:, mask_co_authors][1]

    combined_tensor = torch.cat((unique_authors_tensor, co_author_authors), dim=0)
    unique_authors_tensor_final = torch.unique(combined_tensor, dim=0)

    filtered_authors_tensor = unique_authors_tensor_final[unique_authors_tensor_final != author]
    
    list_co_author_edges.append(filtered_authors_tensor)

# Salva l'insieme finale su disco come 'co_author_top_paper_NUM.json'
output_path = f"../anp_data/processed/co_author_infosphere/co_author_{TYPE}_{YEAR}.pt"

# Salva il tensore in un file .pt
torch.save(list_co_author_edges, output_path)

