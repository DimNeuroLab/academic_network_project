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
ROOT = "../anp_data"
DEVICE = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
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

infosphere_list = []
# Carica i paper popolari fino al YEAR
for number in range(10):
    infosphere_path = f"../anp_data/computed_infosphere/{YEAR}/infosphere_0_1_2_3_4_{YEAR}_{number}.json"
    infosphere_list.extend(json.load(open(infosphere_path)))
 

list_co_author_edges = []
start_time = time.time()  # Registra l'ora di inizio

for author in range(data["author"].num_nodes):
    if author % 10000 == 0:
        elapsed_time = time.time() - start_time  # Tempo trascorso in secondi
        percent_complete = (author / data["author"].num_nodes) * 100
        
        # Calcola il tempo stimato rimanente
        if author > 0:
            time_per_author = elapsed_time / author
            remaining_authors = data["author"].num_nodes - author
            estimated_time_remaining = remaining_authors * time_per_author
            estimated_minutes = estimated_time_remaining / 60
            
            print(f"Progress: {percent_complete:.2f}% completed")
            print(f"Estimated time remaining: {estimated_minutes:.2f} minutes")
        else:
            print(f"Progress: {percent_complete:.2f}% completed")
        
    author_list = []
    if infosphere_list[author] == []:
        list_co_author_edges.append([])
        continue
    else:
        for path_infosphere in infosphere_list[author]:
            for elem in path_infosphere:
                if elem[0] == 'writes':
                    # Aggiungi l'autore alla lista
                    author_list.append(elem[1][0])

    # Converti la lista in un tensore
    author_infosphere_tensor = torch.tensor(author_list).to(DEVICE)

    # Ottieni co-autori già presenti nella sua history
    mask_co_authors = torch.isin(co_author_edges_history[0], author)
    author_history_tensor = co_author_edges_history[:, mask_co_authors][1]

    combined_tensor = torch.cat((author_infosphere_tensor, author_history_tensor), dim=0)
    unique_authors_tensor = torch.unique(combined_tensor, dim=0)

    # Ottieni i co-autori dei co-autori
    mask_co_authors = torch.isin(co_author_edges_history[0], unique_authors_tensor)
    co_author_authors = co_author_edges_history[:, mask_co_authors][1]

    combined_tensor = torch.cat((unique_authors_tensor, co_author_authors), dim=0)
    unique_authors_tensor_final = torch.unique(combined_tensor, dim=0)

    filtered_authors_tensor = unique_authors_tensor_final[unique_authors_tensor_final != author]
    
    list_co_author_edges.append(filtered_authors_tensor)

# Salva l'insieme finale su disco come 'co_author_top_paper_NUM.json'
output_path = f"../anp_data/processed/co_author_infosphere_{YEAR}.pt"

# Salva il tensore in un file .pt
torch.save(list_co_author_edges, output_path)

# Converti il tensore in una lista per la serializzazione JSON
filtered_authors_list = [tensore for tensore in list_co_author_edges]

output_path = f"../anp_data/processed/co_author_infosphere_{YEAR}.json"

# Salva la lista in un file JSON
with open(output_path, 'w') as json_file:
    json.dump(filtered_authors_list, json_file)
