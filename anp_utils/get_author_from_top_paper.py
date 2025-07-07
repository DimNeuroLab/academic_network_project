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

# Constants
BATCH_SIZE = 4096
YEAR = 2019
NUM = 50
ROOT = "../anp_data"
DEVICE = torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu')
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
df = pd.read_csv("../anp_data/raw/sorted_papers.csv")
popular_papers = set(df[df['year'] <= YEAR]['id'][:NUM].values.tolist())
print(f"Number of popular papers: {len(popular_papers)}")

# Carica i topic e seleziona i NUM più popolari
df_topic = pd.read_csv(f"../anp_data/raw/sorted_authors_topics_{YEAR}.csv")
top_topics = df_topic['topic_id'].value_counts().head(NUM).index.tolist()

# Convert popular papers to a tensor for filtering
popular_papers_tensor = torch.tensor(list(popular_papers), device=DEVICE)

# Get mask of connections in `writes` involving popular papers
writes_edges = data['author', 'writes', 'paper'].edge_index.to(DEVICE)
mask_popular_papers = torch.isin(writes_edges[1], popular_papers_tensor)
author_popular_papers = writes_edges[:, mask_popular_papers][0]

 # Ottieni i co-autori dei co-autori
mask_co_authors = torch.isin(co_author_edges_history[0], author_popular_papers)
co_author_authors = co_author_edges_history[:, mask_co_authors][1]

combined_tensor = torch.cat((author_popular_papers, co_author_authors), dim=0)
unique_authors_tensor_final = torch.unique(combined_tensor, dim=0)

# Salva l'insieme finale su disco come 'co_author_top_paper_NUM.json'
output_path = f"../anp_data/processed/co_author_infosphere/co_author_2_{NUM}_{YEAR}.pt"

# Salva il tensore in un file .pt
torch.save(unique_authors_tensor_final, output_path)

