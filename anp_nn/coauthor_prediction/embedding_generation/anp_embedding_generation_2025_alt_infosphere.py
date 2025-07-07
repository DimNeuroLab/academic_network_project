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
ROOT = "../anp_data"
DEVICE = torch.device(f'cuda:{1}' if torch.cuda.is_available() else 'cpu')
from academic_network_project.anp_core.anp_dataset import ANPDataset
from academic_network_project.anp_core.anp_utils import *

# Get command line arguments
infosphere_type = int(sys.argv[1])
infosphere_parameters = sys.argv[2]
drop_percentage = 0

# Current timestamp for model saving
PATH = f"../anp_models/anp_embedding_generation_2025_1_5_False_-1_0.0_2025_04_15_18_25_52/"


# Create ANP dataset
dataset = ANPDataset(root=ROOT)
data = dataset[0]

# Add infosphere data if requested
if infosphere_type != 0:
    if infosphere_type == 1:
        fold = [0, 1, 2, 3, 4]
        fold_string = '_'.join(map(str, fold))
        name_infosphere = f"{infosphere_parameters}_infosphere_{fold_string}_{YEAR}_noisy.pt"

        # Load infosphere
        if os.path.exists(f"{ROOT}/computed_infosphere/{YEAR}/{name_infosphere}"):
            infosphere_edges = torch.load(f"{ROOT}/computed_infosphere/{YEAR}/{name_infosphere}", map_location=DEVICE)
            
             # Drop edges for each type of relationship
            cites_edges = drop_edges(infosphere_edges[CITES], drop_percentage)
            writes_edges = drop_edges(infosphere_edges[WRITES], drop_percentage)
            about_edges = drop_edges(infosphere_edges[ABOUT], drop_percentage)
    
            data['paper', 'infosphere_cites', 'paper'].edge_index = coalesce(cites_edges)
            data['paper', 'infosphere_cites', 'paper'].edge_label = None
            data['author', 'infosphere_writes', 'paper'].edge_index = coalesce(writes_edges)
            data['author', 'infosphere_writes', 'paper'].edge_label = None
            data['paper', 'infosphere_about', 'topic'].edge_index = coalesce(about_edges)
            data['paper', 'infosphere_about', 'topic'].edge_label = None
        else:
            raise Exception(f"{name_infosphere} not found!")
        
    elif infosphere_type == 2:
        infosphere_edge = create_infosphere_top_papers_edge_index(data, int(infosphere_parameters), YEAR)
        data['author', 'infosphere_writes', 'paper'].edge_index = coalesce(infosphere_edge)
        data['author', 'infosphere_writes', 'paper'].edge_label = None

    elif infosphere_type == 3:
        infosphere_parameterss = infosphere_parameters.strip()
        arg_list = ast.literal_eval(infosphere_parameterss)
        if os.path.exists(f"{ROOT}/processed/edge_infosphere_3_{arg_list[0]}_{arg_list[1]}.pt"):
            print("Infosphere 3 edge found!")
            data['author', 'infosphere_writes', 'paper'].edge_index = torch.load(f"{ROOT}/processed/edge_infosphere_3_{arg_list[0]}_{arg_list[1]}.pt", map_location=DEVICE)
            data['author', 'infosphere_writes', 'paper'].edge_label = None
        else:
            print("Generating infosphere 3 edge...")
            infosphere_edge = create_infosphere_top_papers_per_topic_edge_index(data, arg_list[0], arg_list[1], YEAR)
            data['author', 'infosphere_writes', 'paper'].edge_index = coalesce(infosphere_edge)
            data['author', 'infosphere_writes', 'paper'].edge_label = None
            torch.save(data['author', 'infosphere_writes', 'paper'].edge_index, f"{ROOT}/processed/edge_infosphere_3_{arg_list[0]}_{arg_list[1]}.pt")

       
        infosphere_edge = create_infosphere_top_papers_per_topic_edge_index(data, arg_list[0], arg_list[1], YEAR)
        data['author', 'infosphere_writes', 'paper'].edge_index = coalesce(infosphere_edge)
        data['author', 'infosphere_writes', 'paper'].edge_label = None
    
    elif infosphere_type == 4:
        if os.path.exists(f"{ROOT}/processed/rec_edge_5_NAIS.pt"):
            print("Rec edge found!")
            data['author', 'infosphere_writes', 'paper'].edge_index = torch.load(f"{ROOT}/processed/rec_edge_10_NAIS.pt", map_location=DEVICE)
            data['author', 'infosphere_writes', 'paper'].edge_label = None
        else:
            print("Error: Rec edge not found!")
            exit()
    
    elif infosphere_type == 5:
        if os.path.exists(f"{ROOT}/processed/rec_edge_10_LightGCN.pt"):
            print("Rec edge found!")
            data['author', 'infosphere_writes', 'paper'].edge_index = torch.load(f"{ROOT}/processed/rec_edge_10_LightGCN.pt", map_location=DEVICE)
            data['author', 'infosphere_writes', 'paper'].edge_label = None
        else:
            print("Error: Rec edge not found!")
            exit()


# Convert paper features to float and make the graph undirected
data['paper'].x = data['paper'].x.to(torch.float)
data = T.ToUndirected()(data)
data = data.to('cpu')


# Define model components
class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata,
                            num_heads)
            self.convs.append(conv)

        # self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)
    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['author'][row], z_dict['author'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels, 1, 2, data.metadata())
        # self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)
        self.embedding_author = torch.nn.Embedding(data["author"].num_nodes, 32)
        self.embedding_topic = torch.nn.Embedding(data["topic"].num_nodes, 32)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


# Initialize and load the model
model_path = '../anp_models/anp_embedding_generation_2025_1_5_False_-1_0.0_2025_04_15_18_25_52/model.pt'
model = torch.load(model_path, map_location=DEVICE)
model.eval()


# Iterare su tutti gli autori per salvare il loro output dall'encoder
author_embeddings = {}

          
DEVICE = "cpu"
data = data.to(DEVICE)
model.to("cpu")
author_nodes = torch.arange(data['author'].num_nodes, device=DEVICE).to(DEVICE)
with torch.no_grad():
    data['author'].x = model.embedding_author(author_nodes).to(DEVICE)  # Aggiungere embedding autore
    data['topic'].x = model.embedding_topic(torch.arange(data['topic'].num_nodes, device=DEVICE)).to(DEVICE)
    
    z_dict = model.encoder(data.x_dict, data.edge_index_dict)
    author_embeddings = z_dict['author'].cpu().numpy()
    
# Salvataggio delle embedding
author_embeddings_path = os.path.join(PATH, "author_embeddings_" + str(infosphere_type) + "_" + str(infosphere_parameters) + ".npy")
np.save(author_embeddings_path, author_embeddings)

print(f"Author embeddings salvate in {author_embeddings_path}")
