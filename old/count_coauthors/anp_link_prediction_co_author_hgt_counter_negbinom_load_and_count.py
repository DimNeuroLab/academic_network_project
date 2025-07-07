import json
import csv
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
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.utils import coalesce
from tqdm import tqdm

# Constants
infosphere_type = 4
infosphere_parameters = 1
only_new = False
edge_number = 50
drop_percentage = 0.0
BATCH_SIZE = 4096
YEAR = 2019
ROOT = "../anp_data"
DEVICE = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
from academic_network_project.anp_core.anp_dataset import ANPDataset
from academic_network_project.anp_core.anp_utils import *

# Path to the pre-trained model
model_path = '../anp_models_2_11/anp_link_prediction_co_author_hgt_counter_negbinom_1_5_False_-1_0.0_2024_10_27_14_35_46/model.pt'


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


# Try to predict all the future co-author or just the new one (not present in history)
coauthor_function = get_difference_author_edge_year if only_new else get_author_edge_year
coauthor_year = YEAR if only_new else YEAR + 1
coauthor_file = f"{ROOT}/processed/difference_author_edge{coauthor_year}.pt" if only_new \
    else f"{ROOT}/processed/author_edge{coauthor_year}.pt"

# Use existing co-author edge if available, else generate
if os.path.exists(coauthor_file):
    print("Co-author edge found!")
    edge_index = torch.load(coauthor_file, map_location=DEVICE)["author"]
else:
    print("Generating co-author edge...")
    author_edge = coauthor_function(data, coauthor_year, DEVICE)
    edge_index = author_edge["author"]
    torch.save(author_edge, coauthor_file)

count_file = f"{ROOT}/processed/difference_author_edge{coauthor_year}_count.pt" if only_new \
    else f"{ROOT}/processed/author_edge{coauthor_year}_count.pt"

# Use existing co-author edge if available, else generate
if os.path.exists(count_file):
    print("Co-author count found!")
    data['author'].y = torch.load(count_file, map_location=DEVICE)
else:
    print("Generating co-author count...")
    data['author'].y = torch.zeros(data['author'].num_nodes, dtype=torch.long)
    # Count the number of co-author edges for each author
    for author in range(data['author'].num_nodes):
        count = len(edge_index[0][edge_index[0] == author])
        data['author'].y[author] = count
    torch.save(data['author'].y, count_file)

# Convert paper features to float and make the graph undirected
data['paper'].x = data['paper'].x.to(torch.float)
data = T.ToUndirected()(data)
data = data.to('cpu')

autor_data_filtered = anp_simple_filter_data(data, root=ROOT, folds=None, max_year=YEAR)
print(f"Filtered data: {autor_data_filtered}")
author_loader = NeighborLoader(
    data=autor_data_filtered,
    num_neighbors=[100, 50],
    input_nodes='author',
    batch_size=BATCH_SIZE,
    drop_last=False
)
class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata)
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict

class CoauthorCountDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 2)

    def forward(self, z):
        z = self.lin1(z).relu()
        z = self.lin2(z)
        mean_pred, log_dispersion = z[:, 0], z[:, 1]
        mean_pred = torch.nn.functional.softplus(mean_pred)  # Ensure mean is positive
        dispersion_pred = torch.nn.functional.softplus(log_dispersion)  # Ensure dispersion is positive
        return mean_pred, dispersion_pred

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels, 2, 2, data.metadata())
        self.decoder = CoauthorCountDecoder(hidden_channels)
        self.embedding_author = torch.nn.Embedding(data["author"].num_nodes, 32)
        self.embedding_topic = torch.nn.Embedding(data["topic"].num_nodes, 32)

    def forward(self, x_dict, edge_index_dict, input_nodes):
        z_dict = self.encoder(x_dict, edge_index_dict)
        input_node_embeddings = z_dict['author'][input_nodes]
        return self.decoder(input_node_embeddings)

# Initialize model, optimizer, and embeddings
model = torch.load(model_path, map_location=DEVICE)
model.eval()

# Negative Binomial Loss Function
def negative_binomial_loss(predicted_mean, predicted_dispersion, true_counts):
    predicted_mean = torch.clamp(predicted_mean, min=1e-6)  # Prevent log(0)
    predicted_dispersion = torch.clamp(predicted_dispersion, min=1e-6)  # Prevent log(0)
    term1 = torch.lgamma(true_counts + predicted_dispersion) - torch.lgamma(predicted_dispersion) - torch.lgamma(true_counts + 1)
    term2 = predicted_dispersion * (torch.log(predicted_dispersion) - torch.log(predicted_dispersion + predicted_mean))
    term3 = true_counts * (torch.log(predicted_mean) - torch.log(predicted_dispersion + predicted_mean))
    return -(term1 + term2 + term3).mean()


import torch

@torch.no_grad()
def test():
    model.eval()
    total_examples = total_loss = 0
    
    # Liste per accumulare le previsioni
    mean_preds = []
    dispersion_preds = []

    for batch in tqdm(author_loader):
        batch = batch.to(DEVICE)
        num_authors_in_batch = batch['author'].batch_size 
        target = batch['author'].y[0:num_authors_in_batch].float()  # Numero di co-autori

        # Aggiungi le embedding dei nodi per il message passing
        batch['author'].x = model.embedding_author(batch['author'].n_id)
        batch['topic'].x = model.embedding_topic(batch['topic'].n_id)
        
        pred = model(batch.x_dict, batch.edge_index_dict, range(num_authors_in_batch))
        mean_pred, dispersion_pred = pred

        # Aggiungi le previsioni alle liste
        mean_preds.append(mean_pred.cpu())
        dispersion_preds.append(dispersion_pred.cpu())

        # Calcolo della loss
        loss = negative_binomial_loss(mean_pred, dispersion_pred, target)
        total_loss += float(loss) * num_authors_in_batch
        total_examples += num_authors_in_batch

    # Concatena le liste in tensori finali e converti in liste
    mean_preds_list = torch.cat(mean_preds).tolist()
    dispersion_preds_list = torch.cat(dispersion_preds).tolist()

    # Calcola la somma approssimata all'intero più vicino
    sum_rounded = [round(m + d) for m, d in zip(mean_preds_list, dispersion_preds_list)]

    # Salva i dati in un unico file CSV
    name = f'../anp_data/processed/count_{infosphere_type}_{infosphere_parameters}_{YEAR}.csv'
    with open(name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['mean', 'dispersion', 'sum_rounded'])  # Intestazione
        writer.writerows(zip(mean_preds_list, dispersion_preds_list, sum_rounded))

    return total_loss / total_examples, mean_preds_list, dispersion_preds_list


# Main Training Loop
training_loss_list = []
validation_loss_list = []
training_accuracy_list = []
validation_accuracy_list = []
best_val_loss = np.inf
patience = 5
counter = 0

# Training Loop
test()