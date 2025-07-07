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
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.utils import coalesce
from tqdm import tqdm

# Constants
BATCH_SIZE = 4096
YEAR = 2019
ROOT = "../anp_data"
DEVICE = torch.device(f'cuda:{sys.argv[7]}' if torch.cuda.is_available() else 'cpu')
from academic_network_project.anp_core.anp_dataset import ANPDataset
from academic_network_project.anp_core.anp_utils import *

# Get command line arguments
learning_rate = float(sys.argv[1])
infosphere_type = int(sys.argv[2])
infosphere_parameters = sys.argv[3]
only_new = sys.argv[4].lower() == 'true'
edge_number = int(sys.argv[5])
drop_percentage = float(sys.argv[6])

# Current timestamp for model saving
current_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
PATH = f"../anp_models/{os.path.basename(sys.argv[0][:-3])}_{infosphere_type}_{infosphere_parameters}_{only_new}_{edge_number}_{drop_percentage}_{current_date}/"
os.makedirs(PATH)
with open(PATH + 'info.json', 'w') as json_file:
    json.dump({'lr': learning_rate, 'infosphere_type': infosphere_type, 'infosphere_parameters': infosphere_parameters,
               'only_new': only_new, 'edge_number': edge_number, 'drop_percentage': drop_percentage, 'data': []}, json_file)


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
        data['author', 'infosphere', 'paper'].edge_index = coalesce(infosphere_edge)
        data['author', 'infosphere', 'paper'].edge_label = None

    elif infosphere_type == 3:
        infosphere_parameterss = infosphere_parameters.strip()
        arg_list = ast.literal_eval(infosphere_parameterss)
        if os.path.exists(f"{ROOT}/processed/edge_infosphere_3_{arg_list[0]}_{arg_list[1]}.pt"):
            print("Infosphere 3 edge found!")
            data['author', 'infosphere', 'paper'].edge_index = torch.load(f"{ROOT}/processed/edge_infosphere_3_{arg_list[0]}_{arg_list[1]}.pt", map_location=DEVICE)
            data['author', 'infosphere', 'paper'].edge_label = None
        else:
            print("Generating infosphere 3 edge...")
            infosphere_edge = create_infosphere_top_papers_per_topic_edge_index(data, arg_list[0], arg_list[1], YEAR)
            data['author', 'infosphere', 'paper'].edge_index = coalesce(infosphere_edge)
            data['author', 'infosphere', 'paper'].edge_label = None
            torch.save(data['author', 'infosphere', 'paper'].edge_index, f"{ROOT}/processed/edge_infosphere_3_{arg_list[0]}_{arg_list[1]}.pt")

       
        infosphere_edge = create_infosphere_top_papers_per_topic_edge_index(data, arg_list[0], arg_list[1], YEAR)
        data['author', 'infosphere', 'paper'].edge_index = coalesce(infosphere_edge)
        data['author', 'infosphere', 'paper'].edge_label = None

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

sub_graph_train = anp_simple_filter_data(data, root=ROOT, folds=[0, 1, 2, 3], max_year=YEAR)
train_loader = NeighborLoader(
    data=sub_graph_train,
    num_neighbors=[edge_number, -1],
    batch_size=BATCH_SIZE,
    input_nodes=('author')
)

sub_graph_val = anp_simple_filter_data(data, root=ROOT, folds=[4], max_year=YEAR)
val_loader = NeighborLoader(
    data=sub_graph_val,
    num_neighbors=[edge_number, -1],
    batch_size=BATCH_SIZE,
    input_nodes=('author')
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
model = Model(hidden_channels=64).to(DEVICE)
# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Negative Binomial Loss Function
def negative_binomial_loss(predicted_mean, predicted_dispersion, true_counts):
    predicted_mean = torch.clamp(predicted_mean, min=1e-6)  # Prevent log(0)
    predicted_dispersion = torch.clamp(predicted_dispersion, min=1e-6)  # Prevent log(0)
    term1 = torch.lgamma(true_counts + predicted_dispersion) - torch.lgamma(predicted_dispersion) - torch.lgamma(true_counts + 1)
    term2 = predicted_dispersion * (torch.log(predicted_dispersion) - torch.log(predicted_dispersion + predicted_mean))
    term3 = true_counts * (torch.log(predicted_mean) - torch.log(predicted_dispersion + predicted_mean))
    return -(term1 + term2 + term3).mean()

# Training and Testing Functions
def train():
    model.train()
    total_examples = total_loss = 0
    for batch in tqdm(train_loader):
        batch = batch.to(DEVICE)
        target = batch['author'].y[0:BATCH_SIZE].float() # Number of co-authors
        
        # Add node embeddings for message passing
        batch['author'].x = model.embedding_author(batch['author'].n_id)
        batch['topic'].x = model.embedding_topic(batch['topic'].n_id)

        optimizer.zero_grad()
        pred = model(batch.x_dict, batch.edge_index_dict, range(BATCH_SIZE))
        mean_pred, dispersion_pred = pred
        loss = negative_binomial_loss(mean_pred, dispersion_pred, target)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * BATCH_SIZE
        total_examples += BATCH_SIZE

    return total_loss / total_examples

@torch.no_grad()
def test(loader):
    model.eval()
    total_examples = total_loss = 0
    for batch in tqdm(loader):
        batch = batch.to(DEVICE)
        target = batch['author'].y[0:BATCH_SIZE].float() # Number of co-authors
        
        # Add node embeddings for message passing
        batch['author'].x = model.embedding_author(batch['author'].n_id)
        batch['topic'].x = model.embedding_topic(batch['topic'].n_id)

        pred = model(batch.x_dict, batch.edge_index_dict, range(BATCH_SIZE))
        mean_pred, dispersion_pred = pred
        loss = negative_binomial_loss(mean_pred, dispersion_pred, target)

        total_loss += float(loss) * BATCH_SIZE
        total_examples += BATCH_SIZE

    return total_loss / total_examples


# Main Training Loop
training_loss_list = []
validation_loss_list = []
training_accuracy_list = []
validation_accuracy_list = []
best_val_loss = np.inf
patience = 3
counter = 0

# Training Loop
for epoch in range(1, 100):
    train_loss = train()
    confusion_matrix = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    val_loss = test(val_loader)

    # Save the model if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        anp_save(model, PATH, epoch, train_loss, val_loss, 0)
        counter = 0  # Reset the counter if validation loss improves
    else:
        counter += 1
        if counter >= 5: 
            lr_scheduler.step(val_loss)

    # Early stopping check
    if counter >= patience and epoch >= 5:
        print(f'Early stopping at epoch {epoch}.')
        break

    training_loss_list.append(train_loss)
    validation_loss_list.append(val_loss)

    # Print epoch results
    print(f'Epoch: {epoch:02d}, Loss: {train_loss:.4f} - {val_loss:.4f}')

generate_graph(PATH, training_loss_list, validation_loss_list, training_accuracy_list, validation_accuracy_list,
               confusion_matrix)
