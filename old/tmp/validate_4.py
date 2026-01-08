import json
import os
import sys
import ast
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Linear
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HGTConv
from torch_geometric.utils import coalesce
from tqdm import tqdm

# Constants
infosphere_type = 1
infosphere_parameters = 5
only_new = False
edge_number = 50
drop_percentage = 0.0
BATCH_SIZE = 4096  # Reduced batch size for memory optimization
YEAR = 2019
ROOT = "../anp_data"
DEVICE = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

from academic_network_project.anp_core.anp_dataset import ANPDataset
from academic_network_project.anp_core.anp_utils import *

# Path to the pre-trained model
model_path = '../anp_models/anp_link_prediction_co_author_hgt_embedding_faster_1_5_True_50_0.0_2024_10_25_13_05_36/model.pt'


# Create ANP dataset
dataset = ANPDataset(root=ROOT)
data = dataset[0]

# Add infosphere data if it was used in training
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

# Try to predict all the future co-author or just the new ones
coauthor_function = get_difference_author_edge_year if only_new else get_author_edge_year
coauthor_year = YEAR if only_new else YEAR + 1
coauthor_file = f"{ROOT}/processed/difference_author_edge{coauthor_year}.pt" if only_new \
    else f"{ROOT}/processed/author_edge{coauthor_year}.pt"

# Load co-author edges
if os.path.exists(coauthor_file):
    print("Co-author edge found!")
    data['author', 'co_author', 'author'].edge_index = torch.load(coauthor_file, map_location=DEVICE)["author"]
    data['author', 'co_author', 'author'].edge_label = None
else:
    print("Generating co-author edge...")
    author_edge = coauthor_function(data, coauthor_year, DEVICE)
    data['author', 'co_author', 'author'].edge_index = author_edge["author"]
    data['author', 'co_author', 'author'].edge_label = None
    torch.save(author_edge, coauthor_file)

# Convert paper features to float and make the graph undirected
data['paper'].x = data['paper'].x.to(torch.float)
data = T.ToUndirected()(data)
data = data.to('cpu')

# Define the path for saving/loading the data
cache_path = 'author_data_with_neg.pt'

if os.path.exists(cache_path):
    # Load existing processed data
    author_data = torch.load(cache_path)
    print("Loaded processed data from cache.")
else:
    # Create the transform for negative sampling
    transform_data = T.RandomLinkSplit(
        num_val=0,
        num_test=0,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=True,
        edge_types=('author', 'co_author', 'author')
    )
    
    # Apply transformation to get negative edges
    author_data, _, _ = transform_data(data)
    
    # Update edge_index with edge_label_index
    author_data['author', 'co_author', 'author'].edge_index = author_data['author', 'co_author', 'author'].edge_label_index
    
    # Remove the original edge_label_index
    del author_data['author', 'co_author', 'author'].edge_label_index
    
    # Save the processed data
    torch.save(author_data, cache_path)

# Prepare validation data
val_data = anp_simple_filter_data(author_data, root=ROOT, folds=[4], max_year=YEAR)

# Initialize validation loader
val_loader = NeighborLoader(
    data=val_data,
    num_neighbors=[-1, -1],
    input_nodes='author',
    batch_size=BATCH_SIZE,
    drop_last=False
)

# Delete the co-author edge (data will be used for metadata)
del data['author', 'co_author', 'author']

# Define model components (same as training)
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
        self.decoder = EdgeDecoder(hidden_channels)
        self.embedding_author = torch.nn.Embedding(data["author"].num_nodes, 32)
        self.embedding_topic = torch.nn.Embedding(data["topic"].num_nodes, 32)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)

# Initialize and load the model
model = torch.load(model_path, map_location=DEVICE)
model.eval()

def validate():
    confusion_matrix = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    total_examples = total_correct = total_loss = 0
    
    for batch in tqdm(val_loader):
        batch = batch.to(DEVICE)
        edge_label_index = batch['author', 'author'].edge_index
        edge_label = batch['author', 'author'].edge_label
        del batch['author', 'co_author', 'author']

        # Add node embeddings for message passing
        batch['author'].x = model.embedding_author(batch['author'].n_id)
        batch['topic'].x = model.embedding_topic(batch['topic'].n_id)

        embeddings = model.encoder(batch.x_dict, batch.edge_index_dict)

val_loader_emb = NeighborLoader(
    data=val_data,
    num_neighbors=[-1, -1],
    input_nodes='author',
    batch_size=BATCH_SIZE,
)

# Validation function
@torch.no_grad()
def validate():
    confusion_matrix = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    total_examples = total_correct = total_loss = 0
    
    for batch in tqdm(val_loader_emb):
        batch = batch.to(DEVICE)
        edge_label_index = batch['author', 'author'].edge_index
        edge_label = batch['author', 'author'].edge_label
        del batch['author', 'co_author', 'author']

        embeddings = model.encoder({"author": batch['author'].x}, batch.edge_index_dict)

        pred = model.decoder(embeddings, edge_label_index)
        target = edge_label
        loss = F.binary_cross_entropy_with_logits(pred, target)

        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()

        # Calculate accuracy
        pred = pred.clamp(min=0, max=1)
        total_correct += int((torch.round(pred, decimals=0) == target).sum())

        # Update confusion matrix
        for i in range(len(target)):
            if target[i].item() == 0:
                if torch.round(pred, decimals=0)[i].item() == 0:
                    confusion_matrix['tn'] += 1
                else:
                    confusion_matrix['fp'] += 1
            else:
                if torch.round(pred, decimals=0)[i].item() == 1:
                    confusion_matrix['tp'] += 1
                else:
                    confusion_matrix['fn'] += 1

    accuracy = total_correct / total_examples
    loss = total_loss / total_examples
    
    # Calculate additional metrics
    precision = confusion_matrix['tp'] / (confusion_matrix['tp'] + confusion_matrix['fp']) if (confusion_matrix['tp'] + confusion_matrix['fp']) > 0 else 0
    recall = confusion_matrix['tp'] / (confusion_matrix['tp'] + confusion_matrix['fn']) if (confusion_matrix['tp'] + confusion_matrix['fn']) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'loss': loss,
        'confusion_matrix': confusion_matrix,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Run validation and print results
results = validate()
print("\nValidation Results:")
print(f"Accuracy: {results['accuracy']:.4f}")
print(f"Loss: {results['loss']:.4f}")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall: {results['recall']:.4f}")
print(f"F1 Score: {results['f1']:.4f}")
print("\nConfusion Matrix:")
print(f"True Positives: {results['confusion_matrix']['tp']}")
print(f"False Positives: {results['confusion_matrix']['fp']}")
print(f"True Negatives: {results['confusion_matrix']['tn']}")
print(f"False Negatives: {results['confusion_matrix']['fn']}")