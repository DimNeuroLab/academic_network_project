import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.loader import LinkNeighborLoader
from tqdm import tqdm

# Import specifici del progetto
from academic_network_project.anp_core.anp_dataset import ANPDataset
from academic_network_project.anp_core.anp_utils import *
import academic_network_project.anp_nn.models_2025 as models_2025

# Constants
BATCH_SIZE = 4096
YEAR = 2019
ROOT = "../anp_data"
DEVICE = torch.device(f'cuda:{sys.argv[7]}' if torch.cuda.is_available() else 'cpu')

# Get command line arguments
learning_rate = float(sys.argv[1])
infosphere_type = int(sys.argv[2])
infosphere_parameters = sys.argv[3]
only_new = sys.argv[4].lower() == 'true'
edge_number = int(sys.argv[5])
drop_percentage = float(sys.argv[6])
embedding_hindsight = sys.argv[8].lower() == 'true'
sorted_connection = sys.argv[9].lower() == 'true'
GT_infosphere_type = int(sys.argv[10])
GT_infosphere_parameters = sys.argv[11]

string_add = ""
if embedding_hindsight:
    string_add += "_embedding_hindsight"
if sorted_connection:
    string_add += "_sorted"

# Setup path and save config
current_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
PATH = f"../anp_models_50_history_rnu/{os.path.basename(sys.argv[0][:-3])}_{infosphere_type}_{infosphere_parameters}_{only_new}_{edge_number}_{drop_percentage}_{current_date}/"
os.makedirs(PATH, exist_ok=True)

with open(os.path.join(PATH, 'info.json'), 'w') as json_file:
    json.dump({
        'lr': learning_rate, 
        'infosphere_type': infosphere_type, 
        'infosphere_parameters': infosphere_parameters,
        'only_new': only_new, 
        'edge_number': edge_number, 
        'drop_percentage': drop_percentage, 
        'GT_infosphere_type': GT_infosphere_type,
        'GT_infosphere_parameters': GT_infosphere_parameters, 
        'data': []
    }, json_file)

# Create ANP dataset
print("Loading Dataset...")
dataset = ANPDataset(root=ROOT)
data = dataset[0]

# Add infosphere to the dataset
anp_add_infosphere(data=data, infosphere_type=infosphere_type, infosphere_parameters=infosphere_parameters, drop_percentage=drop_percentage,
                             root=ROOT, device=DEVICE, year=YEAR)    
     
# Load the generated connections from the previous ensemble script
coauthor_file = f"{ROOT}/processed/rnu_plus_history_gt_edge_index_{GT_infosphere_type}_{GT_infosphere_parameters}_{YEAR}{string_add}.pt" 
if os.path.exists(coauthor_file):
    print(f"Loading Ground Truth da: {coauthor_file}")
    data['author', 'co_author', 'author'].edge_index = torch.load(coauthor_file, map_location=DEVICE)
    data['author', 'co_author', 'author'].edge_label = None
else:
    print(f"ERRORE: File {coauthor_file} non trovato. Assicurati che lo script precedente abbia terminato la generazione.")
    sys.exit(1)

# Convert paper features to float and make the graph undirected
data['paper'].x = data['paper'].x.to(torch.float)
data = T.ToUndirected()(data)
data = data.to('cpu')

# Training Data
print("Splitting dataset...")
sub_graph_train = anp_simple_filter_data(data, root=ROOT, folds=[0, 1, 2, 3], max_year=YEAR)
transform_train = T.RandomLinkSplit(
    num_val=0,
    num_test=0,
    neg_sampling_ratio=1.0,
    add_negative_train_samples=True,
    edge_types=('author', 'co_author', 'author')
)
train_data, _, _ = transform_train(sub_graph_train)

# Validation Data
sub_graph_val = anp_simple_filter_data(data, root=ROOT, folds=[4], max_year=YEAR)
transform_val = T.RandomLinkSplit(
    num_val=0,
    num_test=0,
    neg_sampling_ratio=1.0,
    add_negative_train_samples=True,
    edge_types=('author', 'co_author', 'author')
)
val_data, _, _ = transform_val(sub_graph_val)

# Data Loaders
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[edge_number, 30],
    edge_label_index=(('author', 'co_author', 'author'), train_data['author', 'co_author', 'author'].edge_label_index),
    edge_label=train_data['author', 'co_author', 'author'].edge_label,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[edge_number, 30],
    edge_label_index=(('author', 'co_author', 'author'), val_data['author', 'co_author', 'author'].edge_label_index),
    edge_label=val_data['author', 'co_author', 'author'].edge_label,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# Initialize Model 2025
print("Initializing models_2025 architecture...")
data_for_model = data.clone()
if ('author', 'co_author', 'author') in data_for_model.edge_types:
    del data_for_model['author', 'co_author', 'author'] # Elimina l'arco target dalla topologia dell'encoder

model = models_2025.Model(hidden_channels=32, data=data_for_model).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_loader), epochs=100, pct_start=0.1, anneal_strategy='cos')

# Training function
def train():
    model.train()
    total_examples = total_correct = total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(DEVICE)
        
        # Corretto: Il Loader posiziona la label nell'edge type ('author', 'co_author', 'author')
        edge_label_index = batch['author', 'co_author', 'author'].edge_label_index
        edge_label = batch['author', 'co_author', 'author'].edge_label
        
        # Previene data leakage: rimuove gli archi co_author dal dizionario passato all'encoder
        del batch['author', 'co_author', 'author']

        batch['author'].x = model.embedding_author(batch['author'].n_id)
        batch['topic'].x = model.embedding_topic(batch['topic'].n_id)

        optimizer.zero_grad()
        pred = model(batch.x_dict, batch.edge_index_dict, edge_label_index)
        target = edge_label
        loss = F.binary_cross_entropy_with_logits(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        lr_scheduler.step()

        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()

        pred = torch.sigmoid(pred)
        total_correct += int((torch.round(pred) == target).sum())

    return total_correct / total_examples, total_loss / total_examples

# Test function
@torch.no_grad()
def test(loader):
    model.eval()
    total_examples = total_correct = total_loss = 0
    for batch in tqdm(loader, desc="Validation"):
        batch = batch.to(DEVICE)
        
        # Corretto: Estrazione dagli archi corretti
        edge_label_index = batch['author', 'co_author', 'author'].edge_label_index
        edge_label = batch['author', 'co_author', 'author'].edge_label
        del batch['author', 'co_author', 'author']

        batch['author'].x = model.embedding_author(batch['author'].n_id)
        batch['topic'].x = model.embedding_topic(batch['topic'].n_id)

        pred = model(batch.x_dict, batch.edge_index_dict, edge_label_index)
        target = edge_label
        loss = F.binary_cross_entropy_with_logits(pred, target)

        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()

        pred = torch.sigmoid(pred)
        total_correct += int((torch.round(pred) == target).sum())

        # Confusion matrix
        for i in range(len(target)):
            if target[i].item() == 0:
                if torch.round(pred, decimals=0)[i].item() == 0:
                    confusion_matrix['tn'] += 1
                else:
                    confusion_matrix['fn'] += 1
            else:
                if torch.round(pred, decimals=0)[i].item() == 1:
                    confusion_matrix['tp'] += 1
                else:
                    confusion_matrix['fp'] += 1

    return total_correct / total_examples, total_loss / total_examples


# Main Training Loop
training_loss_list = []
validation_loss_list = []
training_accuracy_list = []
validation_accuracy_list = []
confusion_matrix = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
best_val_loss = np.inf
patience = 5
counter = 0

print("Starting Training Loop...")
for epoch in range(1, 100):
    train_acc, train_loss = train()
    confusion_matrix = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    val_acc, val_loss = test(val_loader)

    # Save the model if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        anp_save(model, PATH, epoch, train_loss, val_loss, val_acc)
        counter = 0  
    else:
        counter += 1

    # Early stopping check
    if counter >= patience and epoch >= 20:
        print(f'Early stopping at epoch {epoch}.')
        break

    training_loss_list.append(train_loss)
    validation_loss_list.append(val_loss)
    training_accuracy_list.append(train_acc)
    validation_accuracy_list.append(val_acc)

    # Print epoch results
    print(f'Epoch: {epoch:02d}, Loss: {train_loss:.4f} - {val_loss:.4f}, Accuracy: {val_acc:.4f}')

generate_graph(PATH, training_loss_list, validation_loss_list, training_accuracy_list, validation_accuracy_list, confusion_matrix)
print(f"Training complete. Models and graphs saved in {PATH}")