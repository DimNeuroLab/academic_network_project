import json
import os
import sys
import csv
import random
from datetime import datetime
import uuid

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# Experiment UUID
EXPERIMENT_ID = str(uuid.uuid4())

# Constants
BATCH_SIZE = 4096
YEAR = 2019
ROOT = "../anp_data"
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device(f'cuda:{sys.argv[7]}' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available() and 'cuda' in DEVICE.type:
    torch.cuda.set_device(DEVICE)

from academic_network_project.anp_core.anp_dataset import ANPDataset
import academic_network_project.anp_core.anp_utils as anp_utils
import academic_network_project.anp_nn.models as models
from academic_network_project.anp_utils.db_logger import log_experiment, log_epoch

# Get command line arguments
learning_rate = float(sys.argv[1])
infosphere_type = int(sys.argv[2])
infosphere_parameters = sys.argv[3]
only_new = sys.argv[4].lower() == 'true'
edge_number = int(sys.argv[5])
drop_percentage = float(sys.argv[6])
campaign_name = sys.argv[8]

experiment_label = f"{campaign_name}_{infosphere_type}_{infosphere_parameters}"

# Optional arguments with default values
if len(sys.argv) > 11:
    infosphere_type_2 = int(sys.argv[9])
    infosphere_parameters_2 = sys.argv[10]
    infosphere_mix_ratio = float(sys.argv[11])
    print(f"Using second infosphere type: {infosphere_type_2} with parameters {infosphere_parameters_2} and mix ratio {infosphere_mix_ratio}")
    experiment_label += f"_{infosphere_type_2}_{infosphere_parameters_2}_{infosphere_mix_ratio}"
else:
    infosphere_type_2 = None
    infosphere_parameters_2 = None
    infosphere_mix_ratio = None

# Current timestamp for model saving
current_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
PATH = f"../anp_models_{campaign_name}/{os.path.basename(sys.argv[0][:-3])}_{infosphere_type}_{infosphere_parameters}_{only_new}_{edge_number}_{drop_percentage}_{current_date}/"
if infosphere_type_2 is not None:
    PATH = f"../anp_models_{campaign_name}/{os.path.basename(sys.argv[0][:-3])}_{infosphere_type}_{infosphere_parameters}_{infosphere_type_2}_{infosphere_parameters_2}_{infosphere_mix_ratio}_{only_new}_{edge_number}_{drop_percentage}_{current_date}/"
os.makedirs(PATH)
with open(PATH + 'info.json', 'w') as json_file:
    json.dump({'lr': learning_rate, 'infosphere_type': infosphere_type, 'infosphere_parameters': infosphere_parameters,
               'only_new': only_new, 'edge_number': edge_number, 'drop_percentage': drop_percentage,
               'infosphere_type_2': infosphere_type_2, 'infosphere_parameters_2': infosphere_parameters_2, 'infosphere_mix_ratio': infosphere_mix_ratio, 'data': []}, json_file)

print(f"Experiment ID: {EXPERIMENT_ID}")
log_experiment({
    "learning_rate": learning_rate,
    "infosphere_type": infosphere_type,
    "infosphere_parameters": infosphere_parameters,
    "only_new": only_new,
    "edge_number": edge_number,
    "drop_percentage": drop_percentage,
    "infosphere_type_2": infosphere_type_2,
    "infosphere_parameters_2": infosphere_parameters_2,
    "infosphere_mix_ratio": infosphere_mix_ratio,
    "timestamp": current_date
}, campaign=campaign_name, experiment_id=EXPERIMENT_ID, experiment_label=experiment_label)
print(f"Experiment '{experiment_label}' logged.")
# Create ANP dataset
dataset = ANPDataset(root=ROOT)
data = dataset[0]

# Add infosphere
if infosphere_type_2 is None:
    anp_utils.anp_add_infosphere(data=data, infosphere_type=infosphere_type, infosphere_parameters=infosphere_parameters, drop_percentage=drop_percentage,
                                root=ROOT, device=DEVICE, year=YEAR)
else:
    anp_utils.anp_add_infosphere_mix(data=data, infosphere_type=infosphere_type, infosphere_parameters=infosphere_parameters,
                                    infosphere_type_2=infosphere_type_2, infosphere_parameters_2=infosphere_parameters_2,
                                    mix_ratio=infosphere_mix_ratio, drop_percentage=drop_percentage,
                                    root=ROOT, device=DEVICE, year=YEAR)

# Try to predict all the future co-author or just the new one (not present in history)
coauthor_function = anp_utils.get_difference_author_edge_year if only_new else anp_utils.get_author_edge_year
coauthor_year = YEAR if only_new else YEAR + 1
coauthor_file = f"{ROOT}/processed/difference_author_edge{coauthor_year}.pt" if only_new \
    else f"{ROOT}/processed/author_edge{coauthor_year}.pt"

# Use existing co-author edge if available, else generate
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

# Delete the co-author edge (data_copy will be used for data.metadata())
data_copy = data.clone()
del data_copy['author', 'co_author', 'author']


# Training Data
sub_graph_train = anp_utils.anp_simple_filter_data(data, root=ROOT, folds=[0, 1, 2, 3], max_year=YEAR)
transform_train = T.RandomLinkSplit(
    num_val=0,
    num_test=0,
    neg_sampling_ratio=1.0,
    add_negative_train_samples=True,
    edge_types=('author', 'co_author', 'author')
)
train_data, _, _ = transform_train(sub_graph_train)

# Validation Data
sub_graph_val = anp_utils.anp_simple_filter_data(data, root=ROOT, folds=[4], max_year=YEAR)
transform_val = T.RandomLinkSplit(
    num_val=0,
    num_test=0,
    neg_sampling_ratio=1.0,
    add_negative_train_samples=True,
    edge_types=('author', 'co_author', 'author')
)
val_data, _, _ = transform_val(sub_graph_val)

# Define seed edges:
edge_label_index = train_data['author', 'co_author', 'author'].edge_label_index
edge_label = train_data['author', 'co_author', 'author'].edge_label
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[edge_number, 30],
    # neg_sampling_ratio=2.0,
    edge_label_index=(('author', 'co_author', 'author'), edge_label_index),
    edge_label=edge_label,
    batch_size=1024,
    shuffle=True,
)

edge_label_index = val_data['author', 'co_author', 'author'].edge_label_index
edge_label = val_data['author', 'co_author', 'author'].edge_label
val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[edge_number, 30],
    edge_label_index=(('author', 'co_author', 'author'), edge_label_index),
    edge_label=edge_label,
    batch_size=1024,
    shuffle=False,
)

# Delete the co-author edge (data will be used for data.metadata())
del data['author', 'co_author', 'author']


# Initialize model, optimizer, and embeddings
model = models.Model(hidden_channels=32, data=data_copy).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-6
    )

# Training and Testing Functions
def train():
    model.train()
    total_loss = 0
    total_examples = 0
    
    running_cm = np.zeros((2, 2), dtype=int)
    
    all_preds, all_labels = [], []

    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(DEVICE)
        edge_label_index = batch['author', 'author'].edge_label_index
        edge_label = batch['author', 'author'].edge_label
        del batch['author', 'co_author', 'author']

        # Embeddings
        batch['author'].x = model.embedding_author(batch['author'].n_id)
        batch['topic'].x = model.embedding_topic(batch['topic'].n_id)

        optimizer.zero_grad()
        pred = model(batch.x_dict, batch.edge_index_dict, edge_label_index)
        loss = F.binary_cross_entropy_with_logits(pred, edge_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()

        # Calcolo CM per batch
        probs = torch.sigmoid(pred)
        preds_binary = torch.round(probs).long().cpu()
        labels_cpu = edge_label.long().cpu()
        
        batch_cm = confusion_matrix(labels_cpu, preds_binary, labels=[0, 1])
        running_cm += batch_cm

        # Salviamo le predizioni per la tua funzione compute_metrics
        all_preds.append(probs.detach().cpu())
        all_labels.append(edge_label.detach().cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    avg_loss = total_loss / total_examples
    
    return avg_loss, running_cm, all_preds, all_labels


@torch.no_grad()
def test(loader):
    model.eval()
    total_loss = 0
    total_examples = 0
    
    running_cm = np.zeros((2, 2), dtype=int)
    
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Validation"):
        batch = batch.to(DEVICE)
        edge_label_index = batch['author', 'author'].edge_label_index
        edge_label = batch['author', 'author'].edge_label
        del batch['author', 'co_author', 'author']

        batch['author'].x = model.embedding_author(batch['author'].n_id)
        batch['topic'].x = model.embedding_topic(batch['topic'].n_id)

        pred = model(batch.x_dict, batch.edge_index_dict, edge_label_index)
        loss = F.binary_cross_entropy_with_logits(pred, edge_label)

        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()

        probs = torch.sigmoid(pred)
        preds_binary = torch.round(probs).long().cpu()
        labels_cpu = edge_label.long().cpu()

        batch_cm = confusion_matrix(labels_cpu, preds_binary, labels=[0, 1])
        running_cm += batch_cm

        all_preds.append(probs.cpu())
        all_labels.append(edge_label.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    avg_loss = total_loss / total_examples

    return avg_loss, running_cm, all_preds, all_labels

# Main Training Loop
best_val_loss = np.inf
patience = 5
counter = 0
training_loss_list = []
validation_loss_list = []
training_accuracy_list = []
validation_accuracy_list = []

for epoch in range(1, 100):
    train_loss, train_cm, train_preds, train_labels = train()
    val_loss, val_cm, val_preds, val_labels = test(val_loader)
    
    lr_scheduler.step(val_loss)

    train_acc = np.trace(train_cm) / np.sum(train_cm)
    val_acc = np.trace(val_cm) / np.sum(val_cm)
    train_precision, train_recall, train_f1 = anp_utils.compute_metrics(train_labels, train_preds)
    val_precision, val_recall, val_f1 = anp_utils.compute_metrics(val_labels, val_preds)

    # Logging
    training_loss_list.append(train_loss)
    validation_loss_list.append(val_loss)
    training_accuracy_list.append(train_acc)
    validation_accuracy_list.append(val_acc)

    log_epoch(epoch, {
        "cm_train": train_cm.tolist(),
        "cm_val": val_cm.tolist(),
        "train_loss": train_loss,
        "val_loss": val_loss
    }, experiment_id=EXPERIMENT_ID)

    print(f'Epoch {epoch:03d} | '
          f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
          f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | '
          f'Val F1: {val_f1:.4f}')

    # Save the model if validation loss improves
    if val_loss < best_val_loss:
        print(">>> New best model found!")
        best_val_loss = val_loss
        anp_utils.anp_save(model, PATH, epoch, train_loss, val_loss, val_acc)
        
        best_value = {
            'val_loss': val_loss, 'val_acc': val_acc,
            'val_precision': val_precision, 'val_recall': val_recall, 'val_f1': val_f1,
            'train_loss': train_loss, 'train_acc': train_acc,
            'train_precision': train_precision, 'train_recall': train_recall, 'train_f1': train_f1,
            'val_cm': val_cm.tolist()
        }

        # Save labels and predictions
        torch.save({
            'train_labels': train_labels,
            'train_preds': train_preds,
            'val_labels': val_labels,
            'val_preds': val_preds
        }, os.path.join(PATH, "labels_and_predictions.pt"))
        
        counter = 0 
    else:
        counter += 1

    # Early stopping
    if counter >= patience and epoch >= 10:
        print(f'Early stopping at epoch {epoch}.')
        break

    # Print epoch results
    print(f'Epoch: {epoch:02d}, Loss: {train_loss:.4f} - {val_loss:.4f}, Accuracy: {val_acc:.4f}')
anp_utils.generate_graph(PATH, training_loss_list, validation_loss_list, training_accuracy_list, validation_accuracy_list, None, None)
# Write results to CSV
csv_path = os.path.join(PATH, "results.csv")
with open(csv_path, 'w', newline='') as csvfile:
    fieldnames = [
        'fold',
        'last_val_loss', 'last_val_acc',
        'last_val_precision', 'last_val_recall', 'last_val_f1',
        'last_train_loss', 'last_train_acc',
        'last_train_precision', 'last_train_recall', 'last_train_f1',
        'epoch',
        'best_val_loss', 'best_val_acc',
        'best_val_precision', 'best_val_recall', 'best_val_f1',
        'best_train_loss', 'best_train_acc',
        'best_train_precision', 'best_train_recall', 'best_train_f1'
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({
        'fold': '0-4',
        'last_val_loss': val_loss,
        'last_val_acc': val_acc,
        'last_val_precision': val_precision,
        'last_val_recall': val_recall,
        'last_val_f1': val_f1,
        'last_train_loss': train_loss,
        'last_train_acc': train_acc,
        'last_train_precision': train_precision,
        'last_train_recall': train_recall,
        'last_train_f1': train_f1,
        'epoch': epoch,
        'best_val_loss': best_value['val_loss'],
        'best_val_acc': best_value['val_acc'],
        'best_val_precision': best_value['val_precision'],
        'best_val_recall': best_value['val_recall'],
        'best_val_f1': best_value['val_f1'],
        'best_train_loss': best_value['train_loss'],
        'best_train_acc': best_value['train_acc'],
        'best_train_precision': best_value['train_precision'],
        'best_train_recall': best_value['train_recall'],
        'best_train_f1': best_value['train_f1']
    })
