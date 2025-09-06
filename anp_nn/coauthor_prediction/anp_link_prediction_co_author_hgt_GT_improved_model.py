import json
import os
import sys
import csv
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from tqdm import tqdm

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

from academic_network_project.anp_core.anp_dataset import ANPDataset
import academic_network_project.anp_core.anp_utils as anp_utils
import academic_network_project.anp_nn.models as models

# Get command line arguments
learning_rate = float(sys.argv[1])
infosphere_type = int(sys.argv[2])
infosphere_parameters = sys.argv[3]
only_new = sys.argv[4].lower() == 'true'
edge_number = int(sys.argv[5])
drop_percentage = float(sys.argv[6])
sorted_flag = sys.argv[8].lower() == 'true'
embedding_hindsight = sys.argv[9].lower() == 'true'
GT_infosphere_type = int(sys.argv[10])
GT_infosphere_parameters = sys.argv[11]

# Construct GT filename suffix
suffix = ""
if embedding_hindsight:
    suffix += "_embedding_hindsight"
if sorted_flag:
    suffix += "_sorted"

# Setup path and save config
current_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
PATH = f"../anp_models/{suffix}_{GT_infosphere_type}_{GT_infosphere_parameters}_{infosphere_type}_{infosphere_parameters}_{current_date}/"
os.makedirs(PATH)
with open(PATH + 'info.json', 'w') as json_file:
    json.dump({'lr': learning_rate, 'infosphere_type': infosphere_type, 'infosphere_parameters': infosphere_parameters,
               'only_new': only_new, 'edge_number': edge_number, 'drop_percentage': drop_percentage, 'GT_infosphere_type': GT_infosphere_type,
               'GT_infosphere_parameters': GT_infosphere_parameters, 'sorted': sorted_flag, 'embedding_hindsight': embedding_hindsight, 'data': []}, json_file)

# Create ANP dataset
dataset = ANPDataset(root=ROOT)
data = dataset[0]

# Add infosphere
anp_utils.anp_add_infosphere(data=data, infosphere_type=infosphere_type, infosphere_parameters=infosphere_parameters, drop_percentage=drop_percentage,
                             root=ROOT, device=DEVICE, year=YEAR)

# Load ground truth co-authors
gt_filename = f"{ROOT}/processed/NEW_gt_edge_index_{GT_infosphere_type}_{GT_infosphere_parameters}_{YEAR}_{suffix}.pt"
data['author', 'co_author', 'author'].edge_index = torch.load(gt_filename, map_location=DEVICE)
data['author', 'co_author', 'author'].edge_label = None

# Convert paper features to float and make graph undirected
data['paper'].x = data['paper'].x.to(torch.float)
data = T.ToUndirected()(data)
data = data.to('cpu')

# Delete the co-author edge (data_copy will be used for data.metadata())
data_copy = data.clone()
del data_copy['author', 'co_author', 'author']

results = []

for fold in range(5):
    # Filter data
    train_folds = [f for f in range(5) if f != fold]
    val_folds = [fold]
    print("Training on folds:", train_folds, "Validation on fold:", val_folds)
    sub_graph_train = anp_utils.anp_simple_filter_data(data, root=ROOT, folds=train_folds, max_year=YEAR)
    sub_graph_val = anp_utils.anp_simple_filter_data(data, root=ROOT, folds=val_folds, max_year=YEAR)

    transform = T.RandomLinkSplit(
        num_val=0,
        num_test=0,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=True,
        edge_types=('author', 'co_author', 'author')
    )
    train_data, _, _ = transform(sub_graph_train)
    val_data, _, _ = transform(sub_graph_val)

    # Loaders
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[edge_number, 30],
        edge_label_index=(('author', 'co_author', 'author'), train_data['author', 'co_author', 'author'].edge_label_index),
        edge_label=train_data['author', 'co_author', 'author'].edge_label,
        batch_size=1024,
        shuffle=True,
    )
    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[edge_number, 30],
        edge_label_index=(('author', 'co_author', 'author'), val_data['author', 'co_author', 'author'].edge_label_index),
        edge_label=val_data['author', 'co_author', 'author'].edge_label,
        batch_size=1024,
        shuffle=False,
    )

    model = models.Model(hidden_channels=32, data=data_copy).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-6
    )


    def train():
        model.train()
        total_correct = total_loss = total_examples = 0
        all_preds, all_labels = [], []
        for batch in tqdm(train_loader):
            batch = batch.to(DEVICE)
            edge_label_index = batch['author', 'author'].edge_label_index
            edge_label = batch['author', 'author'].edge_label
            del batch['author', 'co_author', 'author']

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
            total_correct += int((torch.round(torch.sigmoid(pred)) == edge_label).sum())

            all_preds.append(torch.sigmoid(pred).cpu())
            all_labels.append(edge_label.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        return total_correct / total_examples, total_loss / total_examples, all_preds, all_labels

    @torch.no_grad()
    def test(loader, fold):
        model.eval()
        all_preds, all_labels = [], []
        total_correct = total_loss = total_examples = 0
        for batch in tqdm(loader):
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
            total_correct += int((torch.round(torch.sigmoid(pred)) == edge_label).sum())
            all_preds.append(torch.sigmoid(pred).cpu())
            all_labels.append(edge_label.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        return total_correct / total_examples, total_loss / total_examples, all_preds, all_labels

    best_val_loss = np.inf
    patience = 3
    counter = 0
    training_loss_list = []
    validation_loss_list = []
    training_accuracy_list = []
    validation_accuracy_list = []
    confusion_matrix = np.zeros((2, 2), dtype=int)
    for epoch in range(1, 51):
        train_acc, train_loss, train_preds, train_labels = train()
        val_acc, val_loss, val_preds, val_labels = test(val_loader, fold)
        lr_scheduler.step(val_loss)

        training_loss_list.append(train_loss)
        validation_loss_list.append(val_loss)
        training_accuracy_list.append(train_acc)
        validation_accuracy_list.append(val_acc)

        # Calcola metriche extra
        train_precision, train_recall, train_f1 = anp_utils.compute_metrics(train_labels, train_preds)
        val_precision, val_recall, val_f1 = anp_utils.compute_metrics(val_labels, val_preds)

        print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
              f'Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}, '
              f'Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')
        if val_loss < best_val_loss:
            print("New best model found!")
            best_val_loss = val_loss
            best_value = {
                'val_loss': val_loss,
                'val_acc': val_acc,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1,
                'train_precision': train_precision,
                'train_recall': train_recall,
                'train_f1': train_f1
            }

            # Save labels and predictions
            labels_path = os.path.join(PATH, f"labels_and_predictions_fold_{fold}.pt")
            torch.save({
                'train_labels': train_labels,
                'train_preds': train_preds,
                'val_labels': val_labels,
                'val_preds': val_preds
            }, labels_path)
            counter = 0
        else:
            counter += 1

        if counter >= patience and epoch >= 4:
            break
    
    anp_utils.generate_graph(PATH, training_loss_list, validation_loss_list, training_accuracy_list, validation_accuracy_list, None, fold=fold)
    results.append({
        'fold': fold,
        'last_val_loss': val_loss,
        'last_val_acc': val_acc,
        'last_train_loss': train_loss,
        'last_train_acc': train_acc,
        'last_val_precision': val_precision,
        'last_val_recall': val_recall,
        'last_val_f1': val_f1,
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

# Write results to CSV
csv_path = os.path.join(PATH, "crossval_results.csv")
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
    for row in results:
        writer.writerow(row)
    # Add average row
    avg_loss = sum(r['last_val_loss'] for r in results) / len(results)
    avg_acc = sum(r['last_val_acc'] for r in results) / len(results)
    avg_train_loss = sum(r['last_train_loss'] for r in results) / len(results)
    avg_train_acc = sum(r['last_train_acc'] for r in results) / len(results)
    avg_precision = sum(r['last_val_precision'] for r in results) / len(results)
    avg_recall = sum(r['last_val_recall'] for r in results) / len(results)
    avg_f1 = sum(r['last_val_f1'] for r in results) / len(results)
    avg_train_precision = sum(r['last_train_precision'] for r in results) / len(results)
    avg_train_recall = sum(r['last_train_recall'] for r in results) / len(results)
    avg_train_f1 = sum(r['last_train_f1'] for r in results) / len(results)
    avg_epoch = sum(r['epoch'] for r in results) / len(results)
    avg_best_val_loss = sum(r['best_val_loss'] for r in results) / len(results)
    avg_best_val_acc = sum(r['best_val_acc'] for r in results) / len(results)
    avg_best_train_loss = sum(r['best_train_loss'] for r in results) / len(results)
    avg_best_train_acc = sum(r['best_train_acc'] for r in results) / len(results)
    avg_best_val_precision = sum(r['best_val_precision'] for r in results) / len(results)
    avg_best_val_recall = sum(r['best_val_recall'] for r in results) / len(results)
    avg_best_val_f1 = sum(r['best_val_f1'] for r in results) / len(results)
    avg_best_train_precision = sum(r['best_train_precision'] for r in results) / len(results)
    avg_best_train_recall = sum(r['best_train_recall'] for r in results) / len(results)
    avg_best_train_f1 = sum(r['best_train_f1'] for r in results) / len(results)

    writer.writerow({
        'fold': 'average',
        'last_val_loss': avg_loss,
        'last_val_acc': avg_acc,
        'last_val_precision': avg_precision,
        'last_val_recall': avg_recall,
        'last_val_f1': avg_f1,
        'last_train_loss': avg_train_loss,
        'last_train_acc': avg_train_acc,
        'last_train_precision': avg_train_precision,
        'last_train_recall': avg_train_recall,
        'last_train_f1': avg_train_f1,
        'epoch': avg_epoch,
        'best_val_loss': avg_best_val_loss,
        'best_val_acc': avg_best_val_acc,
        'best_val_precision': avg_best_val_precision,
        'best_val_recall': avg_best_val_recall,
        'best_val_f1': avg_best_val_f1,
        'best_train_loss': avg_best_train_loss,
        'best_train_acc': avg_best_train_acc,
        'best_train_precision': avg_best_train_precision,
        'best_train_recall': avg_best_train_recall,
        'best_train_f1': avg_best_train_f1
    })

