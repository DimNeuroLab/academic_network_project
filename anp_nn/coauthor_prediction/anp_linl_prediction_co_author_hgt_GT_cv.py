import json
import os
import sys
import ast
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn import Linear, Dropout, BatchNorm1d
from torch_geometric.loader import LinkNeighborLoader
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
GT_infosphere_type = int(sys.argv[8])
GT_infosphere_parameters = sys.argv[9]

# Setup path and save config
current_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
base_path = f"../anp_models/{os.path.basename(sys.argv[0][:-3])}_{infosphere_type}_{infosphere_parameters}_{only_new}_{edge_number}_{drop_percentage}_{current_date}"
os.makedirs(base_path)

# Save config
with open(base_path + '/info.json', 'w') as json_file:
    json.dump({'lr': learning_rate, 'infosphere_type': infosphere_type, 'infosphere_parameters': infosphere_parameters,
               'only_new': only_new, 'edge_number': edge_number, 'drop_percentage': drop_percentage,
               'GT_infosphere_type': GT_infosphere_type, 'GT_infosphere_parameters': GT_infosphere_parameters, 'data': []}, json_file)



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
    
    elif infosphere_type == 4:
        if os.path.exists(f"{ROOT}/processed/rec_edge_10_NAIS.pt"):
            print("Rec edge found!")
            data['author', 'infosphere', 'paper'].edge_index = torch.load(f"{ROOT}/processed/rec_edge_10_NAIS.pt", map_location=DEVICE)
            data['author', 'infosphere', 'paper'].edge_label = None
        else:
            print("Error: Rec edge not found!")
            exit()
    
    elif infosphere_type == 5:
        if os.path.exists(f"{ROOT}/processed/rec_edge_10_LightGCN.pt"):
            print("Rec edge found!")
            data['author', 'infosphere', 'paper'].edge_index = torch.load(f"{ROOT}/processed/rec_edge_10_LightGCN.pt", map_location=DEVICE)
            data['author', 'infosphere', 'paper'].edge_label = None
        else:
            print("Error: Rec edge not found!")
            exit()
            
# Try to predict all the future co-author or just the new one (not present in history)
coauthor_file = f"{ROOT}/processed/gt_edge_index_{GT_infosphere_type}_{GT_infosphere_parameters}_2019_new_3.pt" 
print("Co-author edge found!")
data['author', 'co_author', 'author'].edge_index = torch.load(coauthor_file, map_location=DEVICE)
data['author', 'co_author', 'author'].edge_label = None


# Convert paper features to float and make the graph undirected
data['paper'].x = data['paper'].x.to(torch.float)
data = T.ToUndirected()(data)
data = data.to('cpu')

data_metadata = data.clone()
del data_metadata['author', 'co_author', 'author']

# Model Definitions
class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict({
            node_type: Linear(-1, hidden_channels)
            for node_type in metadata[0]
        })
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HGTConv(hidden_channels, hidden_channels, metadata, num_heads))
            self.norms.append(BatchNorm1d(hidden_channels))

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: self.norms[i](v) for k, v in x_dict.items()}
            x_dict = {k: F.dropout(v, p=0.3, training=self.training) for k, v in x_dict.items()}
        return x_dict

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.dropout = Dropout(p=0.3)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['author'][row], z_dict['author'][col]], dim=-1)
        z = self.dropout(self.lin1(z).relu())
        z = self.lin2(z)
        return z.view(-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels, 1, 2, data_metadata.metadata())
        self.decoder = EdgeDecoder(hidden_channels)
        self.embedding_author = torch.nn.Embedding(data["author"].num_nodes, 32)
        self.embedding_topic = torch.nn.Embedding(data["topic"].num_nodes, 32)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)

# Main cross-validation loop
all_results = {}
for fold in range(4):
    print(f"\nRunning fold {fold}...")
    fold_path = os.path.join(base_path, f"fold_{fold}")
    os.makedirs(fold_path, exist_ok=True)

    # Training Data
    sub_graph_train = anp_simple_filter_data(data, root=ROOT, folds=[fold], max_year=YEAR)
    transform_train = T.RandomLinkSplit(
        num_val=0,
        num_test=0,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=True,
        edge_types=('author', 'co_author', 'author')
    )
    train_data, _, _ = transform_train(sub_graph_train)

    # Define seed edges:
    edge_label_index = train_data['author', 'co_author', 'author'].edge_label_index
    edge_label = train_data['author', 'co_author', 'author'].edge_label
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[edge_number, 30],
        edge_label_index=(('author', 'co_author', 'author'), edge_label_index),
        edge_label=edge_label,
        batch_size=1024,
        shuffle=True,
    )

    val_loader = None
    val_data = None
    test_data = {}
    test_loader = {}

    # Set fold 4 as validation
    val_fold = 4
    if fold != val_fold:
        sub_graph_val = anp_simple_filter_data(data, root=ROOT, folds=[val_fold], max_year=YEAR)
        transform_val = T.RandomLinkSplit(
            num_val=0, num_test=0, neg_sampling_ratio=1.0, add_negative_train_samples=True,
            edge_types=('author', 'co_author', 'author')
        )
        val_data, _, _ = transform_val(sub_graph_val)
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

    for i in range(5):
        if i != fold and i != val_fold:
            sub_graph_test = anp_simple_filter_data(data, root=ROOT, folds=[i], max_year=YEAR)
            transform_test = T.RandomLinkSplit(
                num_val=0, num_test=0, neg_sampling_ratio=1.0, add_negative_train_samples=True,
                edge_types=('author', 'co_author', 'author')
            )
            test_data_t, _, _ = transform_test(sub_graph_test)
            test_data[i] = test_data_t

            edge_label_index = test_data_t['author', 'co_author', 'author'].edge_label_index
            edge_label = test_data_t['author', 'co_author', 'author'].edge_label
            test_loader[i] = LinkNeighborLoader(
                data=test_data_t,
                num_neighbors=[edge_number, 30],
                edge_label_index=(('author', 'co_author', 'author'), edge_label_index),
                edge_label=edge_label,
                batch_size=1024,
                shuffle=False,
            )

    model = Model(hidden_channels=32).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_loader),
                              epochs=100, pct_start=0.1, anneal_strategy='cos')

    best_val_loss = np.inf
    best_model_path = os.path.join(fold_path, "best_model.pt")
    patience = 5
    counter = 0
    val_metrics = []
    best_epoch = 0
    best_val_predictions = []
    best_test_predictions = {}

    for epoch in range(1, 101):
        model.train()
        train_loss_total = 0
        train_correct = 0
        train_total = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} - Training"):
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
            optimizer.step()
            lr_scheduler.step()

            train_loss_total += loss.item() * pred.numel()
            train_correct += (torch.round(torch.sigmoid(pred)) == edge_label).sum().item()
            train_total += pred.numel()

        avg_train_loss = train_loss_total / train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss_total = 0
        val_correct = 0
        val_total = 0
        val_predictions_epoch = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} - Validation"):
                batch = batch.to(DEVICE)
                edge_label_index = batch['author', 'author'].edge_label_index
                edge_label = batch['author', 'author'].edge_label
                del batch['author', 'co_author', 'author']

                batch['author'].x = model.embedding_author(batch['author'].n_id)
                batch['topic'].x = model.embedding_topic(batch['topic'].n_id)

                pred = model(batch.x_dict, batch.edge_index_dict, edge_label_index)
                loss = F.binary_cross_entropy_with_logits(pred, edge_label)
                val_loss_total += loss.item() * pred.numel()
                val_total += pred.numel()
                pred_sigmoid = torch.sigmoid(pred)
                val_correct += (torch.round(pred_sigmoid) == edge_label).sum().item()

                # Store predictions
                for p, l in zip(pred_sigmoid.tolist(), edge_label.tolist()):
                    val_predictions_epoch.append({"pred": p, "label": int(l)})

        avg_val_loss = val_loss_total / val_total
        val_acc = val_correct / val_total

        print(f"Fold {fold} - Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}")

        val_metrics.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            best_val_predictions = val_predictions_epoch
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    # Save validation metrics
    with open(os.path.join(fold_path, 'val_results.json'), 'w') as f:
        json.dump(val_metrics, f)

    # Load best model for test evaluation
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_metrics = {}
    for i in test_loader:
        test_loss_total = 0
        test_correct = 0
        test_total = 0
        test_predictions = []
        with torch.no_grad():
            for batch in tqdm(test_loader[i], desc=f"Fold {fold} - Test Fold {i}"):
                batch = batch.to(DEVICE)
                edge_label_index = batch['author', 'author'].edge_label_index
                edge_label = batch['author', 'author'].edge_label
                del batch['author', 'co_author', 'author']

                batch['author'].x = model.embedding_author(batch['author'].n_id)
                batch['topic'].x = model.embedding_topic(batch['topic'].n_id)

                pred = model(batch.x_dict, batch.edge_index_dict, edge_label_index)
                loss = F.binary_cross_entropy_with_logits(pred, edge_label)
                test_loss_total += loss.item() * pred.numel()
                test_total += pred.numel()
                pred_sigmoid = torch.sigmoid(pred)
                test_correct += (torch.round(pred_sigmoid) == edge_label).sum().item()

                # Store predictions
                for p, l in zip(pred_sigmoid.tolist(), edge_label.tolist()):
                    test_predictions.append({"pred": p, "label": int(l)})

        avg_test_loss = test_loss_total / test_total
        test_acc = test_correct / test_total
        print(f"Fold {fold} - Test Fold {i} - Loss: {avg_test_loss:.4f}, Acc: {test_acc:.4f}")

        # Save metrics
        test_metrics[i] = {
            "test_loss": avg_test_loss,
            "test_acc": test_acc
        }

        best_test_predictions[i] = test_predictions

    # Save best predictions
    with open(os.path.join(fold_path, 'best_val_predictions.json'), 'w') as f:
        json.dump(best_val_predictions, f)
    with open(os.path.join(fold_path, 'best_test_predictions.json'), 'w') as f:
        json.dump(best_test_predictions, f)

    # Save test results summary
    with open(os.path.join(fold_path, 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f)

    all_results[f"fold_{fold}"] = {
        "val_metrics": val_metrics,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss
    }

# Save all fold results
with open(os.path.join(base_path, 'all_results.json'), 'w') as f:
    json.dump(all_results, f)

