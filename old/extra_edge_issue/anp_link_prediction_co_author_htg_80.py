import json
import os
import ast
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from academic_network_project.anp_core.anp_dataset import ANPDataset
from academic_network_project.anp_core.anp_utils import *
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.utils import coalesce
from tqdm import tqdm

# Constants
BATCH_SIZE = 4096
YEAR = 2019
ROOT = "../anp_data"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Get command line arguments
learning_rate = float(sys.argv[1])
infosphere_type = int(sys.argv[2])
infosphere_parameters = sys.argv[3]
only_new = sys.argv[4].lower() == 'true'

# Current timestamp for model saving
current_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
PATH = f"../anp_models/{os.path.basename(sys.argv[0][:-3])}_{current_date}/"
os.makedirs(PATH)
with open(PATH + 'info.json', 'w') as json_file:
    json.dump({'lr': learning_rate, 'infosphere_type': infosphere_type, 'infosphere_parameters': infosphere_parameters,
               'only_new': only_new, 'data': []}, json_file)

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
            infosphere_edges = torch.load(f"{ROOT}/computed_infosphere/{YEAR}/{name_infosphere}")
            data['paper', 'infosphere_cites', 'paper'].edge_index = coalesce(infosphere_edges[CITES])
            data['paper', 'infosphere_cites', 'paper'].edge_label = None
            data['author', 'infosphere_writes', 'paper'].edge_index = coalesce(infosphere_edges[WRITES])
            data['author', 'infosphere_writes', 'paper'].edge_label = None
            data['paper', 'infosphere_about', 'topic'].edge_index = coalesce(infosphere_edges[ABOUT])
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
            data['author', 'infosphere', 'paper'].edge_index = torch.load(f"{ROOT}/processed/edge_infosphere_3_{arg_list[0]}_{arg_list[1]}.pt")
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
    data['author', 'co_author', 'author'].edge_index = torch.load(coauthor_file)
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

# Training Data
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

# Define seed edges:
edge_label_index = train_data['author', 'co_author', 'author'].edge_label_index
edge_label = train_data['author', 'co_author', 'author'].edge_label
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[80, 10],
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
    num_neighbors=[80, 10],
    edge_label_index=(('author', 'co_author', 'author'), edge_label_index),
    edge_label=edge_label,
    batch_size=1024,
    shuffle=False,
)

# Delete the co-author edge (data will be used for data.metadata())
del data['author', 'co_author', 'author']


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
        self.encoder = GNNEncoder(hidden_channels, hidden_channels, 2, 1, data.metadata())
        # self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


# Initialize model, optimizer, and embeddings
model = Model(hidden_channels=32).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
embedding_author = torch.nn.Embedding(data["author"].num_nodes, 32).to(DEVICE)
embedding_topic = torch.nn.Embedding(data["topic"].num_nodes, 32).to(DEVICE)


# Training and Testing Functions
def train():
    model.train()
    total_examples = total_correct = total_loss = 0
    for i, batch in enumerate(tqdm(train_loader)):
        batch = batch.to(DEVICE)
        edge_label_index = batch['author', 'author'].edge_label_index
        edge_label = batch['author', 'author'].edge_label
        del batch['author', 'co_author', 'author']

        # Add node embeddings for message passing
        batch['author'].x = embedding_author(batch['author'].n_id)
        batch['topic'].x = embedding_topic(batch['topic'].n_id)

        optimizer.zero_grad()
        pred = model(batch.x_dict, batch.edge_index_dict, edge_label_index)
        target = edge_label
        loss = F.binary_cross_entropy_with_logits(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()

        # Calculate accuracy
        pred = pred.clamp(min=0, max=1)
        total_correct += int((torch.round(pred, decimals=0) == target).sum())

    return total_correct / total_examples, total_loss / total_examples


@torch.no_grad()
def test(loader):
    model.eval()
    total_examples = total_correct = total_loss = 0
    for i, batch in enumerate(tqdm(loader)):
        batch = batch.to(DEVICE)
        edge_label_index = batch['author', 'author'].edge_label_index
        edge_label = batch['author', 'author'].edge_label
        del batch['author', 'co_author', 'author']

        # Add node embeddings for message passing
        batch['author'].x = embedding_author(batch['author'].n_id)
        batch['topic'].x = embedding_topic(batch['topic'].n_id)

        pred = model(batch.x_dict, batch.edge_index_dict, edge_label_index)
        target = edge_label
        loss = F.binary_cross_entropy_with_logits(pred, target)

        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()

        # Calculate accuracy
        pred = pred.clamp(min=0, max=1)
        total_correct += int((torch.round(pred, decimals=0) == target).sum())

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
patience = 10
counter = 0

# Training Loop
for epoch in range(1, 500):
    train_acc, train_loss = train()
    confusion_matrix = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    val_acc, val_loss = test(val_loader)

    # Save the model if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        anp_save(model, PATH, epoch, train_loss, val_loss, val_acc)
        counter = 0  # Reset the counter if validation loss improves
    else:
        counter += 1

    # Early stopping check
    if counter >= patience:
        print(f'Early stopping at epoch {epoch}.')
        break

    training_loss_list.append(train_loss)
    validation_loss_list.append(val_loss)
    training_accuracy_list.append(train_acc)
    validation_accuracy_list.append(val_acc)

    # Print epoch results
    print(f'Epoch: {epoch:02d}, Loss: {train_loss:.4f} - {val_loss:.4f}, Accuracy: {val_acc:.4f}')

generate_graph(PATH, training_loss_list, validation_loss_list, training_accuracy_list, validation_accuracy_list,
               confusion_matrix)
