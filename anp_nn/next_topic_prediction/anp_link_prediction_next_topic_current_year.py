import torch
import os
import torch.nn.functional as F
import torch_geometric.transforms as T
from academic_network_project.anp_core.anp_dataset import ANPDataset
from academic_network_project.anp_core.anp_utils import *
from torch.nn import Linear
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
from tqdm import tqdm
from datetime import datetime

current_date = datetime.now().strftime("%Y-%m-%d")

BATCH_SIZE = 4096
YEAR = 2019

DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ROOT = "../anp_data"
PATH = f"../anp_models/{sys.argv[0]}_{current_date}/"
if sys.argv[1] == 'True':
    use_link_split = True
else:
    use_link_split = False
lr = float(sys.argv[2])
    
#TODO remove
import shutil
try:
    shutil.rmtree(PATH)
except:
    pass

# Create ANP dataset
dataset = ANPDataset(root=ROOT)
data = dataset[0]

# Use already existing next-topic edge (if exist)
if os.path.exists(f"{ROOT}/processed/next_topic_edge{YEAR}.pt"):
    print("Next-topic edge found!")
    data['author', 'next_topic', 'topic'].edge_index = torch.load(f"{ROOT}/processed/next_topic_edge{YEAR}.pt")
    data['author', 'next_topic', 'topic'].edge_label = None
else:
    print("Generating next-topic edge...")
    data['author', 'next_topic', 'topic'].edge_index = generate_next_topic_edge_year(data, YEAR)
    data['author', 'next_topic', 'topic'].edge_label = None
    torch.save(data['author', 'next_topic', 'topic'].edge_index, f"{ROOT}/processed/next_topic_edge{YEAR}.pt")

# Make paper features float and the graph undirected
data['paper'].x = data['paper'].x.to(torch.float)
data = T.ToUndirected()(data)


if use_link_split == True:
    sub_graph_train, _, _, _ = anp_filter_data(data, root=ROOT, folds=[0, 1, 2, 3, 4], max_year=YEAR, keep_edges=False)
    transform = T.RandomLinkSplit(
    num_val=0.2,
    num_test=0,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=False,
    edge_types=('author', 'next_topic', 'topic')
    )
    train_data, val_data, _= transform(data)

    # Define seed edges:
    edge_label_index = train_data['author', 'next_topic', 'topic'].edge_label_index
    edge_label = train_data['author', 'next_topic', 'topic'].edge_label
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[20, 10],
        neg_sampling_ratio=2.0,
        edge_label_index=(('author', 'next_topic', 'topic'), edge_label_index),
        edge_label=edge_label,
        batch_size=256,
        shuffle=True,
    )

    edge_label_index = val_data['author', 'next_topic', 'topic'].edge_label_index
    edge_label = val_data['author', 'next_topic', 'topic'].edge_label
    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[20, 10],
        edge_label_index=(('author', 'next_topic', 'topic'), edge_label_index),
        edge_label=edge_label,
        batch_size=3 * 256,
        shuffle=False,
    )
else:
    # Train
    # Filter training data
    sub_graph_train, _, _, _ = anp_filter_data(data, root=ROOT, folds=[0, 1, 2, 3 ], max_year=YEAR, keep_edges=False)    
    #sub_graph_train = sub_graph_train.to(DEVICE)

    transform = T.RandomLinkSplit(
        num_val=0,
        num_test=0,
        #disjoint_train_ratio=0.3,
        #neg_sampling_ratio=2.0,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=True,
        edge_types=('author', 'next_topic', 'topic')
    )
    train_data, _, _= transform(sub_graph_train)


    # Validation
    # Filter validation data
    sub_graph_val= anp_simple_filter_data(data, root=ROOT, folds=[4], max_year=YEAR)  
    #sub_graph_val = sub_graph_val.to(DEVICE)

    transform = T.RandomLinkSplit(
        num_val=0,
        num_test=0,
        #neg_sampling_ratio=2.0,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=True,
        edge_types=('author', 'next_topic', 'topic')
    )
    val_data, _, _= transform(sub_graph_val)


    # Define seed edges:
    edge_label_index = train_data['author', 'next_topic', 'topic'].edge_label_index
    edge_label = train_data['author', 'next_topic', 'topic'].edge_label
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[20, 10],
        #neg_sampling_ratio=2.0,
        edge_label_index=(('author', 'next_topic', 'topic'), edge_label_index),
        edge_label=edge_label,
        batch_size=1024,
        shuffle=True,
    )

    edge_label_index = val_data['author', 'next_topic', 'topic'].edge_label_index
    edge_label = val_data['author', 'next_topic', 'topic'].edge_label
    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[20, 10],
        edge_label_index=(('author', 'next_topic', 'topic'), edge_label_index),
        edge_label=edge_label,
        batch_size=1024,
        shuffle=False,
    )

# Delete the next-topic edge (data will be used for data.metadata())
del data['author', 'next_topic', 'topic']

# Initialize weight
weight = None


def weighted_mse_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)
        self.conv3 = SAGEConv((-1, -1), out_channels)
        self.conv4 = SAGEConv((-1, -1), out_channels)
        self.conv5 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        x = self.conv5(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, hidden_channels)
        self.lin4 = Linear(hidden_channels, hidden_channels)
        self.lin5 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['author'][row], z_dict['topic'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z).relu()
        z = self.lin3(z).relu()
        z = self.lin4(z).relu()
        z = self.lin5(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


# Create model, optimizer, and move model to device
# If exist load last checkpoint
if os.path.exists(PATH):
    model, first_epoch = anp_load(PATH)
    first_epoch += 1
else:
    model = Model(hidden_channels=32).to(DEVICE)
    os.makedirs(PATH)
    with open(PATH + 'info.json', 'w') as json_file:
        json.dump([], json_file)
    first_epoch = 1
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
embedding_author = torch.nn.Embedding(data["author"].num_nodes, 32).to(DEVICE)
embedding_topic = torch.nn.Embedding(data["topic"].num_nodes, 32).to(DEVICE)

def train():
    model.train()
    total_examples = total_loss = 0
    for i, batch in enumerate(tqdm(train_loader)):
        batch = batch.to(DEVICE)
        
        edge_label_index = batch['author', 'topic'].edge_label_index
        edge_label = batch['author', 'topic'].edge_label
        del batch['author', 'next_topic', 'topic']
        
        # Add user node features for message passing:
        batch['author'].x = embedding_author(batch['author'].n_id)
        batch['topic'].x = embedding_topic(batch['topic'].n_id)

        optimizer.zero_grad()
        pred = model(batch.x_dict, batch.edge_index_dict, edge_label_index)
        target = edge_label
        loss = weighted_mse_loss(pred, target, weight)
        loss.backward()
        optimizer.step()
        total_examples += len(pred)
        total_loss += float(loss) * len(pred)

    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    model.eval()
    total_examples = total_mse = total_correct = total_loss = 0
    for i, batch in enumerate(tqdm(loader)):
        batch = batch.to(DEVICE)
        
        edge_label_index = batch['author', 'topic'].edge_label_index
        edge_label = batch['author', 'topic'].edge_label
        del batch['author', 'next_topic', 'topic']
        
        # Add user node features for message passing:
        batch['author'].x = embedding_author(batch['author'].n_id)
        batch['topic'].x = embedding_topic(batch['topic'].n_id)

        pred = model(batch.x_dict, batch.edge_index_dict, edge_label_index)
        pred = pred.clamp(min=0, max=1)
        target = edge_label.float()
        rmse = F.mse_loss(pred, target).sqrt()
        loss = weighted_mse_loss(pred, target, weight)
        total_mse += rmse
        total_loss += float(loss) * len(pred)
        total_examples += len(pred)
        total_correct += int((torch.round(pred, decimals=0) == target).sum())

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

    return total_mse / BATCH_SIZE, total_correct / total_examples, total_loss / total_examples


# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

training_loss_list = []
validation_loss_list = []
accuracy_list = []
confusion_matrix = {
    'tp': 0,
    'fp': 0,
    'fn': 0,
    'tn': 0
}

for epoch in range(first_epoch, 51):
    # Train the model
    loss = train()
    
    confusion_matrix = {
        'tp': 0,
        'fp': 0,
        'fn': 0,
        'tn': 0
    }
    
    # Test the model
    val_mse, val_acc, loss_val = test(val_loader)

    # Save the model
    anp_save(model, PATH, epoch, loss, val_mse.item(), val_acc)
    
    training_loss_list.append(loss)
    validation_loss_list.append(loss_val)
    accuracy_list.append(val_acc)
    
    # Print epoch results
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f} - {loss_val:.4f} RMSE: {val_mse:.4f}, Accuracy: {val_acc:.4f}')
    
generate_graph (training_loss_list, validation_loss_list, accuracy_list, confusion_matrix)