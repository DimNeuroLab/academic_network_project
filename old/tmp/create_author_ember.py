import torch
import os
from torch_geometric.utils import coalesce
from academic_network_project.anp_core.anp_dataset import ANPDataset
from academic_network_project.anp_core.anp_utils import drop_edges, create_infosphere_top_papers_edge_index
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import Linear
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HGTConv, Linear
from tqdm import tqdm

# Hardcoded parameters from the filename
infosphere_type = 1
infosphere_parameters = 5
only_new = False
edge_number = 50
drop_percentage = 0.0
BATCH_SIZE = 4096  # Reduced batch size for memory optimization
YEAR = 2019
ROOT = "../anp_data"
DEVICE = torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu')

from academic_network_project.anp_core.anp_dataset import ANPDataset
from academic_network_project.anp_core.anp_utils import *

# Path to the pre-trained model
model_path = '../anp_models/anp_link_prediction_co_author_hgt_embedding_1_5_False_-1_0.0_2024_10_22_17_46_56/model.pt'

# Ensure the correct GPU device is used and clear cache
torch.cuda.set_device(DEVICE)
torch.cuda.empty_cache()



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

# Training Data
subset_dict = {}
mask = data['paper'].x[:, 0] <= YEAR
papers_list_year = torch.where(mask)
subset_dict['paper'] = papers_list_year[0]
data_0 = data.subgraph(subset_dict)

transform_train = T.RandomLinkSplit(
    num_val=0,
    num_test=0,
    neg_sampling_ratio=1.0,
    add_negative_train_samples=True,
    edge_types=('author', 'co_author', 'author')
)
train_data, _, _ = transform_train(data_0)
train_data['author', 'co_author', 'author'].edge_index = train_data['author', 'co_author', 'author'].edge_label_index
del train_data['author', 'co_author', 'author'].edge_label_index

author_loader = NeighborLoader(
    data=train_data,
    num_neighbors=[-1, -1],
    input_nodes='author',
    batch_size=BATCH_SIZE,
    drop_last=False
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
        self.encoder = GNNEncoder(hidden_channels, hidden_channels, 1, 2, data.metadata())
        # self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)
        self.embedding_author = torch.nn.Embedding(data["author"].num_nodes, 32)
        self.embedding_topic = torch.nn.Embedding(data["topic"].num_nodes, 32)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)

# Load the model
model = torch.load(model_path, map_location=DEVICE)
model.eval()  # Set to evaluation mode
model.to(DEVICE)

encoder = model.encoder

# Function to generate author embeddings
@torch.no_grad()
def generate_author_embeddings(model, loader):
    model.eval()
    all_embeddings = []

    for batch in tqdm(loader):
        # Move the batch to the correct device
        batch = batch.to(DEVICE)
        del batch['author', 'co_author', 'author']

        batch['author'].x = model.embedding_author(batch['author'].n_id)
        batch['topic'].x = model.embedding_topic(batch['topic'].n_id)

        # Get embeddings
        embeddings = model.encoder(batch.x_dict, batch.edge_index_dict)

        # Save embeddings for authors
        num_authors_in_batch = batch['author'].batch_size 
        author_embeddings = embeddings['author'][0:num_authors_in_batch]

        # Append to results
        all_embeddings.append(author_embeddings.cpu())

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings

# Generate the embeddings tensor
author_embeddings = generate_author_embeddings(model, author_loader)

# Ensure embeddings are in the correct order
ordered_embeddings = author_embeddings.cpu().detach()

# Check the shape of the tensor
print(ordered_embeddings.shape)
torch.save(ordered_embeddings, 'author_embeddings.pt')
print(data['author'].num_nodes)
