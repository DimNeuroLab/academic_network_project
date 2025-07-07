import os
import torch
from academic_network_project.anp_core.anp_utils import *
from torch_geometric.utils import coalesce
from academic_network_project.anp_core.anp_dataset import ANPDataset

BATCH_SIZE = 4096
YEAR = 2019
ROOT = "../anp_data"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Create ANP dataset
dataset = ANPDataset(root=ROOT)
data = dataset[0]

for infosphere_parameters in range(6):
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

        print(f"--- infosphere {infosphere_parameters} ---")
        print(data)
    else:
        raise Exception(f"{name_infosphere} not found!")