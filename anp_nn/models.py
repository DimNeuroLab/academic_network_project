import torch
from torch.nn import Linear, Dropout, BatchNorm1d
from torch_geometric.nn import HGTConv, Linear
import torch.nn.functional as F


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata, data):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict({
            node_type: Linear(-1, hidden_channels)
            for node_type in data.node_types
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
    def __init__(self, hidden_channels, data):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels, 1, 2, data.metadata(), data)
        self.decoder = EdgeDecoder(hidden_channels)
        self.embedding_author = torch.nn.Embedding(data["author"].num_nodes, 32)
        self.embedding_topic = torch.nn.Embedding(data["topic"].num_nodes, 32)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)
