import torch
from torch.nn import Linear, Dropout, BatchNorm1d
from torch_geometric.nn import HGTConv, Linear
import torch.nn.functional as F


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata, data, dropout_p=0.2):
        super().__init__()
        self.dropout_p = dropout_p

        self.lin_dict = torch.nn.ModuleDict({
            node_type: Linear(-1, hidden_channels)
            for node_type in data.node_types
        })

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(HGTConv(hidden_channels, hidden_channels, metadata, num_heads))
            
            layer_norms = torch.nn.ModuleDict({
                node_type: BatchNorm1d(hidden_channels)
                for node_type in data.node_types
            })
            self.norms.append(layer_norms)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for i, conv in enumerate(self.convs):
            x_residual = x_dict 

            x_dict = conv(x_dict, edge_index_dict)

            next_x_dict = {}
            for node_type, x in x_dict.items():
                res = x_residual[node_type]

                x = F.gelu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)
                x = x + res
                x = self.norms[i][node_type](x)
                next_x_dict[node_type] = x
            
            x_dict = next_x_dict

        return x_dict

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_channels, 1)
        )

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z_src = z_dict['author'][row]
        z_dst = z_dict['author'][col]
        z = torch.cat([z_src, z_dst], dim=-1)
        return self.mlp(z).squeeze(-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, data, dropout_p=0.2):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels=hidden_channels, out_channels=hidden_channels,
                                  num_heads=2, num_layers=2,
                                  metadata=data.metadata(), data=data, dropout_p=dropout_p)
        self.decoder = EdgeDecoder(hidden_channels=hidden_channels)
        self.embedding_author = torch.nn.Embedding(data["author"].num_nodes, 64)
        self.embedding_topic  = torch.nn.Embedding(data["topic"].num_nodes, 64)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)
