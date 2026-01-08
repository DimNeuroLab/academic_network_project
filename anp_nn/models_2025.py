import torch
from torch.nn import Linear, Dropout, BatchNorm1d
from torch_geometric.nn import HGTConv, Linear
import torch.nn.functional as F


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata, data):
        super().__init__()
        # Increase dropout here as discussed in the previous turn
        self.dropout_p = 0.4
        
        # Initial linear projection for input features
        self.lin_dict = torch.nn.ModuleDict({
            node_type: Linear(-1, hidden_channels)
            for node_type in data.node_types
        })

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for _ in range(num_layers):
            # HGT Convolution
            self.convs.append(HGTConv(hidden_channels, hidden_channels, metadata, num_heads))
            
            # Crucial change: Separate BatchNorm for each node type per layer.
            # Authors, topics, and papers have different statistics.
            layer_norms = torch.nn.ModuleDict({
                node_type: BatchNorm1d(hidden_channels)
                for node_type in data.node_types
            })
            self.norms.append(layer_norms)

    def forward(self, x_dict, edge_index_dict):
        # Initial projection
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for i, conv in enumerate(self.convs):
            # SAVE THE RESIDUAL (The input before convolution)
            x_residual = x_dict 

            # 1. Message Passing
            x_dict = conv(x_dict, edge_index_dict)

            # 2. Process results per node type with residual connection
            next_x_dict = {}
            for node_type, x in x_dict.items():
                # Grab the saved residual for this node type
                res = x_residual[node_type]

                # Activation (GELU is often better than ReLU for Transformers)
                x = F.gelu(x)

                # Dropout applied to the crucial path
                x = F.dropout(x, p=self.dropout_p, training=self.training)

                # ADD RESIDUAL CONNECTION
                x = x + res
                
                # Normalize using node-specific BatchNorm
                x = self.norms[i][node_type](x)

                next_x_dict[node_type] = x
            
            # Update dictionary for the next layer
            x_dict = next_x_dict

        return x_dict

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin = torch.nn.Linear(hidden_channels, hidden_channels)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z_src = z_dict['author'][row]
        z_dst = z_dict['author'][col]
        
        z_src = self.lin(z_src)
        
        return (z_src * z_dst).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, data):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels, 1, 2, data.metadata(), data)
        self.decoder = EdgeDecoder(hidden_channels)
        self.embedding_author = torch.nn.Embedding(data["author"].num_nodes, 64)
        self.embedding_topic = torch.nn.Embedding(data["topic"].num_nodes, 64)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)
