"""
anp_dataset.py - ANP Dataset Module

This module provides functionality to load and process data for the Academic Network Project (ANP) dataset.

Functions/Classes:
- load_node_csv: Load node data from a CSV file.
- load_edge_csv: Load edge data from a CSV file.
- IdentityEncoder: Identity encoder class for converting raw column values to PyTorch tensors.
- ANPDataset: Dataset class for the ANP dataset, inheriting from InMemoryDataset.

The ANPDataset class loads and processes node and edge data from CSV files, and prepares it for use in PyTorch-Geometric.
"""

import os
import tarfile
import sys
import urllib

import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset

from academic_network_project.anp_utils.parse_aminer_dataset import extract_dataset



def load_node_csv(path, index_col, encoders=None, **kwargs):
    """
    Load node data from a CSV file.

    Args:
        path (str): Path to the CSV file.
        index_col (str): Name of the index column.
        encoders (dict): Dictionary containing encoders for different columns.
        **kwargs: Additional keyword arguments for pandas.read_csv.

    Returns:
        tuple: Tuple containing node features and list of node indices.
    """
    df = pd.read_csv(path, index_col=index_col, **kwargs).sort_index()
    list = [index for index in df.index.unique().to_list()]

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, list


def load_edge_csv(path, src_index_col, dst_index_col, encoders=None, **kwargs):
    """
    Load edge data from a CSV file.

    Args:
        path (str): Path to the CSV file.
        src_index_col (str): Name of the source index column.
        dst_index_col (str): Name of the destination index column.
        encoders (dict): Dictionary containing encoders for different columns.
        **kwargs: Additional keyword arguments for pandas.read_csv.

    Returns:
        tuple: Tuple containing edge index and edge attributes.
    """
    df = pd.read_csv(path, **kwargs)

    src = []
    dst = []
    for index in range(len(df[src_index_col])):
        node_scr = df[src_index_col][index]
        node_dst = df[dst_index_col][index]
        # if dst_list.get(node_dst) and src_list.get(node_dst):
        src.append(node_scr)
        dst.append(node_dst)
    # else:
    # continue
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


class IdentityEncoder(object):
    """
    Identity encoder class.

    The 'IdentityEncoder' takes the raw column values and converts them to PyTorch tensors.
    """

    # The 'IdentityEncoder' takes the raw column values and converts them to
    # PyTorch tensors.
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)


class ANPDataset(InMemoryDataset):
    """
    ANP Dataset class for academic network project dataset.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an 
            PyTorch-Geometric data object and returns a transformed version. 
            The data object will be transformed before every access. 
            (default: None)
        pre_transform (callable, optional): A function/transform that takes in an 
            PyTorch-Geometric data object and returns a transformed version. 
            The data object will be transformed before being saved to disk. 
            (default: None)
    """

    def __init__(self, root, transform=None, pre_transform=None):
        super(ANPDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['papers.csv', 'cites.csv', 'about.csv', 'writes.csv']

    @property
    def processed_file_names(self):
        return [f'my_data.pt']

    def download(self):
        """
        Download and extract the dataset.
        """
        if not os.path.exists(f"{self.root}/mapping"):
            if not os.path.exists(f"dblp_v14.json"):
                # print("Start download...")
                # urllib.request.urlretrieve("https://originalfileserver.aminer.cn/misc/dblp_v14.tar.gz",
                #                            "dblp_v14.tar.gz")
                # print("Download completed!")

                # with tarfile.open("dblp_v14.tar.gz", 'r:gz') as tar:
                #     tar.extractall()
                sys.exit("Download & extract https://originalfileserver.aminer.cn/misc/dblp_v14.tar.gz")

            print("Start parsing... (It will take some time)")
            extract_dataset(self.root)
            print("Parsing completed!")

    def process(self):
        """
        Process and prepare the dataset.
        """
        # Load data from CSV files
        paper_path = self.raw_paths[0]
        cites_path = self.raw_paths[1]
        about_path = self.raw_paths[2]
        writes_path = self.raw_paths[3]

        author_x, author_list = load_node_csv(writes_path, index_col='author_id')

        paper_x, paper_list = load_node_csv(paper_path, index_col='id',
            encoders={'year': IdentityEncoder(dtype=torch.long), 'citations': IdentityEncoder(dtype=torch.long)})

        topic_x, topic_list = load_node_csv(about_path, index_col='topic_id')

        cites_index, cites_label = load_edge_csv(cites_path, src_index_col='paper1_id', dst_index_col='paper2_id', )

        about_index, about_label = load_edge_csv(about_path, src_index_col='paper_id', dst_index_col='topic_id', )

        writes_index, writes_label = load_edge_csv(writes_path, src_index_col='author_id', dst_index_col='paper_id', )

        data = HeteroData()
        data['author'].num_nodes = len(author_list)  # Authors do not have any features.
        data['topic'].num_nodes = len(topic_list)  # Topics do not have any features.
        data['paper'].x = paper_x
        data['paper']['id'] = torch.Tensor(paper_list)
        data['author']['id'] = torch.Tensor(author_list)
        data['topic']['id'] = torch.Tensor(topic_list)
        data['paper', 'cites', 'paper'].edge_index = cites_index
        data['paper', 'cites', 'paper'].edge_label = cites_label
        data['paper', 'about', 'topic'].edge_index = about_index
        data['paper', 'about', 'topic'].edge_label = about_label
        data['author', 'writes', 'paper'].edge_index = writes_index
        data['author', 'writes', 'paper'].edge_label = writes_label
        print(data)

        data_list = [data]
        data, slices = self.collate(data_list)

        # Save the processed data object
        torch.save((data, slices), self.processed_paths[0])
