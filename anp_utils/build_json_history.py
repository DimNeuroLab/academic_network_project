"""
build_json_history.py

This script generates a JSON file containing the history of co-authors for each author in the ANP dataset.

Usage: python build_json_history.py

Note: Ensure that the ANP dataset is available in the 'anp_data' directory.

Example:
    $ python build_json_history.py
"""

import json
# import cProfile
import io
import sys
import multiprocessing as mp

import torch

from academic_network_project.anp_core.anp_dataset import ANPDataset
from academic_network_project.anp_core.anp_utils import *
 
          

ROOT = "../anp_data"

dataset = ANPDataset(root=ROOT)
data = dataset[0]

sub_graph = anp_simple_filter_data(data, root=ROOT, folds=[4], max_year=2019)  

sub_graph = sub_graph.to(DEVICE)

writes_edge_index = sub_graph['author', 'writes', 'paper'].edge_index
cites_edge_index = sub_graph['paper', 'cites', 'paper'].edge_index
about_edge_index = sub_graph['paper', 'about', 'topic'].edge_index
   
map_history_c = {} 
for i, author_id in enumerate(sub_graph['author'].id):
    if i % 1000 == 0:
        print(f"{i}")
    sub_edge_index, mask = expand_1_hop_edge_index(writes_edge_index, i, flow='target_to_source')
    author_papers = sub_edge_index[1].tolist()
    list_coauthor = set()
    for paper in author_papers:
        # co-authors
        sub_edge_index, mask = expand_1_hop_edge_index(writes_edge_index, paper, flow='source_to_target')
        mask_auth = sub_edge_index[0] != i
        for co_author in sub_edge_index[:, mask_auth][0].tolist():
            list_coauthor.add(int(sub_graph['author'].id[co_author].item()))
    map_history_c[int(author_id.item())] = {'co-authors': list(list_coauthor)}
    
with open('history4co2.json', 'w') as json_file:
        json.dump(map_history_c, json_file)
           

        

