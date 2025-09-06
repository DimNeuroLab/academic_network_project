"""
anp_utils.py - Academic Network Project Utils

This file contains utility functions and helper methods for working with academic network data,
specifically tailored for the Academic Network Project (ANP).

The utilities provided here include functions for data filtering, graph expansion, edge 
generation, model saving/loading, and graph visualization. These functions are essential for 
preprocessing data, generating features, training models, and evaluating results in ANP.

Functions:
- expand_1_hop_edge_index: Expand the edge index to include 1-hop neighbors of a given node.
- expand_1_hop_graph: Expand the graph by adding 1-hop neighbors of the given nodes.
- anp_filter_data: Filter the data based on certain criteria for ANP.
- anp_simple_filter_data: Filter the data based on a simple criteria for ANP.
....
- anp_save: Save the model and associated information for ANP.
- anp_load: Load the saved model for ANP.
- generate_graph: Generate and save graphs based on training and validation metrics for ANP.

"""
import torch
import random
import sys
import numpy as np
import json
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd
import os
import ast

from torch_geometric.utils import coalesce
from sklearn.metrics import precision_score, recall_score, f1_score

PAPER = 0
AUTHOR = 1
TOPIC = 2

CITES = 0
WRITES = 1
ABOUT = 2

MAX_ITERATION = 1



def expand_1_hop_edge_index(edge_index, node, flow):
    """
  Expand the edge index to include 1-hop neighbors of a given node.

  Args:
  - edge_index (Tensor): The edge index tensor.
  - node (int): The node for which neighbors need to be expanded.
  - flow (str): The direction of flow, either 'target_to_source' or 'source_to_target'.

  Returns:
  - expanded_edge_index (Tensor): The expanded edge index tensor.
  - mask (Tensor): The boolean mask indicating the nodes in the expanded edge index.

  """
    # _, sub_edge_index, _, _ = k_hop_subgraph(node, 1, edge_index, flow=flow)
    # Clean
    if flow == 'target_to_source':
        mask = edge_index[0] == node
    else:
        mask = edge_index[1] == node
    return edge_index[:, mask], mask


def expand_1_hop_graph(edge_index, nodes, type, paths):
    """
  Expand the graph by adding 1-hop neighbors of the given nodes.

  Args:
  - edge_index (Tensor): The edge index tensor.
  - nodes (list): List of nodes for which neighbors need to be expanded.
  - type (int): Type of nodes (PAPER, AUTHOR, or TOPIC).
  - paths (dict): Dictionary to store paths.

  Returns:
  - tuple: A tuple containing lists of expanded nodes for papers, authors, and topics.

  """
    if type == PAPER:
        paper_nodes = []
        author_nodes = []
        topic_nodes = []
        for paper in nodes:
            # cited paper
            # sub_edge_index, _ = expand_1_hop_edge_index(edge_index[0], paper, flow='target_to_source')
            # for cited_paper in sub_edge_index[1].tolist():
            #     if not paths[PAPER].get(cited_paper):
            #         paper_nodes.append(cited_paper)
            #         paths[PAPER][cited_paper] = paths[PAPER][paper] + [('cites', [paper, cited_paper])]

            # Since this is used from seed we are intrested in papers that cites it
            sub_edge_index, _ = expand_1_hop_edge_index(edge_index[CITES], paper, flow='source_to_target')
            for cited_paper in sub_edge_index[0].tolist():
                if not paths[PAPER].get(cited_paper):
                    paper_nodes.append(cited_paper)
                    paths[PAPER][cited_paper] = paths[PAPER][paper] + [('cites', [cited_paper, paper])]

            # co-authors
            sub_edge_index, _ = expand_1_hop_edge_index(edge_index[WRITES], paper, flow='source_to_target')
            for co_author in sub_edge_index[0].tolist():
                if not paths[AUTHOR].get(co_author):
                    author_nodes.append(co_author)
                    paths[AUTHOR][co_author] = paths[PAPER][paper] + [('writes', [co_author, paper])]

            # topic
            sub_edge_index, _ = expand_1_hop_edge_index(edge_index[ABOUT], paper, flow='target_to_source')
            for topic in sub_edge_index[1].tolist():
                if not paths[TOPIC].get(topic):
                    topic_nodes.append(topic)
                    paths[TOPIC][topic] = paths[PAPER][paper] + [('about', [paper, topic])]

        return (paper_nodes, author_nodes, topic_nodes)

    elif type == AUTHOR:
        paper_nodes = []
        for author in nodes:
            sub_edge_index, _ = expand_1_hop_edge_index(edge_index, author, flow='target_to_source')
            for paper in sub_edge_index[1].tolist():
                if not paths[PAPER].get(paper):
                    paper_nodes.append(paper)
                    paths[PAPER][paper] = paths[AUTHOR][author] + [('writes', [author, paper])]
        return paper_nodes

    else:
        paper_nodes = []
        for topic in nodes:
            sub_edge_index, _ = expand_1_hop_edge_index(edge_index, topic, flow='source_to_target')
            for paper in sub_edge_index[0].tolist():
                if not paths[PAPER].get(paper):
                    paper_nodes.append(paper)
                    paths[PAPER][paper] = paths[TOPIC][topic] + [('about', [paper, topic])]
        return paper_nodes


def anp_filter_data(data, root, folds, max_year, keep_edges):
    """
  Filter the data based on certain criteria.

  Args:
  - data (Data): The input data.
  - root (str): The root directory.
  - folds (list): List of folds.
  - max_year (int): Maximum year.
  - keep_edges (bool): Whether to keep edges.

  Returns:
  - tuple: A tuple containing filtered data, next-year data, author filter list, and next-year paper list.

  """
    # Old - legacy
    subset_dict = {}
    subset_dict_next_year = {}

    authors_filter_list = []
    for fold in folds:
        df_auth = pd.read_csv(f"{root}/split/authors_{fold}.csv")
        authors_filter_list.extend(df_auth.values.flatten())

    if not keep_edges:
        subset_dict['author'] = subset_dict_next_year['author'] = torch.tensor(authors_filter_list)

    papers_list_next_year = []
    papers_list_year = []
    for i, row in enumerate(data['paper'].x.tolist()):
        if row[0] <= max_year:
            papers_list_year.append(i)
        elif row[0] == max_year + 1:
            papers_list_next_year.append(i)
    subset_dict['paper'] = torch.tensor(papers_list_year)
    papers_list_year.extend(papers_list_next_year)
    subset_dict_next_year['paper'] = torch.tensor(papers_list_year)
    return data.subgraph(subset_dict), data.subgraph(subset_dict_next_year), sorted(
        authors_filter_list), papers_list_next_year


def anp_simple_filter_data(data, root, folds, max_year):
    """
  Filter the data based on a simple criteria.

  Args:
  - data (Data): The input data.
  - root (str): The root directory.
  - folds (list): List of folds.
  - max_year (int): Maximum year.

  Returns:
  - Data: Filtered data.

  """
    subset_dict = {}
    authors_filter_list = []
    if folds:
        for fold in folds:
            df_auth = pd.read_csv(f"{root}/split/authors_{fold}.csv")
            authors_filter_list.extend(df_auth.values.flatten())
        subset_dict['author'] = torch.tensor(authors_filter_list)
    mask = data['paper'].x[:, 0] <= max_year
    papers_list_year = torch.where(mask)
    subset_dict['paper'] = papers_list_year[0]
    return data.subgraph(subset_dict)


def get_author_edge_year(data, year, device):
    """
    Get edges connecting authors and their 2 hops neighbors for a given year.

    Args:
    - data (Data): The input data.
    - year (int): The target year.

    Returns:
    - dict: A dictionary containing tensors for edges between authors, papers, and topics.
    """
    years = data['paper'].x[:, 0]
    mask = years == year
    papers = torch.where(mask)[0]
    edge_index_writes = data['author', 'writes', 'paper'].edge_index.to(device)
    edge_index_about = data['paper', 'about', 'topic'].edge_index.to(device)
    edge_index_cites = data['paper', 'cites', 'paper'].edge_index.to(device)

    src = {"author": [], "paper": [], "topic": []}
    dst = {"author": [], "paper": [], "topic": []}
    dict_tracker = {"author": {}, "paper": {}, "topic": {}}
    time = datetime.now()
    tot = len(papers)

    for i, paper in enumerate(papers):
        if i % 10000 == 0:
            print(f"papers processed: {i}/{tot} - {i / tot * 100:.2f}% - {str(datetime.now() - time)}")
        
        sub_edge_index_writes, _ = expand_1_hop_edge_index(edge_index_writes, paper, flow='source_to_target')
        sub_edge_index_about, _ = expand_1_hop_edge_index(edge_index_about, paper, flow='target_to_source')
        sub_edge_index_cites, _ = expand_1_hop_edge_index(edge_index_cites, paper, flow='target_to_source')

        for author in sub_edge_index_writes[0].tolist():
            for co_author in sub_edge_index_writes[0].tolist():
                if author != co_author and not dict_tracker["author"].get((author, co_author)):
                    dict_tracker["author"][(author, co_author)] = True
                    src["author"].append(author)
                    dst["author"].append(co_author)

            for cited_paper in sub_edge_index_cites[1].tolist():
                if not dict_tracker["paper"].get((author, cited_paper)):
                    dict_tracker["paper"][(author, cited_paper)] = True
                    src["paper"].append(author)
                    dst["paper"].append(cited_paper)

            for topic in sub_edge_index_about[1].tolist():
                if not dict_tracker["topic"].get((author, topic)):
                    dict_tracker["topic"][(author, topic)] = True
                    src["topic"].append(author)
                    dst["topic"].append(topic)

    return {
        "author": torch.tensor([src["author"], dst["author"]]),
        "paper": torch.tensor([src["paper"], dst["paper"]]),
        "topic": torch.tensor([src["topic"], dst["topic"]])
    }


def get_author_edge_history(data, year, device):
    """
    Get edges connecting authors and their 2 hops neighbors up to a given year.

    Args:
    - data (Data): The input data.
    - year (int): The target year.

    Returns:
    - dict: A dictionary containing tensors for edges between authors, papers, and topics.
    """
    years = data['paper'].x[:, 0]
    mask = years <= year
    papers = torch.where(mask)[0]
    edge_index_writes = data['author', 'writes', 'paper'].edge_index.to(device)
    edge_index_about = data['paper', 'about', 'topic'].edge_index.to(device)
    edge_index_cites = data['paper', 'cites', 'paper'].edge_index.to(device)

    src = {"author": [], "paper": [], "topic": []}
    dst = {"author": [], "paper": [], "topic": []}
    dict_tracker = {"author": {}, "paper": {}, "topic": {}}
    time = datetime.now()
    tot = len(papers)

    for i, paper in enumerate(papers):
        if i % 10000 == 0:
            print(f"papers processed: {i}/{tot} - {i / tot * 100:.2f}% - {str(datetime.now() - time)}")
        
        sub_edge_index_writes, _ = expand_1_hop_edge_index(edge_index_writes, paper, flow='source_to_target')
        sub_edge_index_about, _ = expand_1_hop_edge_index(edge_index_about, paper, flow='target_to_source')
        sub_edge_index_cites, _ = expand_1_hop_edge_index(edge_index_cites, paper, flow='target_to_source')

        for author in sub_edge_index_writes[0].tolist():
            for co_author in sub_edge_index_writes[0].tolist():
                if author != co_author and not dict_tracker["author"].get((author, co_author)):
                    dict_tracker["author"][(author, co_author)] = True
                    src["author"].append(author)
                    dst["author"].append(co_author)

            for cited_paper in sub_edge_index_cites[1].tolist():
                if not dict_tracker["paper"].get((author, cited_paper)):
                    dict_tracker["paper"][(author, cited_paper)] = True
                    src["paper"].append(author)
                    dst["paper"].append(cited_paper)

            for topic in sub_edge_index_about[1].tolist():
                if not dict_tracker["topic"].get((author, topic)):
                    dict_tracker["topic"][(author, topic)] = True
                    src["topic"].append(author)
                    dst["topic"].append(topic)

    return {
        "author": torch.tensor([src["author"], dst["author"]]),
        "paper": torch.tensor([src["paper"], dst["paper"]]),
        "topic": torch.tensor([src["topic"], dst["topic"]])
    }

def get_difference_author_edge_year(data, year, device, root="../anp_data"):
    """
    Generate the difference in edges between consecutive years with history, considering all types of nodes.

    Args:
    - data (Data): The input data.
    - year (int): The target year.
    - root (str): The root directory.

    Returns:
    - dict: A dictionary containing the difference in edge indices for each type of node relation (author-paper, paper-topic, paper-paper).
    """
    difference_edge_index = {
        "author": torch.tensor([[], []], dtype=torch.int64).to(device),
        "paper": torch.tensor([[], []], dtype=torch.int64).to(device),
        "topic": torch.tensor([[], []], dtype=torch.int64).to(device)
    }

    edge_types = ["author", "paper", "topic"]
    
    # Load or generate current year edges
    if os.path.exists(f"{root}/processed/author_edge{year}_history.pt"):
        print(f"Current history author edge found!")
        current_edge_data = torch.load(f"{root}/processed/author_edge{year}_history.pt", map_location=device)
    else:
        print(f"Generating current history author edge...")
        current_edge_data = get_author_edge_history(data, year, device)
        torch.save(current_edge_data, f"{root}/processed/author_edge{year}_history.pt")

    # Load or generate next year edges
    if os.path.exists(f"{root}/processed/author_edge{year + 1}.pt"):
        print(f"Next author edge found!")
        next_edge_data = torch.load(f"{root}/processed/author_edge{year + 1}.pt", map_location=device)
    else:
        print(f"Generating next author edge...")
        next_edge_data = get_author_edge_year(data, year + 1, device)
        torch.save(next_edge_data, f"{root}/processed/author_edge{year + 1}.pt")

    for edge_type in edge_types:
        # Compute difference in edges
        set_src = torch.unique(next_edge_data[edge_type][0], sorted=True)
        time = datetime.now()
        tot = len(set_src)
        for i, src in enumerate(set_src):
            if i % 1000 == 0:
                print(f"{edge_type} edge processed: {i}/{tot} - {i / tot * 100:.2f}% - {str(datetime.now() - time)}")

            # Edges in current year for this src node
            mask = current_edge_data[edge_type][0] == src
            dst_old = current_edge_data[edge_type][:, mask][1]

            # Edges in next year for this src node
            mask = next_edge_data[edge_type][0] == src
            dst_new = next_edge_data[edge_type][:, mask][1]

            # Find differences (new edges in next year)
            diff = dst_new[(dst_new.view(1, -1) != dst_old.view(-1, 1)).all(dim=0)]

            for dst in diff:
                difference_edge_index[edge_type] = torch.cat(
                    (difference_edge_index[edge_type], torch.Tensor([[src], [dst]]).to(torch.int64).to(device)), dim=1)

    return difference_edge_index


def create_infosphere_top_papers_edge_index(data, n, year):
    df = pd.read_csv("../anp_data/raw/sorted_papers.csv")
    papers = df[df['year'] <= year]['id'][:n].values
    authors = data['author'].num_nodes
    
    src = []
    dst = []
    for author in range(authors):
        for paper in papers:
            src.append(author)
            dst.append(paper)
    edge_index = torch.tensor([src, dst])
    print(edge_index)
    return edge_index


def create_infosphere_top_papers_per_topic_edge_index(data, topics_per_author, papers_per_topic, year):
    df_papers = pd.read_csv("../anp_data/raw/sorted_papers_about.csv")
    df_papers_filtered = df_papers[df_papers['year'] <= year]
    df_topic = pd.read_csv(f"../anp_data/raw/sorted_authors_topics_{2019}.csv")
    
    papers_dict = df_papers_filtered.groupby('topic_id')['id'].apply(list).to_dict()
    topics_dict = df_topic.groupby('author_id')['topic_id'].apply(list).to_dict()
    
    authors = data['author'].num_nodes
    
    src = []
    dst = []
    for author in range(authors):
        if author in topics_dict:
            topics = topics_dict[author][:topics_per_author]
            
            for topic in topics:
                papers = papers_dict[topic][:papers_per_topic]
                for paper in papers:
                    src.append(author)
                    dst.append(paper)
    
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index

def get_author_paper_edge_by_papers(data, papers):
    """
    Get edges connecting authors and papers for a given list of papers.
    """
    edge_index = data['author', 'writes', 'paper'].edge_index
    # Create a mask selecting edges where the paper is in the top papers list
    if edge_index.device != papers.device:
        edge_index = edge_index.to(papers.device)
    mask = torch.isin(edge_index[1], papers)

    # Filter the edge_index based on the mask
    filtered_edge_index = edge_index[:, mask]
    return filtered_edge_index



def drop_edges(edge_index, drop_percentage, seed=42):
    random.seed(seed)
    num_edges = edge_index.size(1)
    num_edges_to_drop = int(num_edges * drop_percentage)
    
    # Randomly select indices of the edges to drop
    indices_to_drop = random.sample(range(num_edges), num_edges_to_drop)
    
    # Create a mask for the edges to keep
    mask = torch.ones(num_edges, dtype=torch.bool)
    mask[indices_to_drop] = False
    
    # Filter the edges to keep
    edge_index = edge_index[:, mask]
    
    return edge_index


def anp_save(model, path, epoch, loss, loss_val, accuracy):
    """
  Save the model and associated information.

  Args:
  - model: The model to be saved.
  - path (str): The path to save the model.
  - epoch (int): The current epoch.
  - loss (float): The training loss.
  - loss_val (float): The validation loss.
  - accuracy (float): The accuracy.

  Returns:
  - None

  """
    torch.save(model, path + 'model.pt')
    new = {'epoch': epoch, 'loss': loss, 'loss': loss_val, 'accuracy': accuracy}
    with open(path + 'info.json', 'r') as json_file:
        data = json.load(json_file)
    data['data'].append(new)
    with open(path + 'info.json', 'w') as json_file:
        json.dump(data, json_file)


def anp_load(path, device='cpu'):
    """
  Load the saved model.

  Args:
  - path (str): The path to load the model.

  Returns:
  - tuple: A tuple containing the loaded model and the epoch.

  """
    with open(path + 'info.json', 'r') as json_file:
        data = json.load(json_file)
    return torch.load(path + 'model.pt', map_location=device), data[-1]["data"]["epoch"]


def generate_graph(path, training_loss_list, validation_loss_list, training_accuracy_list, validation_accuracy_list,
                   confusion_matrix=None, fold=None):
    """
    Generate and save graphs based on training and validation metrics.

    Args:
    - training_loss_list (list): List of training losses.
    - validation_loss_list (list): List of validation losses.
    - training_accuracy_list (list): List of training accuracies.
    - validation_accuracy_list (list): List of validation accuracies.
    - confusion_matrix (dict): Dictionary containing confusion matrix values.

    Returns:
    - None

    """
    plt.plot(training_loss_list, label='train_loss')
    plt.plot(validation_loss_list, label='validation_loss')
    plt.legend()
    plt.savefig(f'{path}{os.path.basename(sys.argv[0][:-3])}_loss_{fold}.pdf')
    plt.close()

    plt.plot(training_accuracy_list, label='train_accuracy')
    plt.plot(validation_accuracy_list, label='validation_accuracy')
    plt.legend()
    plt.savefig(f'{path}{os.path.basename(sys.argv[0][:-3])}_accuracy_{fold}.pdf')
    plt.close()

    if confusion_matrix:
        array = [[confusion_matrix['tp'], confusion_matrix['fp']], [confusion_matrix['fn'], confusion_matrix['tn']]]
        df_cm = pd.DataFrame(array, index=[i for i in ("POSITIVE", "NEGATIVE")], columns=[i for i in ("POSITIVE", "NEGATIVE")])
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig(f'{path}{os.path.basename(sys.argv[0][:-3])}_CM_{fold}.pdf')
        plt.close()

    value_log = {'training_loss_list': training_loss_list, 'validation_loss_list': validation_loss_list,
        'training_accuracy_list': training_accuracy_list, 'validation_accuracy_list': validation_accuracy_list}

    with open(f'{path}{os.path.basename(sys.argv[0][:-3])}_log_{fold}.json', 'w') as f:
        json.dump(value_log, f, ensure_ascii=False, indent=4)


def anp_add_infosphere(data, infosphere_type, infosphere_parameters, drop_percentage, root, device, year):
    """
    Add infosphere data to the graph based on the specified type and parameters.
    Args:
    - infosphere_type (int): Type of infosphere data to add.
    - infosphere_parameters (str): Parameters for the infosphere data.
    - data (Data): The input data.
    - drop_percentage (float): Percentage of edges to drop.
    Returns:
    - None: The function modifies the data in place.
    """
    # Add infosphere data if requested
    if infosphere_type != 0:
        if infosphere_type == 1:
            fold = [0, 1, 2, 3, 4]
            fold_string = '_'.join(map(str, fold))
            name_infosphere = f"{infosphere_parameters}_infosphere_{fold_string}_{year}_noisy.pt"

            # Load infosphere
            if os.path.exists(f"{root}/computed_infosphere/{year}/{name_infosphere}"):
                infosphere_edges = torch.load(f"{root}/computed_infosphere/{year}/{name_infosphere}", map_location=device)
                
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
            infosphere_edge = create_infosphere_top_papers_edge_index(data, int(infosphere_parameters), year)
            data['author', 'infosphere', 'paper'].edge_index = coalesce(infosphere_edge)
            data['author', 'infosphere', 'paper'].edge_label = None

        elif infosphere_type == 3:
            infosphere_parameterss = infosphere_parameters.strip()
            arg_list = ast.literal_eval(infosphere_parameterss)
            if os.path.exists(f"{root}/processed/edge_infosphere_3_{arg_list[0]}_{arg_list[1]}.pt"):
                print("Infosphere 3 edge found!")
                data['author', 'infosphere', 'paper'].edge_index = torch.load(f"{root}/processed/edge_infosphere_3_{arg_list[0]}_{arg_list[1]}.pt", map_location=device)
                data['author', 'infosphere', 'paper'].edge_label = None
            else:
                print("Generating infosphere 3 edge...")
                infosphere_edge = create_infosphere_top_papers_per_topic_edge_index(data, arg_list[0], arg_list[1], year)
                data['author', 'infosphere', 'paper'].edge_index = coalesce(infosphere_edge)
                data['author', 'infosphere', 'paper'].edge_label = None
                torch.save(data['author', 'infosphere', 'paper'].edge_index, f"{root}/processed/edge_infosphere_3_{arg_list[0]}_{arg_list[1]}.pt")

        
            infosphere_edge = create_infosphere_top_papers_per_topic_edge_index(data, arg_list[0], arg_list[1], year)
            data['author', 'infosphere', 'paper'].edge_index = coalesce(infosphere_edge)
            data['author', 'infosphere', 'paper'].edge_label = None
        
        elif infosphere_type == 4:
            if os.path.exists(f"{root}/processed/rec_edge_5_NAIS.pt"):
                print("Rec edge found!")
                data['author', 'infosphere', 'paper'].edge_index = torch.load(f"{root}/processed/rec_edge_10_NAIS.pt", map_location=device)
                data['author', 'infosphere', 'paper'].edge_label = None
            else:
                print("Error: Rec edge not found!")
                exit()
        
        elif infosphere_type == 5:
            if os.path.exists(f"{root}/processed/rec_edge_10_LightGCN.pt"):
                print("Rec edge found!")
                data['author', 'infosphere', 'paper'].edge_index = torch.load(f"{root}/processed/rec_edge_10_LightGCN.pt", map_location=device)
                data['author', 'infosphere', 'paper'].edge_label = None
            else:
                print("Error: Rec edge not found!")
                exit()


def compute_metrics(labels, preds, threshold=0.5):
    preds_bin = (preds >= threshold).int()
    labels = labels.int()
    precision = precision_score(labels, preds_bin, zero_division=0)
    recall = recall_score(labels, preds_bin, zero_division=0)
    f1 = f1_score(labels, preds_bin, zero_division=0)
    return precision, recall, f1
