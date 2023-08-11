import torch
import anp_expansion
from anp_dataset import ANPDataset
from anp_utils import *

N_NODE = 3
N_CHILDREN = 3


def json_to_edge_list(infosphere):
    infosphere_edge_list = []
    for author_infosphere in infosphere:
        author_infosphere_edge_list = [
            torch.tensor([[],[]]).to(torch.int64).to(DEVICE),
            torch.tensor([[],[]]).to(torch.int64).to(DEVICE),
            torch.tensor([[],[]]).to(torch.int64).to(DEVICE)]
        for path in author_infosphere:
            for element in path:
                match element[0]:
                    case 'cites':
                        author_infosphere_edge_list[CITES] = torch.cat((author_infosphere_edge_list[CITES], torch.Tensor([[element[1][0]],[element[1][1]]]).to(torch.int64).to(DEVICE)), dim=1)
                    case 'writes':
                        author_infosphere_edge_list[WRITES] = torch.cat((author_infosphere_edge_list[WRITES], torch.Tensor([[element[1][0]],[element[1][1]]]).to(torch.int64).to(DEVICE)), dim=1)
                    case 'about':
                        author_infosphere_edge_list[ABOUT] = torch.cat((author_infosphere_edge_list[ABOUT], torch.Tensor([[element[1][0]],[element[1][1]]]).to(torch.int64).to(DEVICE)), dim=1)
        infosphere_edge_list.append(author_infosphere_edge_list)
    return infosphere_edge_list


def finalize_infosphere(fold, year, keep_edges):
    
    authors_infosphere_edge_list = [
            torch.tensor([[],[]]).to(torch.int64).to(DEVICE),
            torch.tensor([[],[]]).to(torch.int64).to(DEVICE),
            torch.tensor([[],[]]).to(torch.int64).to(DEVICE)]
    fold_string = [str(x) for x in fold]
    fold_string = '_'.join(fold_string)
    root = "ANP_DATA"

    dataset = ANPDataset(root=root)
    data = dataset[0]
    sub_graph, _, _, _ = anp_filter_data(data, root=root, folds=fold, max_year=year, keep_edges=keep_edges)
    sub_graph = sub_graph.to(DEVICE)

    try:
        i = 0
        while True:
            print(f"Part {i}")
            with open(f"ANP_DATA/computed_infosphere/infosphere_{fold_string}_{year}_{i}.json", 'r') as json_file:
                part_dict_infosphere =  json.load(json_file)
                part_tensor_infosphere = json_to_edge_list(part_dict_infosphere)
                for author_edge_list in part_tensor_infosphere:
                    if anp_expansion.expand_infosphere(sub_graph, author_edge_list, N_NODE, N_CHILDREN, anp_expansion.random_selection_policy, anp_expansion.random_expansion_policy):
                        authors_infosphere_edge_list[CITES] = torch.cat((authors_infosphere_edge_list[CITES], author_edge_list[CITES]), dim=1)
                        authors_infosphere_edge_list[WRITES] = torch.cat((authors_infosphere_edge_list[WRITES], author_edge_list[WRITES]), dim=1)
                        authors_infosphere_edge_list[ABOUT] = torch.cat((authors_infosphere_edge_list[ABOUT], author_edge_list[ABOUT]), dim=1)
                    else:
                        continue
            i += 1
                    
                
    except FileNotFoundError:
        torch.save(authors_infosphere_edge_list, f"ANP_DATA/computed_infosphere/infosphere_{fold_string}_{year}_expanded.pt")
        
finalize_infosphere([0, 1, 2, 3, 4], 2019, True)
            