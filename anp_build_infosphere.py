import torch
from anp_utils import *
import os
import re


def build_infosphere(fold, year, number):
    DEVICE=torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu')
    authors_infosphere_edge_list = [
            torch.tensor([[],[]]).to(torch.int64).to(DEVICE),
            torch.tensor([[],[]]).to(torch.int64).to(DEVICE),
            torch.tensor([[],[]]).to(torch.int64).to(DEVICE)]
    
    rootdir = "ANP_DATA/computed_infosphere"
    
    fold_string = [str(x) for x in fold]
    fold_string = '_'.join(fold_string)
    
    regex = re.compile(f'^infosphere_{fold_string}_{year}_noisy_.*\.pt$')
    
    if not os.path.exists(f"{rootdir}/fragments_{number}"):
        os.makedirs(f"{rootdir}/fragments_{number}")
    
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if regex.match(file):
                author_edge_list = torch.load(f"{rootdir}/{file}")
                authors_infosphere_edge_list[CITES] = torch.cat((authors_infosphere_edge_list[CITES], author_edge_list[CITES].to(DEVICE)), dim=1)
                authors_infosphere_edge_list[WRITES] = torch.cat((authors_infosphere_edge_list[WRITES], author_edge_list[WRITES].to(DEVICE)), dim=1)
                authors_infosphere_edge_list[ABOUT] = torch.cat((authors_infosphere_edge_list[ABOUT], author_edge_list[ABOUT].to(DEVICE)), dim=1)

                os.rename(f"{rootdir}/{file}", f"{rootdir}/fragments_{number}/{file}")
        break    
    torch.save(authors_infosphere_edge_list, f"ANP_DATA/computed_infosphere/{number}_infosphere_{fold_string}_{year}_noisy.pt")
   
argument = sys.argv[1]
build_infosphere([0, 1, 2, 3, 4], 2019, argument)                 