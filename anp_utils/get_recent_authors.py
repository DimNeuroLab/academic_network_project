import json
import os
import sys
import ast
from datetime import datetime

import numpy as np
import torch
import pandas as pd
from academic_network_project.anp_core.anp_dataset import ANPDataset
from academic_network_project.anp_core.anp_utils import *

# Constants
ROOT = "../anp_data"
YEAR = 2020  # L'anno per cui vogliamo filtrare gli autori
DEVICE = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

# Crea il dataset ANP
dataset = ANPDataset(root=ROOT)
data = dataset[0]

# Filtra i paper pubblicati prima del 2020
mask_before_2020 = (data['paper'].x[:, 0] < YEAR)
papers_before_2020 = set(torch.where(mask_before_2020)[0].tolist())
print(f"Number of papers published before {YEAR}: {len(papers_before_2020)}")

# Filtra i paper pubblicati nel 2020
mask_2020 = (data['paper'].x[:, 0] == YEAR)
papers_2020 = set(torch.where(mask_2020)[0].tolist())
print(f"Number of papers published in {YEAR}: {len(papers_2020)}")

# Filtra gli autori che hanno scritto paper nei due anni
author_paper = data['author', 'writes', 'paper'].edge_index

# Trova gli autori che hanno scritto almeno un paper prima del 2020
authors_before_2020 = set(author_paper[0][torch.isin(author_paper[1], torch.tensor(list(papers_before_2020)))].tolist())

# Trova gli autori che hanno scritto almeno un paper nel 2020
authors_2020 = set(author_paper[0][torch.isin(author_paper[1], torch.tensor(list(papers_2020)))].tolist())

# Trova autori che hanno scritto almeno un paper prima del 2020 e hanno scritto nel 2020
filtered_authors = authors_before_2020.intersection(authors_2020)

# Converti il set di autori in un DataFrame
authors_2020_df = pd.DataFrame(list(filtered_authors), columns=['author_id'])

# Salva il DataFrame su disco come file CSV
authors_2020_df.to_csv("../anp_data/processed/relevant_authors_2020.csv", index=False)
print("Author IDs for 2020 saved to '../anp_data/processed/relevant_authors_2020.csv'")

# Stampa la lunghezza totale degli autori filtrati
total_filtered_authors = len(filtered_authors)
print(f"Total number of filtered authors: {total_filtered_authors}")
