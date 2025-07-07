import torch

# Caricamento del file .pt
index = torch.load('edge_index.pt')

# Stampa della forma e del contenuto
print("Shape:", index.shape if hasattr(index, 'shape') else "Non applicabile")
print("Contenuto:", index)
