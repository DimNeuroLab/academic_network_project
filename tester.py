import os
import csv
import json
import json
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

from anp_history_infosphere_creation import main as gen

if not os.path.exists("output"):
        os.mkdir("output")

fold = 3
year = 2018
#gen(year, fold, True)


def load_csv_mapping(filename):
    mapping = {}
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                mapping[int(row[0])] = row[1]
            except: pass
    return mapping

def print_json_strings(json_data, topic_mapping, author_mapping, paper_mapping):
    for group in json_data:
        for item in group:
            if item[0] == "topic":
                mapping = topic_mapping
            elif item[0] == "author":
                mapping = author_mapping
            elif item[0] == "paper":
                mapping = paper_mapping
            else:
                continue
            number = item[1]
            string = mapping.get(number)
            if string:
                print(f"{item[0]} {number} is {string}")

# Carica i mapping dai file CSV
topic_mapping = load_csv_mapping('ANP_DATA/mapping/topics.csv')
author_mapping = load_csv_mapping('ANP_DATA/mapping/authors.csv')
paper_mapping = load_csv_mapping('ANP_DATA/mapping/papers.csv')

# Esempio di utilizzo
# with open(f"output/missing_seeds_{fold}_{year}.json", 'r') as file:
#     json_data = json.load(file)
#     print_json_strings(json_data, topic_mapping, author_mapping, paper_mapping)
    


import json
import networkx as nx
import matplotlib.pyplot as plt

# Leggi il file JSON
with open(f"output/infosphere_{fold}_{year}.json", 'r') as file:
    data = json.load(file)
# Crea un grafo diretto
G = nx.DiGraph()

# Aggiungi i nodi e gli archi al grafo
for i, author in enumerate(data):
    for sublist in author:
        for edge in sublist:
            if edge[0] == "writes":
                author_id = author_mapping[edge[1][0]]
                paper_id = paper_mapping[edge[1][1]].replace(':', ',')
                G.add_edge(author_id, paper_id, label="writes")
            elif edge[0] == "about":
                paper_id = paper_mapping[edge[1][0]].replace(':', ',')
                topic_id = topic_mapping[edge[1][1]]
                G.add_edge(paper_id, topic_id, label="about")
            elif edge[0] == "cites":
                paper1_id = paper_mapping[edge[1][0]].replace(':', ',')
                paper2_id = paper_mapping[edge[1][1]].replace(':', ',')
                G.add_edge(paper1_id, paper2_id, label="cites")

    # Disegna il grafico
    plt.figure(figsize=(8, 8))
    # use graphviz to find radial layout
    pos = graphviz_layout(G, prog="twopi", root=0)
    # draw nodes, coloring by rtt ping time
    nx.draw(G, pos,
            node_color='lightblue',
            alpha=0.5,
            node_size=15)
    # adjust the plot limits
    xmax = 1.02 * max(xx for xx, yy in pos.values())
    ymax = 1.02 * max(yy for xx, yy in pos.values())
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    
    plt.savefig(f"dummy_name{i}.png")
