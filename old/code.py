import json

import glob
import os

base_path = "/data/sabrina/academic_network_project/anp_models/"
pattern = os.path.join(base_path, "anp_linl_prediction_co_author_hgt_GT_cv*")

# Get all matching directories
folders = [f for f in glob.glob(pattern) if os.path.isdir(f)]

for folder in folders:
    print(f"Processing folder: {folder}")
    for i in range(0, 4):
        # Construct the path to the JSON file
        print(f"Processing fold: {i}")
        json_file_path = os.path.join(folder, f"fold_{i}", "val_results.json")
        
        # Check if the file exists
        if os.path.exists(json_file_path):
            print(f"Found: {json_file_path}")
        else:
            print(f"Not found: {json_file_path}")
            break

        with open(json_file_path, "r") as file:
            data = json.load(file)

        # Inizializza i valori massimi e minimi
        max_train_acc = max_val_acc = float('-inf')
        min_train_loss = min_val_loss = float('inf')

        # Itera sui dati per trovare i valori richiesti
        for epoch_data in data:
            max_train_acc = max(max_train_acc, epoch_data["train_acc"])
            max_val_acc = max(max_val_acc, epoch_data["val_acc"])
            min_train_loss = min(min_train_loss, epoch_data["train_loss"])
            min_val_loss = min(min_val_loss, epoch_data["val_loss"])

        # Stampa i risultati
        print(f"Massima Train Accuracy: {max_train_acc}")
        print(f"Massima Validation Accuracy: {max_val_acc}")
        print(f"Minima Train Loss: {min_train_loss}")
        print(f"Minima Validation Loss: {min_val_loss}")

        # load test results
        test_file_path = os.path.join(folder, f"fold_{i}", "test_results.json")
        if os.path.exists(test_file_path):
            print(f"Found: {test_file_path}")
        else:
            print(f"Not found: {test_file_path}")
        with open(test_file_path, "r") as file:
            test_data = json.load(file)
        
        print(test_data)


# file_path = "/data/sabrina/academic_network_project/anp_models/anp_linl_prediction_co_author_hgt_GT_cv_2_10_False_50_0.0_2025_05_02_04_25_43/fold_3/val_results.json"

# with open(file_path, "r") as file:
#     data = json.load(file)

# Inizializza i valori massimi e minimi
#max_train_acc = max_val_acc = float('-inf')
# min_train_loss = min_val_loss = float('inf')

# # Itera sui dati per trovare i valori richiesti
# for epoch_data in data:
#     max_train_acc = max(max_train_acc, epoch_data["train_acc"])
#     max_val_acc = max(max_val_acc, epoch_data["val_acc"])
#     min_train_loss = min(min_train_loss, epoch_data["train_loss"])
#     min_val_loss = min(min_val_loss, epoch_data["val_loss"])

# # Stampa i risultati
# print(f"Massima Train Accuracy: {max_train_acc}")
# print(f"Massima Validation Accuracy: {max_val_acc}")
# print(f"Minima Train Loss: {min_train_loss}")
# print(f"Minima Validation Loss: {min_val_loss}")