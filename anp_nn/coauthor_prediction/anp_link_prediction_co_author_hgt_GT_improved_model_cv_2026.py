import json
import os
import sys
import csv
import random
import uuid
import math
from datetime import datetime
import copy

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    roc_auc_score, average_precision_score,
    brier_score_loss, ndcg_score
)
from torch_geometric.loader import LinkNeighborLoader
from tqdm import tqdm

EXPERIMENT_ID = str(uuid.uuid4())

BATCH_SIZE = 4096
YEAR = 2019
ROOT = "../anp_data"
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device(f'cuda:{sys.argv[7]}' if len(sys.argv) > 7 and torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available() and 'cuda' in DEVICE.type:
    torch.cuda.set_device(DEVICE)

from academic_network_project.anp_core.anp_dataset import ANPDataset
import academic_network_project.anp_core.anp_utils as anp_utils
import academic_network_project.anp_nn.models_2025 as models_2025
from academic_network_project.anp_utils.db_logger import log_experiment

learning_rate = float(sys.argv[1])
infosphere_type = int(sys.argv[2])
infosphere_parameters = sys.argv[3]
only_new = sys.argv[4].lower() == 'true'
edge_number = int(sys.argv[5])
drop_percentage = float(sys.argv[6])
campaign_name = sys.argv[8] if len(sys.argv) > 8 else "default_campaign"
dropout_p = float(sys.argv[9]) if len(sys.argv) > 9 else 0.2

MAX_EPOCHS = 100
PATIENCE = 5
MIN_EPOCHS_BEFORE_STOP = 10
CV_FOLDS = 4

experiment_label = f"{campaign_name}_{infosphere_type}_{infosphere_parameters}_Complete_Procedure"

current_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
PATH = f"../anp_models_{campaign_name}/{os.path.basename(sys.argv[0][:-3])}_{infosphere_type}_{infosphere_parameters}_{only_new}_{edge_number}_{drop_percentage}_{current_date}/"
os.makedirs(PATH, exist_ok=True)

with open(os.path.join(PATH, 'info.json'), 'w') as json_file:
    json.dump({
        'lr': learning_rate, 'infosphere_type': infosphere_type, 'infosphere_parameters': infosphere_parameters,
        'only_new': only_new, 'edge_number': edge_number, 'drop_percentage': drop_percentage,
        'campaign_name': campaign_name, 'data': []
    }, json_file)

print(f"Experiment ID: {EXPERIMENT_ID}")
log_experiment({
    "learning_rate": learning_rate, "infosphere_type": infosphere_type,
    "infosphere_parameters": infosphere_parameters, "only_new": only_new,
    "edge_number": edge_number, "drop_percentage": drop_percentage,
    "timestamp": current_date
}, campaign=campaign_name, experiment_id=EXPERIMENT_ID, experiment_label=experiment_label)

dataset = ANPDataset(root=ROOT)
data = dataset[0]

anp_utils.anp_add_infosphere(
    data=data, infosphere_type=infosphere_type, infosphere_parameters=infosphere_parameters,
    drop_percentage=drop_percentage, root=ROOT, device=DEVICE, year=YEAR
)

coauthor_function = anp_utils.get_difference_author_edge_year if only_new else anp_utils.get_author_edge_year
coauthor_year = YEAR if only_new else YEAR + 1
coauthor_file = (
    f"{ROOT}/processed/difference_author_edge{coauthor_year}.pt"
    if only_new else
    f"{ROOT}/processed/author_edge{coauthor_year}.pt"
)

if os.path.exists(coauthor_file):
    data['author', 'co_author', 'author'].edge_index = torch.load(coauthor_file, map_location=DEVICE)["author"]
else:
    author_edge = coauthor_function(data, coauthor_year, DEVICE)
    data['author', 'co_author', 'author'].edge_index = author_edge["author"]
    torch.save(author_edge, coauthor_file)
data['author', 'co_author', 'author'].edge_label = None

data['paper'].x = data['paper'].x.to(torch.float)
data = T.ToUndirected()(data)
data = data.to('cpu')

data_copy = data.clone()
del data_copy['author', 'co_author', 'author']


def _make_link_split_transform():
    return T.RandomLinkSplit(
        num_val=0, num_test=0, neg_sampling_ratio=1.0,
        add_negative_train_samples=True,
        edge_types=('author', 'co_author', 'author')
    )


def _make_link_loader(graph_data, shuffle, batch_size=1024):
    return LinkNeighborLoader(
        data=graph_data, num_neighbors=[edge_number, 10],
        edge_label_index=(
            ('author', 'co_author', 'author'),
            graph_data['author', 'co_author', 'author'].edge_label_index
        ),
        edge_label=graph_data['author', 'co_author', 'author'].edge_label,
        batch_size=batch_size, shuffle=shuffle,
        num_workers=4, prefetch_factor=2, persistent_workers=True
    )


def compute_extended_metrics(labels, probs):
    labels_np = labels.long().cpu().numpy()
    probs_np = probs.cpu().numpy()
    preds_np = np.round(probs_np).astype(int)

    acc = accuracy_score(labels_np, preds_np)
    prec = precision_score(labels_np, preds_np, zero_division=0)
    rec = recall_score(labels_np, preds_np, zero_division=0)
    f1 = f1_score(labels_np, preds_np, zero_division=0)
    mcc = matthews_corrcoef(labels_np, preds_np)

    try:
        roc_auc = roc_auc_score(labels_np, probs_np)
    except ValueError:
        roc_auc = 0.5
    try:
        pr_auc = average_precision_score(labels_np, probs_np)
    except ValueError:
        pr_auc = 0.0
    try:
        brier = brier_score_loss(labels_np, probs_np)
    except ValueError:
        brier = 1.0

    return acc, prec, rec, f1, mcc, roc_auc, pr_auc, brier


def compute_ranking_metrics(labels, probs):
    labels_np = labels.long().cpu().numpy()
    probs_np = probs.cpu().numpy()
    if len(probs_np) == 0 or len(np.unique(labels_np)) < 2:
        return 0.0, 0.0, 0.0

    sorted_indices = np.argsort(-probs_np)
    sorted_labels = labels_np[sorted_indices]

    positive_ranks = np.where(sorted_labels == 1)[0] + 1
    mrr = (1.0 / positive_ranks[0]) if len(positive_ranks) > 0 else 0.0

    ndcg_all = ndcg_score([labels_np], [probs_np])
    ndcg_100 = ndcg_score([labels_np], [probs_np], k=100)
    return mrr, ndcg_all, ndcg_100


def compute_precision_at_k_dict(labels, probs, ks=(10, 50, 100, 500)):
    if len(probs) == 0:
        return {f"p_at_{k}": 0 for k in ks}
    sorted_idx = probs.argsort(descending=True)
    return {
        f"p_at_{k}": labels[sorted_idx[:min(k, len(probs))]].float().mean().item()
        for k in ks
    }


def compute_stratified_roc_auc(labels, probs, src_nodes, dst_nodes, degree_tensor, threshold=2):
    labels_np = labels.long().cpu().numpy()
    probs_np = probs.cpu().numpy()
    src_degrees = degree_tensor[src_nodes].cpu().numpy()
    dst_degrees = degree_tensor[dst_nodes].cpu().numpy()

    is_cold_start = (src_degrees <= threshold) | (dst_degrees <= threshold)
    cold_start_auc, hub_auc = 0.5, 0.5
    try:
        if len(np.unique(labels_np[is_cold_start])) > 1:
            cold_start_auc = roc_auc_score(labels_np[is_cold_start], probs_np[is_cold_start])
    except Exception:
        pass
    try:
        if len(np.unique(labels_np[~is_cold_start])) > 1:
            hub_auc = roc_auc_score(labels_np[~is_cold_start], probs_np[~is_cold_start])
    except Exception:
        pass
    return cold_start_auc, hub_auc


def label_smoothed_bce(pred, target, smoothing=0.1):
    target = target * (1 - smoothing) + 0.5 * smoothing
    return F.binary_cross_entropy_with_logits(pred, target)


def train_epoch(model, optimizer, loader):
    model.train()
    total_loss, total_examples = 0, 0
    all_preds, all_labels = [], []
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(DEVICE)
        edge_label_index = batch['author', 'author'].edge_label_index
        edge_label = batch['author', 'author'].edge_label
        del batch['author', 'co_author', 'author']

        batch['author'].x = model.embedding_author(batch['author'].n_id)
        batch['topic'].x = model.embedding_topic(batch['topic'].n_id)

        optimizer.zero_grad()
        pred = model(batch.x_dict, batch.edge_index_dict, edge_label_index)
        loss = label_smoothed_bce(pred, edge_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
        all_preds.append(torch.sigmoid(pred).detach().cpu())
        all_labels.append(edge_label.detach().cpu())

    return total_loss / total_examples, torch.cat(all_preds), torch.cat(all_labels)


@torch.no_grad()
def eval_epoch(model, loader, desc="Evaluating"):
    model.eval()
    total_loss, total_examples = 0, 0
    all_preds, all_labels, all_src, all_dst = [], [], [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = batch.to(DEVICE)
        edge_label_index = batch['author', 'author'].edge_label_index
        edge_label = batch['author', 'author'].edge_label

        global_src = batch['author'].n_id[edge_label_index[0]]
        global_dst = batch['author'].n_id[edge_label_index[1]]
        del batch['author', 'co_author', 'author']

        batch['author'].x = model.embedding_author(batch['author'].n_id)
        batch['topic'].x = model.embedding_topic(batch['topic'].n_id)

        pred = model(batch.x_dict, batch.edge_index_dict, edge_label_index)
        loss = F.binary_cross_entropy_with_logits(pred, edge_label)

        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
        all_preds.append(torch.sigmoid(pred).cpu())
        all_labels.append(edge_label.cpu())
        all_src.append(global_src.cpu())
        all_dst.append(global_dst.cpu())

    return (
        total_loss / total_examples,
        torch.cat(all_preds), torch.cat(all_labels),
        torch.cat(all_src), torch.cat(all_dst)
    )


# ==========================================
# 0. PREPARE TEST SET (STRICT HOLDOUT - FOLD 4)
# ==========================================
print("\n[Step 0] Isolating Strict Test Set (Fold 4)...")
sub_graph_test = anp_utils.anp_simple_filter_data(data, root=ROOT, folds=[4], max_year=YEAR)
test_data, _, _ = _make_link_split_transform()(sub_graph_test)
test_loader = _make_link_loader(test_data, shuffle=False)

num_total_nodes = data['author'].num_nodes
train_degrees = torch.bincount(
    anp_utils.anp_simple_filter_data(
        data, root=ROOT, folds=[0, 1, 2, 3], max_year=YEAR
    )['author', 'co_author', 'author'].edge_index.flatten(),
    minlength=num_total_nodes
)

# ==========================================
# 1. PHASE 1: CV PROFILING
# ==========================================
cv_optimal_epochs = []
results_cv = []
cv_history_all_folds = {} # DIZIONARIO PER LO STORICO DELLA CV

print(f"\n[Step 1] Starting Profiling CV (Folds 0-{CV_FOLDS - 1}) to find Optimal Stopping Epoch...")
for val_fold in range(CV_FOLDS):
    print(f"\n--- Profiling Fold {val_fold} ---")
    train_folds = [f for f in range(CV_FOLDS) if f != val_fold]

    sub_graph_train = anp_utils.anp_simple_filter_data(data, root=ROOT, folds=train_folds, max_year=YEAR)
    sub_graph_val = anp_utils.anp_simple_filter_data(data, root=ROOT, folds=[val_fold], max_year=YEAR)

    cv_train_data, _, _ = _make_link_split_transform()(sub_graph_train)
    cv_val_data, _, _ = _make_link_split_transform()(sub_graph_val)

    cv_train_loader = _make_link_loader(cv_train_data, shuffle=True)
    cv_val_loader = _make_link_loader(cv_val_data, shuffle=False)

    cv_model = models_2025.Model(hidden_channels=32, data=data_copy, dropout_p=dropout_p).to(DEVICE)
    cv_optimizer = torch.optim.Adam(cv_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    cv_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        cv_optimizer, T_max=MAX_EPOCHS, eta_min=1e-6
    )

    best_val_loss = np.inf
    best_val_metrics = None
    patience_counter = 0
    best_epoch_for_this_fold = 1
    
    fold_history = [] # LISTA PER LO STORICO DI QUESTO FOLD

    for epoch in range(1, MAX_EPOCHS + 1):
        # Recuperiamo anche i risultati di train
        t_loss, t_preds, t_labels = train_epoch(cv_model, cv_optimizer, cv_train_loader)
        v_loss, v_preds, v_labels, _, _ = eval_epoch(cv_model, cv_val_loader, "Validating")
        cv_scheduler.step()
        
        # Calcoliamo le metriche estese per train e validation ad ogni epoca
        t_acc, t_prec, t_rec, t_f1, t_mcc, t_roc, t_pr, t_brier = compute_extended_metrics(t_labels, t_preds)
        v_acc, v_prec, v_rec, v_f1, v_mcc, v_roc, v_pr, v_brier = compute_extended_metrics(v_labels, v_preds)

        # Salviamo i risultati dell'epoca
        fold_history.append({
            'epoch': epoch,
            'train_loss': t_loss, 'train_acc': t_acc, 'train_prec': t_prec, 'train_rec': t_rec, 
            'train_f1': t_f1, 'train_mcc': t_mcc, 'train_roc': t_roc, 'train_pr': t_pr, 'train_brier': t_brier,
            'val_loss': v_loss, 'val_acc': v_acc, 'val_prec': v_prec, 'val_rec': v_rec, 
            'val_f1': v_f1, 'val_mcc': v_mcc, 'val_roc': v_roc, 'val_pr': v_pr, 'val_brier': v_brier
        })

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_epoch_for_this_fold = epoch
            patience_counter = 0
            best_val_metrics = (v_acc, v_prec, v_rec, v_f1, v_mcc, v_roc, v_pr, v_brier)
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE and epoch >= MIN_EPOCHS_BEFORE_STOP:
            break
            
    cv_history_all_folds[f"fold_{val_fold}"] = fold_history

    v_acc, _, _, v_f1, v_mcc, v_roc, v_pr, _ = best_val_metrics
    print(f"-> Best epoch for Fold {val_fold}: {best_epoch_for_this_fold} (Val ROC: {v_roc:.4f})")
    cv_optimal_epochs.append(best_epoch_for_this_fold)
    results_cv.append({
        'val_fold': val_fold,
        'best_epoch': best_epoch_for_this_fold,
        'val_loss': best_val_loss,
        'val_acc': v_acc,
        'val_roc': v_roc
    })

target_epochs = math.ceil(sum(cv_optimal_epochs) / len(cv_optimal_epochs))
print(f"\n>>> CV Profiling Complete. Average Optimal Stopping Epoch: {target_epochs} <<<")

# ==========================================
# 2. PHASE 2: MEGA-RETRAINING
# ==========================================
print(f"\n[Step 2] Mega-Retraining final model on Folds 0-{CV_FOLDS - 1} for exactly {target_epochs} epochs...")
sub_graph_mega = anp_utils.anp_simple_filter_data(data, root=ROOT, folds=list(range(CV_FOLDS)), max_year=YEAR)
mega_data, _, _ = _make_link_split_transform()(sub_graph_mega)
mega_train_loader = _make_link_loader(mega_data, shuffle=True)

final_model = models_2025.Model(hidden_channels=32, data=data_copy, dropout_p=dropout_p).to(DEVICE)
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=learning_rate, weight_decay=1e-4)
final_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    final_optimizer, T_max=target_epochs, eta_min=1e-6
)

mega_training_history = [] # LISTA PER LO STORICO DEL MEGA-TRAINING

for epoch in range(1, target_epochs + 1):
    t_loss, t_preds, t_labels = train_epoch(final_model, final_optimizer, mega_train_loader)
    final_scheduler.step()
    
    # Calcoliamo le statistiche
    t_acc, t_prec, t_rec, t_f1, t_mcc, t_roc, t_pr, t_brier = compute_extended_metrics(t_labels, t_preds)
    
    # Salviamo le statistiche dell'epoca
    mega_training_history.append({
        'epoch': epoch,
        'train_loss': t_loss, 'train_acc': t_acc, 'train_prec': t_prec, 'train_rec': t_rec,
        'train_f1': t_f1, 'train_mcc': t_mcc, 'train_roc': t_roc, 'train_pr': t_pr, 'train_brier': t_brier
    })
    
    print(f"Final Retraining Epoch {epoch}/{target_epochs} | Loss: {t_loss:.4f} | Acc: {t_acc:.4f} | ROC: {t_roc:.4f}")

# ==========================================
# 3. PHASE 3: FINAL STRICT TEST EVALUATION
# ==========================================
print("\n[Step 3] Evaluating Mega-Model on Strict Test Set (Fold 4)...")
test_loss, test_preds, test_labels, test_src, test_dst = eval_epoch(
    final_model, test_loader, "Testing Final Model"
)

tst_acc, tst_prec, tst_rec, tst_f1, tst_mcc, tst_roc, tst_pr, tst_brier = compute_extended_metrics(test_labels, test_preds)
tst_mrr, tst_ndcg_all, tst_ndcg_100 = compute_ranking_metrics(test_labels, test_preds)
test_pak = compute_precision_at_k_dict(test_labels, test_preds)
cold_auc, hub_auc = compute_stratified_roc_auc(test_labels, test_preds, test_src, test_dst, train_degrees, threshold=2)

final_results = {
    'target_training_epochs': target_epochs,
    'test_loss': test_loss, 'test_acc': tst_acc, 'test_prec': tst_prec, 'test_rec': tst_rec,
    'test_f1': tst_f1, 'test_mcc': tst_mcc, 'test_roc': tst_roc, 'test_pr': tst_pr,
    'test_brier': tst_brier, 'test_mrr': tst_mrr, 'test_ndcg_all': tst_ndcg_all,
    'test_ndcg_100': tst_ndcg_100, 'test_coldstarQt_auc': cold_auc, 'test_hub_auc': hub_auc,
    **{f"test_{k}": v for k, v in test_pak.items()}
}

print("\n" + "=" * 50)
print("FINAL TEST RESULTS (UNSEEN FOLD 4)")
print("=" * 50)
for metric, val in final_results.items():
    print(f"{metric:>25}: {val:.4f}")
print("=" * 50)

torch.save({
    'model_state': final_model.state_dict(),
    'test_labels': test_labels,
    'test_preds': test_preds
}, os.path.join(PATH, "final_mega_model_and_test_preds.pt"))

with open(os.path.join(PATH, 'final_test_results.json'), 'w') as f:
    json.dump({'cv_profiling': results_cv, 'final_test': final_results}, f, indent=4)

# SALVATAGGIO DEI LOG STORICI PER EPOCA
with open(os.path.join(PATH, 'training_history.json'), 'w') as f:
    json.dump({
        'cv_history': cv_history_all_folds,
        'mega_training_history': mega_training_history
    }, f, indent=4)

csv_path = os.path.join(PATH, "final_test_results.csv")
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=list(final_results.keys()))
    writer.writeheader()
    writer.writerow(final_results)

print(f"All done! Final models and metrics saved to {PATH}")