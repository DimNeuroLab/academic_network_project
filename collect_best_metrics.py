import os
import json
import csv
import glob
import torch
import numpy as np
from pathlib import Path


BASE_PATH = Path(__file__).parent


def precision_at_k(labels: torch.Tensor, preds: torch.Tensor, k: int) -> float:
    k = min(k, len(preds))
    top_k_idx = preds.argsort(descending=True)[:k]
    return labels[top_k_idx].float().mean().item()


def compute_precision_at_k_dict(labels: torch.Tensor, preds: torch.Tensor) -> dict:
    return {f"p_at_{k}": precision_at_k(labels, preds, k) for k in (10, 50, 100, 500)}


def is_2026_model(info: dict, folder_name: str) -> bool:
    return "dropout_p" in info or "_dp" in folder_name


def load_results_csv(results_path: Path) -> dict:
    with open(results_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows[0] if rows else {}


def compute_pak_from_checkpoint(checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    val_labels = checkpoint.get("val_labels")
    val_preds = checkpoint.get("val_preds")
    train_labels = checkpoint.get("train_labels")
    train_preds = checkpoint.get("train_preds")

    result = {}
    if val_labels is not None and val_preds is not None:
        val_pak = compute_precision_at_k_dict(val_labels, val_preds)
        result.update({f"best_val_{k}": v for k, v in val_pak.items()})
    if train_labels is not None and train_preds is not None:
        train_pak = compute_precision_at_k_dict(train_labels, train_preds)
        result.update({f"best_train_{k}": v for k, v in train_pak.items()})
    return result


def collect_model_record(model_dir: Path) -> dict | None:
    info_path = model_dir / "info.json"
    results_path = model_dir / "results.csv"
    checkpoint_path = model_dir / "labels_and_predictions.pt"

    if not info_path.exists() or not results_path.exists():
        return None

    with open(info_path) as f:
        info = json.load(f)

    folder_name = model_dir.name
    campaign_folder = model_dir.parent.name

    results = load_results_csv(results_path)

    is_26 = is_2026_model(info, folder_name)

    record = {
        "campaign_folder": campaign_folder,
        "model_folder": folder_name,
        "model_version": "2026" if is_26 else "2025",
        "lr": info.get("lr"),
        "infosphere_type": info.get("infosphere_type"),
        "infosphere_parameters": info.get("infosphere_parameters"),
        "only_new": info.get("only_new"),
        "edge_number": info.get("edge_number"),
        "drop_percentage": info.get("drop_percentage"),
        "dropout_p": info.get("dropout_p"),
        "infosphere_type_2": info.get("infosphere_type_2"),
        "infosphere_parameters_2": info.get("infosphere_parameters_2"),
        "infosphere_mix_ratio": info.get("infosphere_mix_ratio"),
    }

    for key, val in results.items():
        record[key] = val

    if not is_26 and checkpoint_path.exists():
        pak = compute_pak_from_checkpoint(checkpoint_path)
        record.update(pak)

    return record


def main():
    dropout_dirs = sorted(BASE_PATH.glob("anp_models_dropout*"))

    if not dropout_dirs:
        print(f"No anp_models_dropout* directories found under {BASE_PATH}")
        return

    for campaign_dir in dropout_dirs:
        model_dirs = sorted(
            d for d in campaign_dir.iterdir() if d.is_dir()
        )

        if not model_dirs:
            print(f"No model subdirs in {campaign_dir.name}, skipping.")
            continue

        records = []
        for model_dir in model_dirs:
            rec = collect_model_record(model_dir)
            if rec is not None:
                records.append(rec)
                print(f"  [OK] {model_dir.name}")
            else:
                print(f"  [SKIP] {model_dir.name} — missing info.json or results.csv")

        if not records:
            continue

        all_keys = list(dict.fromkeys(k for r in records for k in r))
        out_csv = campaign_dir / "summary_best_metrics.csv"

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            for rec in records:
                writer.writerow({k: rec.get(k, "") for k in all_keys})

        print(f"\nSaved: {out_csv} ({len(records)} models)\n")


if __name__ == "__main__":
    main()