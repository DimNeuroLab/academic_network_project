import os
from supabase import create_client, Client
from dotenv import load_dotenv
from pathlib import Path

# Load env
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")

# Creiamo il client (questo usa HTTPS -> Porta 443 -> Firewall OK)
supabase: Client = create_client(url, key)

def log_experiment(info: dict, campaign: str, experiment_id: str, experiment_label: str):
    try:
        data = {
            "id": experiment_id,
            "campaign": campaign,
            "label": experiment_label,
            "params": info, # Supabase converte dict in JSONB in automatico
            # created_at lo mette supabase se hai il default, o lo passi tu
        }
        # Invece di SQL INSERT, usiamo .insert()
        response = supabase.table("experiments").insert(data).execute()
        print(f"[API] Experiment {experiment_label} logged.")
    except Exception as e:
        print(f"[API ERROR] {e}")

def log_epoch(epoch: int, metrics: dict, experiment_id: str):
    try:
        data = {
            "experiment_id": experiment_id,
            "epoch": epoch,
            "train_loss": metrics.get('train_loss'),
            "val_loss": metrics.get('val_loss'),
            "val_acc": metrics.get('val_acc'),
            "train_cm": metrics.get('cm_train'),
            "val_cm": metrics.get('cm_val')
        }
        supabase.table("metrics").insert(data).execute()
        print(f"[API] Epoch {epoch} metrics logged.")
    except Exception as e:
        print(f"[API ERROR] {e}")