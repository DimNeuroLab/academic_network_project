import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import re
import numpy as np

# --- CONFIGURATION ---
# INSERISCI QUI IL NOME DELLA CARTELLA GENERATA DAL TUO SCRIPT (es. "consistency_results_a1b2c3d4")
RESULTS_DIR = "consistency_results_0ed3764c"  
OUTPUT_DIR = "."

# --- MAPPING INFOSPHERE ---
INFOSPHERE_MAP = {
    "1": "HIND", "2": "TP", "3": "TPxT", "4": "NAIS", "5": "LGCN",
    "None": "None"
}
INFOSPHERE_ORDER = ["HIND", "TP", "TPxT", "NAIS", "LGCN"]

PALETTE = {
    "HIND": "#1f77b4", "TP": "#ff7f0e", "TPxT": "#2ca02c", 
    "NAIS": "#d62728", "LGCN": "#9467bd"
}

def get_mapped_name(info_id):
    return INFOSPHERE_MAP.get(str(info_id).strip(), str(info_id).strip())

def parse_environment(env_str):
    """Estrae i tipi di infosphere e i loro pesi dall'ambiente misto."""
    match = re.search(r"Env_(\w+)\+(\w+)_mix([\d\.]+)", str(env_str))
    if match:
        t1 = get_mapped_name(match.group(1))
        t2 = get_mapped_name(match.group(2))
        mix_ratio = float(match.group(3))
        # In base alla tua funzione: Type 1 prende mix_ratio, Type 2 prende (1 - mix_ratio)
        return {t1: mix_ratio, t2: 1.0 - mix_ratio}
    return {}

def analyze_cross_environment():
    all_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    if not all_files:
        print(f"Error: Nessun file CSV trovato in {RESULTS_DIR}")
        return

    print(f"Caricamento di {len(all_files)} file di ambiente...")
    
    records = []
    
    # 1. Caricamento e Filtraggio Dati
    for file in all_files:
        df = pd.read_csv(file)
        # Vogliamo SOLO i confronti tra ORIGINAL (Ambiente) e CANDIDATE (Single Model)
        for _, row in df.iterrows():
            info_a = str(row['model_a_info'])
            info_b = str(row['model_b_info'])
            
            is_a_orig = info_a.startswith("ORIGINAL")
            is_b_orig = info_b.startswith("ORIGINAL")
            
            # Se uno è Original e l'altro è Single
            if is_a_orig != is_b_orig:
                env_str = row['evaluated_on_environment']
                single_info = info_b if is_a_orig else info_a
                
                # Estrai il nome del modello Single (es. "Single: 2" -> "TP")
                single_match = re.search(r"Single:\s*(\w+)", single_info)
                if single_match:
                    single_model = get_mapped_name(single_match.group(1))
                    
                    records.append({
                        "Environment": env_str,
                        "Env_Composition": parse_environment(env_str),
                        "Single_Model": single_model,
                        "Correlation": row['pearson_correlation']
                    })

    master_df = pd.DataFrame(records)
    if master_df.empty:
        print("Errore: Nessun record valido ORIGINAL vs CANDIDATE trovato.")
        return
        
    sns.set_style("whitegrid")
    
    # ==========================================
    # CHART A: GLOBAL LEADERBOARD (Media Generale)
    # ==========================================
    print("Generazione Chart A: Global Leaderboard...")
    global_avg = master_df.groupby('Single_Model')['Correlation'].mean().reset_index()
    global_avg = global_avg.sort_values('Correlation', ascending=False)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=global_avg, x='Correlation', y='Single_Model', 
                     palette=[PALETTE.get(x, '#333') for x in global_avg['Single_Model']])
    plt.title("Chart A: Global Simulation Leaderboard\n(Average Pearson Correlation across ALL Mixed Environments)")
    plt.xlim(max(0.0, global_avg['Correlation'].min() - 0.1), 1.0)
    plt.xlabel("Average Pearson Correlation vs ORIGINAL Environment")
    plt.ylabel("Candidate Single Model")
    
    # Aggiungi i valori sulle barre
    for i, v in enumerate(global_avg['Correlation']):
        ax.text(v + 0.005, i, f"{v:.3f}", va='center')
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Chart_A_Global_Leaderboard.png"))
    plt.close()

    # ==========================================
    # CHART B: WEIGHTED TARGET HEATMAP (Chi simula meglio cosa?)
    # ==========================================
    print("Generazione Chart B: Weighted Target Heatmap...")
    # Calcoliamo una media pesata: Quanto è bravo il Modello X a simulare ambienti in cui l'Infosphere Y è presente?
    target_scores = {s: {t: 0.0 for t in INFOSPHERE_ORDER} for s in INFOSPHERE_ORDER}
    target_weights = {s: {t: 0.0 for t in INFOSPHERE_ORDER} for s in INFOSPHERE_ORDER}
    
    for _, row in master_df.iterrows():
        candidate = row['Single_Model']
        corr = row['Correlation']
        env_comp = row['Env_Composition']
        
        if candidate not in target_scores: continue
            
        for target_infosphere, weight in env_comp.items():
            if target_infosphere in INFOSPHERE_ORDER and weight > 0:
                target_scores[candidate][target_infosphere] += corr * weight
                target_weights[candidate][target_infosphere] += weight

    # Costruisci la matrice finale
    matrix_data = pd.DataFrame(index=INFOSPHERE_ORDER, columns=INFOSPHERE_ORDER, dtype=float)
    for cand in INFOSPHERE_ORDER:
        for target in INFOSPHERE_ORDER:
            w_total = target_weights[cand][target]
            if w_total > 0:
                matrix_data.loc[cand, target] = target_scores[cand][target] / w_total
            else:
                matrix_data.loc[cand, target] = np.nan

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_data, annot=True, cmap="YlGnBu", vmax=1.0, fmt=".3f")
    plt.title("Chart B: Infosphere Simulation Proficiency\n(Rows: Candidate Model | Columns: Dominant Infosphere in Environment)")
    plt.ylabel("Candidate (Single Model)")
    plt.xlabel("Target Infosphere to Simulate (Weighted by presence in Env)")
    plt.savefig(os.path.join(OUTPUT_DIR, "Chart_B_Weighted_Target_Heatmap.png"))
    plt.close()

    # ==========================================
    # CHART C: ROBUSTNESS DISTRIBUTION (Boxplot)
    # ==========================================
    print("Generazione Chart C: Robustness Boxplot...")
    plt.figure(figsize=(10, 6))
    
    # Ordina i boxplot in base all'ordine standard
    plot_data = master_df[master_df['Single_Model'].isin(INFOSPHERE_ORDER)]
    
    sns.boxplot(data=plot_data, x='Single_Model', y='Correlation', 
                order=INFOSPHERE_ORDER, palette=PALETTE)
    sns.stripplot(data=plot_data, x='Single_Model', y='Correlation', 
                  order=INFOSPHERE_ORDER, color=".2", alpha=0.5, jitter=True)
    
    plt.title("Chart C: Simulation Robustness & Variance\n(How consistent is each Single Model across different environments?)")
    plt.xlabel("Candidate Single Model")
    plt.ylabel("Pearson Correlation vs Mixed Environments")
    plt.ylim(max(0.0, plot_data['Correlation'].min() - 0.05), 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Chart_C_Robustness_Boxplot.png"))
    plt.close()

    print("\nAnalisi completata! Grafici salvati nella cartella corrente:")
    print("- Chart_A_Global_Leaderboard.png")
    print("- Chart_B_Weighted_Target_Heatmap.png")
    print("- Chart_C_Robustness_Boxplot.png")

if __name__ == "__main__":
    analyze_cross_environment()