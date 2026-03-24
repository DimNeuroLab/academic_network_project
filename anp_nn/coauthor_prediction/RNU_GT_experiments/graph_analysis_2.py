import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import re
import numpy as np
from collections import defaultdict

# --- CONFIGURATION ---
RESULTS_DIR = "consistency_results_8a60cf3c"  
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
        return {t1: mix_ratio, t2: 1.0 - mix_ratio}
    return {}

def analyze_cross_environment():
    all_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    if not all_files:
        print(f"Errore: Nessun file CSV trovato in {RESULTS_DIR}")
        return

    print(f"Caricamento di {len(all_files)} file di ambiente...")
    
    records = []
    
    # 1. Caricamento e Filtraggio Dati
    for file in all_files:
        df = pd.read_csv(file)
        
        for _, row in df.iterrows():
            model_info = str(row['model_info'])
            env_str = str(row['environment_a']) 
            correlation = float(row['pearson_correlation'])
            
            # Estrai il nome del modello Single (es. "Single: 2" -> "TP")
            single_match = re.search(r"Single:\s*(\w+)", model_info)
            if single_match:
                single_model = get_mapped_name(single_match.group(1))
                
                records.append({
                    "Environment": env_str,
                    "Env_Composition": parse_environment(env_str),
                    "Single_Model": single_model,
                    "Correlation": correlation
                })

    master_df = pd.DataFrame(records)
    if master_df.empty:
        print("Errore: Nessun record valido trovato.")
        return
        
    sns.set_style("whitegrid")
    
    # ==========================================
    # CHART A: 5 GRAFICI SEPARATI (Leaderboard)
    # ==========================================
    print("Generazione dei 5 grafici 'Chart A'...")
    for base_model in INFOSPHERE_ORDER:
        model_data = master_df[master_df['Single_Model'] == base_model].copy()
        if model_data.empty: continue
            
        model_data = model_data.groupby('Environment')['Correlation'].mean().reset_index()
        model_data = model_data.sort_values('Correlation', ascending=False)
        
        fig_height = max(6, len(model_data) * 0.25)
        plt.figure(figsize=(10, fig_height))
        
        ax = sns.barplot(data=model_data, x='Correlation', y='Environment', 
                         color=PALETTE.get(base_model, '#333'))
        
        plt.title(f"Chart A: {base_model} - Simulation Leaderboard\n(Correlation across Mixed Environments)")
        min_corr = model_data['Correlation'].min()
        plt.xlim(max(0.0, min_corr - 0.05), 1.0)
        plt.xlabel("Average Pearson Correlation")
        plt.ylabel("Target Mixed Environment")
        
        for i, v in enumerate(model_data['Correlation']):
            ax.text(v + 0.002, i, f"{v:.3f}", va='center', fontsize=8)
            
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"Chart_A_{base_model}_Environments.png"))
        plt.close()

    # ==========================================
    # CHART B: 5 HEATMAP 2D (Pairwise Infospheres)
    # ==========================================
    print("Generazione dei 5 grafici 'Chart B' (Heatmap 2D)...")
    
    for base_model in INFOSPHERE_ORDER:
        model_data = master_df[master_df['Single_Model'] == base_model]
        if model_data.empty: continue
        
        # Inizializziamo una matrice 5x5 vuota
        matrix_data = pd.DataFrame(index=INFOSPHERE_ORDER, columns=INFOSPHERE_ORDER, dtype=float)
        
        # Raccogliamo le correlazioni per ogni coppia di Infosphere
        pair_corrs = defaultdict(list)
        
        for _, row in model_data.iterrows():
            comp_keys = list(row['Env_Composition'].keys())
            corr = row['Correlation']
            
            # Se l'ambiente è un mix di due infosphere
            if len(comp_keys) == 2:
                t1, t2 = comp_keys[0], comp_keys[1]
                pair_corrs[(t1, t2)].append(corr)
                pair_corrs[(t2, t1)].append(corr) # Rende la matrice simmetrica
            # Nel caso di un ambiente puro (se presente nei dati)
            elif len(comp_keys) == 1:
                t1 = comp_keys[0]
                pair_corrs[(t1, t1)].append(corr)
                
        # Calcoliamo la media per ogni coppia e popoliamo la matrice
        for t1 in INFOSPHERE_ORDER:
            for t2 in INFOSPHERE_ORDER:
                if pair_corrs[(t1, t2)]:
                    matrix_data.loc[t1, t2] = np.mean(pair_corrs[(t1, t2)])

        # Generiamo la heatmap 5x5
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix_data, annot=True, cmap="YlGnBu", vmax=1.0, fmt=".3f", 
                    cbar_kws={'label': 'Mean Pearson Correlation'})
        
        plt.title(f"Chart B: {base_model} - Pairwise Simulation Proficiency\n(Average correlation on environments mixing Row & Column)")
        plt.ylabel("Infosphere A in Mix")
        plt.xlabel("Infosphere B in Mix")
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"Chart_B_{base_model}_2D_Heatmap.png"))
        plt.close()

    # ==========================================
    # CHART C: ROBUSTNESS DISTRIBUTION (Boxplot)
    # ==========================================
    print("Generazione Chart C: Robustness Boxplot...")
    plt.figure(figsize=(10, 6))
    
    plot_data = master_df[master_df['Single_Model'].isin(INFOSPHERE_ORDER)]
    
    sns.boxplot(data=plot_data, x='Single_Model', y='Correlation', 
                order=INFOSPHERE_ORDER, palette=PALETTE)
    sns.stripplot(data=plot_data, x='Single_Model', y='Correlation', 
                  order=INFOSPHERE_ORDER, color=".2", alpha=0.5, jitter=True)
    
    plt.title("Chart C: Simulation Robustness & Variance\n(How consistent is each Base Model across all environments?)")
    plt.xlabel("Base Model")
    plt.ylabel("Pearson Correlation vs Mixed Environments")
    plt.ylim(max(0.0, plot_data['Correlation'].min() - 0.05), 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Chart_C_Robustness_Boxplot.png"))
    plt.close()

    print("\nAnalisi completata! I grafici sono stati salvati nella cartella corrente.")

if __name__ == "__main__":
    analyze_cross_environment()