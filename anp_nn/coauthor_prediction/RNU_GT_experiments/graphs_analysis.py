import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os
import numpy as np
from collections import defaultdict

# --- CONFIGURATION ---
INPUT_CSV = "gt_pairwise_consistency_combined_4a79cfe0.csv"  # Assicurati che il nome sia corretto
OUTPUT_DIR = "."  # Current directory

# --- MAPPING INFOSPHERE ---
INFOSPHERE_MAP = {
    "1": "HIND",
    "2": "TP",
    "3": "TPxT",
    "4": "NAIS",
    "5": "LGCN"
}

INFOSPHERE_ORDER = ["HIND", "TP", "TPxT", "NAIS", "LGCN"]

# Palette colori fissa per consistenza visiva tra tutti i grafici
PALETTE = {
    "HIND": "#1f77b4", # Blu
    "TP": "#ff7f0e",   # Arancione
    "TPxT": "#2ca02c", # Verde
    "NAIS": "#d62728", # Rosso
    "LGCN": "#9467bd"  # Viola
}

# --- HELPER FUNCTIONS ---

def custom_sort_types(types_list):
    """Ordina una lista di infosphere rispettando l'ordine numerico originale."""
    return sorted(list(types_list), key=lambda x: INFOSPHERE_ORDER.index(x) if x in INFOSPHERE_ORDER else 999)

def get_mapped_name(info_id):
    info_id_str = str(info_id).strip()
    return INFOSPHERE_MAP.get(info_id_str, info_id_str)

def parse_model_weights(info_str):
    """
    Estrae i pesi. Aggiornato secondo anp_add_infosphere_mix:
    type_1 usa mix_ratio, type_2 usa (1 - mix_ratio).
    """
    info_str = str(info_str)
    weights = {}
    
    if "Single" in info_str:
        match = re.search(r"Single:\s*(\d+|\[.*?\])", info_str)
        if match:
            t1 = get_mapped_name(match.group(1))
            weights[t1] = 1.0
    else:
        match = re.search(r"(\d+|\[.*?\])\+(\d+|\[.*?\])\s*\(mix:\s*([\d\.]+)\)", info_str)
        if match:
            t1 = get_mapped_name(match.group(1)) 
            t2 = get_mapped_name(match.group(2)) 
            ratio = float(match.group(3))
            
            # Logica corretta dal tuo codice PyTorch
            weights[t1] = round(ratio, 1)
            weights[t2] = round(1.0 - ratio, 1)
            
    return weights

def rebuild_info_string(weights_dict):
    if not weights_dict:
        return "Unknown"
    if len(weights_dict) == 1:
        return f"Single: {list(weights_dict.keys())[0]}"
    else:
        keys = list(weights_dict.keys())
        t1, t2 = keys[0], keys[1]
        ratio = weights_dict[t2]
        return f"{t1}+{t2} (mix: {ratio})"

def analyze_results():
    # 1. Load Data
    csv_path = os.path.join(OUTPUT_DIR, INPUT_CSV)
    if not os.path.exists(csv_path):
        print(f"Error: The file {INPUT_CSV} was not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} comparison pairs.")

    # 2. Pre-process Weights
    df['weights_a'] = df['model_a_info'].apply(parse_model_weights)
    df['weights_b'] = df['model_b_info'].apply(parse_model_weights)
    
    df['mapped_info_a'] = df['weights_a'].apply(rebuild_info_string)
    df['mapped_info_b'] = df['weights_b'].apply(rebuild_info_string)

    sns.set_style("whitegrid")
    
    # ==========================================
    # CHART 1: Pure Models Heatmap (Baseline)
    # ==========================================
    print("\nGenerating Chart 1: Pure Models Heatmap...")
    pure_mask = (df['weights_a'].apply(len) == 1) & (df['weights_b'].apply(len) == 1)
    pure_df = df[pure_mask].copy()
    
    if not pure_df.empty:
        edges = []
        for _, row in pure_df.iterrows():
            t1 = list(row['weights_a'].keys())[0]
            t2 = list(row['weights_b'].keys())[0]
            val = row['pearson_correlation']
            edges.append({'T1': t1, 'T2': t2, 'Corr': val})
            edges.append({'T1': t2, 'T2': t1, 'Corr': val}) 

        unique_types = set([x['T1'] for x in edges])
        for t in unique_types:
            edges.append({'T1': t, 'T2': t, 'Corr': 1.0})

        heatmap_df = pd.DataFrame(edges).drop_duplicates()
        heatmap_pivot = heatmap_df.pivot(index='T1', columns='T2', values='Corr')
        
        # FORZIAMO L'ORDINE NUMERICO
        sorted_types_1 = custom_sort_types(unique_types)
        heatmap_pivot = heatmap_pivot.reindex(index=sorted_types_1, columns=sorted_types_1)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_pivot, annot=True, cmap="coolwarm", vmax=1.0, fmt=".3f")
        plt.title("Chart 1: Compatibility of Pure Infospheres")
        plt.savefig("1_pure_compatibility.png")
        plt.close()

    # ==========================================
    # CHART 2: GLOBAL WEIGHTED COMPATIBILITY
    # ==========================================
    print("Generating Chart 2: Global Weighted Compatibility...")
    weighted_corr_sum = defaultdict(lambda: defaultdict(float))
    total_weights = defaultdict(lambda: defaultdict(float))
    all_types_seen = set()

    for _, row in df.iterrows():
        corr = row['pearson_correlation']
        wa, wb = row['weights_a'], row['weights_b']
        
        for type_a, weight_a in wa.items():
            all_types_seen.add(type_a)
            for type_b, weight_b in wb.items():
                all_types_seen.add(type_b)
                interaction_weight = weight_a * weight_b
                
                weighted_corr_sum[type_a][type_b] += corr * interaction_weight
                total_weights[type_a][type_b] += interaction_weight
                
                if type_a != type_b:
                    weighted_corr_sum[type_b][type_a] += corr * interaction_weight
                    total_weights[type_b][type_a] += interaction_weight

    # FORZIAMO L'ORDINE NUMERICO
    sorted_types = custom_sort_types(all_types_seen)
    matrix_data = []

    for t1 in sorted_types:
        row_data = []
        for t2 in sorted_types:
            w_sum = weighted_corr_sum[t1][t2]
            w_total = total_weights[t1][t2]
            row_data.append(w_sum / w_total if w_total > 0 else np.nan)
        matrix_data.append(row_data)

    weighted_df = pd.DataFrame(matrix_data, index=sorted_types, columns=sorted_types)

    plt.figure(figsize=(10, 8))
    sns.heatmap(weighted_df, annot=True, cmap="viridis", vmax=1.0, fmt=".3f")
    plt.title("Chart 2: Global Infosphere Compatibility\n(Weighted by Mixing Ratio)")
    plt.savefig("2_global_weighted_compatibility.png")
    plt.close()

    # ==========================================
    # CHART 3: Rankings
    # ==========================================
    print("Generating Chart 3: Rankings...")
    top_10 = df.nlargest(10, 'pearson_correlation').sort_values('pearson_correlation', ascending=True)
    bottom_10 = df.nsmallest(10, 'pearson_correlation').sort_values('pearson_correlation', ascending=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    labels_top = top_10.apply(lambda x: f"{x['mapped_info_a']} \nvs\n {x['mapped_info_b']}", axis=1)
    labels_bottom = bottom_10.apply(lambda x: f"{x['mapped_info_a']} \nvs\n {x['mapped_info_b']}", axis=1)

    axes[0].barh(labels_top, top_10['pearson_correlation'], color='green', alpha=0.7)
    axes[0].set_xlim(max(0.8, top_10['pearson_correlation'].min() - 0.005), 1.0)
    axes[0].set_title("Top 10 MOST Similar Pairs")
    
    axes[1].barh(labels_bottom, bottom_10['pearson_correlation'], color='red', alpha=0.7)
    axes[1].set_xlim(bottom_10['pearson_correlation'].min() - 0.01, bottom_10['pearson_correlation'].max() + 0.01)
    axes[1].set_title("Top 10 LEAST Similar Pairs")
    
    plt.tight_layout()
    plt.savefig("3_similarity_rankings.png")
    plt.close()

    # ==========================================
    # CHART 4: ISOLATED INFOSPHERE HEATMAP 
    # ==========================================
    print("Generating Chart 4: Isolated Compatibility Heatmap...")
    # Usa già il sorted_types del Chart 2 che è numericamente ordinato
    isolated_matrix_data = pd.DataFrame(index=sorted_types, columns=sorted_types, dtype=float)
    
    for t1 in sorted_types:
        for t2 in sorted_types:
            if t1 == t2:
                isolated_matrix_data.loc[t1, t2] = 1.0
                continue
                
            allowed_keys = {t1, t2}
            
            def is_isolated(row):
                keys_a = set(row['weights_a'].keys())
                keys_b = set(row['weights_b'].keys())
                return keys_a.issubset(allowed_keys) and keys_b.issubset(allowed_keys)
            
            isolated_df = df[df.apply(is_isolated, axis=1)]
            
            if not isolated_df.empty:
                isolated_matrix_data.loc[t1, t2] = isolated_df['pearson_correlation'].mean()
            else:
                isolated_matrix_data.loc[t1, t2] = np.nan

    plt.figure(figsize=(10, 8))
    sns.heatmap(isolated_matrix_data, annot=True, cmap="coolwarm", vmax=1.0, fmt=".3f")
    plt.title("Chart 4: Isolated Compatibility\n(Only interactions between row & col types)")
    plt.savefig("4_isolated_compatibility.png")
    plt.close()

  # ==========================================
    # CHART 5: MIX RATIO TRAJECTORIES (SIDE-BY-SIDE)
    # ==========================================
    print("Generating Chart 5: Mix Ratio Trajectories (Side-by-Side)...")
    
    # Raccogliamo tutte le correlazioni tra Mix e Single
    mix_data = {} 
    
    for _, row in df.iterrows():
        wa, wb = row['weights_a'], row['weights_b']
        corr = row['pearson_correlation']
        
        if len(wa) == 1 and len(wb) == 2:
            m_key = frozenset(wb.items())
            s_key = list(wa.keys())[0]
            if m_key not in mix_data: mix_data[m_key] = {}
            mix_data[m_key][s_key] = corr
        elif len(wb) == 1 and len(wa) == 2:
            m_key = frozenset(wa.items())
            s_key = list(wb.keys())[0]
            if m_key not in mix_data: mix_data[m_key] = {}
            mix_data[m_key][s_key] = corr

    ordered_bases = custom_sort_types(list(set(k for m in mix_data.keys() for k, v in m)))

    for i, base in enumerate(ordered_bases):
        left_records = []
        right_records = []
        
        for m_key, single_corrs in mix_data.items():
            m_dict = dict(m_key)
            if base in m_dict:
                added_type = [k for k in m_dict.keys() if k != base][0]
                ratio_added = m_dict[added_type] # Quantità dell'infosphere aggiunta
                
                # Se abbiamo la correlazione col modello Base puro (Grafo Destro)
                if base in single_corrs:
                    right_records.append({'Added_Type': added_type, 'Mix_Ratio': ratio_added, 'Correlation': single_corrs[base]})
                # Se abbiamo la correlazione col modello Aggiunto puro (Grafo Sinistro)
                if added_type in single_corrs:
                    left_records.append({'Added_Type': added_type, 'Mix_Ratio': ratio_added, 'Correlation': single_corrs[added_type]})
        
        if not right_records and not left_records:
            continue
            
        df_left = pd.DataFrame(left_records)
        df_right = pd.DataFrame(right_records)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        fig.suptitle(f"Degradation Trajectories for Models based on [{base}]", fontsize=16)

        # Plot Sinistro: Correlazione con l'infosphere della Legenda
        if not df_left.empty:
            sns.lineplot(data=df_left, x='Mix_Ratio', y='Correlation', hue='Added_Type', 
                         marker='o', linewidth=2, markersize=8, ax=axes[0], 
                         palette=PALETTE, hue_order=custom_sort_types(df_left['Added_Type'].unique()))
        
        axes[0].set_title(f"Similarity to pure [Added Infosphere]")
        axes[0].set_xlabel(f"Mix Ratio of Added Infosphere")
        axes[0].set_ylabel("Pearson Correlation")
        axes[0].set_xlim(-0.05, 1.05)
        axes[0].set_ylim(0.5, 1.0)
        if axes[0].get_legend():
            axes[0].legend(title="Added Infosphere", loc='lower right')

        # Plot Destro: Correlazione con l'infosphere BASE (Titolo)
        if not df_right.empty:
            sns.lineplot(data=df_right, x='Mix_Ratio', y='Correlation', hue='Added_Type', 
                         marker='o', linewidth=2, markersize=8, ax=axes[1], 
                         palette=PALETTE, hue_order=custom_sort_types(df_right['Added_Type'].unique()))
            
        axes[1].set_title(f"Similarity to pure [{base}]")
        axes[1].set_xlabel(f"Mix Ratio of Added Infosphere")
        axes[1].set_ylabel("") # Sharey nasconde i numeri, togliamo l'etichetta
        axes[1].set_xlim(-0.05, 1.05)
        if axes[1].get_legend():
            axes[1].legend(title="Added Infosphere", loc='lower left')

        safe_base_name = base.replace('[', '').replace(']', '').replace(',', '_').replace(' ', '')
        plt.tight_layout()
        plt.savefig(f"5_trajectory_{i+1}_base_{safe_base_name}.png")
        plt.close()
if __name__ == "__main__":
    analyze_results()