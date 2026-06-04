import subprocess
import sys
import os
import time
from datetime import datetime
from multiprocessing import Pool

from torch import dropout

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# ── GPU config ────────────────────────────────────────────────────────────────
AVAILABLE_GPUS = ["0", "0"]
MAX_CONCURRENT_PROCESSES = len(AVAILABLE_GPUS)

# ── Fixed parameters ──────────────────────────────────────────────────────────
SCRIPT_NAME     = "../anp_nn/coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_cv_2026.py"
LEARNING_RATE  = "0.00001"
ONLY_NEW        = "False"  
EDGE_NUMBER     = "30"
DROP_PERCENTAGE = "0.0"
CAMPAIGN        = "arr_exam_alt"
SORTED = "True"
EMBEDDING_HINDSIGHT = "False"


# ── Configurazioni Infosphere da testare (3 esperimenti totali) ───────────────
INFOSPHERES = [
    {"type": 0, "params": "0"},
]

def execute_experiment(args):
    cmd, label = args
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # cmd[8] corrisponde a sys.argv[7] ovvero la GPU
    gpu_id = cmd[8]

    print(f"[{timestamp}] STARTED: {label} [GPU {gpu_id}]")
    try:
        time.sleep(1)
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(f"[{datetime.now().strftime('%H:%M:%S')}] FINISHED: {label} [GPU {gpu_id}]")
        return f"SUCCESS: {label}"
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR in {label}:\n{e.output}")
        return f"ERROR: {label}"

def run_parallel_experiments():
    commands_list = []

    if not os.path.exists(SCRIPT_NAME):
        print(f"[ERROR] Script non trovato: {SCRIPT_NAME}")
        return

    for info in INFOSPHERES:
            cmd = [
            sys.executable, SCRIPT_NAME,
            LEARNING_RATE,          # argv[1]
            str(info["type"]),      # argv[2]
            str(info["params"]),    # argv[3]
            ONLY_NEW,               # argv[4]
            EDGE_NUMBER,       # argv[5]
            DROP_PERCENTAGE,        # argv[6]
            "TBD",                  # argv[7] — GPU, assegnato sotto
            CAMPAIGN,               # argv[8]
            str(0.0),           # argv[9] — dropout_p (NUOVO)
        ]
        
    label = f"Infosphere={info['type']}_{info['params']} | Campaign={CAMPAIGN}"
    commands_list.append((cmd, label))

    # Assegna GPU in modo ciclico
    for i, (cmd, _) in enumerate(commands_list):
        cmd[8] = AVAILABLE_GPUS[i % len(AVAILABLE_GPUS)]

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting parallel experiments")
    print(f"Total experiments: {len(commands_list)}")
    print(f"Max concurrent processes: {MAX_CONCURRENT_PROCESSES}")
    print("-" * 60)

    with Pool(processes=MAX_CONCURRENT_PROCESSES) as pool:
        results = pool.map(execute_experiment, commands_list)

    print("-" * 60)
    successes = sum(1 for r in results if r.startswith("SUCCESS"))
    print(f"Completati: {successes}/{len(results)}")

if __name__ == "__main__":
    run_parallel_experiments()