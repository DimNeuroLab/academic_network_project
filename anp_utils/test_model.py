import subprocess
import sys
import os
import time
from datetime import datetime
from multiprocessing import Pool

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

AVAILABLE_GPUS = ["0", "0"]
MAX_CONCURRENT_PROCESSES = len(AVAILABLE_GPUS)

LEARNING_RATE  = "0.0001"
ONLY_NEW       = "False"
DROP_PERCENTAGE = "0.0"  

INFOSPHERES = [
    {"type": 0, "params": 0},
    {"type": 1, "params": 5},
    {"type": 2, "params": 10},
    {"type": 3, "params": "[5,2]"},
    {"type": 4, "params": 0},
    {"type": 5, "params": 0},
]

CAMPAIGNS = [
    {
        "script":   "../anp_nn/coauthor_prediction/anp_link_prediction_co_author_hgt.py",
        "dropout":  0.0,
        "campaign": "dropout_study_hgt2025_d00_30",
        "edge_number": 30
    },
     {
        "script":   "../anp_nn/coauthor_prediction/anp_link_prediction_co_author_hgt.py",
        "dropout":  0.0,
        "campaign": "dropout_study_hgt2025_d00_50",
        "edge_number": 50
    },
    {
        "script":   "../anp_nn/coauthor_prediction/anp_link_prediction_co_author_hgt.py",
        "dropout":  0.2,
        "campaign": "dropout_study_hgt2025_d02_30",
        "edge_number": 30
    },
    {
        "script":   "../anp_nn/coauthor_prediction/anp_link_prediction_co_author_hgt.py",
        "dropout":  0.2,
        "campaign": "dropout_study_hgt2025_d02_50",
        "edge_number": 50
    },
     {
        "script":   "../anp_nn/coauthor_prediction/anp_link_prediction_co_author_hgt.py",
        "dropout":  0.4,
        "campaign": "dropout_study_hgt2025_d04_30",
        "edge_number": 30
    },
    {
        "script":   "../anp_nn/coauthor_prediction/anp_link_prediction_co_author_hgt.py",
        "dropout":  0.4,
        "campaign": "dropout_study_hgt2025_d04_50",
        "edge_number": 50
    },
    {
        "script":   "../anp_nn/coauthor_prediction/anp_link_prediction_co_author_hgt_2026.py",
        "dropout":  0.0,
        "campaign": "dropout_study_hgt2026_d00_30",
        "edge_number": 30
    },
     {
        "script":   "../anp_nn/coauthor_prediction/anp_link_prediction_co_author_hgt_2026.py",
        "dropout":  0.0,
        "campaign": "dropout_study_hgt2026_d00_50",
        "edge_number": 50
    },
    {
        "script":   "../anp_nn/coauthor_prediction/anp_link_prediction_co_author_hgt_2026.py",
        "dropout":  0.2,
        "campaign": "dropout_study_hgt2026_d02_30",
        "edge_number": 30
    },
    {
        "script":   "../anp_nn/coauthor_prediction/anp_link_prediction_co_author_hgt_2026.py",
        "dropout":  0.2,
        "campaign": "dropout_study_hgt2026_d02_50",
        "edge_number": 50
    },
     {
        "script":   "../anp_nn/coauthor_prediction/anp_link_prediction_co_author_hgt_2026.py",
        "dropout":  0.4,
        "campaign": "dropout_study_hgt2026_d04_30",
        "edge_number": 30
    },
    {
        "script":   "../anp_nn/coauthor_prediction/anp_link_prediction_co_author_hgt_2026.py",
        "dropout":  0.4,
        "campaign": "dropout_study_hgt2026_d04_50",
        "edge_number": 50
    }
]


def execute_experiment(args):
    cmd, label = args
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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

    for camp in CAMPAIGNS:
        script   = camp["script"]
        dropout  = camp["dropout"]
        edge_number = camp["edge_number"]
        campaign = camp["campaign"]

        if not os.path.exists(script):
            print(f"[WARNING] Script not found, skipping: {script}")
            continue

        for info in INFOSPHERES:
            if (info["type"] == 1 or info["type"] == 0) and script.endswith("hgt.py") and dropout == 0.0:
                print(f"Already processed: {script} with infosphere type {info['type']}")
                continue
            if (info["type"] == 1 or info["type"] == 0) and script.endswith("hgt.py") and dropout == 0.2 and edge_number == 30:
                print(f"Already processed: {script} with infosphere type {info['type']} and dropout {dropout}")
                continue
            cmd = [
                sys.executable, script,
                LEARNING_RATE,          # argv[1]
                str(info["type"]),      # argv[2]
                str(info["params"]),    # argv[3]
                ONLY_NEW,               # argv[4]
                str(edge_number),       # argv[5]
                DROP_PERCENTAGE,        # argv[6]
                "TBD",                  # argv[7] — GPU, assegnato sotto
                campaign,               # argv[8]
                str(dropout),           # argv[9] — dropout_p (NUOVO)
            ]
            label = (
                f"Model={os.path.basename(script)} | "
                f"Dropout={dropout} | "
                f"Infosphere={info['type']} | "
                f"Campaign={campaign}"
            )
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