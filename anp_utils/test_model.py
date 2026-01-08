import subprocess
import sys
import os
import time
from datetime import datetime
from multiprocessing import Pool

# --- EXPERIMENT CONFIGURATION ---

TARGET_SCRIPT = "../anp_nn/coauthor_prediction/anp_link_prediction_co_author_hgt.py"
TARGET_SCRIPT_2 = "../anp_nn/coauthor_prediction/anp_link_prediction_co_author_hgt.py"

# Fixed global parameters
LEARNING_RATE = "0.0001"
ONLY_NEW = "False"
EDGE_NUMBER = "30"
DROP_PERCENTAGE = "0.0"
CAMPAIGN_NAME = "single_infosphere_experiments_6"
CAMPAIGN_NAME_2 = "single_infosphere_experiments_6_3.0"

AVAILABLE_GPUS = ["2", "2", "2"]
MAX_CONCURRENT_PROCESSES = len(AVAILABLE_GPUS)

# Definition of Infospheres (single)
infospheres = [
    {"type": 1, "params": 5},
    {"type": 2, "params": 10},
    {"type": 3, "params": "[5,2]"},
    {"type": 4, "params": 0},
    {"type": 5, "params": 0}
]

def execute_experiment(args):
    cmd, label = args
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # In this configuration, GPU ID is at index 8
    gpu_id = cmd[8]

    print(f"[{timestamp}] STARTED: {label} [GPU {gpu_id}]")

    try:
        time.sleep(1)
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        print(f"[{datetime.now().strftime('%H:%M:%S')}] FINISHED: {label} [GPU {gpu_id}]")
        return f"SUCCESS: {label}"

    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR in {label}:\n{e.output}")
        return f"ERROR: {label}"

def run_parallel_experiments():
    if not os.path.exists(TARGET_SCRIPT):
        print(f"[ERROR] The file '{TARGET_SCRIPT}' was not found.")
        return

    commands_list = []

    # Define the sets of experiments to run
    experiment_sets = [
        (TARGET_SCRIPT, CAMPAIGN_NAME),
        (TARGET_SCRIPT_2, CAMPAIGN_NAME_2)
    ]

    # Generate experiments for both script/campaign sets
    for script, campaign in experiment_sets:
        for info in infospheres:
            cmd = [
                sys.executable, script,
                LEARNING_RATE,                 # argv[1]
                str(info["type"]),             # argv[2]
                str(info["params"]),           # argv[3]
                ONLY_NEW,                      # argv[4]
                EDGE_NUMBER,                   # argv[5]
                DROP_PERCENTAGE,               # argv[6]
                "TBD",                         # argv[7] - Placeholder for GPU
                campaign                       # argv[8]
            ]

            label = f"Script={os.path.basename(script)} | Infosphere={info['type']} | Campaign={campaign}"
            commands_list.append((cmd, label))

    # Assign GPUs cyclically across all generated commands
    for i, (cmd, _) in enumerate(commands_list):
        cmd[8] = AVAILABLE_GPUS[i % len(AVAILABLE_GPUS)]

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting parallel experiments")
    print(f"Total experiments: {len(commands_list)}")
    print(f"Max concurrent processes: {MAX_CONCURRENT_PROCESSES}")
    print("-" * 60)

    # Execute using the multiprocessing pool
    with Pool(processes=MAX_CONCURRENT_PROCESSES) as pool:
        pool.map(execute_experiment, commands_list)

    print("-" * 60)
    print("All experiment sets completed.")

if __name__ == "__main__":
    run_parallel_experiments()