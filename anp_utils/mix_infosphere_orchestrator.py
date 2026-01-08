import subprocess
import sys
import os
import time
from datetime import datetime
from multiprocessing import Pool

# --- EXPERIMENT CONFIGURATION ---

# The name of the main Python script to execute
TARGET_SCRIPT = "../anp_nn/coauthor_prediction/anp_link_prediction_co_author_hgt.py"

# Fixed global parameters
LEARNING_RATE = "0.0001"
ONLY_NEW = "False"
EDGE_NUMBER = "30"
DROP_PERCENTAGE = "0.0"
CAMPAIGN_NAME = "mix_infosphere_experiments_xmas_2025"

# List of available GPU IDs. 
# If you want to run multiple processes on the same GPU, repeat the ID.
# Example for 4 processes on GPU 0: ["0", "0", "0", "0"]
# Example for 2 processes on GPU 0 and 2 processes on GPU 1: ["0", "0", "1", "1"]
AVAILABLE_GPUS = ["2", "2", "2"]

# Maximum number of concurrent processes (should not exceed len(AVAILABLE_GPUS))
MAX_CONCURRENT_PROCESSES = len(AVAILABLE_GPUS)

# Definition of Infospheres and their parameters
# Each dictionary represents a specific infosphere configuration
infospheres = [
    {"type": 1, "params": 5},
    {"type": 2, "params": 10},
    {"type": 3, "params": "[5,2]"},
    {"type": 4, "params": 0},
    {"type": 5, "params": 0}
]

# Mixing ratios to test
mix_ratios = [0.2, 0.4, 0.6, 0.8]

def execute_experiment(args):
    """
    Function executed in parallel by each process in the Pool.
    Args: (cmd, experiment_label)
    """
    cmd, label = args
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # cmd[7] contains the assigned GPU ID
    gpu_id = cmd[8]
    
    print(f"[{timestamp}] STARTED: {label} [GPU {gpu_id}]")
    
    try:
        # Slight delay to prevent race conditions on file resources during startup
        time.sleep(1)
        
        # subprocess.run executes the command and waits for it to finish
        # stdout is captured to keep the main console clean, printed only on error
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] FINISHED: {label} [GPU {gpu_id}]")
        return f"SUCCESS: {label}"
        
    except subprocess.CalledProcessError as e:
        # Print the output only if an error occurs
        error_output = e.output
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR in {label} [GPU {gpu_id}]:\n{error_output}")
        return f"ERROR: {label}"
    
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] FATAL ERROR in {label}: {e}")
        return f"FATAL_ERROR: {label}"


def run_parallel_experiments():
    """
    Prepares the command list and executes them in parallel using multiprocessing.Pool.
    """
    if not os.path.exists(TARGET_SCRIPT):
        print(f"[ERROR] The file '{TARGET_SCRIPT}' was not found. Exiting.")
        return

    commands_list = []
    
    # 1. Generate the list of commands (Anti-repetition logic)
    for i in range(len(infospheres)):
        for j in range(i + 1, len(infospheres)):
            
            info_primary = infospheres[i]
            info_secondary = infospheres[j]
            
            for ratio in mix_ratios:
                # The GPU ID will be assigned later based on the pool slot
                # We use "TBD" as a temporary placeholder
                
                # Command construction based on sys.argv of the target script:
                # 0: script_name
                # 1: lr, 2: type1, 3: param1, 4: only_new, 5: edge_num, 6: drop_pct, 7: gpu
                # 8: type2, 9: param2, 10: mix_ratio
                cmd = [
                    sys.executable, TARGET_SCRIPT,
                    LEARNING_RATE,                  # argv[1]
                    str(info_primary['type']),      # argv[2]
                    str(info_primary['params']),         # argv[3]
                    ONLY_NEW,                       # argv[4]
                    EDGE_NUMBER,                    # argv[5]
                    DROP_PERCENTAGE,                # argv[6]
                    "TBD",                          # argv[7] - GPU Placeholder
                    CAMPAIGN_NAME,                  # argv[8] - Campaign Name
                    str(info_secondary['type']),    # argv[9]
                    str(info_secondary['params']),       # argv[10]
                    str(ratio)                      # argv[11]
                ]
                
                label = f"T1={info_primary['type']}, T2={info_secondary['type']}, Ratio={ratio}"
                commands_list.append((cmd, label))
                
    total_experiments = len(commands_list)
    
    # 2. Assign GPU IDs to commands
    # Cyclic assignment using the AVAILABLE_GPUS list
    for k in range(total_experiments):
        gpu_index = k % len(AVAILABLE_GPUS)
        assigned_gpu_id = AVAILABLE_GPUS[gpu_index]
        
        # Replace the "TBD" placeholder with the actual GPU ID
        commands_list[k][0][8] = assigned_gpu_id 

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting parallel suite.")
    print(f"Total experiments to run: {total_experiments}")
    print(f"Max concurrent processes: {MAX_CONCURRENT_PROCESSES} (GPUs used: {', '.join(sorted(list(set(AVAILABLE_GPUS))))})")
    print("-" * 60)

    # 3. Execute the Pool
    try:
        # Create a pool of workers limited by MAX_CONCURRENT_PROCESSES
        with Pool(processes=MAX_CONCURRENT_PROCESSES) as pool:
            # map_async launches functions and returns results asynchronously
            results = pool.map_async(execute_experiment, commands_list)
            
            # Wait for completion (allows catching KeyboardInterrupt)
            results.get(timeout=None) 
            
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Stop signal detected. Terminating processes...")
        pool.terminate()
        pool.join()
        sys.exit(1)
        
    except Exception as e:
        print(f"\n[FATAL ERROR] Error during Pool execution: {e}")
        pool.terminate()
        pool.join()
        sys.exit(1)

    print("-" * 60)
    print(f"Parallel suite completed. Results saved in their respective directories.")

if __name__ == "__main__":
    run_parallel_experiments()