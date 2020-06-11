import os
import glob
import argparse
from subprocess import call
from datetime import datetime

import config.config_src as config_src

PY_FILE = "main.py"
BASE_EXPERIMENT_DIR = "experiments"

def run_cfg_generator(base_dir):
    cfg_files_dir = os.path.join(base_dir, "config", "files")
    script_path = os.path.join(base_dir, "config", "make_config_files.py")
    call(["python", script_path, "--config_dir", cfg_files_dir])
    return cfg_files_dir

def run_search(base_dir, cfg_files_dir, experiment_dir):
    cfg_files = glob.glob(f"{cfg_files_dir}/*")

    for cfg_i, cfg_file in enumerate(cfg_files):
        call(["python", f"{PY_FILE}", "--base_dir", f"{experiment_dir}", "--file", f"{cfg_file}"])

if __name__ == '__main__':
    # Set base dir
    base_dir = ""

    # Create cfg files
    cfg_files_dir = run_cfg_generator(base_dir)

    # Parameters
    N = config_src.config_opts["num_workers"]
    T = config_src.config_opts["worker_threshold"]
    wb = config_src.config_opts["worker_bias_scalar"]

    # Run simulation
    print("\n---------- Simulating honey bee communication ----------")
    print(f"Parameters: N={N} -- T={T} -- wb={wb}")
    try:
        run_search(base_dir, cfg_files_dir, BASE_EXPERIMENT_DIR)
        print("\nFin.\n")
    except KeyboardInterrupt:
        print("\nCancelling experiments.")
    except Exception as e:
        print("\n ** Exception Occurred")
        print(e)
