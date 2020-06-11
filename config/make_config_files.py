import os
import shutil
import argparse
import numpy as np
from itertools import product
from collections import OrderedDict

import config_src as config_src

def setup_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, default='', help='config file directory')
    return parser.parse_args()

def write_file(path, cfg_opts):
    with open(path, "w") as outfile:
        for key, val in cfg_opts.items():
            if key == "hidden_layers":
                write_str = f"--{key} "
                for val_i in val:
                    write_str += f"{val_i} "
                write_str += "\n"
                outfile.write(write_str)
            else:
                outfile.write(f"--{key} {val}\n")

def main(config_dir):

    cmd_line_args = config_src.config_opts

    # Parse param dictionary into dict containing variations over a parameter and dict containing static parameter values
    variable_cmd_line_args  = {key : val for key, val in cmd_line_args.items() if isinstance(val, list)}
    static_cmd_line_args  = {key : val for key, val in cmd_line_args.items() if not isinstance(val, list)}

    # Ensure ordering for combinations: order by key
    variable_cmd_line_args = OrderedDict(sorted(variable_cmd_line_args.items(), key=lambda t: t[0]))

    # Get combinations
    value_combinations = list(product(*variable_cmd_line_args.values()))

    # Get all combinations
    ordered_keys = variable_cmd_line_args.keys()
    all_hyper_param_combinations = [{key : val for key, val in zip(ordered_keys, combo_val)} for combo_val in value_combinations]

    # Recombine variable combination and static param dicts, then write to file
    for dict_i, combo_dict in enumerate(all_hyper_param_combinations, 1):
        full_dict = {**combo_dict, **static_cmd_line_args}
        write_file(os.path.join(config_dir, f"exp_{dict_i}.cfg"), full_dict)

    # print(f"Number experiments: {len(value_combinations)}")

if __name__ == '__main__':
    opts = setup_opts()

    config_dir = opts.config_dir
    if os.path.exists(config_dir):
        shutil.rmtree(config_dir)
    os.makedirs(config_dir)

    main(config_dir)
