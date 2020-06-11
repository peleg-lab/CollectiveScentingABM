import os
import argparse
import shutil
import numpy as np
from datetime import datetime

import modules.Environment as Environment
import modules.Bees as Bees
import modules.BeeKeeper as BeeKeeper

def config_options():
    class LoadFromFile(argparse.Action):
        def __call__(self, parser, namespace, values, option_sting=None):
            with values as f:
                parser.parse_args(f.read().split(), namespace)
                setattr(namespace, 'config_file', values.name)

    # Instantiate parser
    parser = argparse.ArgumentParser()

    # Setup arguments
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--x_min", type=float, default=-3)
    parser.add_argument("--x_max", type=float, default=3)
    parser.add_argument("--dx", type=float, default=0.01)
    parser.add_argument("--t_min", type=float, default=0)
    parser.add_argument("--t_max", type=float, default=10)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--decay", type=float, default=18.0)
    parser.add_argument("--diffusion_coefficient", type=float, default=0.6)
    parser.add_argument("--queen_x", type=float, default=0)
    parser.add_argument("--queen_y", type=float, default=0)
    parser.add_argument("--queen_bias_scalar", type=float, default=0.0)
    parser.add_argument("--queen_emission_frequency", type=float, default=80)
    parser.add_argument("--queen_initial_concentration", type=float, default=0.0575)
    parser.add_argument("--num_workers", type=int, default=80)
    parser.add_argument("--worker_threshold", type=float, default=0.010)
    parser.add_argument("--worker_bias_scalar", type=float, default=10.0)
    parser.add_argument("--worker_wait_period", type=int, default=80)
    parser.add_argument("--worker_step_size", type=float, default=0.1)
    parser.add_argument("--worker_initial_concentration", type=float, default=0.0575)
    parser.add_argument("--worker_trans_prob", type=float, default=0.4)
    parser.add_argument("--culling_threshold", type=float, default=0.0001)
    parser.add_argument("--sensitivity_mode", type=str, default='none')
    parser.add_argument("--enable_probabilistic", type=bool, default=True)
    parser.add_argument("--space_constraint", type=float, default=0.85)
    parser.add_argument("--t_threshold", type=float, default=100)
    parser.add_argument("--measurements_on", type=bool, default=True)
    parser.add_argument("--save_concentration_maps", type=bool, default=False)
    parser.add_argument("--file", type=open, action=LoadFromFile)

    # Separate
    parser.add_argument("--base_dir", type=str, default="experiments")

    # Read arguments from parser
    args = parser.parse_args()

    return args

def directory(config):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if not hasattr(config, 'config_file'):
        cfg_name = 'test'
    else:
        cfg_name = config.config_file.split(os.path.sep)[-1].replace('.cfg', '')

    # Show params on folder title
    N = config.num_workers
    T = config.worker_threshold
    wb = config.worker_bias_scalar
    seed = config.random_seed
    params_name = f"N{N}_T{T:0.5f}_wb{wb}_seed{seed}"
    model_dir = os.path.join(config.base_dir, f"{params_name}")

    os.makedirs(model_dir, exist_ok=True)

    # Add config file to model dir
    if hasattr(config, 'config_file'):
        shutil.copyfile(config.config_file, os.path.join(model_dir, f"{cfg_name}.cfg"))

    return model_dir

def world_parameters(cfg, model_dir):
    bee_keeper_params = {
        "bee_path"         : os.path.join(model_dir, "bee_hist.h5"),
        "environment_path" : os.path.join(model_dir, "envir_hist.h5"),
        "src_path"         : os.path.join(model_dir, "src_hist.npy"),
        "sleeping"         : not cfg.measurements_on,
        "save_concentration_maps" : cfg.save_concentration_maps
    }

    environment_params = {
        "x_min"      : cfg.x_min,
        "x_max"      : cfg.x_max,
        "dx"         : cfg.dx,
        "t_min"      : cfg.t_min,
        "t_max"      : cfg.t_max,
        "dt"         : cfg.dt,
        "D"          : cfg.diffusion_coefficient,
        "decay_rate" : cfg.decay,
        "culling_threshold" : cfg.culling_threshold
    }

    queen_params  = {
        "num"                : -1,
        "x"                  : cfg.queen_x,
        "y"                  : cfg.queen_y,
        "A"                  : cfg.queen_initial_concentration,
        "wb"                 : cfg.queen_bias_scalar,
        "emission_frequency" : cfg.queen_emission_frequency,
    }

    bee_params = {
        "x_min"            : cfg.x_min,
        "x_max"            : cfg.x_max,
        "init_stddev"      : cfg.space_constraint,
        "A"                : cfg.worker_initial_concentration,
        "threshold"        : cfg.worker_threshold,
        "wb"               : cfg.worker_bias_scalar,
        "wait_period"      : cfg.worker_wait_period,
        "step_size"        : cfg.worker_step_size,
        "probabilistic"    : cfg.enable_probabilistic,
        "trans_prob"       : cfg.worker_trans_prob,
        "sensitivity_mode" : cfg.sensitivity_mode
    }

    world_params = {
        "bee_keeper" : bee_keeper_params,
        "environment" : environment_params,
        "queen" : queen_params,
        "worker" : bee_params
    }

    return world_params

def convert_index_to_xy(idx, idx_min=0, idx_max=600, xy_min=-3, xy_max=3):
    xy = np.interp(idx, [idx_min, idx_max], [xy_min, xy_max])
    return xy

def generate_points_with_min_distance(num_bees, shape, min_dist):
    # Compute grid shape based on number of points
    width_ratio = shape[1] / shape[0]
    num_y = (np.sqrt(num_bees / width_ratio)) + 1
    num_x = (num_bees / num_y) + 1

    # Create regularly spaced points
    x = np.linspace(0., shape[1], num_x)[1:-1]
    y = np.linspace(0., shape[0], num_y)[1:-1]
    coords = np.stack(np.meshgrid(x, y), -1).reshape(-1,2)

    # Compute spacing
    init_dist = np.min((x[1]-x[0], y[1]-y[0]))

    # Perturb points
    max_movement = (init_dist - min_dist) / 2
    noise = np.random.uniform(low=-max_movement,
                            high=max_movement,
                            size=(len(coords), 2))
    coords += noise
    return coords

def create_bees(coords, dim, cfg, bee_params):
    np.random.shuffle(coords)
    bees = []
    for bee_i in range(cfg.num_workers):
        bee_params = bee_params
        bee_params["num"] = bee_i
        bee = Bees.Worker(bee_params)

        # Give bee initial xy position
        bee_x_idx = coords[bee_i][0]
        bee_y_idx = coords[bee_i][1]
        bee_x = convert_index_to_xy(bee_x_idx, idx_min=0, idx_max=dim, xy_min=cfg.x_min, xy_max=cfg.x_max)
        bee_y = convert_index_to_xy(bee_y_idx, idx_min=0, idx_max=dim, xy_min=cfg.x_min, xy_max=cfg.x_max)
        bee.x = bee_x
        bee.y = bee_y

        # Append to list of bee objects
        bees.append(bee)
    return bees

def world_objects(cfg_options, world_params):
    dim = len(np.arange(cfg_options.x_min, cfg_options.x_max, cfg_options.dx))+1

    environment = Environment.Environment(world_params["environment"])
    queen_bee = Bees.Queen(world_params["queen"])
    bee_keeper = BeeKeeper.BeeKeeper(world_params["bee_keeper"])

    # Worker bee objects
    coords = generate_points_with_min_distance(cfg_options.num_workers*2, shape=(dim, dim), min_dist=10)
    bees = create_bees(coords, dim, cfg_options, world_params["worker"])

    world_objs = {
        "environment" : environment,
        "queen_bee" : queen_bee,
        "bees" : bees,
        "bee_keeper" : bee_keeper
    }

    return world_objs
