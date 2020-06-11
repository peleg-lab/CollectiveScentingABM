import numpy as np

config_opts = {
    "verbose"     : True,
    "random_seed" : 42,

    ### WORKER PARAMS ###
    #---- FREE PARAMETERS ----#
    # Vary these values to observe changes in simulation
    "num_workers" : 50,              # min: 0, max: +infinity, recommended: <= 300
    "worker_threshold" : 0.01,       # min: 0, max: +infinity, recommended: 0 to 1.0
    "worker_bias_scalar" : 40.0,     # min: 0, max: +infinity, recommended: 0 to 60
    #-------------------------#

    "worker_wait_period" : 80,
    "worker_step_size" : 0.1,
    "worker_initial_concentration" : [0.0575],
    "worker_trans_prob" : 0.5,
    "enable_probabilistic" : True,

    ### QUEEN PARAMS ###
    "queen_x" : 0,
    "queen_y" : 0,
    "queen_bias_scalar" : 0.0,
    "queen_emission_frequency" : 80,
    "queen_initial_concentration" : 0.0575,

    ### ENVIRONMENT PARAMS ###
    "x_min" : -1.5,
    "x_max" : 1.5,
    "dx" : 0.01,
    "t_min" : 0,
    "t_max" : 25 * (0.05/10),
    "dt" : 0.05 / 10,
    "decay" : [18.0*6],
    "diffusion_coefficient" : [0.6],

    ### OTHER PARAMS ###
    "sensitivity_mode"  : "none",
    "culling_threshold" : 1e-3,
    "space_constraint" : 0.85,
    "t_threshold" : 100,
    "measurements_on" : True
}
