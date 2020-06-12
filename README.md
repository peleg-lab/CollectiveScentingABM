# Collective Olfactory Communication in Honey Bees
# Agent-Based Model 

## Overview:

To become a coherent swarm, worker bees locate their queen by tracking her pheromones, but how can distant individuals exploit these chemical signals which decay rapidly in space and time? In our study [(Nguyen et al. 2020)](https://www.biorxiv.org/content/10.1101/2020.05.23.112540v1), we show that bees use the scenting behavior to collectively create a communication network to propagate pheromone signals. This repository provides Python code for the agent-based model (AMB) that simulates the collective communication that allows bees to localize the queen and form a swarm around her. Full simulations shown in the paper were run on compute clusters; default values for simulation time and arena size provided here results in a smaller and shorter example that could be run locally at a reasonable time.

## Main requirements (versions tested on):
- Python 3.8.3
- NumPy 1.17.4
- H5py 2.10.0
- Matplotlib 3.1.1

The complete list of required packages provided in *requirements.txt*, which you can install in your environment with the command `pip install -r requirements.txt`. Setting up a Python virtual environment, such as [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), is highly recommended.

## Model usage:
To set up parameters for the simulation, navigate to the file *config_src.py* under *config/*. Free parameters of the model discussed in the paper:
- `num_workers`: Number of worker bees in arena (default: 50)
- `worker_threshold`: Concentration threshold that activates a bee from random walk to either scenting or directed walk state (default: 0.01)
- `worker_bias_scalar`: Magnitude of the directional diffusion of pheromone released by a bee (default: 40)

`python run_simulation.py` runs a simulation with the provided parameters from *config/config_src.py*

### Output:
In the folder *experiments*, a subfolder will be created for this particular simulation with the name format *N{num_workers}_T{worker_threshold}_wb{worker_bias_scalar}_seed{random seed}*. A *cfg* file is created in this folder to record the model parameters for this simulation. The file *bee_hist.h5* contains time-series data for the position, state, scenting direction, distance from the queen of all bees. The file *envir_hist.h5* contains the time-series data for the environmental maps of pheromone diffusion and decay created by the collective scenting of the bees.


## Video visualization:
Example: `python make_movie.py -p N50_T0.01000_wb30.0_seed42 -r 5 -s 1` processes the h5 data files to visualize the simulation in a video.

Command line parameters:
- `-p` or `--path`: Path to the experiment folder in the folder *experiments* (default: `N50_T0.01000_wb30.0_seed42`)
- `-r` or `--fps`: Frame per second for output movie (default: 5)
- `-s` or `--stepsize`: Step size for plotting data for output movie (default: 1)

### Output:
Inside the experiment folder (e.g. *experiments/N50_T0.01000_wb30.0_seed42*), a folder named *movie_frames* will be created to store png's of the visualization plots. When the entire dataset is visualized, a movie in the format *N{num_workers}_T{worker_threshold}_wb{worker_bias_scalar}_seed{random seed}.mp4* is created in the experiment folder. Example movie:

<p align="center">
<img src="doc/example.gif" width="400"/>
<p>

Reference:
Nguyen DMT, Iuzzolino ML, Mankel A, Bozek K, Stephens GJ, Peleg O (2020). Flow-Mediated Collective Olfactory
Communication in Honeybee Swarms. bioRxiv 2020.05.23.112540; doi: https://doi.org/10.1101/2020.05.23.112540.
