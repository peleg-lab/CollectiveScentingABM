import warnings
warnings.filterwarnings("ignore")
import sys
import numpy as np

import modules.Setup as Setup

def main(cfg_options, environment, queen_bee, bees, bee_keeper):

    print("Begin simulation...")

    try:
        for global_i, t_i in enumerate(environment):

            if cfg_options.verbose:
                sys.stdout.write(f"\rTimestep: {global_i+1}/{environment.t_grid.shape[0]}")
                sys.stdout.flush()

            # ----------------------------------------------------
            # Step 1: Check for and build sources list for current timestep
            # ----------------------------------------------------
            # Update pheromone list from queen bee
            environment.update_pheromone_sources(queen_bee, t_i)

            # Update pheromone list from worker bees
            for bee_i, bee in enumerate(bees):
                environment.update_pheromone_sources(bee, t_i)

            environment.cull_pheromone_sources(t_i)
            # ----------------------------------------------------

            # ----------------------------------------------------
            # Step 2: Build Concentration map and get gradients
            # ----------------------------------------------------
            # Init concentration map for current timestep to 0's
            # environment.init_concentration_map()
            # Iterate through pheromone sources and build concentration maps
            # -- for each pheromone source, calculate gradient for each bee
            for pheromone_src in environment.pheromone_sources:
                # Update concentration map with x, y, A, dt, etc.
                pheromone_src_C = environment.update_concentration_map(t_i, pheromone_src)

                # Iterate through list of active bees and calculate gradient
                for bee in bees:
                    # If bee is inactive don't need to sense
                    if bee.state != 'inactive':
                        bee.sense_environment(t_i, environment, pheromone_src, pheromone_src_C)
            # ----------------------------------------------------

            # ----------------------------------------------------
            # Step 3: Update bees & environment
            # ----------------------------------------------------
            queen_bee.update()

            for bee_i, bee in enumerate(bees):
                dist_to_queen = bee_keeper.compute_dist_to_queen(bee, queen_bee)
                bee.update(dist_to_queen)

                # Measure and store bee info
                bee_keeper.measure_bees(bee, queen_bee, global_i)

            # Store concentration maps
            bee_keeper.measure_environment(environment)
            # ----------------------------------------------------

            # Take steps (update movement, clear grads, etc)
            queen_bee.step()

            # Enforce no overlapping rule
            np.random.shuffle(bees)
            bee_positions = {f'bee_{bee.num}' : [bee.x, bee.y] for bee_i, bee in enumerate(bees)}

            # Add queen to list to not overlap her
            bee_positions['queen'] = [0,0]

            for bee in bees:
                bee.step(environment, bee_positions)

        # Save data to h5's
        bee_keeper.log_data_to_handy_dandy_notebook()

    except KeyboardInterrupt:
        print("\nEnding early.")
        bee_keeper.log_data_to_handy_dandy_notebook()

if __name__ == '__main__':
    cfg_options = Setup.config_options()
    np.random.seed(cfg_options.random_seed)

    model_dir = Setup.directory(cfg_options)
    world_params = Setup.world_parameters(cfg_options, model_dir)
    world_objects = Setup.world_objects(cfg_options, world_params)
    main(cfg_options, **world_objects)
