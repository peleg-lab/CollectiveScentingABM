import numpy as np
import h5py
import json

class BeeKeeper(object):
    def __init__(self, params):
        self.__set_params(params)

        self.environment_history = []
        self.bee_history = {
            "t"                  : [],
            "bee_i"              : [],
            "x"                  : [],
            "y"                  : [],
            "state"              : [],
            "wx"                 : [],
            "wy"                 : [],
            "distance_from_queen": [],
            "concentration"      : [],
            "threshold_met"      : []
        }

        states = ["random_walk_pre", "emit", "directed_walk", "random_walk_post", 'inactive']
        self.state_encoding = { state : i for i, state in enumerate(states) }

    def __set_params(self, params):
        for key, val in params.items():
            self.__dict__[key] = val

    def measure_environment(self, environment):
        if self.sleeping:
            return
        self.environment_history.append(environment.concentration_map)

    def compute_dist_to_queen(self, bee, queen):
        dist = np.sqrt( (bee.x - queen.x)**2 + (bee.y - queen.y)**2 )
        return dist

    def measure_bees(self, bee, queen, global_i):
        if self.sleeping:
            return

        dist = self.compute_dist_to_queen(bee, queen)

        if bee.state == "wait":
            state = "emit"
        else:
            state = bee.state

        bee_info = {
            "t"                  : global_i,
            "bee_i"              : bee.num,
            "x"                  : bee.x,
            "y"                  : bee.y,
            "state"              : self.state_encoding[state],
            "wx"                 : bee.wx,
            "wy"                 : bee.wy,
            "distance_from_queen": dist,
            "concentration"      : bee.total_C,
            "threshold_met"      : bee.threshold_met,
        }

        self.__update_bee_history(bee_info)


    def __update_bee_history(self, bee_info):
        for key, val in bee_info.items():
            self.bee_history[key].append(val)

    def __update_src_history(self, src_history):
        for key, val in src_history.items():
            self.src_history[key].append(val)

    def __write_environment_data(self):
        with h5py.File(self.environment_path, 'w') as outfile:
            outfile.create_dataset("concentration", data=self.environment_history)

    def __write_bee_data(self):
        with h5py.File(self.bee_path, 'w') as outfile:
            for key, val in self.bee_history.items():
                outfile.create_dataset(key, data=val)

    def log_data_to_handy_dandy_notebook(self):
        if self.sleeping:
            return

        self.__write_environment_data()
        self.__write_bee_data()
