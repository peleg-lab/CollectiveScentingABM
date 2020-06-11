import numpy as np

class Environment(object):
    """
        Pieces of the environment
        ---------------------------
        1. Concentration Map (Grid)
          - Diffusion equation
          - Gradient equation
          - Params: Diffusion coeff
        2. Time-course
    """
    def __init__(self, params):
        self.__set_params(params)

        self.__init_environment_grid()
        self.__init_timecourse()

        # List of 4-tuples ({x, y, wx, wy, A, t0})
        self.pheromone_sources = []

    def __getitem__(self, idx):
        current_t = self.t_grid[idx]
        self.init_concentration_map()
        return current_t

    def __set_params(self, params):
        for key, val in params.items():
            self.__dict__[key] = val

    def __init_environment_grid(self):
        print("Creating concentration map...")
        X1 = np.arange(self.x_min, self.x_max+self.dx, self.dx)
        X2 = np.arange(self.x_min, self.x_max+self.dx, self.dx)
        self.x_grid, self.y_grid = np.meshgrid(X1, X2)

    def __init_timecourse(self):
        print("Creating timecourse...")
        self.t_grid = np.arange(self.t_min, self.t_max, self.dt)

    def update_pheromone_sources(self, bee, t0):
        # Check if bee is active.
        # If so, add info (tuple) to sources list
        if bee.state == "emit":
            # Normalize bias
            d = np.linalg.norm([bee.wx, bee.wy]) + 1e-9

            bee_tuple = {
                "bee_i"  : bee.num,
                "x"      : bee.x,
                "y"      : bee.y,
                "x_grad" : bee.gradient_x,
                "y_grad" : bee.gradient_y,
                "wb"     : bee.wb,
                "wx"     : bee.wx / d,
                "wy"     : bee.wy / d,
                "A"      : bee.A,
                "t0"     : t0
            }
            self.pheromone_sources.append(bee_tuple)

    def cull_pheromone_sources(self, t_i):
        keep_idxs = []
        for pheromone_src_i, pheromone_src in enumerate(self.pheromone_sources):

            delta_t = t_i - pheromone_src['t0']
            delta_t += self.dt
            current_c = self.__diffusion_eq(A=pheromone_src['A'], D=self.D,
                                            x=pheromone_src['x'], x_source=pheromone_src['x'],
                                            y=pheromone_src['y'], y_source=pheromone_src['y'],
                                            wb=pheromone_src['wb'],
                                            wx=pheromone_src['wx'], wy=pheromone_src['wy'],
                                            t=delta_t, decay_rate=self.decay_rate)

            if current_c > self.culling_threshold:
                keep_idxs.append(pheromone_src_i)

        self.pheromone_sources = list(np.array(self.pheromone_sources)[keep_idxs])

    def init_concentration_map(self):
        self.concentration_map = np.zeros([self.x_grid.shape[0], self.x_grid.shape[0]], dtype=np.float32)

    def __diffusion_eq(self, A, D, x, x_source, y, y_source, wb, wx, wy, t, decay_rate):
        term_1 = A / (np.sqrt(t) + 1e-9)
        dx = x - x_source
        dy = y - y_source

        term_2 = (dx - wb*wx * t)**2 + (dy - wb*wy * t)**2
        denom = 4 * D * t
        c = term_1 * (np.exp(-(term_2 / denom) - (decay_rate * t)))
        return c

    def update_concentration_map(self, t_i, pheromone_src):

        delta_t = t_i - pheromone_src['t0']
        delta_t += self.dt
        current_c = self.__diffusion_eq(A=pheromone_src['A'], D=self.D,
                                        x=self.x_grid, x_source=pheromone_src['x'],
                                        y=self.y_grid, y_source=pheromone_src['y'],
                                        wb=pheromone_src['wb'],
                                        wx=pheromone_src['wx'], wy=pheromone_src['wy'],
                                        t=delta_t, decay_rate=self.decay_rate)

        self.concentration_map += current_c

        return current_c

    def __calc_gradient(self, x_sample_pt, y_sample_pt, D, dt, A, x_source, y_source, wx, wy, wb, decay_rate):
        K = -A / (2 * D * dt * np.sqrt(dt) + 1e-5)
        x_term = (x_sample_pt-x_source - wb*wx*dt)**2
        y_term = (y_sample_pt-y_source - wb*wy*dt)**2
        denom = dt*4*D + 1e-5
        exp_term = np.exp(-(x_term + y_term)/denom - decay_rate*dt)
        dc_dx = K * exp_term * (x_sample_pt - x_source - wb*wx*dt)
        dc_dy = K * exp_term * (y_sample_pt - y_source - wb*wy*dt)
        return dc_dx, dc_dy

    def calculate_gradient(self, t_i, bee_x, bee_y, pheromone_src):
        delta_t = t_i - pheromone_src['t0']
        delta_t += self.dt
        dx, dy = self.__calc_gradient(bee_x, bee_y, self.D, delta_t,
                                      pheromone_src['A'],
                                      pheromone_src['x'], pheromone_src['y'],
                                      pheromone_src['wx'], pheromone_src['wy'],
                                      pheromone_src['wb'], self.decay_rate)

        return dx, dy

    def convert_xy_to_index(self, XY):
        index = ((XY - self.x_min) / (self.x_max - self.x_min)) * self.x_grid.shape[0]
        return index
