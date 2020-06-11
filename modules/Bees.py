import numpy as np
import random

class Queen(object):
    def __init__(self, params):
        self.__set_params(params)
        self.state = "emit"
        self.timestep = 0
        self.wx = 0
        self.wy = 0
        self.gradient_x = 0
        self.gradient_y = 0

    def __set_params(self, params):
        for key, val in params.items():
            self.__dict__[key] = val

    def update(self):
        self.state = 'emit' if self.timestep < self.emission_frequency*0.7 else 'wait'

    def step(self):
        if self.timestep == self.emission_frequency:
            self.timestep = 0
        else:
            self.timestep += 1

########################################################################

class Worker(object):
    def __init__(self, params):
        self.__set_params(params)
        self.__init_position()
        self.__init_conditions()

    def __set_params(self, params):
        for key, val in params.items():
            self.__dict__[key] = val

    def __init_position(self):
        self.x = 0
        self.y = 0

    def __init_conditions(self):
        # Counters
        self.timestep = 0
        self.wait_timestep = 0

        # Grads
        self.gradient_x = 0
        self.gradient_y = 0

        self.sensations = []

        self.wx = 0
        self.wy = 0

        # Flags
        self.threshold_met = False

        # State
        self.state = "random_walk_pre"

        self.Cs = []
        self.distances = []

    def __normalize_gradient(self):
        d = np.linalg.norm([self.gradient_x, self.gradient_y])
        self.gradient_x = self.gradient_x / (d + 1e-9)
        self.gradient_y = self.gradient_y / (d + 1e-9)

    def sense_environment(self, t_i, environment, pheromone_src, pheromone_src_C):
        # Calculate Gradient
        # ------------------------------------------------------------------------
        grad = environment.calculate_gradient(t_i, self.x, self.y, pheromone_src)
        # ------------------------------------------------------------------------

        # Calculate concentration at bee.x, bee.y given individual pheromone src
        # ------------------------------------------------------------------------
        x_bee = environment.convert_xy_to_index(self.x)
        y_bee = environment.convert_xy_to_index(self.y)
        concentration_at_bee = pheromone_src_C[int(y_bee), int(x_bee)]
        # ------------------------------------------------------------------------

        # Omitting very low sources, threshold at lowest T in our search
        if concentration_at_bee > 1e-3:

            # Calculate distance between bees
            self_bee_position = np.array([x_bee, y_bee])

            x_bee_src = environment.convert_xy_to_index(pheromone_src['x'])
            y_bee_src = environment.convert_xy_to_index(pheromone_src['y'])

            src_bee_position = np.array([x_bee_src, y_bee_src])

            distance_between_bees = np.linalg.norm(src_bee_position-self_bee_position)

            # Update bee's sensation
            sensation = {
                "bee_i"    : pheromone_src['bee_i'],
                "C"        : concentration_at_bee,
                "grad"     : grad,
                "distance" : distance_between_bees
            }
            self.sensations.append(sensation)

    def __update_gradient(self, grad):
        grad_x, grad_y = grad
        self.gradient_x += grad_x
        self.gradient_y += grad_y

    def __update_bias(self):
        self.wx = -self.gradient_x
        self.wy = -self.gradient_y

    def __check_arena_boundary(self, environment, dim):
        return environment.x_min < dim < environment.x_max

    def compute_euclidean(self, next_x, next_y, bee_xy):
        bee_x = bee_xy[0]
        bee_y = bee_xy[1]
        dist = np.sqrt((bee_x - next_x)**2 + (bee_y - next_y)**2)
        return dist

    def check_collision(self, next_x, next_y, bee_positions):
        # Check collisions
        distances = []
        for bee_key, bee_xy in bee_positions.items():
            # Skip itself
            if bee_key == f'bee_{self.num}':
                continue
            d_i = self.compute_euclidean(next_x, next_y, bee_xy)
            distances.append(d_i)
        distances = np.array(distances)
        threshold_distance = self.step_size
        able_to_move = np.all(distances > threshold_distance)
        return able_to_move

    def __update_movement(self, dx, dy, environment, bee_positions):
        # dx, dy are unit vector components for heading direction
        next_x = self.x + self.step_size * dx
        next_y = self.y + self.step_size * dy

        # Check for arena boundaries
        move_in_x = self.__check_arena_boundary(environment, next_x)
        move_in_y = self.__check_arena_boundary(environment, next_y)

        if not move_in_x:
            next_x = self.x
        if not move_in_y:
            next_y = self.y

        # Check collisions for the initial intended direction
        able_to_move = self.check_collision(next_x, next_y, bee_positions)

        if able_to_move:
            self.x = next_x
            self.y = next_y
            bee_positions[f'bee_{self.num}'] = [self.x, self.y]

        # Allow bee that can't move to first position to try moving to another
        else:
            # Enter loop to pick another direction within some boundaries
            num_tries_per_level = 5
            orig_grad = np.array([dx, dy])
            step_size = self.step_size
            thetas = [self.comp_delta(orig_grad, 45), self.comp_delta(orig_grad, 60), self.comp_delta(orig_grad, 90), self.comp_delta(orig_grad, 130)]
            steps = [self.comp_step(step_size, 0.98), self.comp_step(step_size, 0.80), self.comp_step(step_size, 0.70)]

            # Loop over angles to pick a new angle
            for level_i, (theta_range, step_range) in enumerate(zip(thetas, steps), 1):
                # Loop over step sizes to pick a new step size
                for try_i in range(num_tries_per_level):
                    random_angle = np.random.uniform(*theta_range)
                    random_step = np.random.uniform(*step_range)

                    # New angle and step size
                    new_grad_x = np.cos(np.deg2rad(random_angle))
                    new_grad_y = np.sin(np.deg2rad(random_angle))

                    # Normalize new grad
                    new_grad = np.array([new_grad_x, new_grad_y])
                    new_grad = new_grad / np.linalg.norm(new_grad)
                    new_grad = random_step * new_grad
                    new_grad_x, new_grad_y = new_grad

                    # New bee position
                    new_next_x = self.x + new_grad_x
                    new_next_y = self.y + new_grad_y

                    self.new_grad_x = new_grad_x
                    self.new_grad_y = new_grad_y

                    # Check for arena boundaries
                    move_in_x = self.__check_arena_boundary(environment, new_next_x)
                    move_in_y = self.__check_arena_boundary(environment, new_next_y)
                    if not move_in_x:
                        new_next_x = self.x
                    if not move_in_y:
                        new_next_y = self.y

                    # Check collision
                    able_to_move = self.check_collision(new_next_x, new_next_y, bee_positions)
                    if able_to_move:
                        self.x = new_next_x
                        self.y = new_next_y
                        bee_positions[f'bee_{self.num}'] = [self.x, self.y]
                        return

    def comp_delta(self, orig_grad, theta):
        orig_grad_theta = np.rad2deg(np.arctan2(orig_grad[1], orig_grad[0]))
        grad_1 = orig_grad_theta - theta/2
        grad_2 = orig_grad_theta + theta/2
        return grad_1, grad_2

    def comp_step(self, orig_step, decay):
        lower_bound = orig_step * decay
        return lower_bound, orig_step

    def __check_src_contributions(self):
        self.Cs = []
        self.distances = []
        for sensation in self.sensations:
            self.Cs.append(sensation['C'])
            self.distances.append(sensation['distance'])

    def update(self, dist_to_queen=100):
        """
            Three modes:
                1. No distinction between any bees (e.g., pheromones from queen are treated equal to workers)
                2. Distinction between queen and workers
                3. Distinction between queen and workers, as well as distinction between workers
        """
        if dist_to_queen <= -1 and self.state != 'random_walk_pre' and self.state != 'random_walk_post':
            self.state = 'inactive'

        if self.state != 'inactive':
            if self.sensitivity_mode == 'none':
                self.__determine_sensation_effects__mode_1()
            elif self.sensitivity_mode == 'queen_worker':
                # 2. Distinction between queen and workers
                self.__determine_sensation_effects__mode_2()
            elif self.sensitivity_mode == 'all':
                # 3. Distinction between queen and workers, as well as distinction between workers
                self.__determine_sensation_effects__mode_3()

        # Check if threshold met or not
        # ------------------------------------------------
        if self.total_C > self.threshold:
            self.threshold_met = True
            # Don't compute gradient and bias when in emitting state
            # But compute when emitting is over
            if self.state != 'emit' and self.state != 'inactive':
                self.__update_gradient(self.total_grads)
                self.__normalize_gradient()
                self.__update_bias()
                self.__check_src_contributions()
        else:
            self.threshold_met = False
        # ------------------------------------------------
        self.__update_state(dist_to_queen)

    def __determine_sensation_effects__mode_1(self):
        """
            # 1. No distinction between any bees (e.g., pheromones from queen are treated equal to workers)
        """
        self.total_C = 0
        self.total_grads = np.array([0.0, 0.0])
        for sensation in self.sensations:
            self.total_C += sensation['C']
            self.total_grads += np.array(sensation['grad'])

    def __determine_sensation_effects__mode_2(self):
        """
            2. Distinction between queen and workers
               - A. Check queen concentration - if > threshold,
        """
        queen_C = 0
        queen_grads = np.array([0.0, 0.0])

        worker_C = 0
        worker_grads = np.array([0.0, 0.0])
        for sensation in self.sensations:
            bee_i = sensation['bee_i']

            # Queen
            if bee_i == -1:
                queen_C += sensation['C']
                queen_grads += np.array(sensation['grads'])
            # Workers
            else:
                worker_C += sensation['C']
                worker_grads += np.array(sensation['grads'])

        self.total_C = 0
        self.total_grads = np.array([0.0, 0.0])

        # Queen
        if queen_C > self.threshold:
            self.total_C += queen_C
            self.total_grads += queen_grads
        if worker_C > self.threshold:
            self.total_C += worker_C
            self.total_grads += worker_grads

    def __determine_sensation_effects__mode_3(self):
        """
            3. Distinction between queen and workers, as well as distinction between workers
        """
        combined_sensations = {}
        for sensation in self.sensations:
            bee_i = sensation['bee_i']
            if bee_i in combined_sensations:
                combined_sensations[bee_i]['C'] += sensation['C']
                combined_sensations[bee_i]['grads'] += np.array(sensation['grads'])
            else:
                combined_sensations[bee_i] = {
                    "C"     : sensation['C'],
                    "grads" : np.array(sensation['grads'])
                }

        self.total_C = 0
        self.total_grads = np.array([0.0, 0.0])
        for sensation in combined_sensations.values():
            if sensation['C'] > self.threshold:
                self.total_C += sensation['C']
                self.total_grads += np.array(sensation['grads'])


    def __update_state(self, dist_to_queen):
        self.next_state = None

        if self.state == "random_walk_pre" or self.state == "random_walk_post":
            if self.threshold_met:
                random_draw = np.random.uniform(0, 1)
                if random_draw <= self.trans_prob:
                    self.next_state = "emit"
                    self.wait_timestep = 0
                else:
                    ######## Only this for not emitting
                    self.next_state = "directed_walk"

        # Scent for half (or whatever fraction) of wait_period, and for the rest wait in place for pheromone to decay...
        elif self.state == "emit":
            if self.wait_period*0.5 <= self.wait_timestep <= self.wait_period:
                self.next_state = "wait"
            self.wait_timestep += 1

        # In wait, it would just stand still
        # Then we don't need emit_final state to resume calculating gradient
        elif self.state == "wait":
            if self.wait_timestep > self.wait_period:
                self.next_state = "directed_walk"
            self.wait_timestep += 1

        elif self.state == "directed_walk":
            if self.threshold_met:
                random_draw = np.random.uniform(0, 1)
                if random_draw <= self.trans_prob:
                    self.next_state = "emit"
                    self.wait_timestep = 0
            else:
                self.next_state = "random_walk_post"

        elif self.state == 'inactive':
            self.next_state = 'inactive'

        # Check for case where state hasnt't changed
        if self.next_state is None:
            self.next_state = self.state

    def __clear(self):
        self.gradient_x = 0
        self.gradient_y = 0
        self.sensations = []

    def step(self, environment, bee_positions):
        self.state = self.next_state

        if self.state == "random_walk_pre" or self.state == "random_walk_post":
            random_sign_x = np.random.choice([-1, 0, 1])
            random_sign_y = np.random.choice([-1, 0, 1])

            # Update movement
            self.__update_movement(random_sign_x, random_sign_y, environment, bee_positions)

        elif self.state == "directed_walk":
            self.__update_movement(self.gradient_x, self.gradient_y, environment, bee_positions)

        # Normalize and clear out gradient for timestep
        if self.state != "emit":
            self.__clear()

        self.timestep += 1
