import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import csv
import copy
import os
from pathlib import Path
import sys
import time
import yaml
import multiprocessing as mp
from dataclasses import asdict

import casadi as ca
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
import yaml
from scipy.interpolate import CubicSpline
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.spatial.transform import Rotation as Rot

from poly_fly.optimal_planner.polytopes import Cable, Obs, Polytope, Quadrotor, SquarePayload
import poly_fly.utils.plot as plotter
from poly_fly.optimal_planner.global_planner import solve_with_polytopes
from poly_fly.utils.utils import (
    MPC,
    dictToClass,
    quaternion_multiplication_casadi,
    rotation_inverse_casadi,
    vec2asym,
    yamlToDict,
    rotation_matrix_from_a_to_b,
    rpy_from_a_to_b,
    get_yaw_along_trajectory
)
from poly_fly.data_io.utils import save_csv_arrays, save_params, save_all, BASE_DIR, PARAMS_DIR, \
    GIFS_DIR, CSV_DIR, IMG_DIR, get_rotation_matrix_from_quat
from poly_fly.data_io.enums import DatasetKeys, AttrKeys


class Planner:
    # states are
    # xL, yL, zL
    # xLdot, yLdot, zLdot
    # xLddot, yLddot, zLddot
    # inputs are
    # xL_jerk, yL_jerk, zL_jerk
    N_STATES = 9
    N_INPUTS = 3

    def __init__(self, params, plot=False):
        self.params = params
        self.plot = plot
        self.global_planner_explored_nodes = None  # Initialize explored nodes storage

        x_sym = ca.MX.sym('x', self.N_STATES, 1)
        u_sym = ca.MX.sym('u', self.N_INPUTS, 1)
        dt_sym = ca.MX.sym('dt', 1)
        self.rk4_function = ca.Function(
            'rk4', [x_sym, u_sym, dt_sym], [self.dynamics_rk45(x_sym, u_sym, dt_sym)]
        )

    def clear_obstacles(self):
        """
        Remove all obstacles currently stored in `self.params.obstacles` and
        reset the internal list used during `initialize_variables`.
        """
        self.params.obstacles.clear()

    def add_obstacle_box(self, *, x, y, z, xl, yl, zl, name=None):
        """
        Append an axis-aligned rectangular prism (center + half-sizes).

        Returns the obstacle’s key so the caller can reference it later.
        """
        if name is None:
            name = f"obs_{len(self.params.obstacles)}"
        self.params.obstacles[name] = dict(x=x, y=y, z=z, l=xl, b=yl, h=zl)
        return name

    def set_start_state(self, pos, vel=(0, 0, 0), acc=(0, 0, 0)):
        """
        Overwrite `params.initial_state` and keep the cached `self.state`
        (used in constraints) in sync when it already exists.
        """
        self.params.initial_state = list(pos) + list(vel) + list(acc)

    def set_end_state(self, pos, vel=(0, 0, 0), acc=(0, 0, 0)):
        """
        Overwrite both `params.end_state` and the alias `params.x_des`.
        """
        end_vec = list(pos) + list(vel) + list(acc)
        self.params.end_state = end_vec
        self.params.x_des = end_vec

    def set_xy_bounds(self, x_min, x_max, y_min, y_max):
        self.params.state_min[0] = x_min
        self.params.state_min[1] = y_min
        self.params.state_max[0] = x_max
        self.params.state_max[1] = y_max

    def set_margin(self, margin):
        self.params.margin = margin

    def fdot(self, x, u):
        x_dot = ca.MX(self.N_STATES, 1)
        x_dot[0] = x[3, 0]
        x_dot[1] = x[4, 0]
        x_dot[2] = x[5, 0]

        x_dot[3] = x[6, 0]
        x_dot[4] = x[7, 0]
        x_dot[5] = x[8, 0]

        x_dot[6] = u[0, 0]
        x_dot[7] = u[1, 0]
        x_dot[8] = u[2, 0]

        return x_dot

    def dynamics_rk45(self, x, u, dt=None):
        if dt is None:
            dt = self.params.dt

        k1 = self.fdot(x, u)
        k2 = self.fdot(x + dt / 2 * k1, u)
        k3 = self.fdot(x + dt / 2 * k2, u)
        k4 = self.fdot(x + dt * k3, u)
        x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return x_next

    def dynamics_rk3(self, x, u, dt=None):
        if dt is None:
            dt = self.params.dt

        # ── RK-3 stages ──────────────────────────────────────────────────────────
        k1 = self.fdot(x, u)  # f(x      , u)
        k2 = self.fdot(x + 0.5 * dt * k1, u)  # f(x+½dt k1, u)
        k3 = self.fdot(x - dt * k1 + 2 * dt * k2, u)  # f(x-dt k1+2dt k2,u)

        # ── RK-3 update ─────────────────────────────────────────────────────────
        x_next = x + dt / 6 * (k1 + 4 * k2 + k3)

        return x_next

    def get_minimum_dist(self, A_1, B_1, A_2, B_2):
        opti = ca.Opti()
        point_1 = opti.variable(A_1.shape[-1], 1)
        point_2 = opti.variable(A_2.shape[-1], 1)
        cost = 0

        const1 = A_1 @ point_1 <= B_1
        const2 = A_2 @ point_2 <= B_2
        opti.subject_to(const1)
        opti.subject_to(const2)

        dist_vec = point_1 - point_2
        cost += dist_vec.T @ dist_vec

        opti.minimize(cost)
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}

        opti.solver("ipopt", option)
        opt_sol = opti.solve()

        dist = opt_sol.value(ca.norm_2(dist_vec))

        # Divide by 2 due to dual formulation setup
        if dist > 0:
            lamb_obs = opt_sol.value(opti.dual(const1)) / 2 * dist
            lamb_load = opt_sol.value(opti.dual(const2)) / 2 * dist

        else:
            dist = -1
            lamb_obs = np.zeros(shape=(A_1.shape[0],))
            lamb_load = np.zeros(shape=(A_2.shape[0],))

        return dist, lamb_obs, lamb_load

    def set_state(self, state):
        self.state = state

    def initialize_variables(self, state, plot_global_planner=False):
        self.obstacles = []
        for key in self.params.obstacles.keys():
            wall = self.params.obstacles[key]
            self.obstacles.append(
                Obs(wall['x'], wall['y'], wall['z'], wall['l'], wall['b'], wall['h'])
            )

        self.payload = SquarePayload(self.params)
        self.cable = Cable(self.params)
        self.quad = Quadrotor(self.params)

        # self.state = np.array([0] * self.N_STATES)
        self.variables = dict()
        self.min_time = self.params.min_time
        self.frames = []  # Store frames for GIF creation

        self.variables["x"] = self.opti.variable(self.N_STATES, self.params.horizon + 1)
        self.variables["u"] = self.opti.variable(self.N_INPUTS, self.params.horizon)

        for j in range(self.params.horizon):
            for i in range(self.N_INPUTS):
                self.opti.set_initial(self.variables["u"][i, j], 0)

        if self.params.use_global_planner:
            self.params.payload_pos_init = self.get_global_plan(plot=plot_global_planner)

        # initial guess
        # Perform linear interpolation between payload initialization points
        num_points = len(self.params.payload_pos_init)

        if num_points == 0:
            self.interpolated_positions = np.zeros((3, self.params.horizon + 1))
            self.interpolated_velocities = np.zeros((3, self.params.horizon + 1))
            print("No initialization available")
            return

        time_points = np.linspace(0, self.params.horizon + 1, num_points, endpoint=True)
        interpolated_positions = np.zeros((3, self.params.horizon + 1))

        for k in range(3):  # x, y, z positions
            interpolated_positions[k, :] = np.interp(
                np.arange(self.params.horizon + 1),
                time_points,
                [pos[k] for pos in self.params.payload_pos_init],
            )

        # Set initial values for x positions using the interpolated positions
        for j in range(self.params.horizon + 1):
            for k in range(3):  # x, y, z positions
                self.opti.set_initial(self.variables["x"][k, j], interpolated_positions[k, j])

        # initialize velocities
        interpolated_velocities = np.zeros((3, self.params.horizon + 1))
        if len(self.params.payload_vel_init) == 0:
            # print("No velocity initialization available")
            # Calculate constant velocities based on the interpolated position direction
            for k in range(3):  # x, y, z velocities
                interpolated_velocities[k, :-1] = (
                    np.diff(interpolated_positions[k, :]) / self.params.dt
                )
                interpolated_velocities[k, -1] = interpolated_velocities[
                    k, -2
                ]  # Maintain last velocity
        else:
            assert len(self.params.payload_vel_init) == len(self.params.payload_pos_init)
            print("Velocity initialization available")
            for k in range(3):  # x, y, z velocities
                interpolated_velocities[k, :] = np.interp(
                    np.arange(self.params.horizon + 1),
                    time_points,
                    [vel[k] for vel in self.params.payload_vel_init],
                )
            # Set initial values for x velocities using the interpolated velocities
            for j in range(self.params.horizon + 1):
                for k in range(3):
                    self.opti.set_initial(
                        self.variables["x"][k + 3, j], interpolated_velocities[k, j]
                    )

        # Set initial values for x velocities
        for j in range(self.params.horizon + 1):
            for k in range(3):  # x, y, z velocities
                self.opti.set_initial(self.variables["x"][k + 3, j], interpolated_velocities[k, j])

        # initialize inputs
        num_points = len(self.params.payload_input_init)
        time_points = np.linspace(0, self.params.horizon, num_points, endpoint=True)
        interpolated_inputs = np.zeros((3, self.params.horizon))
        if len(self.params.payload_input_init) == 0:
            # print("No input initialization available")
            for j in range(self.params.horizon):
                for k in range(3):
                    self.opti.set_initial(self.variables["u"][k, j], 0)
        else:
            assert len(self.params.payload_input_init) == len(self.params.payload_pos_init) - 1
            print("Input initialization available")
            for k in range(3):  # x, y, z inputs
                interpolated_inputs[k, :] = np.interp(
                    np.arange(self.params.horizon),
                    time_points,
                    [input[k] for input in self.params.payload_input_init],
                )
            # Set initial values for x inputs using the interpolated inputs
            for j in range(self.params.horizon):
                for k in range(3):
                    self.opti.set_initial(self.variables["u"][k, j], interpolated_inputs[k, j])

        # Save the interpolated positions and velocities for visualization
        self.interpolated_positions = interpolated_positions
        self.interpolated_velocities = interpolated_velocities
        self.interpolated_inputs = interpolated_inputs

        if self.min_time:
            self.variables["t"] = self.opti.variable(self.params.horizon)
            self.opti.set_initial(self.variables["t"], [0.09] * self.params.horizon)

    def get_global_plan(self, plot=False):
        result = solve_with_polytopes(self.params, show_animation=plot)

        # Handle backward compatibility - check if explored nodes are returned
        if len(result) == 3:
            rx, ry, explored_nodes = result
            # Convert explored nodes to numpy array for plotting
            if explored_nodes:
                self.global_planner_explored_nodes = np.array(explored_nodes)
            else:
                self.global_planner_explored_nodes = np.array([]).reshape(0, 2)
        else:
            # Fallback for older version that only returns rx, ry
            rx, ry = result
            self.global_planner_explored_nodes = np.array([]).reshape(0, 2)

        payload_pos_init = []

        # payload_pos_init.append([])
        for i in range(len(rx) - 1, 0, -1):
            payload_pos_init.append([rx[i], ry[i], 0.0])

        return payload_pos_init

    def init_opt_vars(self, keys, values):
        for key, value in zip(keys, values):
            if key not in self.variables.keys():
                print("Creating new var")
                self.variables[key] = self.opti.variable(*value.shape)

            self.opti.set_initial(self.variables[key], value)

    def add_initial_condition_constraint(self, xdes_tol=0.05):
        for i in range(self.N_STATES):
            self.opti.subject_to(self.variables["x"][i, 0] == self.state[i])

        for i in range(self.N_STATES):
            self.opti.subject_to(self.variables["x"][i, -1] <= self.params.end_state[i] + xdes_tol)
            self.opti.subject_to(self.variables["x"][i, -1] >= self.params.end_state[i] - xdes_tol)

    def add_state_and_input_constraint(self):
        # Constraints on the state read from yaml file
        for i in range(self.params.horizon):
            for j in range(self.variables["x"].shape[0]):
                self.opti.subject_to(self.variables["x"][j, i] <= self.params.state_max[j])
                self.opti.subject_to(self.params.state_min[j] <= self.variables["x"][j, i])

        # input constraints
        for i in range(self.params.horizon):
            for j in range(self.variables["u"].shape[0]):
                self.opti.subject_to(self.variables["u"][j, i] <= self.params.input_max[j])
                self.opti.subject_to(self.params.input_min[j] <= self.variables["u"][j, i])

        for j in range(self.variables["u"].shape[0]):
            self.opti.subject_to(self.variables["u"][j, 0] == 0)
            self.opti.subject_to(self.variables["u"][j, -1] == 0)

        self.opti.subject_to(self.variables["x"][6:8, 0] == 0)
        self.opti.subject_to(self.variables["x"][3:8, -1] == 0)

        if self.min_time:
            for i in range(self.params.horizon):
                if i > 0:
                    self.opti.subject_to(
                        (self.variables["t"][i] - self.variables["t"][i - 1]) ** 2 <= 0.05**2
                    )

    def add_dynamics_constraint(self):
        for i in range(self.params.horizon):
            if self.min_time:
                self.opti.subject_to(
                    self.variables["x"][:, i + 1]
                    == self.rk4_function(
                        self.variables["x"][:, i],
                        self.variables["u"][:, i],
                        self.variables["t"][i],
                    )
                )
            else:
                self.opti.subject_to(
                    self.variables["x"][:, i + 1]
                    == self.rk4_function(self.variables["x"][:, i], self.variables["u"][:, i]),
                    self.params.dt,
                )

            if self.min_time:
                self.opti.subject_to(self.variables["t"][i] >= self.params.min_dt)
                self.opti.subject_to(self.variables["t"][i] <= self.params.max_dt)

    def add_cost(self):
        u = self.variables["u"]
        x = self.variables["x"]
        self.cost = 0

        # stay close to initial guess
        for i in range(self.params.horizon):
            for j in range(3):
                self.cost += (
                    self.params.Qi[j]
                    * (x[j, i] - self.interpolated_positions[j, i]) ** 2
                    / self.params.horizon
                )

        # for i in range(self.params.horizon - 1):
        #     for j in range(3):
        #         self.cost += self.params.Qu * (u[j, i + 1] - u[j, i]) ** 2 / self.params.horizon
        u_diff = u[:, 1:] - u[:, :-1]
        self.cost += self.params.Qu * ca.sum1(ca.sum2(u_diff**2)) / self.params.horizon

        if self.min_time:
            # for i in range(self.params.horizon):
            #     self.cost += self.variables["t"][i] * self.params.Q_dt / self.params.horizon
            self.cost += ca.sum1(self.variables["t"]) * self.params.Q_dt / self.params.horizon

    def get_cable_rotation(self, acc_in, symbolic=False):
        gravity = 9.81
        if symbolic:
            R = ca.MX(3, 3)
            p = ca.MX(3, 1)
            acc = ca.MX(3, 1)
            acc[0, 0] = acc_in[0]
            acc[1, 0] = acc_in[1]
            acc[2, 0] = acc_in[2] + gravity
            acc_norm_sq = acc_in[0] ** 2 + acc_in[1] ** 2 + (acc_in[2] + gravity) ** 2

        else:
            R = np.zeros((3, 3))
            p = np.zeros((3, 1))
            acc = np.array([acc_in[0], acc_in[1], acc_in[2] + gravity])
            acc_norm_sq = acc_in[0] ** 2 + acc_in[1] ** 2 + (acc_in[2] + gravity) ** 2

        p = -acc / (acc_norm_sq**0.5)
        x = p[0]
        y = p[1]
        z = p[2]

        # Calculate the rotation of box from the cable direction
        # the box z' direction is taken along the vector because it should be cable length
        # first choose x along 1,0,0 , find y' by -cross(z,x) and x' by cross(y,z)

        R[0, 0] = (z**2 + y**2 + 1e-5) ** 0.5
        R[0, 1] = 0
        R[0, 2] = x
        R[1, 0] = -y * x / (z**2 + y**2 + 1e-5) ** 0.5
        R[1, 1] = z / (z**2 + y**2 + 1e-5) ** 0.5
        R[1, 2] = y
        R[2, 0] = -z * x / (z**2 + y**2 + 1e-5) ** 0.5
        R[2, 1] = -y / (z**2 + y**2 + 1e-5) ** 0.5
        R[2, 2] = z

        return R, p

    def add_payload_to_obstacles_constraints(self, idx, obs_geo):
        A_obs, B_obs = obs_geo.get_convex_rep()
        A_load, B_load = self.payload.get_convex_rep()

        self.variables[f"lamb_obs_load_{idx}"] = self.opti.variable(
            A_obs.shape[0], self.params.horizon
        )
        self.variables[f"lamb_load_{idx}"] = self.opti.variable(
            A_load.shape[0], self.params.horizon
        )

        # lambda = lambda_obs
        self.variables[f"omega_load_{idx}"] = self.opti.variable(self.params.horizon, 1)
        for i in range(self.params.horizon):
            distance_to_obstacle = np.linalg.norm(
                self.interpolated_positions[0:3, i] - obs_geo.get_center()
            )
            if distance_to_obstacle > self.params.constraint_min_dist or not self.inside_tube[idx]:
                continue

            R_load = np.eye(3)
            xL = self.variables["x"][0:3, i]

            # Lagrange Multiplier Constraints
            self.opti.subject_to(self.variables[f"lamb_obs_load_{idx}"][:, i] >= 0)
            self.opti.subject_to(self.variables[f"lamb_load_{idx}"][:, i] >= 0)

            # Get CBF value at initialized point
            # get current value of cbf
            cbf_query_state = self.interpolated_positions[:, i]
            cbf_eval_state = np.array(
                [[cbf_query_state[0]], [cbf_query_state[1]], [cbf_query_state[2]]]
            )

            # CBF constraints for payload
            self.opti.subject_to(
                -ca.mtimes(B_load.T, self.variables[f"lamb_load_{idx}"][:, i])
                + ca.mtimes(
                    (ca.mtimes(A_obs, xL) - B_obs).T, self.variables[f"lamb_obs_load_{idx}"][:, i]
                )
                >= self.params.margin
            )
            self.opti.subject_to(
                ca.mtimes(A_load.T, self.variables[f"lamb_load_{idx}"][:, i])
                + ca.mtimes(
                    ca.mtimes(R_load.T, A_obs.T), self.variables[f"lamb_obs_load_{idx}"][:, i]
                )
                == 0
            )
            temp = ca.mtimes(A_obs.T, self.variables[f"lamb_obs_load_{idx}"][:, i])
            self.opti.subject_to(ca.mtimes(temp.T, temp) <= 1)

            self.opti.subject_to(self.variables[f"omega_load_{idx}"][i] >= 0)
            self.opti.set_initial(self.variables[f"omega_load_{idx}"][i], 0.1)

    def add_cable_to_obstacles_constraints(self, idx, obs):
        A_obs, B_obs = obs.get_convex_rep()
        A_quad, B_quad = self.cable.get_convex_rep()

        acc = np.zeros(3)
        R, p = self.get_cable_rotation(acc, False)

        T = (
            np.array(self.state[0:3]).T
            - np.array(self.params.cable_length) / 2.0 * np.array([p[0], p[1], p[2]]).T
        )
        T = T.reshape(3, 1)

        cbf_curr, lamb_obs_curr, lamb_load_curr = self.get_minimum_dist(
            A_obs,
            B_obs,
            np.dot(A_quad, R.T),
            np.dot(np.dot(A_quad, R.T), T) + B_quad,
        )

        self.variables[f"lamb_obs_cable_{idx}"] = self.opti.variable(
            A_quad.shape[0], self.params.horizon
        )
        self.variables[f"lamb_cable_{idx}"] = self.opti.variable(
            B_quad.shape[0], self.params.horizon
        )
        self.variables[f"omega_cable_{idx}"] = self.opti.variable(self.params.horizon, 1)

        for i in range(self.params.horizon):
            distance_to_obstacle = np.linalg.norm(
                self.interpolated_positions[0:3, i] - obs.get_center()
            )
            if distance_to_obstacle > self.params.constraint_min_dist or not self.inside_tube[idx]:
                continue

            acc = self.variables["x"][6:9, i]
            R_sym, p_sym = self.get_cable_rotation(acc, True)
            x_k = self.variables["x"][0:3, i] - self.params.cable_length / 2.0 * p_sym

            acc_numeric = np.zeros(3)
            R_numeric, p_numeric = self.get_cable_rotation(acc_numeric, False)
            cbf_query_state = self.interpolated_positions[:, i]
            T_numeric = (
                np.array(cbf_query_state[0:3]).T
                - np.array(self.params.cable_length)
                / 2.0
                * np.array([p_numeric[0], p_numeric[1], p_numeric[2]]).T
            )
            T_numeric = T_numeric.reshape(3, 1)

            self.opti.subject_to(self.variables[f"lamb_obs_cable_{idx}"][:, i] >= 0)
            self.opti.subject_to(self.variables[f"lamb_cable_{idx}"][:, i] >= 0)
            self.opti.subject_to(
                -ca.mtimes(B_quad.T, self.variables[f"lamb_cable_{idx}"][:, i])
                + ca.mtimes(
                    (ca.mtimes(A_obs, x_k) - B_obs).T, self.variables[f"lamb_obs_cable_{idx}"][:, i]
                )
                >= self.params.margin
            )
            self.opti.subject_to(
                ca.mtimes(A_quad.T, self.variables[f"lamb_cable_{idx}"][:, i])
                + ca.mtimes(
                    ca.mtimes(R_sym.T, A_obs.T), self.variables[f"lamb_obs_cable_{idx}"][:, i]
                )
                == 0
            )
            temp = ca.mtimes(A_obs.T, self.variables[f"lamb_obs_cable_{idx}"][:, i])
            self.opti.subject_to(ca.mtimes(temp.T, temp) <= 1)
            self.opti.subject_to(self.variables[f"omega_cable_{idx}"][i] >= 0)
            self.opti.set_initial(self.variables[f"omega_cable_{idx}"][i], 0.1)

    def add_quadrotor_to_obstacles_constraints(self, idx, obs):
        A_obs, B_obs = obs.get_convex_rep()
        A_quad, B_quad = self.quad.get_convex_rep()

        acc = np.zeros(3)
        R, p = self.get_cable_rotation(acc, False)

        T = (
            np.array(self.state[0:3]).T
            - (np.array(self.params.cable_length) + self.params.robot_height / 2.0)
            * np.array([p[0], p[1], p[2]]).T
        )
        T = T.reshape(3, 1)

        cbf_curr, lamb_obs_curr, lamb_load_curr = self.get_minimum_dist(
            A_obs,
            B_obs,
            np.dot(A_quad, R.T),
            np.dot(np.dot(A_quad, R.T), T) + B_quad,
        )

        self.variables[f"lamb_obs_quad_{idx}"] = self.opti.variable(
            A_quad.shape[0], self.params.horizon
        )
        self.variables[f"lamb_quad_{idx}"] = self.opti.variable(
            B_quad.shape[0], self.params.horizon
        )
        self.variables[f"omega_quad_{idx}"] = self.opti.variable(self.params.horizon, 1)

        for i in range(self.params.horizon):
            distance_to_obstacle = np.linalg.norm(
                self.interpolated_positions[0:3, i] - obs.get_center()
            )
            if distance_to_obstacle > self.params.constraint_min_dist or not self.inside_tube[idx]:
                continue

            acc = self.variables["x"][6:9, i]
            _, p_sym = self.get_cable_rotation(acc, True)
            x_k = (
                self.variables["x"][0:3, i]
                - (np.array(self.params.cable_length) + self.params.robot_height / 2.0) * p_sym
            )
            R_sym = self.compute_quadrotor_rotation_matrix_no_jrk(
                acc, self.params, symbolic=True, use_robot_rotation=self.params.use_robot_rotation
            )

            acc_numeric = np.zeros(3)
            R_numeric, p_numeric = self.get_cable_rotation(acc_numeric, False)
            cbf_query_state = self.interpolated_positions[:, i]
            T_numeric = (
                np.array(cbf_query_state[0:3]).T
                - (np.array(self.params.cable_length) + self.params.robot_height / 2.0)
                * np.array([p_numeric[0], p_numeric[1], p_numeric[2]]).T
            )
            T_numeric = T_numeric.reshape(3, 1)

            self.opti.subject_to(self.variables[f"lamb_obs_quad_{idx}"][:, i] >= 0)
            self.opti.subject_to(self.variables[f"lamb_quad_{idx}"][:, i] >= 0)
            self.opti.subject_to(
                -ca.mtimes(B_quad.T, self.variables[f"lamb_quad_{idx}"][:, i])
                + ca.mtimes(
                    (ca.mtimes(A_obs, x_k) - B_obs).T, self.variables[f"lamb_obs_quad_{idx}"][:, i]
                )
                >= self.params.margin
            )
            self.opti.subject_to(
                ca.mtimes(A_quad.T, self.variables[f"lamb_quad_{idx}"][:, i])
                + ca.mtimes(
                    ca.mtimes(R_sym.T, A_obs.T), self.variables[f"lamb_obs_quad_{idx}"][:, i]
                )
                == 0
            )
            temp = ca.mtimes(A_obs.T, self.variables[f"lamb_obs_quad_{idx}"][:, i])
            self.opti.subject_to(ca.mtimes(temp.T, temp) <= 1)
            self.opti.subject_to(self.variables[f"omega_quad_{idx}"][i] >= 0)
            self.opti.set_initial(self.variables[f"omega_quad_{idx}"][i], 0.1)

    def setup(self, viz_cb=False, warm=False, plot_global_planner=False):
        
        self.opti = ca.Opti()
        option = {
            "verbose": False,
            "ipopt.print_level": 0, # 5 for prints
            "ipopt.max_iter": 1000,  # Increase the maximum number of iterations
            "ipopt.warm_start_init_point": "yes",
            "ipopt.warm_start_bound_push": 1e-3,
            "ipopt.warm_start_bound_frac": 1e-9,
            "ipopt.warm_start_slack_bound_frac": 1e-3,
            "ipopt.warm_start_slack_bound_push": 1e-3,
            "ipopt.warm_start_mult_bound_push": 1e-3,
            "ipopt.fast_step_computation": "yes",
            "ipopt.acceptable_tol": 1e-4,
            "expand": True,
            "ipopt.hessian_approximation": 'limited-memory',  
            # "ipopt.linear_solver": "ma57",  # Use faster linear solver
            # "ipopt.hsllib": "libhsl.so"
            # "ipopt.mu_strategy": "adaptive",
            # "ipopt.limited_memory_update_type": "bfgs",  # More robust than SR1
            # "ipopt.mu_init: 0.1"
        }
        if warm:
            option["ipopt.mu_init"] = 1e-6
            option["ipopt.bound_relax_factor"] = 1e-9

        self.opti.solver("ipopt", option)

        if viz_cb:
            self.opti.callback(self.visualize_callback)

        self.initialize_variables(
            self.params.initial_state, plot_global_planner=plot_global_planner
        )
        self.set_state(self.params.initial_state)
        self.add_initial_condition_constraint()
        self.add_state_and_input_constraint()
        self.add_dynamics_constraint()
        self.min_obs_distances, self.inside_tube = self.find_tube_distances()

        # PLOT HELPER: visualize obstacles inside (red) vs outside (blue) the tube with path overlay
        if self.plot:
            self.plot_tube_obstacle_selection()

        for idx, obs in enumerate(self.obstacles):
            self.add_payload_to_obstacles_constraints(idx, obs)
            self.add_cable_to_obstacles_constraints(idx, obs)
            self.add_quadrotor_to_obstacles_constraints(idx, obs)

        self.add_cost()

    def find_tube_distances(self):
        # Prepare outputs
        self.min_obs_distances = []
        self.inside_tube = []

        # Graceful handling if not initialized yet
        if not hasattr(self, "obstacles") or self.obstacles is None:
            raise Exception
        if not hasattr(self, "interpolated_positions") or self.interpolated_positions is None:
            raise Exception

        # Positions along the planned path: shape (3, N)
        pos = np.asarray(self.interpolated_positions)
        if pos.ndim != 2 or pos.shape[0] < 3:
            raise Exception

        # Tube threshold (fallback to +inf if missing)
        tube_thresh = self.params.tube_distance

        # For each obstacle, compute min distance to the path (center + vertices)
        for obs in self.obstacles:
            # Center-based distance
            center = np.asarray(obs.get_center()).reshape(-1)[:3]
            diffs_center = pos[:3, :].T - center[None, :]
            dists_center = np.linalg.norm(diffs_center, axis=1)
            dmin_center = float(np.min(dists_center)) if dists_center.size > 0 else float("inf")

            # Vertex-based distance (optional layer)
            dmin_vertices = float("inf")
            verts = obs.get_vertices()
            # Compute min over all vertices
            for v in np.asarray(verts):
                v = np.asarray(v).reshape(1, 3)
                diffs_v = pos[:3, :].T - v  # (N,3) - (1,3)
                dists_v = np.linalg.norm(diffs_v, axis=1)
                dv = float(np.min(dists_v)) if dists_v.size > 0 else float("inf")
                if dv < dmin_vertices:
                    dmin_vertices = dv
    
            # Overall min distance: consider both center and corners
            dmin = min(dmin_center, dmin_vertices)

            self.min_obs_distances.append(dmin)
            self.inside_tube.append(bool(dmin < tube_thresh))

        return self.min_obs_distances, self.inside_tube

    def plot_tube_obstacle_selection(self):
        """
        Plot interpolated XY path and color obstacles based on tube membership:
        - Inside tube: red
        - Outside tube: blue
        Uses the same edge-tracing style as solve_with_polytopes.
        """
        if not hasattr(self, "inside_tube") or not hasattr(self, "interpolated_positions"):
            return

        ox_in, oy_in, ox_out, oy_out = [], [], [], []
        step = 0.1

        # Use params.obstacles to build obstacle edges similar to solve_with_polytopes
        for idx, key in enumerate(self.params.obstacles.keys()):
            ob = self.params.obstacles[key]
            x, y = ob["x"], ob["y"]
            l, b = ob["l"], ob["b"]
            x_min, x_max = x - l / 2.0, x + l / 2.0
            y_min, y_max = y - b / 2.0, y + b / 2.0

            tgt_ox, tgt_oy = (ox_in, oy_in) if (idx < len(self.inside_tube) and self.inside_tube[idx]) else (ox_out, oy_out)

            # Horizontal edges
            n_l = max(int(round(l / step)), 1)
            for i in range(n_l + 1):
                xi = x_min + i * (l / n_l)
                tgt_ox.append(xi); tgt_oy.append(y_min)
                tgt_ox.append(xi); tgt_oy.append(y_max)

            # Vertical edges
            n_b = max(int(round(b / step)), 1)
            for j in range(n_b + 1):
                yj = y_min + j * (b / n_b)
                tgt_ox.append(x_min); tgt_oy.append(yj)
                tgt_ox.append(x_max); tgt_oy.append(yj)

        # Plot
        plt.figure()
        # Interpolated XY path
        if self.interpolated_positions is not None and self.interpolated_positions.shape[1] > 0:
            plt.plot(self.interpolated_positions[0, :], self.interpolated_positions[1, :], "-g", label="interpolated path")
            # Mark start/end
            plt.plot(self.interpolated_positions[0, 0], self.interpolated_positions[1, 0], "og", label="start")
            plt.plot(self.interpolated_positions[0, -1], self.interpolated_positions[1, -1], "xb", label="goal")

        if len(ox_in) > 0:
            plt.plot(ox_in, oy_in, ".r", label="obstacles (inside tube)")
        if len(ox_out) > 0:
            plt.plot(ox_out, oy_out, ".b", label="obstacles (outside tube)")

        # Axes and limits if available
        try:
            plt.xlim(self.params.state_min[0], self.params.state_max[0])
            plt.ylim(self.params.state_min[1], self.params.state_max[1])
        except Exception:
            pass

        plt.title(f"Tube selection: {sum(self.inside_tube)} inside, {len(self.inside_tube) - sum(self.inside_tube)} outside")
        plt.grid(True)
        plt.axis("equal")
        plt.legend()
        plt.show()

    def visualize_callback(self, iteration):
        """Callback function to visualize trajectories every 25 iterations and save frames for GIF."""
        if iteration % 50 == 0:
            print("Saving frame for iteration:", iteration)

            # Extract current values of variables using self.opti.debug
            sol = self.opti.debug.value(self.variables["x"])
            u = self.opti.debug.value(self.variables["u"])

            # Calculate the current total optimized time
            total_time = 0
            if self.min_time:
                for i in range(self.params.horizon):
                    total_time += self.opti.debug.value(self.variables["t"][i])

            # Create a figure for the current frame with higher DPI
            fig = plt.figure(figsize=(10, 8), dpi=150)  # Increased DPI for higher resolution
            ax = fig.add_subplot(111, projection='3d')

            sol_values = {
                "x": self.opti.debug.value(self.variables["x"]),
                "u": self.opti.debug.value(self.variables["u"]),
                "t": self.opti.debug.value(self.variables["t"]),
            }
            for i in range(self.params.horizon):
                sol_values["t"][i] = self.opti.debug.value(self.variables["t"][i])

            interpolated_time, interpolated_x, interpolated_u = interpolate_distance(
                self.params, sol_values
            )

            plotter.plot_result(
                self.params,
                interpolated_x,
                interpolated_u,
                self.differential_flatness,
                self.compute_quadrotor_rotation_matrix_no_jrk,
                ax=ax,  # Pass the axis to plotter
                show=False,  # Ensure the plot is not displayed
                title=f"Iteration: {iteration}, Total Time: {total_time:.2f}s",
            )

            # Save the frame as an image array
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            self.frames.append(image)

            plt.close(fig)  # Close the figure to avoid displaying it during optimization

    def save_gif(self, file_dir):
        """Save the collected frames as a GIF."""
        if not self.frames:
            print("No frames collected to create a GIF.")
            return

        subdirectory = Path(file_dir).parts[0]
        filename_yaml = Path(file_dir).name
        filename = Path(file_dir).stem
        filedir_gif = os.path.join(GIFS_DIR, subdirectory, filename) + ".gif"
        folder = Path(os.path.join(GIFS_DIR, subdirectory))
        folder.mkdir(parents=True, exist_ok=True)

        # Save the animation as a GIF using Pillow
        images = [Image.fromarray(frame) for frame in self.frames]
        images[0].save(filedir_gif, save_all=True, append_images=images[1:], duration=600, loop=0)
        print(f"GIF saved at {filedir_gif}")

        # Clear the frames after saving
        self.frames = []

    def optimize(self, max_iter=1e6):
        self.opti.minimize(self.cost)
        success = True

        try:
            t1 = time.time()
            opt_sol = self.opti.solve()
            opt_time = time.time() - t1
            print("Optimization completed in {:.2f} seconds".format(opt_time))

            # Check the solver status
            stats = self.opti.stats()
            return_status = stats.get('return_status', 'Status not available')
            print("Solver Return Status:", return_status)

            # Check the number of iterations
            iterations = stats.get('iter_count', 'Iteration count not available')
            print("Number of iterations:", iterations)

        except RuntimeError as e:
            print("Solver failed:", str(e))
            opt_sol = self.opti.debug
            success = False
            iterations = 99999
            opt_time = 999

        sol_values = {}
        for key in self.variables.keys():
            try:
                sol_values[key] = opt_sol.value(self.variables[key]).T
            except RuntimeError:
                pass

        sol_values["lam_g"] = opt_sol.value(self.opti.lam_g)
        sol_values["xx"] = opt_sol.value(self.opti.x)

        return sol_values, opt_sol, success, iterations, opt_time

    def warm_start(self, init_sol_values):
        for key in self.variables.keys():
            self.opti.set_initial(self.variables[key], init_sol_values[key])

        self.opti.set_value(self.opti.lam_g, init_sol_values["lam_g"])
        self.opti.set_initial(self.opti.lam_g, init_sol_values["lam_g"])
        self.opti.set_initial(self.opti.x, init_sol_values["xx"])

    @staticmethod
    def differential_flatness(x, v, a, jrk, params):
        gravity = 9.81
        acc = np.array([a[0], a[1], a[2] + gravity])
        acc_norm = np.linalg.norm(acc)
        acc_norm_sq = acc_norm**2
        acc_norm_cub = acc_norm**3

        yaw = 0  # Assuming yaw is zero for simplicity
        yaw_dot = 0  # Assuming yaw_dot is zero for simplicity

        p = -acc / acc_norm
        acc_dot_jrk = np.dot(acc, jrk)
        dot_acc_norm = acc_dot_jrk / acc_norm
        ddot_acc_norm = np.linalg.norm(jrk) ** 2 / acc_norm - (acc_dot_jrk**2) / acc_norm_cub

        pdot = -jrk / acc_norm + dot_acc_norm * acc / acc_norm_sq
        pddot = (
            2.0 * dot_acc_norm * jrk / acc_norm_sq
            + ddot_acc_norm * acc / acc_norm_sq
            - 2.0 * (dot_acc_norm**2) * acc / acc_norm_cub
        )

        force = (
            params.mass_load + params.mass_quad
        ) * acc - params.mass_quad * params.cable_length * pddot

        b3c = force / np.linalg.norm(force)
        b2d = np.array([-np.sin(yaw), np.cos(yaw), 0])
        b1c = np.cross(b2d, b3c) / np.linalg.norm(np.cross(b2d, b3c))
        b2c = np.cross(b3c, b1c)

        R = np.column_stack((b1c, b2c, b3c))
        orientation = R  # Rotation matrix
        quat = Rot.from_matrix(R).as_quat() # returns x, y, z, w
        
        pos_payload = np.array([x[0], x[1], x[2]])
        pos_quad = pos_payload + (force / np.linalg.norm(force)) * params.cable_length

        vel_payload = np.array([v[0], v[1], v[2]])
        acc_payload = np.array([a[0], a[1], a[2]])
        vel_quad = vel_payload - params.cable_length * pdot
        acc_quad = acc_payload - params.cable_length * pddot

        payload_vector = (pos_payload - pos_quad)/params.cable_length

        # Rotation matrix from quad to payload
        # so that R[:,2] is the z axis of the quadrotor
        # and -payload_vector is the z axis of the payload (cable direction
        rpy = rpy_from_a_to_b(R[:, 2], -payload_vector)
        # rpy = np.array([0, 0, 0]) # because z axis of quad is always aligned with cable direction in differential flatnness

        return {
            "orientation": orientation,
            "quat": quat,
            "pos_quad": pos_quad,
            "vel_quad": vel_quad,
            "acc_quad": acc_quad,
            "payload_rpy": rpy,
        }
    
    @staticmethod
    def rpy_from_A_to_B(A, B, eps=1e-12):
        a = A / np.linalg.norm(A)
        b = B / np.linalg.norm(B)
        v = np.cross(a, b)
        c = float(np.dot(a, b))
        s = np.linalg.norm(v)

        if s < eps:
            if c > 0:  # aligned
                return np.eye(3)
            # opposite: choose any axis orthogonal to a
            e = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            u = np.cross(a, e); u /= np.linalg.norm(u)
            return -np.eye(3) + 2.0 * np.outer(u, u)

        vx = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
        rot_mat = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))

        yaw, pitch, roll = Rot.from_matrix(rot_mat).as_euler('zyx', degrees=False)
        # roll, pitch, yaw = Rot.from_matrix(rot_mat).as_euler('xyz', degrees=False)

        return np.array([roll, pitch, yaw])

    @staticmethod
    def compute_quadrotor_rotation_matrix_no_jrk(
        a, params, yaw_des=0, yaw_dot_des=0, symbolic=False, use_robot_rotation=True
    ):
        """
        Computes the quadrotor's rotation matrix based on payload's acceleration and yaw.

        Parameters:
            a (list): Acceleration [ax, ay, az].
            yaw_des (float): Desired yaw.
            yaw_dot_des (float): Desired yaw rate.
            symbolic (bool): If True, create a CasADi symbolic expression for R.

        Returns:
            np.ndarray or casadi.MX: Rotation matrix (3x3).
        """
        # TODO Remove and integrate into diff flatness method above
        gravity = 9.81
        if symbolic:
            acc = ca.MX(3, 1)
            acc[0, 0] = a[0]
            acc[1, 0] = a[1]
            acc[2, 0] = a[2] + gravity
            yaw = ca.MX(yaw_des)
            yaw_dot = ca.MX(yaw_dot_des)
            force = (params.mass_quad + params.mass_load) * acc
            force_norm = ca.norm_2(force)
            b3c = force / force_norm
            b2d = ca.vertcat(-ca.sin(yaw), ca.cos(yaw), 0)  # Fixed creation of b2d
            b1c = ca.cross(b2d, b3c) / ca.norm_2(ca.cross(b2d, b3c))
            b2c = ca.cross(b3c, b1c)
            R = ca.horzcat(b1c, b2c, b3c)
        else:
            acc = np.array([a[0], a[1], a[2] + 9.81])
            yaw = yaw_des
            yaw_dot = yaw_dot_des
            force = (params.mass_quad + params.mass_load) * acc
            force_norm = np.linalg.norm(force)
            b3c = force / force_norm if force_norm > 1e-6 else np.array([0, 0, 1])
            b2d = np.array([-np.sin(yaw), np.cos(yaw), 0])
            b1c = np.cross(b2d, b3c) / np.linalg.norm(np.cross(b2d, b3c))
            b2c = np.cross(b3c, b1c)
            R = np.column_stack((b1c, b2c, b3c))

        if not use_robot_rotation:
            return np.eye(3)

        return R

    def compute_quadrotor_rotation_matrix(self, a, j, yaw_des, yaw_dot_des, mQ, mL):
        """
        Computes the quadrotor's rotation matrix based on payload's position, velocity, acceleration, and jerk.

        Parameters:
            a (list): Acceleration [ax, ay, az].
            j (list): Jerk [jx, jy, jz].
            yaw_des (float): Desired yaw.
            yaw_dot_des (float): Desired yaw rate.
            mQ (float): Quadrotor mass.
            mL (float): Payload mass.
            cable_stable_length (float): Cable length.

        Returns:
            np.ndarray: Rotation matrix (3x3).
        """
        acc = np.array([a[0], a[1], a[2] + 9.81])
        jrk = np.array([j[0], j[1], j[2]])
        yaw = yaw_des
        yaw_dot = yaw_dot_des

        p = -acc / np.linalg.norm(acc)
        acc_norm = np.linalg.norm(acc)
        acc_norm_sq = acc_norm**2
        acc_norm_cub = acc_norm**3

        acc_dot_jrk = np.dot(acc, jrk)
        dot_acc_norm = acc_dot_jrk / acc_norm
        ddot_acc_norm = np.linalg.norm(jrk) ** 2 / acc_norm - (acc_dot_jrk**2) / acc_norm_cub

        pdot = -jrk / acc_norm + dot_acc_norm * acc / acc_norm_sq
        pddot = (
            2.0 * dot_acc_norm * jrk / acc_norm_sq
            + ddot_acc_norm * acc / acc_norm_sq
            - 2.0 * (dot_acc_norm**2) * acc / acc_norm_cub
        )

        force = (
            self.params.mass_quad + self.params.mass_load
        ) * acc - self.params.mass_quad * self.params.cable_length * pddot
        force_norm = np.linalg.norm(force)

        b2d = np.array([-np.sin(yaw), np.cos(yaw), 0])
        b3c = force / force_norm if force_norm > 1e-6 else np.array([0, 0, 1])
        b1c = np.cross(b2d, b3c) / np.linalg.norm(np.cross(b2d, b3c))
        b2c = np.cross(b3c, b1c)

        R = np.column_stack((b1c, b2c, b3c))
        return R


def interpolate(params, sol_values, dt=0.01):
    # consume time-major arrays: x (N+1, Mx), u (N, Mu), t ~ (N,)
    sol_x, sol_u = sol_values["x"], sol_values["u"]

    # t may be absent if not min-time; flatten to 1-D
    if "t" in sol_values and sol_values["t"] is not None:
        sol_t = np.asarray(sol_values["t"]).reshape(-1)
    else:
        sol_t = np.full(params.horizon, params.dt, dtype=float)

    # Compute cumulative time (length N+1)
    cumulative_time = np.concatenate([[0.0], np.cumsum(sol_t)])

    # Interpolation using CubicSpline
    time_points = np.array(cumulative_time)
    interpolated_time = np.arange(0, cumulative_time[-1], dt)

    # keep outputs component-major (state/input first, time second)
    Nt = len(interpolated_time)
    interpolated_x = np.zeros((Nt, sol_x.shape[1]))
    interpolated_u = np.zeros((Nt, sol_u.shape[1]))

    # Interpolate states: per component over time-major series
    for i in range(sol_x.shape[1]):
        cs = CubicSpline(time_points, sol_x[:, i], bc_type='natural')
        interpolated_x[:, i] = cs(interpolated_time)

    # Interpolate inputs (have one less time point)
    for i in range(sol_u.shape[1]):
        cs = CubicSpline(time_points[:-1], sol_u[:, i], bc_type='natural')
        interpolated_u[:, i] = cs(interpolated_time)

    # find time index in interpolated_time that is closest to the last time point
    last_time_index = np.argmin(np.abs(interpolated_time - cumulative_time[-1]))
    return (
        interpolated_time[:last_time_index],
        interpolated_x[:last_time_index, :],
        interpolated_u[:last_time_index, :],
    )


def interpolate_distance(
    params,
    sol_values,
    ds=0.3,  # [m] arc-length step
    pos_idx=(0, 1, 2),  # rows in sol_x that hold x-, y-, z-position
    append_zero=True,
):
    interp_t, interp_x, interp_u = interpolate(params, sol_values, dt=0.01)

    N = interp_x.shape[0]
    indices = [0]  # always keep the very first point
    s_cum = [0.0]  # cumulative distance for the kept points
    d_accum = 0.0  # distance since last kept point

    # ---- walk through fine samples -----------------------------------------
    for i in range(1, N):
        step = np.linalg.norm(interp_x[i, pos_idx] - interp_x[i - 1, pos_idx])
        d_accum += step
        if d_accum >= ds:  # keep this sample
            indices.append(i)
            s_cum.append(s_cum[-1] + d_accum)
            d_accum = 0.0

    # ---- ensure final knot is included -------------------------------------
    if indices[-1] != N - 1:
        indices.append(N - 1)
        # add the leftover distance since last kept sample
        last_step = np.linalg.norm(interp_x[N - 1, pos_idx] - interp_x[indices[-2], pos_idx])
        s_cum.append(s_cum[-1] + last_step)

    indices = np.array(indices, dtype=int)

    # ---- gather the selected samples ---------------------------------------
    sel_t = interp_t[indices]
    sel_x = interp_x[indices, :]
    sel_u = interp_u[indices, :]  # same indices → same length

    # ---- optionally append a zero-input column -----------------------------
    if append_zero:
        zero_col = np.zeros((sel_u.shape[1], 1))
        sel_u = np.hstack([sel_u, zero_col])
        sel_x = np.hstack([sel_x, sel_x[-1:, :]])  # repeat last state
        sel_t = np.append(sel_t, sel_t[-1])
        s_cum.append(s_cum[-1])  # distance unchanged

    sel_s = np.asarray(s_cum)

    return sel_t, sel_x, sel_u


def save_result(file_dir, params, sol_values, dt=0.002, plot=False):
    print(f"file_dir: {file_dir}")
    subdirectory = Path(file_dir).parts[0]
    filename_yaml = Path(file_dir).name
    filename = Path(file_dir).stem
    filedir_csv = os.path.join(CSV_DIR, subdirectory, filename) + ".csv"
    filedir_params = os.path.join(PARAMS_DIR, subdirectory, filename) + ".yaml"

    folder = Path(os.path.join(CSV_DIR, subdirectory))
    folder.mkdir(parents=True, exist_ok=True)

    sol_x, sol_u = sol_values["x"], sol_values["u"]

    # t may be absent if not min-time; flatten to 1-D
    if "t" in sol_values and sol_values["t"] is not None:
        sol_t = np.asarray(sol_values["t"]).reshape(-1)
    else:
        sol_t = np.full(params.horizon, params.dt, dtype=float)

    # Compute cumulative time
    cumulative_time = np.concatenate([[0.0], np.cumsum(sol_t)])
    time_points = np.array(cumulative_time)

    # Interpolate results
    interpolated_time, interpolated_x, interpolated_u = interpolate(params, sol_values, dt=dt)
    
    if plot:
        fig, axs = plt.subplots(sol_x.shape[1] + sol_u.shape[1], 1, figsize=(10, 15))
        for i in range(sol_x.shape[1]):  # Plot states
            axs[i].plot(time_points, sol_x[:, i], 'o-', label=f'Original State {i}')
            axs[i].plot(
                interpolated_time, interpolated_x[:, i], '-', label=f'Interpolated State {i}'
            )
            axs[i].set_title(f'State {i}')
            axs[i].legend()

        for i in range(sol_u.shape[1]):  # Plot inputs
            axs[sol_x.shape[1] + i].plot(
                time_points[:-1], sol_u[:, i], 'o-', label=f'Original Input {i}'
            )
            axs[sol_x.shape[1] + i].plot(
                interpolated_time, interpolated_u[:, i], '-', label=f'Interpolated Input {i}'
            )
            axs[sol_x.shape[1] + i].set_title(f'Input {i}')
            axs[sol_x.shape[1] + i].legend()

        plt.tight_layout()
        plt.show()

    # update z height for real world experiments
    # interpolated_x[2, :] += 0.3  # Adjust z height for real-world experiments
    # interpolated_x[0, :] -= 1.0  # Adjust z height for real-world experiments
    # interpolated_x[1, :] += 0.0  # Adjust z height for real-world experiments

    interpolated_quad_x = np.zeros_like(interpolated_x)
    interpolated_payload_rpy = np.zeros((interpolated_x.shape[0], 3))
    interpolated_quad_quat = np.zeros((interpolated_x.shape[0], 4))

    for i in range(interpolated_x.shape[0]):
        x = interpolated_x[i, :3]
        v = interpolated_x[i, 3:6]
        a = interpolated_x[i, 6:9]
        jrk = interpolated_u[i, :3]
        result = Planner.differential_flatness(x, v, a, jrk, params)
        interpolated_quad_x[i, :] = np.concatenate((result["pos_quad"], result["vel_quad"], result["acc_quad"]))
        interpolated_quad_quat[i, :] = result["quat"]
        interpolated_payload_rpy[i, :] = result["payload_rpy"]

    # save_csv_arrays(filedir_csv, interpolated_time, interpolated_x, interpolated_u, interpolated_quad_x)
    # save_params(filedir_params, params)

    interpolated_yaw, interpolated_quad_quat = get_yaw_along_trajectory(interpolated_x[:, :3], interpolated_x[:, 3:6], interpolated_quad_quat, interpolated_time)
    interpolated_rot_mat = get_rotation_matrix_from_quat(interpolated_quad_quat)
    save_all({
        AttrKeys.STEM: filename,
        AttrKeys.CSV_SUBDIRECTORY: subdirectory,
        DatasetKeys.TIME: interpolated_time,
        DatasetKeys.SOL_X: interpolated_x,
        DatasetKeys.SOL_U: interpolated_u,
        DatasetKeys.SOL_QUAD_X: interpolated_quad_x,
        DatasetKeys.SOL_QUAD_QUAT: interpolated_quad_quat,
        DatasetKeys.SOL_PAYLOAD_RPY: interpolated_payload_rpy,
        DatasetKeys.PARAMS: params,
        DatasetKeys.ROT_MAT: interpolated_rot_mat,
    })

def save_images(
    file_dir, params, sol_values, differential_flatness, compute_quadrotor_rotation_matrix_no_jrk
):
    file_ext = ".png"
    subdirectory = Path(file_dir).parts[0]
    filename_yaml = Path(file_dir).name
    filename = Path(file_dir).stem
    filedir_img = os.path.join(IMG_DIR, subdirectory, filename) + file_ext
    folder = Path(os.path.join(IMG_DIR, subdirectory))
    folder.mkdir(parents=True, exist_ok=True)

    plotter.plot_result(
        params,
        sol_values["x"],  # transpose to component-major for plotting
        sol_values["u"],
        differential_flatness,
        compute_quadrotor_rotation_matrix_no_jrk,
        save=True,
        show=False,
        save_filename=filedir_img,
    )

    interpolated_time, interpolated_x, interpolated_u = interpolate_distance(params, sol_values)
    filedir_img = os.path.join(IMG_DIR, subdirectory, filename) + "_ortho" + file_ext

    plotter.plot_result(
        params,
        sol_values["x"],
        sol_values["u"],
        differential_flatness,
        compute_quadrotor_rotation_matrix_no_jrk,
        save=True,
        show=False,
        ortho_view=True,
        save_filename=filedir_img,
    )

    interpolated_time, interpolated_x, interpolated_u = interpolate_distance(params, sol_values)
    filedir_img = os.path.join(IMG_DIR, subdirectory, filename) + "_interpolated" + file_ext
    plotter.plot_result(
        params,
        sol_values["x"],  # transpose
        sol_values["u"],
        differential_flatness,
        compute_quadrotor_rotation_matrix_no_jrk,
        save=True,
        show=False,
        save_filename=filedir_img,
    )


def run(
    relative_path,
    init_sol_values=None,
    plot=True,
    plot_times=False,
    plot_interpolated=False,
    save_fig=False,
):
    print("------------------------------------")
    print(f"Solving {relative_path}")
    print("------------------------------------")
    file_dir = os.path.join(PARAMS_DIR, relative_path)
    params = dictToClass(MPC, yamlToDict(file_dir))
    planner = Planner(params, plot=False)
    planner.setup(viz_cb=False)

    if init_sol_values is not None:
        print("CALLING WARM START")
        planner.warm_start(init_sol_values)

    if plot:
        plotter.plot_interpolated_positions_and_obstacles(
            planner.params, planner.interpolated_positions
        )
    sol_values, sol_opt, success, iterations, opt_time = planner.optimize()

    total_time = 1000
    if success and planner.params.min_time:
        total_time = 0
        for i in range(planner.params.horizon):
            total_time += sol_opt.value(planner.variables["t"][i])
        print(f"Total time = {total_time}")

    path_length = 999
    if success:
        # time-major: x (N+1, M); take position columns [:3]
        pos = sol_values["x"][:, :3]
        deltas = np.diff(pos, axis=0)
        segment_lengths = np.linalg.norm(deltas, axis=1)
        path_length = segment_lengths.sum()

    sol_x, sol_u = sol_values["x"], sol_values["u"]

    if save_fig:
        save_images(
            relative_path,
            planner.params,
            sol_values,
            planner.differential_flatness,
            planner.compute_quadrotor_rotation_matrix_no_jrk,
        )

    if plot:
        plotter.plot_result(
            planner.params,
            sol_x,  # transpose to component-major
            sol_u,
            planner.differential_flatness,
            planner.compute_quadrotor_rotation_matrix_no_jrk,
        )

    # if plot_interpolated:
    #     interpolated_time, interpolated_x, interpolated_u = interpolate_distance(
    #         planner.params, sol_values
    #     )
    #     plotter.plot_result(
    #         planner.params,
    #         interpolated_x,
    #         interpolated_u,
    #         planner.differential_flatness,
    #         planner.compute_quadrotor_rotation_matrix_no_jrk,
    #     )

    if plot_times:
        plotter.plot_resuls_times(planner, params, sol_values)

    save_result(relative_path, params, sol_values, plot=plot)
    planner.save_gif(relative_path)

    return total_time, sol_opt, sol_values, iterations, opt_time, path_length


def experiments():
    yamls = [
        "autotrans/maze_1.yaml",
        "autotrans/maze_2.yaml",
        "autotrans/maze_3.yaml",
        "autotrans/maze_4.yaml",
        "autotrans/maze_5.yaml",
        "autotrans/maze_6.yaml",
        "autotrans/maze_7.yaml",
        "autotrans/maze_8.yaml",
        "autotrans/maze_9.yaml",
        "orientation_benefits/collision.yaml",
        "orientation_benefits/no_collision.yaml",
    ]

    total_times = []
    iterations = []
    opt_times = []
    path_lengths = []
    for path_to_yaml in yamls:
        total_time, opt_sol, opt_sol_values, iteration, opt_time, path_length = run(
            path_to_yaml, plot=True, plot_interpolated=False, plot_times=False, save_fig=False
        )
        total_times.append(total_time)
        iterations.append(iteration)
        opt_times.append(opt_time)
        path_lengths.append(path_length)

    print("\n\n")
    for i in range(len(yamls)):
        print(
            f"{yamls[i]}: Traj: {total_times[i]:.2f} , Iterations: {iterations[i]}, Opt Time: {opt_times[i]:.2f}(s), Path Length: {path_lengths[i]:.2f}(m)"
        )


def weight_study():
    yamls = [
        "init_weight_study/maze_1_0.yaml",
        "init_weight_study/maze_1_1.yaml",
        "init_weight_study/maze_1_5.yaml",
        "init_weight_study/maze_2_0.yaml",
        "init_weight_study/maze_2_1.yaml",
        "init_weight_study/maze_2_5.yaml",
        "init_weight_study/maze_3_0.yaml",
        "init_weight_study/maze_3_1.yaml",
        "init_weight_study/maze_3_5.yaml",
        "init_weight_study/maze_4_0.yaml",
        "init_weight_study/maze_4_1.yaml",
        "init_weight_study/maze_4_5.yaml",
        "init_weight_study/maze_5_0.yaml",
        "init_weight_study/maze_5_1.yaml",
        "init_weight_study/maze_5_5.yaml",
        "init_weight_study/maze_6_0.yaml",
        "init_weight_study/maze_6_1.yaml",
        "init_weight_study/maze_6_5.yaml",
        "init_weight_study/maze_7_0.yaml",
        "init_weight_study/maze_7_1.yaml",
        "init_weight_study/maze_7_5.yaml",
        "init_weight_study/maze_8_0.yaml",
        "init_weight_study/maze_8_1.yaml",
        "init_weight_study/maze_8_5.yaml",
    ]

    total_times = []
    for path_to_yaml in yamls:
        total_time, opt_sol, opt_sol_values, iteration, opt_time = run(path_to_yaml, plot=True)
        total_times.append(total_time)

    for i in range(len(yamls)):
        print(f"{yamls[i]}: {total_times[i]} ")


def one_run(core_ids):
    print(f"using core {core_ids}")
    print(f"PID {os.getpid()} on cores {core_ids}")

    os.sched_setaffinity(0, core_ids)  # Linux only
    os.environ["OMP_NUM_THREADS"] = "1"  # keep BLAS/HSL single-threaded
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    relative_path = "long_range/trees2.yaml"

    print("------------------------------------")
    print(f"Solving {relative_path}")
    print("------------------------------------")
    file_dir = os.path.join(PARAMS_DIR, relative_path)
    params = dictToClass(MPC, yamlToDict(file_dir))

    planner = Planner(params, plot=False)
    planner.setup(warm=False)
    sol_values, sol_opt, success, iterations, opt_time = planner.optimize()


def run_mp():
    t1 = time.time()

    mp.set_start_method("spawn", force=True)  # or "forkserver"
    core_sets = [{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}]  # adapt
    core_sets = [{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}, {12, 13}]  # adapt
    core_sets = [{i} for i in range(14)]  # 14 workers → cores 0‑13

    with mp.Pool(processes=len(core_sets)) as pool:
        results = pool.starmap(one_run, zip(core_sets))

    t2 = time.time()
    print(f"Total time taken: {t2 - t1:.2f} seconds")
    print(f"Efficiency {len(core_sets)/ (t2 - t1)}")


if __name__ == "__main__":
    # run_mp()
    experiments()
    # weight_study()
