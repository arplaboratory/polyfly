"""
Random tree-field generator + trajectory optimisation demo
----------------------------------------------------------

* Creates 8 random cuboid obstacles (‘tree trunks’) in the X–Y plane.
* Ensures none collides with the start position (0,0,0) or with each other.
* Chooses the nominal goal (8,0,0) if it is collision–free; otherwise
  searches nearby Y locations until a free one is found.
* Runs the full optimal-planner pipeline and plots the resampled result.
"""

import os
import random
import argparse
import multiprocessing as mp
import time
import numpy as np
from time import perf_counter as _timer
from pathlib import Path
import typing 
from typing import Optional

import poly_fly.utils.plot as plotter
from poly_fly.optimal_planner.planner import (
    Planner,
    interpolate_distance,
    dictToClass,
    yamlToDict,
    save_result,
)
from poly_fly.utils.utils import MPC
from poly_fly.optimal_planner.global_planner import OpenSetEmptyException

from poly_fly.data_io.utils import save_csv_arrays, save_params, BASE_DIR, PARAMS_DIR, \
    GIFS_DIR, CSV_DIR, IMG_DIR
from poly_fly.forest_planner.forest_params import ForestParamsLarge, ForestParamsSmall, \
    FOREST_SMALL_OBS, FOREST_LARGE_OBS

MAX_PROCS = 16

class ObstacleGenerationException(RuntimeError):
    pass


class ForestGenerator:
    # ───────────────────────── initialisation ──────────────────────────
    def __init__(self, seed=None, show_plot=True):
        # reproducibility
        self.seed = seed  # set to an int for repeatability
        if self.seed is not None:
            random.seed(self.seed)
        self.plot = show_plot

        # forest-related params now live in ForestParamsLarge
        # default so legacy entry points still work; can be overridden via set_forest_params()
        self.forest_params = ForestParamsLarge()

        self.Q_dt = 1000
        self.margin = None 

        self.base_yaml = os.path.join(PARAMS_DIR, "forests") + "/base.yaml" #Path(os.environ["POLYFLY_DIR"]) / "params" / "forests" / "base.yaml"
        self.params = dictToClass(MPC, yamlToDict(self.base_yaml))
        self.planner = Planner(self.params, plot=show_plot)
        self.planner.clear_obstacles()

    # ───────────────────────── helper utilities ─────────────────────────
    @staticmethod
    def _sx(box):  # size in x
        return box.get("xl", box.get("l"))

    @staticmethod
    def _sy(box):  # size in y
        return box.get("yl", box.get("b"))

    def _box_overlap(self, a, b) -> bool:
        return abs(a["x"] - b["x"]) <= self._sx(a) + self._sx(b) and abs(
            a["y"] - b["y"]
        ) <= self._sy(a) + self._sy(b)

    def _point_in_box(self, px, py, box) -> bool:
        return abs(px - box["x"]) <= self._sx(box) and abs(py - box["y"]) <= self._sy(box)

    def _rand_y(self, dist_mode: str, gauss_sigma: float = 2.0) -> float:
        """Draw a Y-coordinate according to the chosen distribution."""
        if dist_mode == "uniform":
            return random.uniform(*self.forest_params.y_range)
        if dist_mode == "gauss":
            while True:
                y = random.gauss(0.0, self.forest_params.gauss_sigma_y)
                if self.forest_params.y_obs_range[0] <= y <= self.forest_params.y_obs_range[1]:
                    return y
        raise ValueError(f"Unknown distribution: {dist_mode!r}")

    def _rand_x(self, dist_mode: str, gauss_sigma: float = 2.0) -> float:
        """Draw an X-coordinate according to the chosen distribution."""
        if dist_mode == "uniform":
            return random.uniform(*self.forest_params.x_obs_range)
        if dist_mode == "gauss":
            while True:
                x = random.gauss(0.0, self.forest_params.gauss_sigma_x)
                if self.forest_params.x_obs_range[0] <= x <= self.forest_params.x_obs_range[1]:
                    return x
        raise ValueError(f"Unknown distribution: {dist_mode!r}")

    def _make_box(self, size_l_rng, size_b_rng, dist_mode_x: typing.Optional[str] = None,
                  dist_mode_y: typing.Optional[str] = None):
        """Return one random cuboid dict (x,y,z,xl,yl,zl)."""
        dm_x = dist_mode_x or self.forest_params.dist_mode_x
        dm_y = dist_mode_y or self.forest_params.dist_mode_y
        return dict(
            x=self._rand_x(dm_x),
            y=self._rand_y(dm_y),
            z=0.0,
            xl=random.uniform(*size_l_rng),
            yl=random.uniform(*size_b_rng),
            zl=self.forest_params.trunk_height / 2.0,
        )

    # ───────────────────────── public API ───────────────────────────────
    def set_goal(self, x: float, y: float, z, buffer: float = 1.0) -> None:
        """
        Update goal in forest_params and validate bounds.
        """
        if not (self.forest_params.x_range[0] <= x <= self.forest_params.x_range[1]):
            raise ValueError(f"x={x} outside workspace {self.forest_params.x_range}")
        if not (self.forest_params.y_range[0] <= y <= self.forest_params.y_range[1]):
            raise ValueError(f"y={y} outside workspace {self.forest_params.y_range}")

        self.forest_params.goal_x = x
        self.forest_params.goal_y = y
        self.forest_params.goal_z = z

        # Sanity-check the remaining space
        if (self.forest_params.x_obs_range[1] <= self.forest_params.x_obs_range[0] or
            self.forest_params.y_obs_range[1] <= self.forest_params.y_obs_range[0]):
            raise RuntimeError("Goal buffer leaves no room for obstacles!")

    def set_margin(self, margin: float):
        assert margin > 0
        self.margin = margin

    def generate_forest(
        self,
        goal_x: Optional[float] = None,
    ):
        """Create `n_large + n_small` trunks plus one fixed central trunk.
        """
        n_large = self.forest_params.n_large_obs
        n_small = self.forest_params.n_small_obs
        goal_x = self.forest_params.goal_x if goal_x is None else goal_x

        obstacles: list[dict] = []
        large_obstacles: list[dict] = []
        small_obstacles: list[dict] = []

        # 0) fixed central trunk
        if self.forest_params.central_trunk:
            obstacles.append(
                dict(
                    x=goal_x / 2.0,
                    y=0.0,
                    z=0.0,
                    xl=self.forest_params.large_size_l_range[1],
                    yl=self.forest_params.large_size_b_range[1],
                    zl=self.forest_params.trunk_height / 2.0,
                )
            )

        while len(large_obstacles) < n_large:
            cand = self._make_box(
                self.forest_params.large_size_l_range,
                self.forest_params.large_size_b_range,
                dist_mode_x=self.forest_params.dist_mode_x,
                dist_mode_y=self.forest_params.dist_mode_y,
            )

            if self._point_in_box(0.0, 0.0, cand):
                continue
            if any(self._box_overlap(cand, prev) for prev in large_obstacles):
                continue
            if any(self._box_overlap(cand, prev) for prev in obstacles):
                continue

            large_obstacles.append(cand)

        while len(small_obstacles) < n_small:
            cand = self._make_box(
                self.forest_params.small_size_l_range,
                self.forest_params.small_size_b_range,
                dist_mode_x=self.forest_params.dist_mode_x,
                dist_mode_y=self.forest_params.dist_mode_y,
            )

            if self._point_in_box(0.0, 0.0, cand):
                continue
            if any(self._box_overlap(cand, prev) for prev in large_obstacles):
                continue
            if any(self._box_overlap(cand, prev) for prev in small_obstacles):
                continue
            if any(self._box_overlap(cand, prev) for prev in obstacles):
                continue

            small_obstacles.append(cand)

        obstacles = obstacles + large_obstacles + small_obstacles 

        return obstacles

    def generate_checkerboard_forest(
        self,
        density: float,
        *,
        xl: Optional[float] = None,
        yl: Optional[float] = None,
        skip_start: bool = True,
        x_grid_density: float = 2.5,
        y_grid_density: float = 1.2,
    ):
        """Make a uniform checkerboard of identical box obstacles."""
        xl = 1.0 if xl is None else xl
        yl = 1.5 if yl is None else yl

        # spacing derived from density
        s_x = (x_grid_density * xl) / density
        s_y = (y_grid_density * yl) / density

        x_centres = np.arange(self.forest_params.x_obs_range[0] + xl,
                              self.forest_params.x_obs_range[1] - xl + 1e-6, s_x)
        y_centres = np.arange(self.forest_params.y_obs_range[0] + yl,
                              self.forest_params.y_obs_range[1] - yl + 1e-6, s_y)

        obstacles: list[dict] = []
        for i, x in enumerate(x_centres):
            for j, y in enumerate(y_centres):
                if (i + j) % 2:  # checkerboard rule
                    continue

                cand = dict(
                    x=x,
                    y=y,
                    z=0.0,
                    xl=xl,
                    yl=yl,
                    zl=self.forest_params.trunk_height / 2.0,
                )

                if skip_start and self._point_in_box(0.0, 0.0, cand):
                    continue
                obstacles.append(cand)

        if not obstacles:
            raise ObstacleGenerationException(
                "No obstacles generated — check density or workspace size."
            )
        return obstacles

    def set_forest(self, forest):
        self.forest = forest

    def set_horizon(self, horizon):
        self.horizon = horizon

    def set_forest_params(self, forest_params: ForestParamsLarge):
        """Set forest configuration (sizes, sampling modes, trunk height, etc.)."""
        self.forest_params = forest_params

    def pick_goal(
        self,
        obstacles,
        goal_x: Optional[float] = None,
        goal_y: Optional[float] = None,
        goal_z: Optional[float] = None,
    ):
        """Return a collision-free goal near (goal_x, goal_y, goal_z)."""
        goal_x = self.forest_params.goal_x if goal_x is None else goal_x
        goal_y = self.forest_params.goal_y if goal_y is None else goal_y
        goal_z = self.forest_params.goal_z if goal_z is None else goal_z

        # 1) try nominal goal
        if not any(self._point_in_box(goal_x, goal_y, obs) for obs in obstacles):
            return (goal_x, goal_y, goal_z)

        # 2) scan candidate y offsets
        print("Finding y candidates")
        for y in self.forest_params.goal_y_candidates:
            if not any(self._point_in_box(goal_x, y, obs) for obs in obstacles):
                return (goal_x, y, goal_z)

        # 3) last resort: move slightly upstream in x
        x_vals = np.linspace(goal_x - 0.5, goal_x - 2.0, 16)
        for x in x_vals:
            if not any(self._point_in_box(x, goal_y, obs) for obs in obstacles):
                return (x, goal_y, goal_z)

        raise RuntimeError("Could not find a collision-free goal.")

    def set_planner_params(self):
        self.planner.clear_obstacles()

        for obs in self.forest:
            self.planner.add_obstacle_box(**obs)  # now matches the helper’s signature
        self.planner.set_start_state((0.0, 0.0, 0.0))
        self.planner.set_end_state(
            self.pick_goal(self.forest,
                           goal_x=self.forest_params.goal_x,
                           goal_y=self.forest_params.goal_y)
        )
        self.planner.set_xy_bounds(
            self.forest_params.x_range[0], self.forest_params.x_range[1],
            self.forest_params.y_range[0], self.forest_params.y_range[1]
        )
        self.planner.Q_dt = self.Q_dt
        self.planner.set_margin(self.margin)

    def solve(self, goal_x=None, goal_y=None, goal_z=None, start_state=None, start_vel_state=None, retry=False):
        goal_x = self.forest_params.goal_x if goal_x is None else goal_x
        goal_y = self.forest_params.goal_y if goal_y is None else goal_y
        goal_z = self.forest_params.goal_z if goal_z is None else goal_z

        self.set_goal(goal_x, goal_y, goal_z)
        self.set_planner_params()

        if start_state is not None and start_vel_state is not None:
            self.planner.set_start_state(start_state, vel=start_vel_state)
        elif start_state is not None:
            self.planner.set_start_state(start_state)

        try:
            self.planner.setup(plot_global_planner=self.plot, viz_cb=False)
        except OpenSetEmptyException as e:
            print(e)
            return None 
        
        try:
            sol_values, opt_sol, success, iterations, opt_time = self.planner.optimize()
        except OpenSetEmptyException:
            return None

        if retry and not success:
            print("Retrying with more longer horizon")
            print(f"Old horizon: {self.horizon}")

            self.planner.params.horizon = self.planner.params.horizon + 20
            self.planner.setup(plot_global_planner=self.plot, viz_cb=False)

            try:
                sol_values, opt_sol, success, iterations, opt_time = self.planner.optimize()
            except OpenSetEmptyException:
                return None

        return sol_values, opt_sol, success, iterations, opt_time

    def run(self, goal_x_list, goal_y_list, show_plot=True, filename="example"):
        forest = self.set_forest(self.generate_forest())

        for goal_x in goal_x_list:
            for goal_y in goal_y_list:
                result = self.solve(goal_x=goal_x, goal_y=goal_y)
                if result is None:
                    print(f"optimization failed for {filename}")
                    continue

                sol_values, opt_sol, success, iterations, opt_time = result
                print("success", success)
                if not success:
                    print(f"optimization failed for {filename}")
                    continue
                
                if show_plot:
                    t, x, u = interpolate_distance(self.planner.params, sol_values, ds=0.25, append_zero=False)
                    plotter.plot_result(
                        self.planner.params,
                        x,
                        u,
                        self.planner.differential_flatness,
                        self.planner.compute_quadrotor_rotation_matrix_no_jrk,
                    )
                    plotter.plot_results_2d(self.planner.params, t, x, u, self.planner.differential_flatness)

                    sol_x, sol_u = sol_values["x"], sol_values["u"]
                    sol_t = np.zeros((1, self.planner.params.horizon))
                    for i in range(self.planner.params.horizon):
                        sol_t[:, i] = sol_values["t"][i]
                    cumulative_time = [0]
                    for t in sol_t[0]:
                        cumulative_time.append(cumulative_time[-1] + t)
                    time_points = np.array(cumulative_time[:])  # Exclude the last point
                    plotter.plot_result(
                        self.planner.params,
                        sol_x,
                        sol_u,
                        self.planner.differential_flatness,
                        self.planner.compute_quadrotor_rotation_matrix_no_jrk,
                    )

                relative_path = "forests/" + filename + ".test"
                save_result(relative_path, self.planner.params, sol_values, plot=show_plot)


# ------------------------------------------------------------
# Main routine
# ------------------------------------------------------------
def main(show_plot=True):
    gen = ForestGenerator(seed=1)
    goal_x_list = [15]
    goal_y_list = [-2]
    gen.run(goal_x_list, goal_y_list, filename="ex")


def main_multiple_forests(show_plot=True):
    seeds = [4, 3]
    # seeds = [3514696575, 100]
    for i in range(2):
        gen = ForestGenerator(show_plot=show_plot, seed=seeds[i])
        goal_x_list = [15]
        goal_y_list = [-2]
        gen.run(goal_x_list, goal_y_list, filename=f"example_{i}", show_plot=show_plot)


def run_one_seed(
    goal: tuple,
    start: tuple,
    seed: int,
    idx: int,
    planner_margin, 
    init_vel,
    show_plot: bool = False,
    core_id: Optional[int] = None,
    forest_type: int = 2,
):
    """
    Worker: build a ForestGenerator with `seed`, set goal, vary start,
    and run once with a unique output filename.

    Returns:
        (filename, elapsed_seconds, idx, seed, core_id, success)
    """
    # Limit BLAS/OpenMP threads per process to avoid oversubscription
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = os.environ.get(var, "1")

    # CPU affinity pinning (Linux) — always on for this script
    if core_id is not None:
        try:
            if hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(0, {int(core_id)})
                print(f"[worker {idx:03d}] pinned to CPU {core_id}")
            else:
                print(f"[worker {idx:03d}] CPU pinning not supported; continuing.")
        except Exception as e:
            print(f"[worker {idx:03d}] could not set affinity to {core_id}: {e}")

    # Do the work (measure only the generation+run block)
    t0 = _timer()
    gen = ForestGenerator(show_plot=show_plot, seed=seed)

    if forest_type == FOREST_SMALL_OBS["id"]:
        gen.set_forest_params(ForestParamsSmall())
        forest = gen.set_forest(gen.generate_forest())

    elif forest_type == FOREST_LARGE_OBS["id"]:
        gen.set_forest_params(ForestParamsLarge())
        forest = gen.set_forest(gen.generate_forest())
    else:
        raise Exception()
    
    gen.set_goal(*goal)
    gen.set_margin(planner_margin)
    
    # Build a filesystem-safe suffix for goal, margin and init velocity
    gx, gy, gz = (str(v).replace('.', 'p').replace('-', 'n') for v in goal)
    goal_tag = f"g{gx}_{gy}_{gz}"
    margin_tag = f"m{planner_margin}".replace('.', 'p').replace('-', 'n')
    iv = tuple(init_vel)
    iv_tag = "v" + "_".join(str(v).replace('.', 'p').replace('-', 'n') for v in iv)

    fname = f"forest_{idx:03d}_f{forest_type}_{goal_tag}_{margin_tag}_{iv_tag}_s{seed}"
    success = False
    sol_values = None

    # Use the provided start state
    try:
        result = gen.solve(start_state=start, start_vel_state=init_vel)
    except OpenSetEmptyException as e:
        print("open set exception {e}")
        result = None 

    if result is not None:
        sol_values, opt_sol, success, iterations, opt_time = result
        if success:
            relative_path = "forests/" + fname + ".test"
            save_result(relative_path, gen.planner.params, sol_values, plot=show_plot, dt=0.1)
        else:
            print(f"optimization failed for {fname}")
    dt = _timer() - t0

    if show_plot and success:
        sol_x, sol_u = sol_values["x"], sol_values["u"]
        plotter.plot_result(
            gen.planner.params,
            sol_x,
            sol_u,
            gen.planner.differential_flatness,
            gen.planner.compute_quadrotor_rotation_matrix_no_jrk,
        )

    return (fname, dt, idx, seed, core_id, success)


def worker_loop(worker_id: int, show_plot: bool, pin: bool,
                optimizations_to_run, results, opt_lock, res_lock, forest_type=2) -> None:
    """
    Pop jobs from optimizations_to_run until empty, run them, and append to results.
    Uses opt_lock/res_lock to synchronize access to shared lists.
    """
    core_id = worker_id if pin else None
    while True:
        # Pull a job
        opt_lock.acquire()
        try:
            if len(optimizations_to_run) == 0:
                return
            job = optimizations_to_run.pop()  # LIFO pop
        finally:
            opt_lock.release()

        # Run the optimization
        res = run_one_seed(
            job["goal"], job["start"], job["seed"], job["idx"], 
            job["planner_margin"], job["init_vel"],
            show_plot, core_id, forest_type=forest_type)

        # Store the result
        res_lock.acquire()
        try:
            results.append(res)
        finally:
            res_lock.release()


def append_job_lists(
    goals,
    starts,
    seeds,
    planner_margins,
    init_vels,
    input_iterations: int,
    env_seeds,
    goal_set,
    planner_margin_sets,
    init_vel_sets,
    start_states,
):
    """
    Append all (start × goal × planner_margin × init_vel) combos for each iteration/seed
    into the provided lists and return them.
    """
    assert len(env_seeds) == input_iterations
    for i in range(input_iterations):
        s = env_seeds[i]
        for start in start_states:
            for g in goal_set:
                for m in planner_margin_sets:
                    for v in init_vel_sets:
                        goals.append(g)
                        starts.append(start)
                        seeds.append(s)
                        planner_margins.append(m)
                        init_vels.append(v)
    return goals, starts, seeds, planner_margins, init_vels


def main_mp(
    input_iterations: int = 2, show_plot: bool = False, pin: bool = False, use_mp=False, 
    base_seed: Optional[int] = None, forest_type = 2) -> None:
    """
    Generate `iterations` random forests in parallel (batched).
      - Max concurrent workers: 8 (capped)
      - Seeds: random 32-bit ints (deterministic if base_seed is provided)
      - Goal:  (15, 0)
      - Filenames: forest_{idx:03d}_s{seed}
      - Pinning: always on; round-robin cores 0..(batch_size-1)
      - Prints per-run elapsed times at the end
    """
    # If provided, set global RNG seeds so generated per-run seeds are reproducible
    if base_seed is not None:
        random.seed(base_seed)
        np.random.seed(base_seed)

    # generate environments seeds 
    env_seeds = []
    for i in range(input_iterations):
        s = random.getrandbits(32)
        env_seeds.append(s)

    goals = []
    starts = []
    seeds = []
    planner_margins = []
    init_vels = []

    start_states = [(0, 0, 0)]
    goal_set = [(15, 0, 0), (15, -3, 0), (15, 3, 0)]
    planner_margin_sets = [0.5]
    init_vel_sets = [(0, 0, 0), (1, 0, 0)]
    goals, starts, seeds, planner_margins, init_vels = append_job_lists(
        goals, starts, seeds, planner_margins, init_vels,
        input_iterations,
        env_seeds,
        goal_set,
        planner_margin_sets,
        init_vel_sets,
        start_states,
    )

    start_states = [(0, 0, 0)]
    goal_set = [(15, 0, 0), (15, -3, 0), (15, 3, 0)]
    planner_margin_sets = [0.6]
    init_vel_sets = [(0, 0, 0)]
    goals, starts, seeds, planner_margins, init_vels = append_job_lists(
        goals, starts, seeds, planner_margins, init_vels,
        input_iterations,
        env_seeds,
        goal_set,
        planner_margin_sets,
        init_vel_sets,
        start_states,
    )
    iterations = len(goals)

    print("========================================\n")
    print("Total number of runs:", iterations)
    print("========================================\n")

    # for debugging
    if False:
        seeds = []
        for i in range(input_iterations):
            seeds.append(2415427529)
            seeds.append(2415427529)
            seeds.append(2415427529)

    assert len(seeds) == len(goals)

    # goals = [15, 0, 0] * len(seeds)
    # Choose batch size: <= min(iterations, MAX_PROCS, os.cpu_count)
    # TODO Each process should loop through N jobs
    # or maintain a global set of seeds, start, pos and each thread pull from that once its done
    cpu_total = os.cpu_count() or 1
    batch_size = min(len(seeds), MAX_PROCS, cpu_total)
    t0_all = time.time()
    results_list = []  # local copy for summaries

    if use_mp:
        mp.set_start_method("spawn", force=True)

        # Create manager-backed shared lists and locks
        manager = mp.Manager()
        optimizations_to_run = manager.list()
        results = manager.list()
        opt_lock = manager.Lock()
        res_lock = manager.Lock()

        # Fill the global work list with all unique (goal,start,seed,idx) combos
        for idx, (g, st, s) in enumerate(zip(goals, starts, seeds)):
            optimizations_to_run.append({
                "goal": g,
                "start": st,
                "seed": s,
                "idx": idx,
                "planner_margin": planner_margins[idx],
                "init_vel": init_vels[idx],
            })

        # Spawn batch_size worker processes
        procs = []
        for wid in range(batch_size):
            p = mp.Process(
                target=worker_loop,
                args=(wid, show_plot, pin, optimizations_to_run, results, opt_lock, res_lock, forest_type),
            )
            p.start()
            procs.append(p)

        # Wait for all workers to finish
        for p in procs:
            p.join()

        # Snapshot manager results into a plain list for downstream summaries
        results_list = list(results)

    else:
        batches_total = len(seeds)  # 1 run per "batch"
        for idx in range(len(goals)):
            g = goals[idx]
            st = starts[idx]
            s = seeds[idx]
            pm = planner_margins[idx]
            iv = init_vels[idx]
            core_id = 0 if pin else None  # harmless in single-process mode
            b0 = time.time()
            results_list.append(run_one_seed(g, st, s, idx, pm, iv, show_plot, core_id, forest_type))
            bdt = time.time() - b0

            collected = len(results_list)
            batch_ok = 1
            batch_idx = collected
            succ_total = sum(1 for r in results_list if r[-1])
            print_batch_status(
                batch_idx, batches_total, bdt, batch_ok, collected, succ_total, iterations, t0_all
            )
    total_dt = time.time() - t0_all

    # ------------------------------------------------------------------
    # Summary prints
    # ------------------------------------------------------------------
    print(
        f"[generate_forest.main] Completed {iterations} runs in {total_dt:.2f}s "
        f"(batch size={batch_size}, pinning={bool(pin)})."
    )

    # Sort by idx to get stable ordering
    results_list.sort(key=lambda r: r[2])

    print("\n[Per-run timings]")
    for fname, dt, idx, seed, core_id, success in results_list:
        status = "OK" if success else "FAIL"
        print(f"  {idx:03d}  {fname}  seed={seed}  core={core_id}  time={dt:.2f}s  status={status}")

    # Success summary
    succ_count = sum(1 for r in results_list if r[-1])
    if iterations > 0:
        rate = 100.0 * succ_count / iterations
        print(f"\n[Success] {succ_count}/{iterations} trajectories succeeded ({rate:.1f}%).")

    # Optional quick stats
    if results_list:
        total_single = sum(r[1] for r in results_list)
        avg = total_single / len(results_list)
        print(f"[Timing stats] sum(single)={total_single:.2f}s  avg={avg:.2f}s/run")

def main_mp_deprecated(
    input_iterations: int = 2, show_plot: bool = False, pin: bool = False, use_mp=False, base_seed: Optional[int] = None, forest_type: int = 2
) -> None:
    """
    Generate `iterations` random forests in parallel (batched).
      - Max concurrent workers: 8 (capped)
      - Seeds: random 32-bit ints (deterministic if base_seed is provided)
      - Goal:  (15, 0)
      - Filenames: forest_{idx:03d}_s{seed}
      - Pinning: always on; round-robin cores 0..(batch_size-1)
      - Prints per-run elapsed times at the end
    """
    # If provided, set global RNG seeds so generated per-run seeds are reproducible
    if base_seed is not None:
        random.seed(base_seed)
        np.random.seed(base_seed)

    # Define start states (2 per seed)
    start_states = [
        (0, 0, 0),
        # (0, -1, 0),
    ]

    # Define goal states (3 per seed, as before)
    goal_set = [(15, 0, 0), (15, -2, 0), (15, 2, 0)]

    # Build flattened lists for all (start × goal) combos per seed
    goals = []
    starts = []
    seeds = []
    for i in range(input_iterations):
        s = random.getrandbits(32) if base_seed is None else random.getrandbits(32)
        for start in start_states:
            for g in goal_set:
                goals.append(g)
                starts.append(start)
                seeds.append(s)

    iterations = len(goals)

    # for debugging
    if False:
        seeds = []
        for i in range(input_iterations):
            seeds.append(2415427529)
            seeds.append(2415427529)
            seeds.append(2415427529)

    assert len(seeds) == len(goals)

    # goals = [15, 0, 0] * len(seeds)
    # Choose batch size: <= min(iterations, MAX_PROCS, cpu_total)
    cpu_total = os.cpu_count() or 1
    batch_size = min(len(seeds), MAX_PROCS, cpu_total)
    t0_all = time.time()
    results = []  # collect (fname, dt, idx, seed, core_id, success)

    if use_mp:
        mp.set_start_method("spawn", force=True)
        batches_total = (len(seeds) + batch_size - 1) // batch_size

        # Process in batches of `batch_size`
        for batch_start in range(0, len(seeds), batch_size):
            batch_seeds = seeds[batch_start : batch_start + batch_size]
            batch_goals = goals[batch_start : batch_start + batch_size]
            batch_starts = starts[batch_start : batch_start + batch_size]

            # Build jobs; pinning to cores 0..(batch_len-1)
            jobs = []
            for j, s in enumerate(batch_seeds):
                idx = batch_start + j
                core_id = j if pin else None
                jobs.append((batch_goals[j], batch_starts[j], s, idx, show_plot, core_id, forest_type))
                print(f"goal: {batch_goals[j]}, start: {batch_starts[j]}, seed {s}")

            b0 = time.time()
            with mp.Pool(processes=len(batch_seeds)) as pool:
                batch_results = pool.starmap(run_one_seed, jobs)
            bdt = time.time() - b0

            results.extend(batch_results)
            collected = len(results)
            batch_ok = len(batch_results)
            batch_idx = (batch_start // batch_size) + 1
            succ_total = sum(1 for r in results if r[-1])
            print_batch_status(
                batch_idx, batches_total, bdt, batch_ok, collected, succ_total, iterations, t0_all
            )
    else:
        batches_total = len(seeds)  # 1 run per "batch"
        for idx, (g, st, s) in enumerate(zip(goals, starts, seeds)):
            core_id = 0 if pin else None  # harmless in single-process mode
            b0 = time.time()
            results.append(run_one_seed(g, st, s, idx, show_plot, core_id, forest_type))
            bdt = time.time() - b0

            collected = len(results)
            batch_ok = 1
            batch_idx = collected
            # compute successes so far
            succ_total = sum(1 for r in results if r[-1])
            print_batch_status(
                batch_idx, batches_total, bdt, batch_ok, collected, succ_total, iterations, t0_all
            )
    total_dt = time.time() - t0_all

    # ------------------------------------------------------------------
    # Summary prints
    # ------------------------------------------------------------------
    print(
        f"[generate_forest.main] Completed {iterations} runs in {total_dt:.2f}s "
        f"(batch size={batch_size}, pinning={bool(pin)})."
    )

    # Sort by idx to get stable ordering
    results.sort(key=lambda r: r[2])

    print("\n[Per-run timings]")
    for fname, dt, idx, seed, core_id, success in results:
        status = "OK" if success else "FAIL"
        print(f"  {idx:03d}  {fname}  seed={seed}  core={core_id}  time={dt:.2f}s  status={status}")

    # Success summary
    succ_count = sum(1 for r in results if r[-1])
    if iterations > 0:
        rate = 100.0 * succ_count / iterations
        print(f"\n[Success] {succ_count}/{iterations} trajectories succeeded ({rate:.1f}%).")

    # Optional quick stats
    if results:
        total_single = sum(r[1] for r in results)
        avg = total_single / len(results)
        print(f"[Timing stats] sum(single)={total_single:.2f}s  avg={avg:.2f}s/run")


def print_args_summary(args) -> None:
    cpu_total = os.cpu_count() or 1
    if args.use_mp:
        batch_size = min(MAX_PROCS, cpu_total)
        mode = f"multiprocessing (batch_size={batch_size}, pinning={'on' if args.pin else 'off'})"
    else:
        batch_size = 1
        mode = "sequential"

    print("\n=== Run configuration ===")
    print(f"seeds (iterations): {args.iterations}")
    print(f"mode              : {mode}")
    print(f"forest_type       : {args.forest_type}")
    print(f"plotting          : {'on' if args.plot else 'off'}")
    print(f"system cores      : {cpu_total}  | MAX_PROCS cap: {MAX_PROCS}")
    print(f"POLYFLY_DIR       : {os.environ.get('POLYFLY_DIR', '<unset>')}")
    print(f"base RNG seed     : {args.seed if args.seed is not None else '<random>'}")
    print("========================================\n", flush=True)


def print_batch_status(
    batch_idx: int,
    batches_total: int,
    batch_dt: float,
    batch_ok: int,
    collected: int,
    succ_total: int,
    total: int,
    t0_all: float,
) -> None:
    pct = 100.0 * (collected / total) if total else 0.0
    succ_pct = 100.0 * (succ_total / collected) if collected else 0.0
    elapsed = time.time() - t0_all
    rate = collected / elapsed if elapsed > 0 else 0.0
    print(
        (
            f"[batch {batch_idx}/{batches_total}]  +{batch_ok} ok in {batch_dt:.2f}s  "
            f"| collected {collected}/{total} ({pct:.1f}%)  "
            f"| success {succ_total}/{collected} ({succ_pct:.1f}%)  "
            f"| avg throughput {rate:.2f} traj/s"
        ),
        flush=True,
    )


if __name__ == "__main__":
    #  Example usage 
    # python generate_forest.py -n 5 --mp

    mp.freeze_support()  # safe no-op on Linux; helpful on Windows/CI

    ap = argparse.ArgumentParser(
        description="Generate forests and optimize trajectories for 3 goals per seed."
    )
    ap.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=2,
        help="number of seeds to generate (each seed runs goals: (15,0), (15,-2), (15,2))",
    )
    ap.add_argument("--plot", action="store_true", help="enable plotting of solutions (slower)")
    # Choose multiprocessing or sequential (default: sequential)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--mp", dest="use_mp", action="store_true", help="use multiprocessing with batching"
    )
    mode.add_argument(
        "--sequential",
        dest="use_mp",
        action="store_false",
        help="run sequentially in a single process (default)",
    )
    ap.set_defaults(use_mp=False)

    ap.add_argument(
        "--pin",
        action="store_true",
        help="pin workers to CPU cores 0..(batch_size-1) when using --mp",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=2,
        help="base RNG seed for reproducible forest generation (default: random)",
    )
    ap.add_argument(
        "--forest-type",
        type=int,
        default=2,
        help="Defines the type of forest",
    )

    args = ap.parse_args()
    print_args_summary(args)

    main_mp(
        input_iterations=args.iterations,
        show_plot=args.plot,
        pin=args.pin,
        use_mp=args.use_mp,
        base_seed=args.seed,
        forest_type=args.forest_type
    )

