
#!/usr/bin/env python3
"""
Trajectory tracker for a 3rd-order integrator robot.

Pipeline (per the user's spec):
1) Initialize the system in the start state = first state from the loaded trajectory.
2) For time i = 0 ... N-2:
   - desired_next = desired_states[i+1]
   - cmd = controller.generate_control(present_state, desired_next)
   - next_state = simulate_robot(present_state, cmd, dt_i)     # RK4 on 3rd-order integrator
   - present_state = next_state
3) Stops when the final desired step is reached.

CSV expectations:
- Produced by the planner's `save_result(...)`, but we parse robustly.
- Must contain a "time" column.
- Must contain columns named "sol_x_0" ... "sol_x_8" (9 states).
- Extra columns (e.g., "sol_u_*") are ignored.

Usage:
    python trajectory_tracker.py --csv /path/to/your/traj.csv
    # With the built-in PD controller (jerk = Kp*pos_err + Kv*vel_err + Ka*acc_err):
    python trajectory_tracker.py --csv /path/to/your/traj.csv --use-demo-pd --kp 10 --kv 6 --ka 2

Library:
    from trajectory_tracker import load_trajectory, simulate_robot, track_trajectory, PDJerkController
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Low-level dynamics helpers (mirror planner.fdot & planner.dynamics_rk45)
# State x = [x, y, z, xdot, ydot, zdot, xddot, yddot, zddot]  (9,)
# Input u = [jx, jy, jz]  (jerk)  (3,)
# ──────────────────────────────────────────────────────────────────────────────

N_STATES = 9
N_INPUTS = 3


def _fdot(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Continuous-time RHS for the 3rd-order integrator (matches planner.fdot).

    x_dot[0:3] = v
    x_dot[3:6] = a
    x_dot[6:9] = j (the control input)
    """
    x = np.asarray(x).reshape(N_STATES,)
    u = np.asarray(u).reshape(N_INPUTS,)

    x_dot = np.zeros_like(x)
    x_dot[0:3] = x[3:6]
    x_dot[3:6] = x[6:9]
    x_dot[6:9] = u
    return x_dot


def simulate_robot(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """
    One RK4 step of the 3rd-order integrator, matching planner.dynamics_rk45 logic.
    """
    x = np.asarray(x).reshape(N_STATES,)
    u = np.asarray(u).reshape(N_INPUTS,)

    k1 = _fdot(x, u)
    k2 = _fdot(x + 0.5 * dt * k1, u)
    k3 = _fdot(x + 0.5 * dt * k2, u)
    k4 = _fdot(x + dt * k3, u)
    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x_next


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory loading
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LoadedTrajectory:
    time: np.ndarray           # shape (N,)
    desired_states: np.ndarray # shape (N, 9)
    desired_inputs: Optional[np.ndarray] = None  # shape (N, 3) or (N-1, 3)


def load_trajectory(csv_path: str) -> LoadedTrajectory:
    """
    Load a trajectory CSV and return time, desired states, and (optionally) desired inputs.
    Required: "time", "sol_x_0..8"
    Optional: "sol_u_0..2"
    """
    # Read header to find column indices
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)

    # Map column names → indices
    name_to_idx = {name.strip(): i for i, name in enumerate(header)}

    # Required columns
    if 'time' not in name_to_idx:
        raise ValueError("CSV missing required 'time' column.")

    x_cols = []
    for k in range(9):
        key = f"sol_x_{k}"
        if key not in name_to_idx:
            raise ValueError(f"CSV missing required state column '{key}'.")
        x_cols.append(name_to_idx[key])

    # Optional input columns
    u_cols = []
    have_all_u = True
    for k in range(3):
        key = f"sol_u_{k}"
        if key not in name_to_idx:
            have_all_u = False
            break
        u_cols.append(name_to_idx[key])

    # Load full numeric table (skip header)
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)

    # If file has just a single row, np.loadtxt returns 1D; promote to 2D.
    if data.ndim == 1:
        data = data.reshape(1, -1)

    time = data[:, name_to_idx['time']].astype(float)
    desired_states = data[:, x_cols].astype(float)  # (N, 9)

    desired_inputs = None
    if have_all_u:
        desired_inputs = data[:, u_cols].astype(float)  # (N, 3)

    return LoadedTrajectory(time=time, desired_states=desired_states, desired_inputs=desired_inputs)


# ──────────────────────────────────────────────────────────────────────────────
# Controller interface + a simple PD jerk example
# ──────────────────────────────────────────────────────────────────────────────

class ControllerInterface:
    """Any controller must implement `generate_control(present_state, next_state)` → u (3,)"""
    def generate_control(self, present_state: np.ndarray, next_state: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class PDJerkController(ControllerInterface):
    """
    Very simple preview controller for a 3rd-order integrator.
    Computes jerk directly from P-D-A error (pos/vel/acc).

        u = Kp*(xdes - x) + Kv*(vdes - v) + Ka*(ades - a)

    Gains may be tuned depending on your dt and trajectory smoothness.
    """
    Kp: float = 10.0
    Kv: float = 6.0
    Ka: float = 2.0

    def generate_control(self, present_state: np.ndarray, next_state: np.ndarray) -> np.ndarray:
        x = np.asarray(present_state).reshape(N_STATES,)
        xd = np.asarray(next_state).reshape(N_STATES,)

        pos_err = xd[0:3] - x[0:3]
        vel_err = xd[3:6] - x[3:6]
        acc_err = xd[6:9] - x[6:9]

        # jerk command
        u = self.Kp * pos_err + self.Kv * vel_err + self.Ka * acc_err
        return u


# ──────────────────────────────────────────────────────────────────────────────
# Tracking loop
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrackResult:
    tracked_states: np.ndarray  # \(N, 9\)
    commands: np.ndarray        # \(N-1, 3\)


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────
def _ensure_matplotlib():
    import matplotlib  # noqa: F401
    try:
        # In case user runs headless
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def visualize_3d(traj: LoadedTrajectory, result: TrackResult, title: str = "", savepath: Optional[str] = None):
    """
    One chart per figure (no subplots). 3D desired vs tracked position (x,y,z).
    """
    plt = _ensure_matplotlib()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xd = traj.desired_states
    xt = result.tracked_states

    ax.plot(xd[:, 0], xd[:, 1], xd[:, 2], label="desired")
    ax.plot(xt[:, 0], xt[:, 1], xt[:, 2], label="tracked")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title:
        ax.set_title(title)
    ax.legend()

    if savepath:
        fig.savefig(savepath, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def visualize_position_time(traj: LoadedTrajectory, result: TrackResult, title: str = "", savepath: Optional[str] = None):
    """
    One chart per figure (no subplots). Plot x(t), y(t), z(t) for desired & tracked on a single axes.
    """
    plt = _ensure_matplotlib()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    t = traj.time
    xd = traj.desired_states[:, 0:3]
    xt = result.tracked_states[:, 0:3]

    # Desired
    ax.plot(t, xd[:, 0], label="x_des")
    ax.plot(t, xd[:, 1], label="y_des")
    ax.plot(t, xd[:, 2], label="z_des")
    # Tracked
    ax.plot(t, xt[:, 0], label="x_trk")
    ax.plot(t, xt[:, 1], label="y_trk")
    ax.plot(t, xt[:, 2], label="z_trk")

    ax.set_xlabel("time")
    ax.set_ylabel("position")
    if title:
        ax.set_title(title)
    ax.legend(ncol=2)

    if savepath:
        fig.savefig(savepath, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()




def track_trajectory_with_csv_inputs(
    traj: LoadedTrajectory,
    init_state: Optional[np.ndarray] = None,
) -> TrackResult:
    """
    Debug mode: use the CSV-provided inputs as the command at each i (jerk at time i).
    Assumes inputs correspond to transitions i -> i+1.
    - Accepts desired_inputs of shape (N,3) or (N-1,3). Requires at least N-1 rows.
    """
    if traj.desired_inputs is None:
        raise ValueError("CSV debug mode requested but CSV does not contain sol_u_0..2 columns.")

    time = traj.time
    xdes = traj.desired_states
    U = np.asarray(traj.desired_inputs)
    N = xdes.shape[0]

    if U.shape[0] < N - 1 or U.shape[1] != 3:
        raise ValueError(f"CSV inputs shape invalid: got {U.shape}, need at least {(N-1, 3)}.")

    # Initialize present state
    x = np.copy(xdes[0] if init_state is None else np.asarray(init_state).reshape(N_STATES,))

    tracked = np.zeros((N, N_STATES), dtype=float)
    tracked[0, :] = x
    cmds = np.zeros((N-1, N_INPUTS), dtype=float)

    for i in range(N - 1):
        dt = float(time[i + 1] - time[i])
        if dt <= 0:
            raise ValueError(f"Non-positive dt detected between rows {i} and {i+1}: dt={dt}")

        u = U[i, :]  # use input at time i
        x = simulate_robot(x, u, dt)

        cmds[i, :] = u
        tracked[i + 1, :] = x

    return TrackResult(tracked_states=tracked, commands=cmds)


def track_trajectory(
    traj: LoadedTrajectory,
    controller: ControllerInterface,
    init_state: Optional[np.ndarray] = None,
) -> TrackResult:
    """
    Run the tracking loop over the loaded trajectory.
    - If `init_state` is None, use the first desired state as the initial condition.
    - dt is taken from successive time differences.
    """
    time = traj.time
    xdes = traj.desired_states  # (N, 9)
    N = xdes.shape[0]
    if N < 2:
        raise ValueError("Trajectory must contain at least two rows.")

    # Initialize present state
    x = np.copy(xdes[0] if init_state is None else np.asarray(init_state).reshape(N_STATES,))

    tracked = np.zeros((N, N_STATES), dtype=float)
    tracked[0, :] = x
    cmds = np.zeros((N-1, N_INPUTS), dtype=float)

    for i in range(N - 1):
        desired_next = xdes[i + 1]
        dt = float(time[i + 1] - time[i])
        if dt <= 0:
            raise ValueError(f"Non-positive dt detected between rows {i} and {i+1}: dt={dt}")

        u = controller.generate_control(x, desired_next)  # per user-specified interface
        x = simulate_robot(x, u, dt)

        cmds[i, :] = u
        tracked[i + 1, :] = x

    return TrackResult(tracked_states=tracked, commands=cmds)



def compare_both_modes(csv_path: str, kp: float = 10.0, kv: float = 6.0, ka: float = 2.0, outdir: Optional[str] = None, show: bool = False):
    """
    Convenience: run PD tracking and (if available) CSV-input tracking and save figures.
    Returns a dict with keys present among {"pd", "csv"} mapping to TrackResult.
    """
    import os
    traj = load_trajectory(csv_path)
    bname = os.path.splitext(os.path.basename(csv_path))[0]
    if outdir is None:
        outdir = os.path.dirname(csv_path) or "."

    results = {}

    # PD mode
    controller: ControllerInterface = PDJerkController(Kp=kp, Kv=kv, Ka=ka)
    res_pd = track_trajectory(traj, controller)
    results["pd"] = res_pd
    pd_3d_path = os.path.join(outdir, f"{bname}_pd_3d.png")
    pd_pos_path = os.path.join(outdir, f"{bname}_pd_pos_time.png")
    visualize_3d(traj, res_pd, title=f"{bname} — PD tracking (3D)", savepath=pd_3d_path)
    visualize_position_time(traj, res_pd, title=f"{bname} — PD tracking (pos vs time)", savepath=pd_pos_path)
    if show:
        visualize_3d(traj, res_pd, title=f"{bname} — PD tracking (3D)")
        visualize_position_time(traj, res_pd, title=f"{bname} — PD tracking (pos vs time)")

    # CSV-input mode, if inputs present
    if traj.desired_inputs is not None and traj.desired_inputs.shape[0] >= traj.desired_states.shape[0] - 1:
        res_csv = track_trajectory_with_csv_inputs(traj)
        results["csv"] = res_csv
        csv_3d_path = os.path.join(outdir, f"{bname}_csv_3d.png")
        csv_pos_path = os.path.join(outdir, f"{bname}_csv_pos_time.png")
        visualize_3d(traj, res_csv, title=f"{bname} — CSV-input tracking (3D)", savepath=csv_3d_path)
        visualize_position_time(traj, res_csv, title=f"{bname} — CSV-input tracking (pos vs time)", savepath=csv_pos_path)
        if show:
            visualize_3d(traj, res_csv, title=f"{bname} — CSV-input tracking (3D)")
            visualize_position_time(traj, res_csv, title=f"{bname} — CSV-input tracking (pos vs time)")
    return results



def compare_csv_inputs_vs_positions(csv_path: str, outdir: Optional[str] = None, show: bool = False):
    """
    Compare:
      (1) The trajectory obtained by forward-simulating from the first state using the CSV inputs at time i, and
      (2) The positions directly specified in the CSV (desired positions).
    Always saves two figures (3D path and position-vs-time). Optionally also shows them.
    """
    import os
    traj = load_trajectory(csv_path)
    if traj.desired_inputs is None or traj.desired_inputs.shape[0] < traj.desired_states.shape[0] - 1:
        raise ValueError("CSV does not contain valid 'sol_u_0..2' inputs for each step (need at least N-1 rows).")

    res_cmd = track_trajectory_with_csv_inputs(traj)

    bname = os.path.splitext(os.path.basename(csv_path))[0]
    if outdir is None:
        outdir = os.path.dirname(csv_path) or "."

    # File paths
    out_3d  = os.path.join(outdir, f"{bname}_csvcmd_vs_positions_3d.png")
    out_pos = os.path.join(outdir, f"{bname}_csvcmd_vs_positions_pos_time.png")

    # Save figures
    visualize_3d(traj, res_cmd, title=f"{bname} — Sim from CSV inputs vs CSV positions (3D)", savepath=out_3d)
    visualize_position_time(traj, res_cmd, title=f"{bname} — Sim from CSV inputs vs CSV positions (pos vs time)", savepath=out_pos)

    # Optionally display
    if show:
        visualize_3d(traj, res_cmd, title=f"{bname} — Sim from CSV inputs vs CSV positions (3D)")
        visualize_position_time(traj, res_cmd, title=f"{bname} — Sim from CSV inputs vs CSV positions (pos vs time)")

    return {"result_csv_inputs": res_cmd, "out_3d": out_3d, "out_pos": out_pos}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Track a CSV trajectory with a 3rd-order integrator robot.")
    p.add_argument("--csv", required=True, help="Path to trajectory CSV (with 'time' + 'sol_x_0..8' and optionally 'sol_u_0..2').")
    mode = p.add_mutually_exclusive_group(required=False)
    mode.add_argument("--use-demo-pd", action="store_true",
                      help="Use built-in PD jerk controller (u=Kp*e_pos + Kv*e_vel + Ka*e_acc).")
    mode.add_argument("--use-csv-inputs", action="store_true",
                      help="DEBUG: Use 'sol_u_0..2' from the CSV as the command at time i.")
    p.add_argument("--kp", type=float, default=10.0, help="Kp for demo PD controller.")
    p.add_argument("--kv", type=float, default=6.0, help="Kv for demo PD controller.")
    p.add_argument("--ka", type=float, default=2.0, help="Ka for demo PD controller.")
    p.add_argument("--plot", action="store_true", help="Save plots alongside the CSV (3D and pos-vs-time).")
    p.add_argument("--compare-both", action="store_true",
                   help="Run PD tracking and CSV-input tracking (if inputs exist) and save plots for both.")
    p.add_argument("--compare-csv-inputs-vs-positions", action="store_true",
                   help="Compare sim trajectory from CSV inputs against positions in the CSV; saves plots.")
    p.add_argument("--show", action="store_true", help="Display figures in a window (if a display is available).")
    return p




def main():
    args = _build_argparser().parse_args()

    # If user asked to compare both, do it up front and exit.
    if args.compare_both:
        compare_both_modes(args.csv, kp=args.kp, kv=args.kv, ka=args.ka, show=args.show)
        print("Completed compare-both run. Figures saved next to the CSV.")
        return

    # If user asked to compare csv inputs vs csv positions, do it and exit.
    if args.compare_csv_inputs_vs_positions:
        info = compare_csv_inputs_vs_positions(args.csv, show=args.show)
        print(f"Saved: {info['out_3d']}{info['out_pos']}")
        return

    # Otherwise, ensure a single mode was selected.
    if not (args.use_csv_inputs or args.use_demo_pd):
        raise SystemExit("Choose a mode: --use-csv-inputs or --use-demo-pd; or use --compare-csv-inputs-vs-positions / --compare-both.")

    traj = load_trajectory(args.csv)

    # Single-mode run
    if args.use_csv_inputs:
        result = track_trajectory_with_csv_inputs(traj)
        mode = "CSV-inputs"
    else:
        controller: ControllerInterface = PDJerkController(Kp=args.kp, Kv=args.kv, Ka=args.ka)
        result = track_trajectory(traj, controller)
        mode = "PD"

    # Print quick summary
    print(f"Loaded {traj.desired_states.shape[0]} desired states.")
    print(f"Tracked states shape: {result.tracked_states.shape}, commands shape: {result.commands.shape}")

    # Plot/save as requested
    import os
    bname = os.path.splitext(os.path.basename(args.csv))[0]
    outdir = os.path.dirname(args.csv) or "."

    if args.plot:
        out1 = os.path.join(outdir, f"{bname}_{mode.lower()}_3d.png")
        out2 = os.path.join(outdir, f"{bname}_{mode.lower()}_pos_time.png")
        visualize_3d(traj, result, title=f"{bname} — {mode} tracking (3D)", savepath=out1)
        visualize_position_time(traj, result, title=f"{bname} — {mode} tracking (pos vs time)", savepath=out2)
        print(f"Saved: {out1}{out2}")

    if args.show:
        visualize_3d(traj, result, title=f"{bname} — {mode} tracking (3D)")
        visualize_position_time(traj, result, title=f"{bname} — {mode} tracking (pos vs time)")

if __name__ == "__main__":
    main()