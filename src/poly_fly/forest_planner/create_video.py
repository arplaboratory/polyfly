#!/usr/bin/env python3
"""
create_video.py

Create a time-consistent video (MP4 or GIF fallback) from saved CSV trajectories
exported by poly_fly.optimal_planner.planner.save_result.

Also supports batch processing and a multi-CSV inspect mode:
- Process a whole folder of CSVs:            --csvs_folder forests/    (alias: --csv_folder)
- Inspect each CSV one by one:               --csvs_folder forests/ --inspect
- One viewer with CSV switcher + step slider:
                                             --csvs_folder forests/ --inspect --combined
  (The 'csv_selection' slider switches the entire scene, including obstacles,
   payload/quad geometry, and axis limits, to the selected trajectory.)

Examples
--------
# Default (auto-pick newest CSV in csvs/, write MP4 if ffmpeg is available)
python create_video.py

# Single CSV
python create_video.py --csv csvs/autotrans/maze_1.csv

# Batch over a folder
python create_video.py --csvs_folder forests/

# Batch + inspect each (opens a viewer per CSV)
python create_video.py --csvs_folder forests/ --inspect

# One viewer with CSV switcher + step scrubbing
python create_video.py --csv_folder forests/ --inspect --combined

# Control playback & format
python create_video.py --fps 60 --speed 1.0 [--gif] [--ortho] [--no-video]
"""

from __future__ import annotations
import argparse
import torch
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

from poly_fly.optimal_planner.polytopes import Obs, SquarePayload, Quadrotor
import poly_fly.utils.plot as plotter
from poly_fly.optimal_planner.planner import Planner
from poly_fly.data_io.utils import ZARR_DIR, DEPTH_SCALE_FACTOR
from poly_fly.data_io.zarr_io import load_zarr_folder
from poly_fly.deep_poly_fly.model.policy import load_model_from_checkpoint
from poly_fly.data_io.utils import load_normalization_stats, process_rotation_matrix
from poly_fly.data_io.enums import DatasetKeys as DK

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

def draw_obstacles(ax, params, obstacle_color='#708090', obstacle_alpha=0.2):
    """Face-by-face obstacle rendering like plot.plot_result."""
    collections = []
    if not hasattr(params, "obstacles"):
        return collections
    for ob in params.obstacles.values():
        box = Obs(ob["x"], ob["y"], ob["z"], ob["l"], ob["b"], ob["h"])
        verts = box.get_vertices()
        hull = ConvexHull(verts)
        for simplex in hull.simplices:
            face = [verts[i] for i in simplex]
            poly = Poly3DCollection([face], alpha=obstacle_alpha, facecolors=obstacle_color)
            collections.append(poly)
    return collections


def set_axes_limits(ax, params, sol_x):
    # Use state bounds if present, otherwise derive from data + padding
    try:
        xlim = (params.state_min[0], params.state_max[0])
        ylim = (params.state_min[1], params.state_max[1])
        zmin = min(params.state_min[2], float(sol_x[:, 2].min()) - 0.25)
        zmax = max(params.state_max[2], float(sol_x[:, 2].max()) + 0.25)
        zlim = (zmin, zmax)
    except Exception:
        pad = 0.5
        xlim = (float(sol_x[:, 0].min()) - pad, float(sol_x[:, 0].max()) + pad)
        ylim = (float(sol_x[:, 1].min()) - pad, float(sol_x[:, 1].max()) + pad)
        zlim = (float(sol_x[:, 2].min()) - pad, float(sol_x[:, 2].max()) + pad)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    try:
        plotter.set_box_equal(ax)
    except Exception:
        pass
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(False)


def build_dynamic_artists(ax, params):
    """Create (but don't yet place) artists for payload, quadrotor, and cable line."""
    payload_color = "#485000"
    quad_color = "#DE8C2A"
    cable_color = "black"

    payload_shape = SquarePayload(params).get_vertices()
    payload_poly = Poly3DCollection([], alpha=0.8, facecolors=payload_color)

    quad_shape = Quadrotor(params).get_vertices()
    quad_poly = Poly3DCollection([], alpha=0.7, facecolors=quad_color)

    (cable_line,) = ax.plot([], [], [], linewidth=2, color=cable_color)

    ax.add_collection3d(payload_poly)
    ax.add_collection3d(quad_poly)

    return dict(
        payload_poly=payload_poly,
        quad_poly=quad_poly,
        cable_line=cable_line,
        payload_shape=payload_shape,
        quad_shape=quad_shape,
    )


def update_dynamic_artists(artists, params, p_payload, p_quad, acc_payload):
    """Update faces + line for one frame."""
    payload_poly = artists["payload_poly"]
    quad_poly = artists["quad_poly"]
    cable_line = artists["cable_line"]
    payload_shape = artists["payload_shape"]
    quad_shape = artists["quad_shape"]

    # payload faces via convex hull (translated only)
    pv = payload_shape + np.asarray(p_payload)[None, :]
    ph = ConvexHull(pv)
    p_faces = [[pv[i] for i in simplex] for simplex in ph.simplices]
    payload_poly.set_verts(p_faces)

    # quad: rotate then translate; rotation from planner's helper
    rot_mat = Planner.compute_quadrotor_rotation_matrix_no_jrk(
        acc_payload, params, use_robot_rotation=True
    )
    qv = (quad_shape @ rot_mat.T) + np.asarray(p_quad)[None, :]
    qh = ConvexHull(qv)
    q_faces = [[qv[i] for i in simplex] for simplex in qh.simplices]
    quad_poly.set_verts(q_faces)

    # cable as a simple segment (lightweight)
    xs = [p_payload[0], p_quad[0]]
    ys = [p_payload[1], p_quad[1]]
    zs = [p_payload[2], p_quad[2]]
    cable_line.set_data(xs, ys)
    cable_line.set_3d_properties(zs)

def _nearest_time_index(t_array, t_val):
    """Return nearest index in t_array to t_val. If t_array is None or empty, return 0."""
    if t_array is None:
        return 0
    t = np.asarray(t_array, dtype=float)
    if t.size == 0:
        return 0
    return int(np.clip(np.abs(t - float(t_val)).argmin(), 0, t.size - 1))

def _depth_frame_to_image(frame):
    """
    Convert a depth frame (float) to an 8-bit image for display.
    Assumes frame is in [0,1] range; clips and scales to [0,255].
    """
    if frame is None:
        return None
    return (frame / DEPTH_SCALE_FACTOR * 255.0).astype(np.uint8)


# ────────────────────────────────────────────────────────────────────────────────
# Interactive viewers
# ────────────────────────────────────────────────────────────────────────────────

def interactive_multi_inspect(datasets, ortho=False, dpi=200, policy_nn=False, policy_config=None, plot_step_size=5):
    """
    Multi-trajectory viewer with a synced depth viewer.
    - datasets: list of dicts with keys:
        { 'name', 'params', 'time', 'sol_x', 'sol_u', 'quad_pos', 'path',
          optional: 'depth_time', 'depth', 'future_robot_pos', 'future_payload_pos' }
    Added:
      plot_step_size: int > 0. Every plot_step_size time steps, plot the next
        plot_step_size predicted horizon samples (mode 0) aligned on the time axis.
        For time k we plot predicted[k, 0, 0:plot_step_size] at time indices k .. k+plot_step_size-1.
    """
    fig = plt.figure(figsize=(10, 8), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # Depth/quaternion/rotation/velocity window state (persistent)
    depth_fig = None
    depth_ax = None
    depth_im = None
    depth_cbar = None
    quat_ax = None
    rot_ax = None 
    vel_ax = None
    pvel_ax = None

    # UI axes
    ax_csv = plt.axes([0.15, 0.02, 0.70, 0.03], facecolor='lightgoldenrodyellow')
    ax_step = plt.axes([0.15, 0.07, 0.70, 0.03], facecolor='lightgoldenrodyellow')
    ax_play = plt.axes([0.88, 0.07, 0.05, 0.03])
    ax_pause = plt.axes([0.88, 0.03, 0.05, 0.03])

    s_file = Slider(ax_csv, 'file', 0, len(datasets) - 1, valinit=0, valfmt="%0.0f")
    s_step = None
    b_play = Button(ax_play, 'Play')
    b_pause = Button(ax_pause, 'Pause')

    if policy_nn:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy_model = load_model_from_checkpoint(
            "/home/mrunal/Documents/poly_fly/data/zarr/models/forests/policy_ep0100.pth",
            "/home/mrunal/Documents/poly_fly/src/poly_fly/deep_poly_fly/model/config.yaml",
            device,
        )
    else:
        policy_model = None

    norm = None
    if policy_model is not None:
        use_rotation_mat = policy_model.use_rotation_mat
        dict_mean, dict_std = load_normalization_stats()
        keys = [
            DK.PAYLOAD_VEL,
            DK.ROBOT_VEL,
            DK.ROBOT_GOAL_RELATIVE,
            DK.ROT_MAT if use_rotation_mat else DK.ROBOT_QUAT,

        ]
        state_mean = np.concatenate([np.asarray(dict_mean[k]).ravel() for k in keys]).astype(np.float32)
        state_std  = np.concatenate([np.asarray(dict_std[k]).ravel()  for k in keys]).astype(np.float32)
        depth_mean = float(np.asarray(dict_mean[DK.DEPTH]))
        depth_std = float(np.asarray(dict_std[DK.DEPTH]))

        stats = {
            "fut_pos": DK.FUTURE_ROBOT_POS,
            "fut_quat": DK.FUTURE_QUATERNION,
            "fut_rot_mat": DK.FUTURE_ROT_MAT,
            "fut_payload_pos": DK.FUTURE_PAYLOAD_POS,
            "fut_robot_vel": DK.FUTURE_ROBOT_VEL,
            "fut_payload_vel": DK.FUTURE_PAYLOAD_VEL
        }
        norm = {
            "state_mean": np.asarray(state_mean, dtype=np.float32),
            "state_std":  np.asarray(state_std,  dtype=np.float32),
            "depth_mean": np.asarray(depth_mean, dtype=np.float32),
            "depth_std":  np.asarray(depth_std,  dtype=np.float32),
        }
        for name, key in stats.items():
            norm[f"{name}_mean"] = np.asarray(dict_mean[key], dtype=np.float32)
            norm[f"{name}_std"]  = np.asarray(dict_std[key],  dtype=np.float32)
            
    state = {
        "csv_idx": 0,
        "artists": None,
        "obs_polys": [],
        "playing": False,
        "s_step": None,
        "future_scatter": None,
        "future_payload_scatter": None,
        "predicted_robot_scatters": None,
        "predicted_payload_scatters": None,
    }

    if plot_step_size < 1:
        plot_step_size = 1

    def ensure_depth_window(ds):
        nonlocal depth_fig, depth_ax, depth_im, depth_cbar, quat_ax, rot_ax, vel_ax, pvel_ax

        dtime = ds.get(DK.DEPTH_TIME, ds.get(DK.TIME, None))
        ddata = ds.get(DK.DEPTH, None)
        qdata = ds.get(DK.ROBOT_QUAT, None)
        rotdata = ds.get(DK.ROT_MAT, None)
        vdata = ds.get(DK.ROBOT_VEL, None)
        pvdata = ds.get(DK.PAYLOAD_VEL, None)

        has_depth = ddata is not None and dtime is not None and len(dtime) > 0 and len(ddata) > 0

        if depth_fig is None:
            depth_fig, (depth_ax, quat_ax, rot_ax, vel_ax, pvel_ax) = plt.subplots(1, 5, figsize=(15, 4), dpi=dpi)

        depth_ax.clear()
        quat_ax.clear()
        rot_ax.clear()
        vel_ax.clear()
        pvel_ax.clear()
        depth_im = None
        if depth_cbar is not None:
            try:
                depth_cbar.remove()
            except Exception:
                pass

        # Depth subplot (left)
        if has_depth:
            depth_ax.set_title("Depth")
            depth_im = depth_ax.imshow(_depth_frame_to_image(np.asarray(ddata[0])), cmap='plasma', vmin=0, vmax=255)
            cbar = depth_fig.colorbar(depth_im, ax=depth_ax, fraction=0.046, pad=0.04)
            cbar.set_label("Depth (scaled 0-255)")
            cbar.set_ticks([0, 64, 128, 192, 255])
            depth_cbar = cbar
            depth_ax.axis('off')
        else:
            depth_ax.set_title("No depth available")
            depth_ax.text(0.5, 0.5, "No depth", ha="center", va="center", transform=depth_ax.transAxes)
            depth_ax.axis('off')

        # Quaternion subplot (index 1)
        q = np.asarray(qdata)
        labels = ["x", "y", "z", "w"]
        colors = ["tab:purple", "tab:blue", "tab:orange", "tab:green"]
        for j in range(4):
            quat_ax.plot(q[:, j], label=labels[j], color=colors[j], linewidth=1.2)
            # UPDATED predicted overlays (mode 0) every plot_step_size
            if policy_model is not None and not policy_model.use_rotation_mat:
                pquat = ds.get("predicted_quat_nn", None)
                pquat_arr = np.asarray(pquat)
                assert pquat_arr.ndim == 4 and pquat_arr.shape[-1] == 4, f"shape is {pquat_arr.shape}"
                N = pquat_arr.shape[0]
                for k in range(0, N, plot_step_size):
                    seg_len = min(plot_step_size, N - k)
                    # horizon slice 0:seg_len at time k
                    pred_seg = pquat_arr[k, 0, 0:seg_len, j]
                    t_idx = np.arange(k, k + seg_len)
                    label_pred = f"{labels[j]}_pred" if k == 0 else None
                    quat_ax.plot(
                        t_idx,
                        pred_seg,
                        color=colors[j],
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.6,
                        label=label_pred,
                    )
        quat_ax.set_title("Robot quaternion (all steps)")
        quat_ax.set_xlabel("step")
        quat_ax.set_ylabel("value")
        quat_ax.grid(True, alpha=0.3)
        quat_ax.legend(loc="best", fontsize=8)

        # Rotation 6D subplot (index 2)
        r6 = np.asarray(rotdata)
        labels_6 = [f"r6d_{i+1}" for i in range(6)]
        colors_6 = ["tab:red", "tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown"]
        for j in range(6):
            rot_ax.plot(r6[:, j], label=labels_6[j], color=colors_6[j], linewidth=1.2)
            if policy_model is not None and policy_model.use_rotation_mat:
                prot = ds.get("predicted_rot_mat_nn", None)
                if prot is not None:
                    prot_arr = np.asarray(prot)
                    assert prot_arr.ndim == 4 and prot_arr.shape[-1] == 6
                    N = prot_arr.shape[0]
                    for k in range(0, N, plot_step_size):
                        seg_len = min(plot_step_size, N - k)
                        pred_seg = prot_arr[k, 0, 0:seg_len, j]
                        t_idx = np.arange(k, k + seg_len)
                        label_pred = f"{labels_6[j]}_pred" if k == 0 else None
                        rot_ax.plot(
                            t_idx,
                            pred_seg,
                            color=colors_6[j],
                            linestyle="--",
                            linewidth=1.0,
                            alpha=0.6,
                            label=label_pred,
                        )
        rot_ax.set_title("Rotation (6D) components (all steps)")
        rot_ax.set_xlabel("step")
        rot_ax.set_ylabel("value")
        rot_ax.grid(True, alpha=0.3)
        rot_ax.legend(loc="best", fontsize=8)

        # Robot velocity subplot (index 3)
        if vdata is not None:
            v = np.asarray(vdata)
            vlabels = ["vx", "vy", "vz"]
            vcolors = ["tab:cyan", "tab:pink", "tab:olive"]
            for j in range(3):
                vel_ax.plot(v[:, j], label=vlabels[j], color=vcolors[j], linewidth=1.2)
                if policy_model is not None:
                    pvel = ds.get("predicted_robot_vel_nn", None)
                    if pvel is not None:
                        pvel_arr = np.asarray(pvel)
                        assert pvel_arr.ndim == 4 and pvel_arr.shape[-1] == 3, f"shape is {pvel_arr.shape}"
                        N = pvel_arr.shape[0]
                        for k in range(0, N, plot_step_size):
                            seg_len = min(plot_step_size, N - k)
                            pred_seg = pvel_arr[k, 0, 0:seg_len, j]
                            t_idx = np.arange(k, k + seg_len)
                            label_pred = f"{vlabels[j]}_pred" if k == 0 else None
                            vel_ax.plot(
                                t_idx,
                                pred_seg,
                                color=vcolors[j],
                                linestyle="--",
                                linewidth=1.0,
                                alpha=0.6,
                                label=label_pred,
                            )
            vel_ax.set_title("Robot velocity (all steps)")
            vel_ax.set_xlabel("step")
            vel_ax.set_ylabel("m/s")
            vel_ax.grid(True, alpha=0.3)
            vel_ax.legend(loc="best", fontsize=8)
        else:
            vel_ax.set_title("No velocity available")
            vel_ax.axis('off')

        # Payload velocity subplot (index 4)
        if pvdata is not None:
            pv = np.asarray(pvdata)
            pvlabels = ["pvx", "pvy", "pvz"]
            pvcolors = ["tab:blue", "tab:orange", "tab:green"]
            for j in range(3):
                pvel_ax.plot(pv[:, j], label=pvlabels[j], color=pvcolors[j], linewidth=1.2)
                if policy_model is not None:
                    ppvel = ds.get("predicted_payload_vel_nn", None)
                    if ppvel is not None:
                        ppvel_arr = np.asarray(ppvel)
                        assert ppvel_arr.ndim == 4 and ppvel_arr.shape[-1] == 3, f"shape is {ppvel_arr.shape}"
                        N = ppvel_arr.shape[0]
                        for k in range(0, N, plot_step_size):
                            seg_len = min(plot_step_size, N - k)
                            pred_seg = ppvel_arr[k, 0, 0:seg_len, j]
                            t_idx = np.arange(k, k + seg_len)
                            label_pred = f"{pvlabels[j]}_pred" if k == 0 else None
                            pvel_ax.plot(
                                t_idx,
                                pred_seg,
                                color=pvcolors[j],
                                linestyle="--",
                                linewidth=1.0,
                                alpha=0.6,
                                label=label_pred,
                            )
            pvel_ax.set_title("Payload velocity (all steps)")
            pvel_ax.set_xlabel("step")
            pvel_ax.set_ylabel("m/s")
            pvel_ax.grid(True, alpha=0.3)
            pvel_ax.legend(loc="best", fontsize=8)
        else:
            pvel_ax.set_title("No payload velocity available")
            pvel_ax.axis('off')

        depth_fig.tight_layout()
        depth_fig.canvas.draw_idle()
        return depth_im

    def clear_obstacles():
        for p in state["obs_polys"]:
            try:
                p.remove()
            except ValueError:
                # Raised if the artist is not present in the Axes' container
                pass
        state["obs_polys"] = []

    def clear_artists():
        a = state["artists"]
        if a is not None:
            # Remove main dynamic artists
            for k in ["payload_poly", "quad_poly", "cable_line"]:
                try:
                    artist = a.get(k, None)
                except AttributeError:
                    artist = None
                if artist is not None:
                    try:
                        artist.remove()
                    except ValueError:
                        # Raised if already detached / not in Axes' container
                        pass

        # Remove single scatter artists
        for k in ["future_scatter", "future_payload_scatter"]:
            sc = state.get(k, None)
            if sc is not None:
                try:
                    sc.remove()
                except ValueError:
                    pass
                state[k] = None

        # Remove list-based scatter artists (predicted)
        for k in ["predicted_robot_scatters", "predicted_payload_scatters"]:
            lst = state.get(k, None)
            if lst:
                for sc in lst:
                    try:
                        sc.remove()
                    except ValueError:
                        pass
            state[k] = []

        state["artists"] = None

    def rebuild_step_slider(N):
        nonlocal s_step
        ax_step.cla()
        s = Slider(ax_step, 'step', 0, max(N - 1, 0), valinit=0, valfmt="%0.0f")
        state["s_step"] = s
        return s

    def get_predicted_nn(ds):
        time = ds.get(DK.TIME, ds.get(DK.DEPTH_TIME))
        payload_vel = ds[DK.PAYLOAD_VEL]
        robot_pos = ds[DK.ROBOT_POS]
        robot_vel = ds[DK.ROBOT_VEL]
        robot_quat = ds[DK.ROBOT_QUAT]
        robot_goal_relative = ds[DK.ROBOT_GOAL_RELATIVE] 
        robot_rot_mat = ds[DK.ROT_MAT]

        n_modes = policy_model.n_prediction_modes
        horizon = policy_model.horizon
        use_rotation_mat = policy_model.use_rotation_mat
        assert time.shape[0] == robot_pos.shape[0]
        predicted_robot_pos_nn = np.zeros((robot_pos.shape[0], n_modes, horizon, 3), dtype=np.float32)
        predicted_robot_quat_nn = np.zeros((robot_pos.shape[0], n_modes, horizon, 4), dtype=np.float32)
        predicted_payload_pos_nn = np.zeros((robot_pos.shape[0], n_modes, horizon, 3), dtype=np.float32)
        predicted_rot_mat_nn = np.zeros((robot_pos.shape[0], n_modes, horizon, 6), dtype=np.float32)
        predicted_robot_vel_nn = np.zeros((robot_pos.shape[0], n_modes, horizon, 3), dtype=np.float32)
        predicted_payload_vel_nn = np.zeros((robot_pos.shape[0], n_modes, horizon, 3), dtype=np.float32)

        for i in range(robot_pos.shape[0]):
            depth = ds[DK.DEPTH][i, :, :]
            # depth[:, :] = 0
            robot_vel_i = robot_vel[i, :]
            payload_vel_i = payload_vel[i, :]
            robot_quat_i = robot_quat[i, :]
            robot_rot_mat_i = robot_rot_mat[i, :]
            robot_goal_relative_i = robot_goal_relative[i, :]

            if use_rotation_mat:
                state_vec = np.concatenate([payload_vel_i, robot_vel_i, robot_goal_relative_i, robot_rot_mat_i], axis=0).astype(np.float32)
            else:
                state_vec = np.concatenate([payload_vel_i, robot_vel_i, robot_goal_relative_i, robot_quat_i], axis=0).astype(np.float32)

            state_normalized = (state_vec - norm["state_mean"]) / (norm["state_std"] + 1e-6)
            depth_normalized = (depth - norm["depth_mean"]) / (norm["depth_std"] + 1e-6)
            
            depth_tensor = torch.tensor(depth_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            state_tensor = torch.tensor(state_normalized, dtype=torch.float32).unsqueeze(0)  # (1, D)
            depth_tensor = depth_tensor.to(device)
            state_tensor = state_tensor.to(device)
            
            # set depth to zero 
            # depth_tensor[:, :, :] = 1
            with torch.no_grad():
                predicted_traj = policy_model(state_tensor, depth_tensor)[0]  # first output is predicted_traj
                # predicted_traj: (1, n_modes, horizon, D)
                pred = predicted_traj[0].cpu().numpy().astype(np.float32)  # (n_modes, H, D)

            # Unnormalize only the parts we need using per-horizon stats
            # Defensive check: ensure horizon matches stats shapes
            if norm["fut_pos_mean"].shape[0] != horizon or norm["fut_quat_mean"].shape[0] != horizon:
                raise ValueError(f"Future stats horizon mismatch: stats H_pos={norm['fut_pos_mean'].shape[0]}, "
                                 f"H_quat={norm['fut_quat_mean'].shape[0]}, model H={horizon}")

            # Position: [:, :, :3], Quaternion: [:, :, 3:7]
            pred_pos = pred[:, :, :3]
            pred_payload_pos = pred[:, :, 3:6]
            un_pos = pred_pos * (norm["fut_pos_std"][None, :, :] + 1e-6) + norm["fut_pos_mean"][None, :, :]
            un_payload_pos = pred_payload_pos * (norm["fut_payload_pos_std"][None, :, :] + 1e-6) + norm["fut_payload_pos_mean"][None, :, :]

            if use_rotation_mat:
                pred_rot_mat = pred[:, :, 6:12] # (n_modes, H, 6)
                un_rot_mat = pred_rot_mat * (norm["fut_rot_mat_std"][None, :, :] + 1e-6) + norm["fut_rot_mat_mean"][None, :, :]
                
                un_quat = np.zeros((pred.shape[0], pred.shape[1], 4), dtype=np.float32)
                for j in range(pred.shape[0]):
                    un_quat[j, :, :] = R.from_matrix((process_rotation_matrix(un_rot_mat[j, :, :]))).as_quat()
            else:
                pred_quat = pred[:, :, 6:10]
                un_quat = pred_quat * (norm["fut_quat_std"][None, :, :] + 1e-6) + norm["fut_quat_mean"][None, :, :]

            if use_rotation_mat:
                pred_robot_vel = pred[:, :, 12:15]
                pred_payload_vel = pred[:, :, 15:18]
            else:
                pred_robot_vel = pred[:, :, 10:13]
                pred_payload_vel = pred[:, :, 13:16]
            un_robot_vel = pred_robot_vel * (norm["fut_robot_vel_std"][None, :, :] + 1e-6) + norm["fut_robot_vel_mean"][None, :, :]
            un_payload_vel = pred_payload_vel * (norm["fut_payload_vel_std"][None, :, :] + 1e-6) + norm["fut_payload_vel_mean"][None, :, :]  

            predicted_robot_pos_nn[i, :, :, :] = un_pos
            predicted_robot_quat_nn[i, :, :, :] = un_quat
            predicted_payload_pos_nn[i, :, :, :] = un_payload_pos
            predicted_robot_vel_nn[i, :, :, :] = un_robot_vel
            predicted_payload_vel_nn[i, :, :, :] = un_payload_vel

            if use_rotation_mat:
                predicted_rot_mat_nn[i, :, :, :] = un_rot_mat
        
        # vectorized: world positions = relative + current pose
        predicted_robot_pos_world_nn = predicted_robot_pos_nn + robot_pos[:, None, None, :]

        predicted_payload_pos_world_nn = np.zeros_like(predicted_payload_pos_nn)
        for i in range(robot_pos.shape[0]):
            for j in range(n_modes):
                for k in range(horizon):
                    rot = R.from_quat(predicted_robot_quat_nn[i, j, k, :]).as_matrix()
                    predicted_payload_pos_world_nn[i, j, k, :] = predicted_robot_pos_world_nn[i, j, k, :] + rot @ predicted_payload_pos_nn[i, j, k, :]

        # Convert predicted positions from relative-to-current to world by adding current p
        res = {
            "predicted_robot_pos_nn": predicted_robot_pos_world_nn, #predicted_robot_pos_nn + robot_pos[:, None, None, :],
            "predicted_robot_quat_nn": predicted_robot_quat_nn,
            "predicted_payload_pos_nn": predicted_payload_pos_world_nn, #predicted_robot_pos_nn + robot_pos[:, None, None, :] + predicted_payload_pos_nn,
            "predicted_rot_mat_nn": predicted_rot_mat_nn,
            "predicted_robot_vel_nn": predicted_robot_vel_nn,
            "predicted_payload_vel_nn": predicted_payload_vel_nn
        }
        return res 

    def apply_dataset(k):
        """Rebuild scene for dataset k and (re)attach depth/quaternion window."""
        state["csv_idx"] = k
        ds = datasets[k]
        params = ds.get(DK.PARAMS, None)
        time = ds.get(DK.TIME, ds.get(DK.DEPTH_TIME))
        sol_x = ds.get(DK.SOL_X, None)
        quad = ds.get(DK.ROBOT_POS, None)
        future = ds.get(DK.FUTURE_ROBOT_POS_WORLD)
        future_payload = ds.get(DK.FUTURE_PAYLOAD_POS_WORLD)

        if policy_model is not None:
            result = get_predicted_nn(ds)
            ds.update({
                "predicted_robot_pos_nn": result["predicted_robot_pos_nn"],
                "predicted_payload_pos_nn": result["predicted_payload_pos_nn"],
                "predicted_quat_nn": result["predicted_robot_quat_nn"],   # make available to ensure_depth_window
                "predicted_rot_mat_nn": result["predicted_rot_mat_nn"],
                "predicted_robot_vel_nn": result["predicted_robot_vel_nn"],
                "predicted_payload_vel_nn": result["predicted_payload_vel_nn"],
            })
            
        # Remove old scene elements
        clear_obstacles()
        clear_artists()
        # Also remove any existing legend to avoid stacking legends
        try:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        except Exception:
            pass

        # Obstacles for this dataset
        obs_list = draw_obstacles(ax, params)
        for poly in obs_list:
            ax.add_collection3d(poly)
        state["obs_polys"] = obs_list

        # Axis + view
        set_axes_limits(ax, params, sol_x)
        if ortho:
            ax.view_init(elev=90, azim=-180)
            try:
                ax.set_proj_type('ortho')
            except Exception:
                pass

        # New dynamic artists using this dataset's geometry
        state["artists"] = build_dynamic_artists(ax, params)

        # NEW: add future position scatters (only if provided)
        if isinstance(future, np.ndarray) and future.ndim == 3 and future.shape[2] == 3 and future.shape[0] > 0:
            pts0 = future[0]
            if pts0.size > 0:
                state["future_scatter"] = ax.scatter(
                    pts0[:, 0], pts0[:, 1], pts0[:, 2],
                    s=12, c="tab:blue", alpha=0.8, depthshade=True, label="future robot"
                )

        if isinstance(future_payload, np.ndarray) and future_payload.ndim == 3 and future_payload.shape[2] == 3 and future.shape[0] > 0:
            ppts0 = future_payload[0]
            if ppts0.size > 0:
                state["future_payload_scatter"] = ax.scatter(
                    ppts0[:, 0], ppts0[:, 1], ppts0[:, 2],
                    s=12, c="tab:orange", alpha=0.8, depthshade=True, label="future payload"
                )

        # NEW (multi-mode predicted robot scatters)
        pred_pos = ds.get("predicted_robot_pos_nn", None)
        if isinstance(pred_pos, np.ndarray):
            N, M, H, A = pred_pos.shape
            markers = ["x", "v", "^", "s", "P", "*", "D", "o", "<", ">", "h"]  # up to 11 modes
            assert N > 0 and M > 0 and H > 0 and A == 3
 
            cmap = plt.cm.get_cmap("viridis", M)
            state["predicted_robot_scatters"] = []
            for m in range(M):
                pts = pred_pos[0, m, :, :]  # (H,3)
                sc = ax.scatter(
                    pts[:, 0], pts[:, 1], pts[:, 2],
                    s=10, c=[cmap(m)], marker=markers[m], alpha=0.85,
                    depthshade=True, label=f"pred robot m{m}"
                )
                state["predicted_robot_scatters"].append(sc)
  
            pred_payload = ds.get("predicted_payload_pos_nn", None)
            assert isinstance(pred_payload, np.ndarray) and pred_payload.shape == (N, M, H, 3)
            payload_cmap = plt.cm.get_cmap("plasma", M)
            state["predicted_payload_scatters"] = []
            for m in range(M):
                ppts = pred_payload[0, m, :, :]  # (H,3)
                scp = ax.scatter(
                    ppts[:, 0], ppts[:, 1], ppts[:, 2],
                    s=10, c=[payload_cmap(m)], marker=markers[m],
                    alpha=0.7, depthshade=True, label=f"pred payload m{m}"
                )
                state["predicted_payload_scatters"].append(scp)

            # dedupe legend
            handles, labels = ax.get_legend_handles_labels()
            assert handles
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="best")

        # Ensure depth/quaternion window for this dataset (reuses one persistent figure)
        nonlocal depth_im
        depth_im = ensure_depth_window(ds)
        # (Re)build step slider for this dataset length
        s = rebuild_step_slider(sol_x.shape[0])

        def draw_at(i):
            i = int(np.clip(i, 0, sol_x.shape[0] - 1))
            pL = sol_x[i, 0:3]
            aL = sol_x[i, 6:9]
            pQ = quad[i, :]
            update_dynamic_artists(state["artists"], params, pL, pQ, aL)
            ax.set_title(
                f"[{k+1}/{len(datasets)}] {ds['name']}  |  t={time[i]:.3f}s  (step {i+1}/{sol_x.shape[0]})"
            )

            # Update future scatters only if they exist
            if state.get("future_scatter", None) is not None and isinstance(future, np.ndarray) and i < future.shape[0]:
                pts = future[i]
                if pts.ndim == 2 and pts.shape[1] == 3 and pts.shape[0] > 0:
                    state["future_scatter"]._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])

            if state.get("future_payload_scatter", None) is not None and isinstance(future_payload, np.ndarray) and i < future_payload.shape[0]:
                ppts = future_payload[i]
                if ppts.ndim == 2 and ppts.shape[1] == 3 and ppts.shape[0] > 0:
                    state["future_payload_scatter"]._offsets3d = (ppts[:, 0], ppts[:, 1], ppts[:, 2])

            # UPDATED: per-mode horizon scatter refresh (points for current time i only)
            pred_pos = ds.get("predicted_robot_pos_nn", None)
            if state["predicted_robot_scatters"] and isinstance(pred_pos, np.ndarray):
                if i < pred_pos.shape[0]:
                    M = pred_pos.shape[1]
                    for m, sc in enumerate(state["predicted_robot_scatters"]):
                        if m < M:
                            pts = pred_pos[i, m, :, :]  # (H,3)
                            sc._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
            # ADD: refresh payload scatters for current step
            pred_payload = ds.get("predicted_payload_pos_nn", None)
            if state["predicted_payload_scatters"] and isinstance(pred_payload, np.ndarray):
                if i < pred_payload.shape[0]:
                    M = pred_payload.shape[1]
                    for m, scp in enumerate(state["predicted_payload_scatters"]):
                        if m < M:
                            ppts = pred_payload[i, m, :, :]  # (H,3)
                            scp._offsets3d = (ppts[:, 0], ppts[:, 1], ppts[:, 2])

            # Depth update only (quat is static per CSV)
            if depth_im is not None and ds.get(DK.DEPTH, None) is not None:
                di = _nearest_time_index(ds.get(DK.DEPTH_TIME, None), time[i])
                depth_im.set_data(_depth_frame_to_image(np.asarray(ds[DK.DEPTH][di])))
                if depth_fig is not None:
                    depth_fig.canvas.draw_idle()
            fig.canvas.draw_idle()

        # Bind step slider to draw
        s.on_changed(lambda val: draw_at(val))
        # Initial draw
        draw_at(0)
        fig.canvas.draw_idle()

    # Play/pause
    def on_play(_):
        state["playing"] = True

    def on_pause(_):
        state["playing"] = False

    b_play.on_clicked(on_play)
    b_pause.on_clicked(on_pause)

    # Keyboard shortcuts: left/right to step, space to toggle play
    def on_key(event):
        s = state["s_step"]
        if s is None:
            return
        i = int(round(s.val))
        if event.key == "right":
            s.set_val(min(i + 1, int(s.valmax)))
        elif event.key == "left":
            s.set_val(max(i - 1, 0))
        elif event.key == " ":
            state["playing"] = not state["playing"]

    fig.canvas.mpl_connect("key_press_event", on_key)

    def on_file_change(val):
        k = int(np.clip(int(round(val)), 0, len(datasets) - 1))
        apply_dataset(k)

    s_file.on_changed(on_file_change)

    # Timer loop
    timer = fig.canvas.new_timer(interval=20)  # ~50 Hz UI loop

    def on_timer(_):
        if not state["playing"]:
            return
        s = state["s_step"]
        if s is None:
            return
        i = int(round(s.val))
        if i >= int(s.valmax):
            state["playing"] = False
            return
        s.set_val(i + 1)

    timer.add_callback(on_timer, None)
    timer.start()

    # Boot with dataset 0
    apply_dataset(0)
    plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=60.0, help="Target FPS for output video")
    ap.add_argument(
        "--speed", type=float, default=1.0, help="Playback speed multiplier (1.0 = real time)"
    )
    ap.add_argument(
        "--gif", action="store_true", help="Force GIF output (use if ffmpeg unavailable)"
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output file path (defaults next to CSV as videos/<sub>/<name>.mp4)",
    )
    ap.add_argument("--dpi", type=int, default=200, help="Figure DPI when writing frames")
    ap.add_argument("--ortho", action="store_true", help="Top-down orthographic view")
    ap.add_argument(
        "--inspect",
        action="store_true",
        help="Open an interactive 3D viewer with a time slider after processing",
    )
    ap.add_argument(
        "--no-video",
        action="store_true",
        help="Skip writing video; only run interactive viewer if --inspect is set",
    )
    ap.add_argument(
        "--inspect_folder",
        action="store_true",
        help="Apply --inspect to every CSV when processing a folder",
    )
    ap.add_argument(
        "--combined",
        action="store_true",
        help="With --csv_folder and --inspect, open one viewer with a csv_selection slider",
    )
    ap.add_argument(
        "--depth",
        action="store_true",
        help="",
    )
    # NEW: Zarr dataset input
    ap.add_argument(
        "--zarr",
        type=str,
        default=None,
        help="Path to a Zarr dataset directory; if set, visualize/load from Zarr instead of CSVs",
    )
    ap.add_argument(
        "--policy_nn",
        action="store_true",
        help="Enable policy neural net predictions (predicted_robot_pos_nn).",
    )
    ap.add_argument(
        "--plot_step_size",
        type=int,
        default=5,
        help="Interval and horizon length for plotting predicted trajectories (mode 0).",
    )

    args = ap.parse_args()

    # NEW: Zarr mode (takes precedence)
    if args.zarr:
        zpath = Path(args.zarr)
        zpath = (ZARR_DIR / zpath).with_suffix(".zarr")
        # Multi-view over all Zarr trajectories
        if args.inspect and args.combined:
            datasets = load_zarr_folder(zpath, args, limit=100, random_sample=True)
            interactive_multi_inspect(
                datasets,
                ortho=args.ortho,
                dpi=args.dpi,
                policy_nn=args.policy_nn,
                plot_step_size=args.plot_step_size,
            )
        else:
            raise NotImplementedError("Currently, only --inspect --combined mode is supported with --zarr")

if __name__ == "__main__":
    # Example usage 
    # python create_video.py --inspect --no-video --combined --csv_folder forests
    # python create_video.py --inspect --no-video --combined --csv forests/forest_011_s3130833813.csv --yaml forests/forest_011_s3130833813.yaml
    main()
