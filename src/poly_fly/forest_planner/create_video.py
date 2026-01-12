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
python create_video.py --csv csvs/experiments/maze_1.csv

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
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import yaml  # read optional sidecar params
import random  # for random CSV sampling

from poly_fly.utils.utils import MPC, dictToClass, yamlToDict
from poly_fly.optimal_planner.polytopes import Obs, SquarePayload, Quadrotor
import poly_fly.utils.plot as plotter
from poly_fly.optimal_planner.planner import Planner
from poly_fly.data_io.utils import (
    load,
    guess_yaml_from_csv,
    find_base_dirs,
)
from poly_fly.data_io.enums import DatasetKeys as DK, AttrKeys


# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────


def newest_csv(csv_dir: Path) -> Path:
    all_csv = list(csv_dir.rglob("*.csv"))
    if not all_csv:
        raise FileNotFoundError(f"No CSV files found under {csv_dir}")
    return max(all_csv, key=lambda p: p.stat().st_mtime)


def draw_obstacles(ax, params, obstacle_color='#708090', obstacle_alpha=0.2):
    """Face-by-face obstacle rendering like plot.plot_result."""
    collections = []
    if not hasattr(params, "obstacles"):
        return collections
    for key in params.obstacles.keys():
        ob = params.obstacles[key]
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
    R = Planner.compute_quadrotor_rotation_matrix_no_jrk(
        acc_payload, params, use_robot_rotation=True
    )
    qv = (quad_shape @ R.T) + np.asarray(p_quad)[None, :]
    qh = ConvexHull(qv)
    q_faces = [[qv[i] for i in simplex] for simplex in qh.simplices]
    quad_poly.set_verts(q_faces)

    # cable as a simple segment (lightweight)
    xs = [p_payload[0], p_quad[0]]
    ys = [p_payload[1], p_quad[1]]
    zs = [p_payload[2], p_quad[2]]
    cable_line.set_data(xs, ys)
    cable_line.set_3d_properties(zs)


def _choose_frame_indices_by_time(t, fps, speed=1.0):
    """Return indices so that successive frames are ~1/(fps)*speed apart in time."""
    t = np.asarray(t, dtype=float)
    if t.size == 0:
        return np.array([], dtype=int)
    if t.size == 1:
        return np.array([0], dtype=int)
    dt_target = (1.0 / max(fps, 1e-6)) * max(speed, 1e-6)
    t0, tN = t[0], t[-1]
    frame_times = np.arange(t0, tN + 1e-9, dt_target)
    idx = np.searchsorted(t, frame_times, side='left')
    idx[idx >= t.size] = t.size - 1
    return np.unique(np.concatenate(([0], idx, [t.size - 1]))).astype(int)


def _nearest_time_index(t_array, t_val):
    """Return nearest index in t_array to t_val. If t_array is None or empty, return 0."""
    if t_array is None:
        return 0
    t = np.asarray(t_array, dtype=float)
    if t.size == 0:
        return 0
    return int(np.clip(np.abs(t - float(t_val)).argmin(), 0, t.size - 1))


# ────────────────────────────────────────────────────────────────────────────────
# Interactive viewers
# ────────────────────────────────────────────────────────────────────────────────


def interactive_inspect(params, time, sol_x, sol_u, quad_pos, ortho=False, dpi=200):
    """Single-trajectory viewer with a step slider and Play/Pause."""
    fig = plt.figure(figsize=(10, 8), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # Obstacles (static)
    obs_list = draw_obstacles(ax, params)
    for poly in obs_list:
        ax.add_collection3d(poly)

    set_axes_limits(ax, params, sol_x)

    if ortho:
        ax.view_init(elev=90, azim=-180)
        try:
            ax.set_proj_type('ortho')
        except Exception:
            pass

    # Dynamic artists
    artists = build_dynamic_artists(ax, params)

    N = sol_x.shape[0]
    t = np.asarray(time, dtype=float)

    # Slider UI
    ax_step = plt.axes([0.15, 0.07, 0.70, 0.03], facecolor='lightgoldenrodyellow')
    s_idx = Slider(ax_step, 'step', 0, N - 1, valinit=0, valfmt="%0.0f")

    # Play/Pause buttons
    ax_play = plt.axes([0.88, 0.07, 0.05, 0.03])
    ax_pause = plt.axes([0.88, 0.03, 0.05, 0.03])
    b_play = Button(ax_play, 'Play')
    b_pause = Button(ax_pause, 'Pause')
    playing = {"on": False}

    def draw_at(i):
        i = int(np.clip(i, 0, N - 1))
        pL = sol_x[i, 0:3]
        aL = sol_x[i, 6:9]
        pQ = quad_pos[i, :]
        update_dynamic_artists(artists, params, pL, pQ, aL)
        ax.set_title(f"t = {t[i]:.3f} s  (step {i+1}/{N})")
        fig.canvas.draw_idle()

    s_idx.on_changed(lambda val: draw_at(val))
    b_play.on_clicked(lambda evt: playing.__setitem__("on", True))
    b_pause.on_clicked(lambda evt: playing.__setitem__("on", False))

    # Keyboard shortcuts: left/right to step, space to toggle play
    def on_key(event):
        i = int(round(s_idx.val))
        if event.key == "right":
            s_idx.set_val(min(i + 1, N - 1))
        elif event.key == "left":
            s_idx.set_val(max(i - 1, 0))
        elif event.key == " ":
            playing["on"] = not playing["on"]

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Initial draw
    draw_at(0)

    # Simple timer-based autoplay
    timer = fig.canvas.new_timer(interval=20)  # ~50 Hz loop

    def on_timer(_):
        if not playing["on"]:
            return
        i = int(round(s_idx.val))
        if i >= N - 1:
            playing["on"] = False
            return
        s_idx.set_val(i + 1)

    timer.add_callback(on_timer, None)
    timer.start()
    plt.show()


def interactive_multi_csv_inspect(datasets, ortho=False, dpi=200):
    """
    Multi-trajectory viewer (no depth subplot).
    - datasets: list of dicts with keys:
        { 'name', 'params', 'time', 'sol_x', 'sol_u', 'quad_pos', 'path',
          optional: 'future_robot_pos', 'future_payload_pos', 'robot_quat' }
    """
    fig = plt.figure(figsize=(10, 8), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # Quaternion window state (persistent). Depth plotting removed.
    depth_fig = None  # kept as handle for the external figure
    depth_ax = None  # unused now, but kept to minimize surgery
    data_im = None  # no longer used
    depth_cbar = None  # no longer used
    quat_ax = None
    vel_ax = None
    pvel_ax = None
    quat_lines = []

    # UI axes
    ax_csv = plt.axes([0.15, 0.02, 0.70, 0.03], facecolor='lightgoldenrodyellow')
    ax_step = plt.axes([0.15, 0.07, 0.70, 0.03], facecolor='lightgoldenrodyellow')
    ax_play = plt.axes([0.88, 0.07, 0.05, 0.03])
    ax_pause = plt.axes([0.88, 0.03, 0.05, 0.03])

    s_csv = Slider(ax_csv, 'csv_selection', 0, len(datasets) - 1, valinit=0, valfmt="%0.0f")
    s_step = None
    b_play = Button(ax_play, 'Play')
    b_pause = Button(ax_pause, 'Pause')

    state = {
        "csv_idx": 0,
        "artists": None,
        "obs_polys": [],
        "playing": False,
        "s_step": None,
        # NEW: scatter artists
        "future_scatter": None,  # robot future
        "future_payload_scatter": None,  # payload future
    }

    def ensure_data_window(ds):
        """
        Create/refresh the external figure for quaternion visualization only.
        Depth image plotting has been removed.
        """
        nonlocal depth_fig, depth_ax, data_im, depth_cbar, quat_ax, quat_lines, vel_ax, pvel_ax

        qdata = ds.get("robot_quat", None)

        has_quat = (
            qdata is not None and getattr(qdata, "shape", None) is not None and qdata.shape[-1] == 4
        )
        vdata = ds["sol_quad_x"][:, 3:6]
        pvdata = ds["sol_x"][:, 3:6]

        # import pdb; pdb.set_trace()
        # Create persistent figure: keep 1x2 but only use the right subplot for quaternions.
        if depth_fig is None:
            # depth_fig, (depth_ax, quat_ax) = plt.subplots(1, 1, figsize=(10, 4), dpi=dpi)
            depth_fig, (quat_ax, vel_ax, pvel_ax) = plt.subplots(1, 3, figsize=(10, 4), dpi=dpi)

        # Clear prior content
        # depth_ax.clear()
        quat_ax.clear()
        vel_ax.clear()
        pvel_ax.clear()
        data_im = None
        if depth_cbar is not None:
            try:
                depth_cbar.remove()
            except Exception:
                pass

        # Quaternion subplot (right)
        quat_lines = []
        if has_quat:
            q = np.asarray(qdata)
            labels = ["w", "x", "y", "z"]
            colors = ["tab:purple", "tab:blue", "tab:orange", "tab:green"]
            for j in range(4):
                (line,) = quat_ax.plot(q[:, j], label=labels[j], color=colors[j], linewidth=1.2)
                quat_lines.append(line)
            quat_ax.set_title("Robot quaternion (all steps)")
            quat_ax.set_xlabel("step")
            quat_ax.set_ylabel("value")
            quat_ax.grid(True, alpha=0.3)
            quat_ax.legend(loc="best", fontsize=8)
        else:
            quat_ax.set_title("No robot_quat available")
            quat_ax.text(0.5, 0.5, "No quat", ha="center", va="center", transform=quat_ax.transAxes)
            quat_ax.grid(False)

        # Robot velocity subplot (index 3)
        if vdata is not None:
            v = np.asarray(vdata)
            vlabels = ["vx", "vy", "vz"]
            vcolors = ["tab:cyan", "tab:pink", "tab:olive"]
            for j in range(3):
                vel_ax.plot(v[:, j], label=vlabels[j], color=vcolors[j], linewidth=1.2)

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

        # Nothing to return for depth image anymore; keep API but return None.
        return None

    def clear_obstacles():
        for p in state["obs_polys"]:
            try:
                p.remove()
            except Exception:
                pass
        state["obs_polys"] = []

    def clear_artists():
        a = state["artists"]
        if a is None:
            pass
        else:
            try:
                a["payload_poly"].remove()
            except Exception:
                pass
            try:
                a["quad_poly"].remove()
            except Exception:
                pass
            try:
                a["cable_line"].remove()
            except Exception:
                pass
        # NEW: remove future scatters if present
        if state["future_scatter"] is not None:
            try:
                state["future_scatter"].remove()
            except Exception:
                pass
            state["future_scatter"] = None
        if state["future_payload_scatter"] is not None:
            try:
                state["future_payload_scatter"].remove()
            except Exception:
                pass
            state["future_payload_scatter"] = None
        state["artists"] = None

    def rebuild_step_slider(N):
        nonlocal s_step
        ax_step.cla()
        s = Slider(ax_step, 'step', 0, max(N - 1, 0), valinit=0, valfmt="%0.0f")
        state["s_step"] = s
        return s

    def apply_dataset(k):
        """Rebuild scene for dataset k and (re)attach quaternion window (no depth)."""
        nonlocal data_im
        state["csv_idx"] = k
        ds = datasets[k]
        params = ds["params"]
        time = ds["time"]
        sol_x = ds["sol_x"]
        quad = ds["quad_pos"]
        future = ds.get("future_robot_pos")
        future_payload = ds.get("future_payload_pos")

        clear_obstacles()
        clear_artists()

        obs_list = draw_obstacles(ax, params)
        for poly in obs_list:
            ax.add_collection3d(poly)
        state["obs_polys"] = obs_list

        set_axes_limits(ax, params, sol_x)
        if ortho:
            ax.view_init(elev=90, azim=-180)
            try:
                ax.set_proj_type('ortho')
            except Exception:
                pass

        state["artists"] = build_dynamic_artists(ax, params)

        # Future scatters unchanged
        if (
            isinstance(future, np.ndarray)
            and future.ndim == 3
            and future.shape[2] == 3
            and future.shape[0] > 0
        ):
            pts0 = future[0]
            if pts0.size > 0:
                state["future_scatter"] = ax.scatter(
                    pts0[:, 0],
                    pts0[:, 1],
                    pts0[:, 2],
                    s=12,
                    c="tab:blue",
                    alpha=0.8,
                    depthshade=True,
                    label="future robot",
                )

        if (
            isinstance(future_payload, np.ndarray)
            and future_payload.ndim == 3
            and future_payload.shape[2] == 3
            and future_payload.shape[0] > 0
        ):
            ppts0 = future_payload[0]
            if ppts0.size > 0:
                state["future_payload_scatter"] = ax.scatter(
                    ppts0[:, 0],
                    ppts0[:, 1],
                    ppts0[:, 2],
                    s=12,
                    c="tab:orange",
                    alpha=0.8,
                    depthshade=True,
                    label="future payload",
                )

        # Ensure only quaternion window is drawn
        ensure_data_window(ds)
        data_im = None  # explicitly no depth image

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

            # Future scatters unchanged
            if (
                state.get("future_scatter", None) is not None
                and isinstance(future, np.ndarray)
                and i < future.shape[0]
            ):
                pts = future[i]
                if pts.ndim == 2 and pts.shape[1] == 3 and pts.shape[0] > 0:
                    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
                    state["future_scatter"]._offsets3d = (xs, ys, zs)

            if (
                state.get("future_payload_scatter", None) is not None
                and isinstance(future_payload, np.ndarray)
                and i < future_payload.shape[0]
            ):
                ppts = future_payload[i]
                if ppts.ndim == 2 and ppts.shape[1] == 3 and ppts.shape[0] > 0:
                    px, py, pz = ppts[:, 0], ppts[:, 1], ppts[:, 2]
                    state["future_payload_scatter"]._offsets3d = (px, py, pz)

            # Depth update removed – only 3D + quat now
            fig.canvas.draw_idle()

        s.on_changed(lambda val: draw_at(val))
        draw_at(0)
        fig.canvas.draw_idle()

    # Play/pause
    def on_play(_):
        state["playing"] = True

    def on_pause(_):
        state["playing"] = False

    b_play.on_clicked(on_play)
    b_pause.on_clicked(on_pause)

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

    def on_csv_change(val):
        k = int(np.clip(int(round(val)), 0, len(datasets) - 1))
        apply_dataset(k)

    s_csv.on_changed(on_csv_change)

    timer = fig.canvas.new_timer(interval=20)

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

    apply_dataset(0)
    plt.show()


# ────────────────────────────────────────────────────────────────────────────────
# Batch helpers and video writer
# ────────────────────────────────────────────────────────────────────────────────


def build_datasets_from_folder(folder_path: Path, args, limit=10):
    """
    Load every CSV in folder (recursive), compute quad positions.
    Returns list of dicts: { 'name', 'params', 'time', 'sol_x', 'sol_u', 'quad_pos', 'path' }
    """
    base, csv_dir, params_dir = find_base_dirs()

    if not folder_path.is_absolute():
        folder_path = (csv_dir / folder_path).resolve()
    if not folder_path.exists():
        raise FileNotFoundError(f"CSV folder not found: {folder_path}")

    # Gather candidates
    all_csvs = sorted(folder_path.rglob("*.csv"))
    if not all_csvs:
        raise FileNotFoundError(f"No CSVs found in folder: {folder_path}")

    if getattr(args, "combined", False) and len(all_csvs) > limit:
        total = len(all_csvs)
        all_csvs = random.sample(all_csvs, k=limit)
        print(f"[MULTI-VIEW] --combined: randomly sampled {len(all_csvs)} CSVs from {total} total.")

    datasets = []
    for cp in all_csvs:
        yaml_path = guess_yaml_from_csv(cp, params_dir) if not args.yaml else Path(args.yaml)
        data = load(cp, yaml_path)
        time = data["time"]
        sol_x = data["sol_x"]
        sol_u = data["sol_u"]
        sol_quad_x = data["sol_quad_x"]
        sol_robot_quat = data["sol_quad_quat"]
        params = data["params"]

        quad_pos = sol_quad_x[:, :3]

        ds = dict(
            name=cp.name,
            params=params,
            time=time,
            sol_x=sol_x,
            sol_u=sol_u,
            quad_pos=quad_pos,
            path=cp,
            sol_quad_x=sol_quad_x,
        )
        ds["robot_quat"] = sol_robot_quat
        datasets.append(ds)
    return datasets


def run_for_csv(csv_path: Path, args):
    base, csv_dir, params_dir = find_base_dirs()

    if not csv_path.is_absolute():
        csv_path = (csv_dir / csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Params: prefer sidecar next to the CSV; fallback to mapped YAML or --yaml
    if args.yaml:
        yaml_path = Path(args.yaml)
        if not yaml_path.is_absolute():
            yaml_path = (params_dir / yaml_path).resolve()
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML not found: {yaml_path}")
    else:
        yaml_path = guess_yaml_from_csv(csv_path, params_dir)
    time, sol_x, sol_u, sol_quad_x, sol_quad_quat, payload_rpy, params = load(csv_path, yaml_path)

    quad_pos = sol_quad_x[:, :3]

    # Output path
    if args.out:
        out_path = Path(args.out)
        if args.csvs_folder or args.csv_folder:
            stem = csv_path.stem
            out_path = out_path.with_name(out_path.stem + f"_{stem}").with_suffix(out_path.suffix)
    else:
        rel = csv_path.relative_to(find_base_dirs()[1])  # relative to csvs/
        out_root = base / "videos"
        out_path = (out_root / rel).with_suffix(".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Figure and axes
    fig = plt.figure(figsize=(10, 8), dpi=args.dpi)
    ax = fig.add_subplot(111, projection="3d")

    # Obstacles (static)
    obs_list = draw_obstacles(ax, params)
    for poly in obs_list:
        ax.add_collection3d(poly)

    set_axes_limits(ax, params, sol_x)

    # Orthographic / top-down option
    if args.ortho:
        ax.view_init(elev=90, azim=-180)
        try:
            ax.set_proj_type('ortho')
        except Exception:
            pass

    # Dynamic artists
    artists = build_dynamic_artists(ax, params)

    # Time-decimate to match FPS/speed (skip points so frames ≈ total_time*fps/speed)
    frame_idx = _choose_frame_indices_by_time(time, args.fps, args.speed)
    t = np.asarray(time, dtype=float)[frame_idx]
    dt = np.diff(t, prepend=t[0])
    repeats = np.maximum(1, np.round((dt * args.fps) / max(args.speed, 1e-6)).astype(int))

    # Select writer
    writer = None
    use_gif = args.gif
    if not use_gif:
        try:
            writer = animation.FFMpegWriter(fps=args.fps)
        except Exception:
            use_gif = True
    if use_gif:
        writer = animation.PillowWriter(fps=args.fps)
        out_path = out_path.with_suffix(".gif")

    # Write video (unless disabled)
    if not args.no_video:
        print(
            f"[INFO] Writing {'GIF' if use_gif else 'MP4'} to: {out_path} (fps={args.fps}, speed={args.speed})"
        )
        with writer.saving(fig, str(out_path), dpi=args.dpi):
            for k in frame_idx:
                pL = sol_x[k, 0:3]
                aL = sol_x[k, 6:9]
                pQ = quad_pos[k, :]
                update_dynamic_artists(artists, params, pL, pQ, aL)
                for _ in range(int(repeats[np.where(frame_idx == k)[0][0]])):
                    writer.grab_frame()
        print(f"[DONE] Saved: {out_path}")

    # Interactive inspect if requested
    if args.inspect or getattr(args, "inspect_folder", False):
        plt.close(fig)
        interactive_inspect(params, time, sol_x, sol_u, quad_pos, ortho=args.ortho, dpi=args.dpi)
    else:
        plt.close(fig)


def main():
    base, csv_dir, params_dir = find_base_dirs()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv", type=str, default=None, help="Path to CSV under csvs/ (default: newest)"
    )
    ap.add_argument(
        "--yaml", type=str, default=None, help="Override params YAML path (under params/)"
    )
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
        "--csvs_folder",
        type=str,
        default=None,
        help="Folder under csvs/ containing CSVs to process in batch",
    )
    ap.add_argument("--csv_folder", type=str, default=None, help="Alias for --csvs_folder")
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
    args = ap.parse_args()

    # Folder mode
    folder_arg = args.csvs_folder or args.csv_folder
    if folder_arg:
        folder_path = Path(folder_arg)
        if not folder_path.is_absolute():
            folder_path = (csv_dir / folder_path).resolve()
        if not folder_path.exists():
            raise FileNotFoundError(f"CSV folder not found: {folder_path}")

        # NEW: One viewer with CSV selector (obstacles + trajectory swap)
        if args.inspect and args.combined:
            print(f"[MULTI-VIEW] Loading all trajectories from {folder_path}")
            datasets = build_datasets_from_folder(folder_path, args)
            interactive_multi_csv_inspect(datasets, ortho=args.ortho, dpi=args.dpi)
            return

        # Otherwise: regular batch per-file processing (and per-file inspect if requested)
        csv_files = sorted(folder_path.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSVs found in folder: {folder_path}")
        print(f"[BATCH] Processing {len(csv_files)} CSVs under {folder_path}")
        for i, cp in enumerate(csv_files, 1):
            print(f"[{i}/{len(csv_files)}] {cp}")
            run_for_csv(cp, args)
        return

    # Single-file path resolution (existing behavior)
    csv_path = Path(args.csv) if args.csv else newest_csv(csv_dir)
    run_for_csv(csv_path, args)


if __name__ == "__main__":
    # Example usage
    # python create_video.py --inspect --no-video --combined --csv_folder forests
    # python create_video.py --inspect --no-video --combined --csv forests/forest_011_s3130833813.csv --yaml forests/forest_011_s3130833813.yaml
    main()
