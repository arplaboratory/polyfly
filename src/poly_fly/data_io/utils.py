from __future__ import annotations
import argparse
import os
import sys
import csv
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation as R

from pathlib import Path
from dataclasses import asdict

from poly_fly.utils.utils import MPC, dictToClass, vec2asym, yamlToDict

import random

from poly_fly.data_io.enums import DatasetKeys as DK, AttrKeys

try:
    BASE_DIR = os.environ['POLYFLY_DIR']
except KeyError:
    raise EnvironmentError("Required environment variable 'POLYFLY_DIR' is not set.")

PARAMS_DIR = os.path.join(BASE_DIR, "data", "params")
GIFS_DIR = os.path.join(BASE_DIR, "data", "gifs")
CSV_DIR = os.path.join(BASE_DIR, "data", "csvs")
IMG_DIR = os.path.join(BASE_DIR, "data", "imgs")
DEPTH_DIR = os.path.join(BASE_DIR, "data", "depth")
ZARR_DIR = os.path.join(BASE_DIR, "data", "zarr")
DEPTH_SCALE_FACTOR = 10.0  # scale depth images by this factor to get meters

NORMALIZATION_STATS_DIR = os.path.join(BASE_DIR, "data", "normalization_stats")
DEFAULT_NORMALIZATION_STATE_KEYS = [
    "payload_vel",
    "robot_vel",
    "robot_quat",
    "robot_goal_relative",
    "rot_mat",
]
DEFAULT_NORMALIZATION_FUTURE_KEYS = [
    "future_robot_pos",
    "future_payload_pos",
    "future_quaternion",
    "future_robot_vel",
    "future_payload_vel",
    "future_rot_mat",
]


def find_base_dirs() -> tuple[Path, Path, Path]:
    base = Path(os.environ.get("POLYFLY_DIR", Path.cwd()))
    csv_dir = base / "data" / "csvs"
    params_dir = base / "data" / "params"
    return base, csv_dir, params_dir


def load_all_csvs_and_params(csv_subdirectory) -> dict[str, list]:
    """
    In the CSV subdirectory, loop through each CSV file, find the matching YAML,
    load the data using `load(csv_path, yaml_path)`, and return a dict:
      key   = CSV stem (filename without extension)
      value = list containing the return values of `load`
    """
    csv_root = Path(CSV_DIR) / str(csv_subdirectory)
    yaml_root = Path(PARAMS_DIR) / str(csv_subdirectory)

    if not csv_root.exists():
        raise FileNotFoundError(f"CSV subdirectory not found: {csv_root}")
    if not yaml_root.exists():
        # not fatal; we may still map via guess_yaml_from_csv
        print(f"[warn] YAML subdirectory not found: {yaml_root} — will try to guess per CSV")

    out: dict[str, list] = {}
    csv_files = sorted(csv_root.glob("*.csv"))
    for csv_path in csv_files:
        print(f"loading csv {csv_path}")
        stem = csv_path.stem
        yaml_path = yaml_root / f"{stem}.yaml"
        if not yaml_path.exists():
            # fallback to guesser using global PARAMS_DIR
            yaml_path = guess_yaml_from_csv(csv_path, Path(PARAMS_DIR))
        if not yaml_path.exists():
            print(f"[skip] No YAML found for CSV: {csv_path} (expected {yaml_path})")
            continue

        try:
            loaded = load(str(csv_path), str(yaml_path))  # returns tuple
            out[stem] = list(loaded)
        except Exception as e:
            print(f"[error] Failed to load {csv_path} with {yaml_path}: {e}")

    return out


def load_all_csvs_and_params_names(csv_subdirectory) -> dict[str, list]:
    """
    Return two aligned lists:
      - csv_files: list[Path] of CSV files under CSV_DIR/<csv_subdirectory>
      - yaml_paths: list[Path] of matched YAML files under PARAMS_DIR/<csv_subdirectory> (or guessed)
    """
    csv_root = Path(CSV_DIR) / str(csv_subdirectory)
    yaml_root = Path(PARAMS_DIR) / str(csv_subdirectory)

    if not csv_root.exists():
        raise FileNotFoundError(f"CSV subdirectory not found: {csv_root}")
    if not yaml_root.exists():
        print(f"[warn] YAML subdirectory not found: {yaml_root} — will try to guess per CSV")

    csv_files = sorted(csv_root.glob("*.csv"))
    yaml_paths = []

    for csv_path in csv_files:
        print(f"loading csv {csv_path}")
        stem = csv_path.stem
        yaml_path = yaml_root / f"{stem}.yaml"

        if not yaml_path.exists():
            # fallback to guesser using global PARAMS_DIR
            yaml_path = guess_yaml_from_csv(csv_path, Path(PARAMS_DIR))

        if not yaml_path.exists():
            raise Exception(f"[skip] No YAML found for CSV: {csv_path} (expected {yaml_path})")

        yaml_paths.append(yaml_path)

    return csv_files, yaml_paths


def load(csv_path, yaml_path, depth_path=None):
    data = load_csv(csv_path)
    params = load_params(yaml_path)

    # Attach params and YAML metadata
    data[DK.PARAMS] = params
    data[DK.YAML_FILE] = str(yaml_path)
    data[DK.YAML_FILE_NAME] = Path(yaml_path).name

    if depth_path:
        depth = load_depth_data(depth_path)
        data[DK.DEPTH] = depth
    else:
        data[DK.DEPTH] = None

    return data


def load_csv(csv_path):
    """
    Read the CSV produced by `save_csv_arrays`.

    Returns:
        time            : (N,)
        sol_x           : (9, N)     # payload pos/vel/acc
        sol_u           : (3, N)     # jerk inputs
        sol_quad_x      : (9, N)     # quad pos/vel/acc
        sol_quad_quat   : (4, N)     # quad quaternion (w,x,y,z). If missing → zeros
        sol_payload_rpy : (3, N)     # payload rpy
    """
    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        raise ValueError("Expected 'time' column in CSV.")
    time = df["time"].to_numpy(dtype=float)
    N = time.shape[0]

    # Collect column groups by prefix
    x_cols = [c for c in df.columns if c.startswith(f"{DK.SOL_X}_")]
    u_cols = [c for c in df.columns if c.startswith(f"{DK.SOL_U}_")]
    quad_x_cols = [c for c in df.columns if c.startswith(f"{DK.SOL_QUAD_X}_")]
    quad_quat_cols = [c for c in df.columns if c.startswith(f"{DK.SOL_QUAD_QUAT}_")]
    payload_rpy_cols = [c for c in df.columns if c.startswith(f"{DK.SOL_PAYLOAD_RPY}_")]
    rot_mat_cols = [c for c in df.columns if c.startswith(f"{DK.ROT_MAT}_")]

    # Validate presence and expected counts for mandatory groups
    if len(x_cols) != 9:
        raise ValueError(f"Expected 9 payload state columns 'sol_x_*'; got {len(x_cols)}")
    if len(u_cols) != 3:
        raise ValueError(f"Expected 3 input columns 'sol_u_*'; got {len(u_cols)}")
    if len(quad_x_cols) != 9:
        raise ValueError(f"Expected 9 quad state columns 'sol_quad_x_*'; got {len(quad_x_cols)}")

    # Quaternion columns are optional but if present must be 4
    if len(quad_quat_cols) not in (0, 4):
        raise ValueError(
            f"Expected either 0 or 4 quad quaternion columns 'sol_quad_quat_*'; got {len(quad_quat_cols)}"
        )

    # Payload RPY columns are optional but if present must be 3
    if len(payload_rpy_cols) > 0 and len(payload_rpy_cols) != 3:
        raise ValueError(
            f"Expected 3 payload rpy columns 'sol_payload_rpy_*'; got {len(payload_rpy_cols)}"
        )

    # Sort each group numerically by suffix
    x_cols = sorted(x_cols, key=lambda c: int(c.split("_")[-1]))
    u_cols = sorted(u_cols, key=lambda c: int(c.split("_")[-1]))
    quad_x_cols = sorted(quad_x_cols, key=lambda c: int(c.split("_")[-1]))
    quad_quat_cols = sorted(quad_quat_cols, key=lambda c: int(c.split("_")[-1]))
    payload_rpy_cols = sorted(payload_rpy_cols, key=lambda c: int(c.split("_")[-1]))
    rot_mat_cols = sorted(rot_mat_cols, key=lambda c: int(c.split("_")[-1]))

    sol_x = df[x_cols].to_numpy(dtype=float)
    sol_u = df[u_cols].to_numpy(dtype=float)
    sol_quad_x = df[quad_x_cols].to_numpy(dtype=float)

    assert len(quad_quat_cols) == 4
    sol_quad_quat = df[quad_quat_cols].to_numpy(dtype=float)

    assert len(payload_rpy_cols) == 3
    sol_payload_rpy = df[payload_rpy_cols].to_numpy(dtype=float)

    assert len(rot_mat_cols) == 6
    sol_rot_mat = df[rot_mat_cols].to_numpy(dtype=float)

    # Extra sanity checks: lengths must match time length
    if sol_x.shape[0] != N:
        raise ValueError(f"sol_x has {sol_x.shape[0]} samples but time has {N}.")
    if sol_u.shape[0] != N:
        raise ValueError(f"sol_u has {sol_u.shape[0]} samples but time has {N}.")
    if sol_quad_x.shape[0] != N:
        raise ValueError(f"sol_quad_x has {sol_quad_x.shape[0]} samples but time has {N}.")
    if sol_quad_quat.shape[0] != N:
        raise ValueError(f"sol_quad_quat has {sol_quad_quat.shape[0]} samples but time has {N}.")
    if sol_payload_rpy.shape[0] != N:
        raise ValueError(
            f"sol_payload_rpy has {sol_payload_rpy.shape[0]} samples but time has {N}."
        )

    return {
        DK.TIME: time,
        DK.SOL_X: sol_x,
        DK.SOL_U: sol_u,
        DK.SOL_QUAD_X: sol_quad_x,
        DK.SOL_QUAD_QUAT: sol_quad_quat,
        DK.SOL_PAYLOAD_RPY: sol_payload_rpy,
        DK.ROT_MAT: sol_rot_mat,
    }


def load_obstacle_info_from_yaml(yaml_path):
    """
    Load obstacle information (position + dimensions) starting from a yaml path.

    Returns:
        np.ndarray with dtype [('x', float), ('y', float), ('z', float),
                               ('xL', float), ('yL', float), ('zL', float)]
        One row per obstacle.
    """
    params = load_params(yaml_path)
    return load_obstacle_info_from_params(params)


def load_obstacle_info_from_params(params):
    """
    Simplified: assumes params is an MPC object and params.obstacles is a dict:
        obstacle_id -> { 'x','y','l','b', optional: 'z','h','xL','yL','zL' }
    Returns structured ndarray with fields (x,y,z,xL,yL,zL).
    """
    dtype = np.dtype(
        [
            ("x", "f8"),
            ("y", "f8"),
            ("z", "f8"),
            ("xL", "f8"),
            ("yL", "f8"),
            ("zL", "f8"),
        ]
    )
    if not params.obstacles:
        return np.zeros(0, dtype=dtype)

    records = []
    for _, obs in params.obstacles.items():
        x = obs["x"]
        y = obs["y"]
        z = obs.get("z", 0.0)
        # Allow either explicit xL/yL/zL or legacy l/b/h
        xL = obs.get("l", np.inf)
        yL = obs.get("b", np.inf)
        zL = obs.get("h", np.inf)
        records.append((float(x), float(y), float(z), float(xL), float(yL), float(zL)))

    return np.array(records, dtype=dtype)


def save_csv_arrays(data: dict):
    """
    Save CSV using a dictionary payload.
    Required enum keys:
      - AttrKeys.CSV_FILEPATH: str (full output path to CSV)
      - DK.TIME: (T,)
      - DK.SOL_X: (T, Nx)
      - DK.SOL_U: (T, Nu)
      - DK.SOL_QUAD_X: (T, Nqx)
      - DK.SOL_QUAD_QUAT: (T, 4)
      - DK.SOL_PAYLOAD_RPY: (T, 3)
    """
    required = [
        AttrKeys.CSV_FILEPATH,
        DK.TIME,
        DK.SOL_X,
        DK.SOL_U,
        DK.SOL_QUAD_X,
        DK.SOL_QUAD_QUAT,
        DK.SOL_PAYLOAD_RPY,
        DK.ROT_MAT,
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"save_csv_arrays missing required keys: {missing}")

    csv_filepath = data[AttrKeys.CSV_FILEPATH]
    times = data[DK.TIME]
    sol_x = data[DK.SOL_X]
    sol_u = data[DK.SOL_U]
    sol_quad_x = data[DK.SOL_QUAD_X]
    sol_robot_quat = data[DK.SOL_QUAD_QUAT]
    sol_payload_rpy = data[DK.SOL_PAYLOAD_RPY]
    sol_rot_mat = data[DK.ROT_MAT].reshape(times.shape[0], 6)

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)

    print(f"writing to {csv_filepath}")
    with open(csv_filepath, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write header using enums
        writer.writerow(
            [DK.TIME]
            + [f'{DK.SOL_X}_{i}' for i in range(sol_x.shape[1])]
            + [f'{DK.SOL_U}_{i}' for i in range(sol_u.shape[1])]
            + [f'{DK.SOL_QUAD_X}_{i}' for i in range(sol_quad_x.shape[1])]
            + [f'{DK.SOL_QUAD_QUAT}_{i}' for i in range(sol_robot_quat.shape[1])]
            + [f'{DK.SOL_PAYLOAD_RPY}_{i}' for i in range(sol_payload_rpy.shape[1])]
            + [f'{DK.ROT_MAT}_{i}' for i in range(sol_rot_mat.shape[1])]
        )

        # Write data rows
        for idx, t in enumerate(times):
            row = (
                [t]
                + sol_x[idx, :].tolist()
                + sol_u[idx, :].tolist()
                + sol_quad_x[idx, :].tolist()
                + sol_robot_quat[idx, :].tolist()
                + sol_payload_rpy[idx, :].tolist()
                + sol_rot_mat[idx, :].tolist()
            )
            writer.writerow(row)


def save_depth_data(filename, times, depth, subdirectory: str = "depth") -> str:
    """
    Save a stack of depth frames aligned with 'times' to a compressed NPZ.
    - filename: stem to use for the saved file (no extension)
    - times: 1-D array-like of timestamps (length T)
    - depth: numpy array of shape (T, H, W) or (T, H, W, C), float preferred
    - subdirectory: subfolder under DEPTH_DIR where to place the 'depth' outputs
    Returns the full path to the saved file.
    """
    times = np.asarray(times)
    depth = np.asarray(depth)
    if times.ndim != 1:
        raise ValueError(f"times must be 1-D, got shape {times.shape}")
    if depth.shape[0] != times.shape[0]:
        raise ValueError(f"Depth length {depth.shape[0]} must match times length {times.shape[0]}.")

    min_value = np.min(depth)
    if min_value < 0:
        print(depth)
        raise ValueError(f"Depth contains negative values (min {min_value}).")

    out_dir = Path(DEPTH_DIR) / str(subdirectory)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{filename}.npz"
    print(f"SAVING TO {out_path}")
    np.savez_compressed(out_path, time=times, depth=depth.astype(np.float16, copy=False))
    return str(out_path)


def load_depth_data(p):
    """
    Load depth NPZ saved by save_depth_data.
    Expects file at: DEPTH_DIR/<subdirectory>/<filename>.npz
    Returns:
      time:  (T,) float array
      depth: (T, H, W) or (T, H, W, C) float array
    """
    print(f"LOADING depth data {p}")
    if not p.exists():
        raise FileNotFoundError(f"Depth file not found: {p}")
    with np.load(p) as data:
        time = data["time"]
        depth = data["depth"]
    return time, depth


def guess_yaml_from_csv(csv_path: Path, params_dir: Path) -> Path:
    try:
        parts = csv_path.parts
        if "csvs" in parts:
            idx = parts.index("csvs")
            rel = Path(*parts[idx + 1 :])
        else:
            rel = csv_path
        return params_dir / rel.with_suffix(".yaml")
    except Exception:
        return params_dir / csv_path.with_suffix(".yaml").name


def load_params(yaml_path: Path):
    """
    Build MPC params object.
    Preference
      1) If csv_path provided and a sidecar '<csv>.params.yaml' exists → use it
      2) Otherwise load mapped params YAML (yaml_path) or raise if missing
    """
    if type(yaml_path) == str:
        yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Could not locate params YAML: {yaml_path}")
    return dictToClass(MPC, yamlToDict(str(yaml_path)))


def save_params(params_path, params):
    with open(params_path, "w") as f_params:
        yaml.safe_dump(asdict(params), f_params, sort_keys=False)


def save_all(data: dict):
    """
    Save all artifacts (CSV and YAML) using a single dictionary.
    Required enum keys:
      - AttrKeys.STEM
      - AttrKeys.CSV_SUBDIRECTORY
      - DK.TIME
      - DK.SOL_X
      - DK.SOL_U
      - DK.SOL_QUAD_X
      - DK.SOL_QUAD_QUAT
      - DK.SOL_PAYLOAD_RPY
      - DK.PARAMS
    """
    required = [
        AttrKeys.STEM,
        AttrKeys.CSV_SUBDIRECTORY,
        DK.TIME,
        DK.SOL_X,
        DK.SOL_U,
        DK.SOL_QUAD_X,
        DK.SOL_QUAD_QUAT,
        DK.SOL_PAYLOAD_RPY,
        DK.PARAMS,
        DK.ROT_MAT,
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"save_all missing required keys: {missing}")

    filename = data[AttrKeys.STEM]
    subdirectory = data[AttrKeys.CSV_SUBDIRECTORY]
    interpolated_time = data[DK.TIME]
    interpolated_x = data[DK.SOL_X]
    interpolated_u = data[DK.SOL_U]
    interpolated_quad_x = data[DK.SOL_QUAD_X]
    robot_quat = data[DK.SOL_QUAD_QUAT]
    interpolated_payload_rpy = data[DK.SOL_PAYLOAD_RPY]
    params = data[DK.PARAMS]
    interpolated_rot_mat = data[DK.ROT_MAT]

    # Ensure directories exist
    os.makedirs(os.path.join(CSV_DIR, subdirectory), exist_ok=True)
    os.makedirs(os.path.join(PARAMS_DIR, subdirectory), exist_ok=True)

    filedir_csv = os.path.join(CSV_DIR, subdirectory, filename) + ".csv"
    filedir_params = os.path.join(PARAMS_DIR, subdirectory, filename) + ".yaml"

    save_csv_arrays(
        {
            AttrKeys.CSV_FILEPATH: filedir_csv,
            DK.TIME: interpolated_time,
            DK.SOL_X: interpolated_x,
            DK.SOL_U: interpolated_u,
            DK.SOL_QUAD_X: interpolated_quad_x,
            DK.SOL_QUAD_QUAT: robot_quat,
            DK.SOL_PAYLOAD_RPY: interpolated_payload_rpy,
            DK.ROT_MAT: interpolated_rot_mat,
        }
    )
    save_params(filedir_params, params)


def find_csvs(
    num_files,
    yaml_dir="/home/mrunal/Documents/poly_fly/data/params/forests",
    csv_dir="/home/mrunal/Documents/poly_fly/data/csvs/forests",
):
    """
    Return two lists (yaml_paths, csv_paths) of length `num_files` with matching stems:
      - yaml:  {yaml_dir}/forest_*.yaml
      - csv:   {csv_dir}/forest_*.csv
    Pairs are chosen in lexicographic order by YAML filename.
    Raises if not enough matching pairs exist.
    """
    ydir = Path(yaml_dir)
    cdir = Path(csv_dir)
    if not ydir.is_dir():
        raise FileNotFoundError(f"YAML dir not found: {yaml_dir}")
    if not cdir.is_dir():
        raise FileNotFoundError(f"CSV dir not found: {csv_dir}")

    candidates = sorted(p for p in ydir.glob("forest_*.yaml") if p.is_file())
    # Randomize YAML candidates to pick random (yaml,csv) pairs
    random.shuffle(candidates)

    pairs = []
    for y in candidates:
        stem = y.stem  # e.g., forest_008_s443094727
        c = cdir / f"{stem}.csv"
        if c.exists():
            pairs.append((str(y), str(c)))
            if len(pairs) >= num_files:
                break

    if len(pairs) < num_files:
        raise FileNotFoundError(
            f"Requested {num_files} (yaml,csv) pairs but found {len(pairs)}. "
            f"yaml_dir={yaml_dir}, csv_dir={csv_dir}"
        )
    yaml_paths, csv_paths = zip(*pairs)
    return list(yaml_paths), list(csv_paths)


def extract_future_trajectory(future_horizon: int, x):
    if future_horizon <= 0:
        raise ValueError("future_horizon must be a positive integer")
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"x must have shape [N, M], got {x.shape}")

    N = x.shape[0]
    out = np.zeros((N, future_horizon, x.shape[1]))
    for i in range(N - 1):
        start_idx = i + 1
        end_idx = min(start_idx + future_horizon, N)
        cur_horizon_length = end_idx - start_idx
        out[i, :cur_horizon_length, :] = x[start_idx:end_idx, :]

        if end_idx < start_idx + future_horizon:
            out[i, cur_horizon_length:future_horizon, :] = x[-1, :]

    out[-1, :, :] = x[-1, :]

    return out


def make_trajectory_relative(x, base_x):
    x = np.asarray(x)
    assert x.ndim == 3
    assert x.shape[0] == base_x.shape[0]
    assert x.shape[2] == base_x.shape[1]

    N = x.shape[0]
    out = np.zeros_like(x)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j, :] = x[i, j, :] - base_x[i, :]

    return out


def extract_relative_payload_pos(future_robot_pos, future_payload_pos, future_quaternion):
    x_payload_wrt_robot_in_world = future_payload_pos - future_robot_pos
    x_payload_wrt_robot_in_body = np.zeros_like(x_payload_wrt_robot_in_world)

    # TODO Vectoriz implementation
    for i in range(future_quaternion.shape[0]):
        for j in range(future_quaternion.shape[1]):
            R_mat = R.from_quat(future_quaternion[i, j, :]).as_matrix()
            x_payload_wrt_robot_in_body[i, j, :] = R_mat.T @ x_payload_wrt_robot_in_world[i, j, :]

    return x_payload_wrt_robot_in_body


def make_vel_trajectory_relative(v, base_q):
    v = np.asarray(v)
    assert v.ndim == 3
    assert v.shape[0] == base_q.shape[0]
    assert v.shape[2] == 3

    N = v.shape[0]
    out = np.zeros_like(v)
    r_inv = R.from_quat(base_q).inv()  # (x,y,z,w)

    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            out[i, j, :] = r_inv[i].apply(v[i, j, :])

    return out


# Write a test for extract_future_trajectory
def test_extract_future_trajectory():
    x = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    future_horizon = 3
    expected_output = np.array(
        [
            [[1, 1], [2, 2], [3, 3]],
            [[1, 1], [2, 2], [3, 3]],
            [[1, 1], [2, 2], [3, 3]],
            [[1, 1], [2, 2], [3, 3]],
            [[4, 4], [4, 4], [4, 4]],
        ]
    )
    output = extract_future_trajectory(future_horizon, x, relative=True)
    assert np.array_equal(output, expected_output), f"Expected {expected_output}, but got {output}"

    print("test_extract_future_trajectory passed.")


# write a test for make_trajectory_relative
def test_make_trajectory_relative():
    x = np.array([[[2, 2, 2], [4, 5, 6], [7, 8, 9]], [[1, 1, 1], [5, 6, 7], [8, 9, 10]]])
    expected_output = np.array(
        [[[0, 0, 0], [3, 3, 3], [6, 6, 6]], [[0, 0, 0], [3, 3, 3], [6, 6, 6]]]
    )
    output = make_trajectory_relative(x)
    assert np.array_equal(output, expected_output), f"Expected {expected_output}, but got {output}"

    print("test_make_trajectory_relative passed.")


# ----------------------- Normalization stats I/O -----------------------
def _npz_file_path(filename: str = "normalization_stats.npz") -> str:
    return os.path.join(NORMALIZATION_STATS_DIR, filename)


def save_normalization_stats(
    dict_mean: dict, dict_std: dict, filename: str = "normalization_stats.npz"
):
    """
    Save normalization stats (means and stds) into a single NPZ file.
    Expects keys: {'depth'} ∪ required_state_keys ∪ required_future_keys.
    """
    required_state_keys = DEFAULT_NORMALIZATION_STATE_KEYS
    required_future_keys = DEFAULT_NORMALIZATION_FUTURE_KEYS

    required = {"depth"} | set(required_state_keys) | set(required_future_keys)
    missing_mean = required - set(dict_mean.keys())
    missing_std = required - set(dict_std.keys())
    if missing_mean:
        raise ValueError(f"mean dict missing keys: {sorted(missing_mean)}")
    if missing_std:
        raise ValueError(f"std dict missing keys: {sorted(missing_std)}")

    os.makedirs(NORMALIZATION_STATS_DIR, exist_ok=True)
    payload = {}
    for k in sorted(required):
        payload[f"mean__{k}"] = np.asarray(dict_mean[k])
        payload[f"std__{k}"] = np.asarray(dict_std[k])
    np.savez_compressed(_npz_file_path(filename), **payload)


def load_normalization_stats(filename: str = "normalization_stats.npz"):
    """
    Load normalization stats from NPZ and return (dict_mean, dict_std).
    """
    required_state_keys = DEFAULT_NORMALIZATION_STATE_KEYS
    required_future_keys = DEFAULT_NORMALIZATION_FUTURE_KEYS
    path = _npz_file_path(filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Normalization stats file not found: {path}")

    dict_mean, dict_std = {}, {}
    with np.load(path, allow_pickle=False) as data:
        for name in data.files:
            if name.startswith("mean__"):
                key = name[len("mean__") :]
                dict_mean[key] = data[name]
            elif name.startswith("std__"):
                key = name[len("std__") :]
                dict_std[key] = data[name]

    required = {"depth"} | set(required_state_keys) | set(required_future_keys)
    missing_mean = required - set(dict_mean.keys())
    missing_std = required - set(dict_std.keys())
    if missing_mean:
        raise ValueError(f"Loaded mean dict missing keys: {sorted(missing_mean)}")
    if missing_std:
        raise ValueError(f"Loaded std dict missing keys: {sorted(missing_std)}")

    return dict_mean, dict_std


def get_rotation_matrix_from_quat(quat):
    """
    Convert quaternion (x,y,z, w) to rotation matrix (3x3).
    Inout is shape (N,4)
    """
    mats = R.from_quat(quat).as_matrix()
    # extract first 2 columns of the rotation matrix
    return mats[:, :, :2]


def process_rotation_matrix(rot_mat):
    """
    Input is (N, 6) representing first two columns of rotation matrix.
    Return (N, 3, 3)
    """
    N = rot_mat.shape[0]
    if rot_mat.shape[1] != 6:
        raise ValueError(f"Expected rot_mat shape (N,6), got {rot_mat.shape}")

    rot_mat_33 = np.zeros((N, 3, 3))
    rot_mat_33[:, :, :2] = rot_mat.reshape(N, 3, 2)
    # compute third column as cross product of first two columns
    rot_mat_33[:, :, 2] = np.cross(rot_mat_33[:, :, 0], rot_mat_33[:, :, 1])
    return rot_mat_33
