import os
from pathlib import Path
from dataclasses import is_dataclass, asdict
import numpy as np
import argparse
import random
import yaml 

import zarr
from numcodecs import Blosc, VLenUTF8

from poly_fly.data_io.utils import (
    CSV_DIR,
    PARAMS_DIR,
    DEPTH_DIR,
    load,
    guess_yaml_from_csv,
    load_depth_data, load_params
)
from poly_fly.data_io.utils import extract_future_trajectory, make_trajectory_relative, extract_relative_payload_pos
from poly_fly.data_io.utils import ZARR_DIR
from poly_fly.data_io.enums import DatasetKeys as DK, AttrKeys as AK, GroupNames as GN

def build_dataset(csv_subdirectory, cfg, out_path= None, ):
    horizon = cfg["model"]["policy"]["horizon"]
    csv_root = Path(CSV_DIR) / str(csv_subdirectory)
    params_root = Path(PARAMS_DIR)
    depth_root = Path(DEPTH_DIR) / str(csv_subdirectory)
    assert csv_root.exists()

    if out_path is None:
        data_dir = Path(CSV_DIR).parent  # .../data
        out_path = data_dir / "zarr" / f"{csv_subdirectory}.zarr"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    root = zarr.group(store=zarr.DirectoryStore(str(out_path)), overwrite=True)

    csv_files = sorted(csv_root.rglob("*.csv"))
    assert csv_files

    tid = 0
    for csv_path in csv_files:
        stem = csv_path.stem
        # Map YAML
        yaml_path = guess_yaml_from_csv(csv_path, params_root)
        if not yaml_path.exists():
            continue 
            # Skip if params missing
            raise Exception(f"[skip] No YAML for {csv_path} (expected {yaml_path})")
        
        # Load trajectory
        data = load(str(csv_path), str(yaml_path))
        time = data[DK.TIME]
        sol_x = data[DK.SOL_X]
        sol_u = data[DK.SOL_U]
        sol_quad_x = data[DK.SOL_QUAD_X]
        sol_quad_quat = data[DK.SOL_QUAD_QUAT]
        sol_payload_rpy = data[DK.SOL_PAYLOAD_RPY]
        params = data[DK.PARAMS]
        rot_mat = data[DK.ROT_MAT]

        T = int(len(time))
        if T == 0:
            raise Exception(f"[skip] Empty trajectory in {csv_path}")

        # Build payload position (x,y,z)
        payload_pos = sol_x[:, 0:3].astype(np.float32, copy=False)
        payload_vel = sol_x[:, 3:6].astype(np.float32, copy=False)

        # Build robot state parts
        robot_px = sol_quad_x[:, 0:3]
        robot_v = sol_quad_x[:, 3:6]
        robot_q = sol_quad_quat  # (T,4) [w,x,y,z]
        robot_goal = np.asarray(params.end_state[:3])
        robot_rot_mat = rot_mat[:, :]

        # future trajectories 
        future_robot_pos_world = extract_future_trajectory(horizon, robot_px)
        future_payload_pos_world = extract_future_trajectory(horizon, payload_pos)
        future_quaternion = extract_future_trajectory(horizon, sol_quad_quat[:, :])
        future_rot_mat = extract_future_trajectory(horizon, rot_mat[:, :])  # (T, 6)
        future_payload_pos = extract_relative_payload_pos(future_robot_pos_world, future_payload_pos_world, future_quaternion)
        future_robot_pos = make_trajectory_relative(future_robot_pos_world, robot_px)
        future_robot_vel = extract_future_trajectory(horizon, sol_quad_x[:, 3:6])
        future_payload_vel = extract_future_trajectory(horizon, sol_x[:, 3:6])

        robot_goal_relative = robot_goal - robot_px 
        u = sol_u.astype(np.float32, copy=False)

        assert future_robot_pos.shape[0] == T
        assert future_payload_pos.shape[0] == T
        assert future_quaternion.shape[0] == T
        assert future_robot_vel.shape[0] == T
        assert future_payload_vel.shape[0] == T
        assert future_rot_mat.shape[0] == T

        depth_path = depth_root / f"{stem}.npz"
        assert depth_path.exists()
        depth_time, depth = load_depth_data(depth_path)
        depth_time = np.asarray(depth_time, dtype=float).reshape(-1)
        time_f = np.asarray(time, dtype=float).reshape(-1)
        # Assert exact time correspondence and proceed using the same indices
        assert depth_time.shape[0] == time_f.shape[0], (
            f"Depth and traj length mismatch for {stem}: "
            f"{depth_time.shape[0]} vs {time_f.shape[0]}"
        )
        assert np.allclose(depth_time, time_f, rtol=0.0, atol=1e-9), (
            f"Depth timestamps do not match trajectory for {stem}"
        )
        depth_aligned = depth.astype(np.float32, copy=False)

        # Create group and datasets
        g = root.require_group(f"{GN.TRAJS}/{stem}")
        g.attrs.update({
            AK.T: T,
            AK.STEM: stem,
            AK.YAML_FILE_NAME: Path(yaml_path).name
            # TODO Can I add params here?
        })

        # Save full YAML path as a scalar UTF-8 string dataset
        yaml_path_str = str(Path(yaml_path).resolve())
        if DK.YAML_FILE in g:
            del g[DK.YAML_FILE]
        g.create_dataset(
            DK.YAML_FILE,
            data=yaml_path_str,
            dtype=object,
            object_codec=VLenUTF8()
        )

        # Datasets with chunking
        g.create_dataset(DK.TIME, data=time.astype(np.float32, copy=False), chunks=(min(4096, T),), compressor=compressor)
        g.create_dataset(DK.PAYLOAD_POS, data=payload_pos, chunks=(min(1024, T), 3), compressor=compressor)
        g.create_dataset(DK.PAYLOAD_VEL, data=payload_vel, chunks=(min(1024, T), 3), compressor=compressor)
        g.create_dataset(DK.ROBOT_POS, data=robot_px.astype(np.float32, copy=False), chunks=(min(1024, T), 3), compressor=compressor)
        g.create_dataset(DK.ROBOT_VEL, data=robot_v.astype(np.float32, copy=False), chunks=(min(1024, T), 3), compressor=compressor)
        g.create_dataset(DK.ROBOT_QUAT, data=robot_q.astype(np.float32, copy=False), chunks=(min(1024, T), 4), compressor=compressor)
        g.create_dataset(DK.ROT_MAT, data=robot_rot_mat.astype(np.float32, copy=False), chunks=(min(256, T), 6), compressor=compressor)
        g.create_dataset(DK.U, data=u, chunks=(min(1024, T), 3), compressor=compressor)
        g.create_dataset(DK.ROBOT_GOAL, data=robot_goal.astype(np.float32, copy=False), compressor=compressor)
        g.create_dataset(DK.ROBOT_GOAL_RELATIVE, data=robot_goal_relative.astype(np.float32, copy=False), compressor=compressor)

        # Save future trajectories
        r_chunks = (min(256, T),) + future_robot_pos.shape[1:]
        p_chunks = (min(256, T),) + future_payload_pos.shape[1:]
        q_chunks = (min(256, T),) + future_quaternion.shape[1:]
        rot_mat_chunks = (min(256, T),) + future_rot_mat.shape[1:]
        depth_chunks = (min(64, T),) + depth_aligned.shape[1:]

        g.create_dataset(DK.FUTURE_ROBOT_POS, data=future_robot_pos.astype(np.float32, copy=False), chunks=r_chunks, compressor=compressor)
        g.create_dataset(DK.FUTURE_ROBOT_POS_WORLD, data=future_robot_pos_world.astype(np.float32, copy=False), chunks=r_chunks, compressor=compressor)
        g.create_dataset(DK.FUTURE_PAYLOAD_POS, data=future_payload_pos.astype(np.float32, copy=False), chunks=p_chunks, compressor=compressor)
        g.create_dataset(DK.FUTURE_PAYLOAD_POS_WORLD, data=future_payload_pos_world.astype(np.float32, copy=False), chunks=p_chunks, compressor=compressor)
        g.create_dataset(DK.FUTURE_QUATERNION, data=future_quaternion.astype(np.float32, copy=False), chunks=q_chunks, compressor=compressor)
        g.create_dataset(DK.FUTURE_ROT_MAT, data=future_rot_mat.astype(np.float32, copy=False), chunks=rot_mat_chunks, compressor=compressor)
        g.create_dataset(DK.FUTURE_ROBOT_VEL, data=future_robot_vel.astype(np.float32, copy=False), chunks=r_chunks, compressor=compressor)
        g.create_dataset(DK.FUTURE_PAYLOAD_VEL, data=future_payload_vel.astype(np.float32, copy=False), chunks=p_chunks, compressor=compressor)
        g.create_dataset(DK.DEPTH, data=depth_aligned.astype(np.float32, copy=False), chunks=depth_chunks, compressor=compressor)

        print(f"[zarr] wrote traj {stem}, T={T}")
        tid += 1

    root.attrs.update({AK.NUM_TRAJS: tid, AK.CSV_SUBDIRECTORY: csv_subdirectory})
    print(f"[zarr] dataset complete: {out_path} (num_trajs={tid})")
    return str(out_path), params

def _load_traj_from_group(g):
    """Load a single trajectory tuple from a Zarr group g."""
    time = np.asarray(g[DK.TIME][:], dtype=np.float32)
    payload = np.asarray(g[DK.PAYLOAD_POS][:], dtype=np.float32)
    payload_v = np.asarray(g[DK.PAYLOAD_VEL][:], dtype=np.float32)
    robot_pos = np.asarray(g[DK.ROBOT_POS][:], dtype=np.float32)
    robot_vel = np.asarray(g[DK.ROBOT_VEL][:], dtype=np.float32)
    robot_quat = np.asarray(g[DK.ROBOT_QUAT][:], dtype=np.float32)
    robot_rot_mat = np.asarray(g[DK.ROT_MAT][:], dtype=np.float32)
    sol_u = np.asarray(g[DK.U][:], dtype=np.float32)
    
    T = int(time.shape[0])
    sol_x = np.zeros((T, 9), dtype=np.float32)
    sol_x[:, 0:3] = payload
    sol_x[:, 3:6] = payload_v

    sol_quad_x = np.zeros((T, 9), dtype=np.float32)
    sol_quad_x[:, 0:3] = robot_pos
    sol_quad_x[:, 3:6] = robot_vel

    sol_payload_rpy = np.zeros((T, 3), dtype=np.float32)

    yaml_file_val = g[DK.YAML_FILE][()]
    if isinstance(yaml_file_val, bytes):
        yaml_file_val = yaml_file_val.decode("utf-8")
    params = load_params(str(yaml_file_val))

    future_robot_pos = np.asarray(g[DK.FUTURE_ROBOT_POS][:], dtype=np.float32)
    future_robot_pos_world = np.asarray(g[DK.FUTURE_ROBOT_POS_WORLD][:], dtype=np.float32)
    future_payload_pos = np.asarray(g[DK.FUTURE_PAYLOAD_POS][:], dtype=np.float32)
    future_payload_pos_world = np.asarray(g[DK.FUTURE_PAYLOAD_POS_WORLD][:], dtype=np.float32)
    future_quaternion = np.asarray(g[DK.FUTURE_QUATERNION][:], dtype=np.float32)
    future_rot_mat = np.asarray(g[DK.FUTURE_ROT_MAT][:], dtype=np.float32)
    future_robot_vel = np.asarray(g[DK.FUTURE_ROBOT_VEL][:], dtype=np.float32)
    future_payload_vel = np.asarray(g[DK.FUTURE_PAYLOAD_VEL][:], dtype=np.float32)

    # Return a dictionary instead of a tuple
    return {
        DK.TIME: time,
        DK.SOL_X: sol_x,
        DK.SOL_U: sol_u,
        DK.SOL_QUAD_X: sol_quad_x,
        DK.ROBOT_QUAT: robot_quat.astype(np.float32, copy=False),
        DK.ROT_MAT: robot_rot_mat.astype(np.float32, copy=False),
        DK.SOL_PAYLOAD_RPY: sol_payload_rpy,
        DK.PARAMS: params,
        DK.FUTURE_ROBOT_POS: future_robot_pos,
        DK.FUTURE_ROBOT_POS_WORLD: future_robot_pos_world,
        DK.FUTURE_PAYLOAD_POS: future_payload_pos,
        DK.FUTURE_PAYLOAD_POS_WORLD: future_payload_pos_world,
        DK.FUTURE_QUATERNION: future_quaternion,
        DK.FUTURE_ROT_MAT: future_rot_mat.astype(np.float32, copy=False),
        DK.FUTURE_ROBOT_VEL: future_robot_vel,
        DK.FUTURE_PAYLOAD_VEL: future_payload_vel,
    }

def _load_dataset(zarr_path, limit=None):
    # Open Zarr
    root = zarr.open_group(store=zarr.DirectoryStore(str(zarr_path)), mode="r")
    trajs_grp = root[GN.TRAJS]

    # Determine the list of trajectory keys to load (avoid zarr-specific filters)
    all_keys = sorted(trajs_grp.group_keys())
    if limit is not None:
        keys = all_keys[:limit]
    else:
        keys = all_keys

    # Return list of dictionaries instead of tuples
    return [_load_traj_from_group(trajs_grp[k]) for k in keys], trajs_grp, keys

def load_zarr_folder(zarr_path: Path, args, limit=None, random_sample=False):
    """
    Load every trajectory in a Zarr dataset and package for the multi-view inspector.
    Returns list of dicts: { 'name', 'params', 'time', 'sol_x', 'sol_u', 'quad_pos', 'path',
                             optional: 'depth_time', 'depth', 'future_robot_pos', 'future_payload_pos', 'future_quaternion' }

    If args.lazy_depth_only or args.lazy is True:
      - Avoid materializing arrays; attach zarr array handles (e.g., 'depth') for on-demand reading.
      - Only minimal fields are populated (name, path, depth).
    """
    print(f"loading Zarr dataset from {zarr_path}")
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr dataset not found: {zarr_path}")

    lazy = bool(getattr(args, "lazy_depth_only", False) or getattr(args, "lazy", False))
    if lazy:
        root = zarr.open_group(store=zarr.DirectoryStore(str(zarr_path)), mode="r")
        trajs_grp = root[GN.TRAJS]

        all_keys = sorted(trajs_grp.group_keys())
        keys = all_keys if limit is None else all_keys[:limit]
        if random_sample:
           keys = random.sample(keys, k=len(keys))

        print(f"building dataset (lazy) with {len(keys)} trajectories...")
        datasets = []
        for key in keys:
            g = trajs_grp[key]
            name = g.attrs.get(AK.STEM, key)
            yaml_val = g[DK.YAML_FILE][()]
            if isinstance(yaml_val, bytes):
                yaml_val = yaml_val.decode("utf-8")

            ds = {
                DK.NAME: name,
                DK.PATH: zarr_path,
                DK.TIME: g[DK.TIME],
                DK.PAYLOAD_POS: g[DK.PAYLOAD_POS],
                DK.PAYLOAD_VEL: g[DK.PAYLOAD_VEL],
                DK.ROBOT_POS: g[DK.ROBOT_POS],
                DK.ROBOT_VEL: g[DK.ROBOT_VEL],
                DK.ROBOT_QUAT: g[DK.ROBOT_QUAT],
                DK.ROT_MAT: g[DK.ROT_MAT],
                DK.U: g[DK.U],
                DK.FUTURE_ROBOT_POS: g[DK.FUTURE_ROBOT_POS],
                DK.FUTURE_ROBOT_POS_WORLD: g[DK.FUTURE_ROBOT_POS_WORLD],
                DK.FUTURE_PAYLOAD_POS: g[DK.FUTURE_PAYLOAD_POS],
                DK.FUTURE_PAYLOAD_POS_WORLD: g[DK.FUTURE_PAYLOAD_POS_WORLD],
                DK.FUTURE_QUATERNION: g[DK.FUTURE_QUATERNION],
                DK.FUTURE_ROT_MAT: g[DK.FUTURE_ROT_MAT],
                DK.FUTURE_ROBOT_VEL: g[DK.FUTURE_ROBOT_VEL],
                DK.FUTURE_PAYLOAD_VEL: g[DK.FUTURE_PAYLOAD_VEL],
                DK.DEPTH: g[DK.DEPTH],
                DK.YAML_FILE: yaml_val,
                DK.ROBOT_GOAL: g[DK.ROBOT_GOAL],
                DK.ROBOT_GOAL_RELATIVE: g[DK.ROBOT_GOAL_RELATIVE],
            }

            datasets.append(ds)
        return datasets

    res, trajs_grp, keys = _load_dataset(str(zarr_path), limit=limit)
    idxs = list(range(len(keys)))
    if getattr(args, "combined", False) and limit is not None and len(idxs) > limit:
        total = len(idxs)
        idxs = sorted(random.sample(idxs, k=limit))

    print(f"building dataset with {len(idxs)} trajectories...")
    datasets = []
    for i in idxs:
        key = keys[i]
        g = trajs_grp[key]
        name = g.attrs.get(AK.STEM, key)
        traj = res[i]
        quad_pos = traj[DK.SOL_QUAD_X][:, :3]

        ds = dict()
        ds[DK.NAME] = name
        ds[DK.PARAMS] = traj[DK.PARAMS]
        ds[DK.SOL_X] = traj[DK.SOL_X]
        ds[DK.SOL_U] = traj[DK.SOL_U]
        ds[DK.ROBOT_GOAL] = np.asarray(g[DK.ROBOT_GOAL][:3], dtype=np.float32)
        ds[DK.PAYLOAD_POS] = traj[DK.SOL_X][:, :3]
        ds[DK.PAYLOAD_VEL] = traj[DK.SOL_X][:, 3:6]
        ds[DK.ROBOT_POS] = quad_pos
        ds[DK.ROBOT_VEL] = traj[DK.SOL_QUAD_X][:, 3:6]
        ds[DK.ROBOT_QUAT] = traj[DK.ROBOT_QUAT]
        ds[DK.ROT_MAT] = traj[DK.ROT_MAT]
        ds[DK.DEPTH] = np.asarray(g[DK.DEPTH][:])
        ds[DK.DEPTH_TIME] = np.asarray(g[DK.TIME][:], dtype=float)
        ds[DK.FUTURE_ROBOT_POS] = traj[DK.FUTURE_ROBOT_POS]
        ds[DK.FUTURE_ROBOT_POS_WORLD] = traj[DK.FUTURE_ROBOT_POS_WORLD]
        ds[DK.FUTURE_PAYLOAD_POS] = traj[DK.FUTURE_PAYLOAD_POS]
        ds[DK.FUTURE_PAYLOAD_POS_WORLD] = traj[DK.FUTURE_PAYLOAD_POS_WORLD]
        ds[DK.FUTURE_QUATERNION] = traj[DK.FUTURE_QUATERNION]
        ds[DK.FUTURE_ROT_MAT] = traj[DK.FUTURE_ROT_MAT]
        ds[DK.FUTURE_ROBOT_VEL] = traj[DK.FUTURE_ROBOT_VEL]
        ds[DK.FUTURE_PAYLOAD_VEL] = traj[DK.FUTURE_PAYLOAD_VEL]
        ds[DK.ROBOT_GOAL_RELATIVE] = ds[DK.ROBOT_GOAL] - ds[DK.ROBOT_POS]
        ds[DK.YAML_FILE_NAME] = name
        datasets.append(ds)

    return datasets


if __name__ == "__main__":
    # CLI: load a dataset by subdirectory
    parser = argparse.ArgumentParser(description="Load a Zarr dataset by subdirectory.")
    parser.add_argument("--dir", "-d", dest="subdir", required=True, help="Subdirectory name under ZARR_DIR")
    parser.add_argument("--policy-config", type=str, required=True, help="Path to Policy YAML config (required)")
    args = parser.parse_args()

    subdir = args.subdir
    with open(args.policy_config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    build_dataset(subdir, cfg)
