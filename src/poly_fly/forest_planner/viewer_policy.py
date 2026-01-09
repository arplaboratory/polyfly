import matplotlib.image
import numpy as np
import random
import time
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
from PIL import Image
import matplotlib
import torch
import random
import matplotlib.pyplot as plt   # <-- add
from aerial_gym.utils.math import quat_from_euler_xyz_tensor  # if you want to use Euler -> quat
from isaacgym import gymtorch  # add (for pushing DOF states)
from poly_fly.data_io.utils import load_obstacle_info_from_yaml, load_csv, find_csvs, BASE_DIR, PARAMS_DIR, \
    GIFS_DIR, CSV_DIR, IMG_DIR
from scipy.spatial.transform import Rotation as Rot
import os
from pathlib import Path
# Add missing helpers from IO
from poly_fly.data_io.utils import load_all_csvs_and_params, save_depth_data, load_all_csvs_and_params_names, DEPTH_SCALE_FACTOR
from poly_fly.forest_planner.forest_params import ForestParamsLarge, ForestParamsSmall
from poly_fly.deep_poly_fly.model.policy import load_model_from_checkpoint
from poly_fly.controller.mpc import MPC 
# ADD: import normalization stats loader
from poly_fly.data_io.utils import load_normalization_stats
from poly_fly.data_io.enums import DatasetKeys

SAMPLING_STRIDE = 1 # do not change 

class MotionCollector:
    def __init__(self, num_envs: int = 2, device: str = "cuda:0", seed: int = 0,
        headless: bool = False, use_warp: bool = True, viz_depth: bool = True):
        self.num_envs = num_envs
        self.device = device
        self.seed = seed
        self.headless = not viz_depth
        self.use_warp = use_warp
        self.viz_depth = viz_depth
        # Defer env creation until YAML inspection
        self.env_manager = None
        self._selected_env_name = None
        # NEW: policy model handle & step counter
        self.policy_model = None
        self._policy_step = 0
        self.mpc = MPC()
        # seed RNGs
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)

        # interactive plotting if visualizing depth
        if self.viz_depth:
            plt.ion()

    # ----- methods migrated from top-level helpers -----
    def move_asset(self, env_id: int, asset_id: int, new_pos, new_quat=None):
        state_tensor = self.env_manager.global_tensor_dict["env_asset_state_tensor"]
        state_tensor[env_id, asset_id, 0:3] = torch.as_tensor(new_pos, device=state_tensor.device)
        if new_quat is not None:
            state_tensor[env_id, asset_id, 3:7] = torch.as_tensor(new_quat, device=state_tensor.device)
        state_tensor[env_id, asset_id, 7:13] = 0.0
        self.env_manager.IGE_env.write_to_sim()
        if self.env_manager.use_warp:
            self.env_manager.warp_env.reset_idx(torch.as_tensor([env_id], device=state_tensor.device))

    def set_cubes(self, env_id: int, positions, quats=None):
        state = self.env_manager.global_tensor_dict["env_asset_state_tensor"]
        device = state.device
        assert len(positions) == 3, "Expecting 3 cube positions"
        for cube_idx, pos in enumerate(positions):
            state[env_id, cube_idx, 0:3] = torch.as_tensor(pos, device=device)
            if quats is not None:
                state[env_id, cube_idx, 3:7] = torch.as_tensor(quats[cube_idx], device=device)
            state[env_id, cube_idx, 7:13] = 0.0
        self.env_manager.IGE_env.write_to_sim()
        if self.env_manager.use_warp:
            self.env_manager.warp_env.reset_idx(torch.as_tensor([env_id], device=device))

    def set_obstacles_from_yaml(self, env_id: int, yaml_path: str):
        obstacle_info = load_obstacle_info_from_yaml(yaml_path)
        state_tensor = self.env_manager.global_tensor_dict["env_asset_state_tensor"]
        num_assets_total = state_tensor.shape[1]
        asset_dicts = self.env_manager.global_asset_dicts[env_id]

        logger.info(f"YAML obstacles: {len(obstacle_info)} | Env assets: {num_assets_total}")
        assert len(obstacle_info) == num_assets_total -1
        
        # The first asset is the floor
        start_asset_id = 1
        max_placeable = max(0, num_assets_total - start_asset_id)
        if len(obstacle_info) > max_placeable:
            logger.warning(
                f"Too many obstacles ({len(obstacle_info)}) for available asset slots ({max_placeable}). "
                f"Only the first {max_placeable} will be used."
            )

        placed = []
        for i, obs in enumerate(obstacle_info):
            if i >= max_placeable:
                break
            asset_id = start_asset_id + i
            pos = [obs["x"], obs["y"], obs["z"]]

            if asset_id < len(asset_dicts):
                ad = asset_dicts[asset_id]
                isaac_asset = ad.get("isaacgym_asset", None)
                if isaac_asset is not None and hasattr(isaac_asset, "name"):
                    base_name = isaac_asset.name
                else:
                    base_name = ad.get("name", f"asset_{asset_id}")
            else:
                base_name = f"asset_{asset_id}"

            logger.debug(f"Moving asset {asset_id} ('{base_name}') to position {pos}")
            self.move_asset(env_id=env_id, asset_id=asset_id, new_pos=pos)
            placed.append({"asset_id": asset_id, "name": base_name, "pos": pos})
        return placed

    def print_env_asset_index_map(self):
        state = self.env_manager.global_tensor_dict["env_asset_state_tensor"]
        num_assets = state.shape[1]
        print(f"Total env assets per env: {num_assets}")
        am = self.env_manager.asset_manager
        printed = False
        cand_attrs = [
            "env_asset_specs", "asset_specs", "ordered_asset_specs",
            "env_asset_name_list", "asset_name_list"
        ]
        for attr in cand_attrs:
            if hasattr(am, attr):
                specs = getattr(am, attr)
                print(f"\nFrom asset_manager.{attr}:")
                for idx, s in enumerate(specs):
                    name = getattr(s, "name", s)
                    count = getattr(s, "num_assets", None)
                    if count is not None:
                        pass
                printed = True
        print("\nInitial positions (env 0):")
        pos0 = state[0, :, 0:3].cpu()
        for idx in range(num_assets):
            print(f"AssetIdx {idx}: pos {pos0[idx].tolist()}")

    def set_robot_pose(self, env_ids, positions, quat_xyzw_list=None, euler_rpy_list=None,
                       zero_vel=True, dof_positions_list=None, dof_velocities_list=None):
        gtd = self.env_manager.global_tensor_dict
        state = gtd["robot_state_tensor"]
        env_ids = list(env_ids) if isinstance(env_ids, (list, tuple, range)) else [int(env_ids)]
        positions = list(positions)
        if quat_xyzw_list is None and euler_rpy_list is not None:
            eulers = torch.as_tensor(euler_rpy_list, device=self.env_manager.device, dtype=torch.float32)
            quats = quat_from_euler_xyz_tensor(eulers)
            quat_xyzw_list = [quats[i] for i in range(quats.shape[0])]
        elif quat_xyzw_list is None:
            quat_xyzw_list = [torch.tensor([0, 0, 0, 1.0], device=self.env_manager.device, dtype=torch.float32)] * len(env_ids)

        for k, env_id in enumerate(env_ids):
            pos_k = torch.as_tensor(positions[k], device=state.device, dtype=state.dtype)
            quat_k = torch.as_tensor(quat_xyzw_list[k], device=state.device, dtype=state.dtype)
            state[env_id, 0:3] = pos_k
            state[env_id, 3:7] = quat_k
            if zero_vel:
                state[env_id, 7:13] = 0.0

        if "dof_state_tensor" in gtd and gtd["dof_state_tensor"] is not None and dof_positions_list is not None:
            dof_state = gtd["dof_state_tensor"]
            num_dofs = dof_state.shape[1]
            for k, env_id in enumerate(env_ids):
                q_list = dof_positions_list[k] if dof_positions_list is not None else None
                if q_list is None:
                    continue
                n_set = min(len(q_list), num_dofs)
                dof_state[env_id, :n_set, 0] = torch.as_tensor(q_list[:n_set],
                                                               device=dof_state.device,
                                                               dtype=dof_state.dtype)
                if dof_velocities_list is not None and dof_velocities_list[k] is not None:
                    v_list = dof_velocities_list[k]
                    dof_state[env_id, :n_set, 1] = torch.as_tensor(v_list[:n_set],
                                                                   device=dof_state.device,
                                                                   dtype=dof_state.dtype)
                elif zero_vel:
                    dof_state[env_id, :n_set, 1] = 0.0
            unfolded = gtd["unfolded_dof_state_tensor"]
            self.env_manager.IGE_env.gym.set_dof_state_tensor(
                self.env_manager.IGE_env.sim,
                gymtorch.unwrap_tensor(unfolded),
            )

        self.env_manager.IGE_env.write_to_sim()
        self.env_manager.robot_manager.robot.update_states()

    def check_asset_names(self, env_id: int = 0, expected_first: str = "floor", verbose: bool = True):
        gym = self.env_manager.IGE_env.gym
        env_handle = self.env_manager.IGE_env.env_handles[env_id]
        handles = self.env_manager.IGE_env.asset_handles[env_id]
        asset_dicts = self.env_manager.global_asset_dicts[env_id]

        if verbose:
            print(f"[DEBUG] num actor handles={len(handles)}, num asset dict entries={len(asset_dicts)}")

        names = []
        for i, asset_handle in enumerate(handles):
            actor_name = gym.get_actor_name(env_handle, asset_handle)
            if i < len(asset_dicts):
                ad = asset_dicts[i]
                isaac_asset = ad.get("isaacgym_asset", None)
                if isaac_asset is not None and hasattr(isaac_asset, "name"):
                    base_name = isaac_asset.name
                else:
                    base_name = ad.get("name", "unknown")
            else:
                base_name = "<no asset_dict entry>"
            names.append(base_name)
            if verbose:
                print(i, actor_name, base_name)

        if names:
            first = names[0]
            if first != expected_first:
                raise RuntimeError(f"First asset name '{first}' != expected '{expected_first}'")
        return names

    @staticmethod
    def adjust_yaw(curr_yaw, prev_yaw, zero_thr=0.05, pi_thr=0.05):
        adjustment = 0
        if abs(curr_yaw - prev_yaw) > 2.5:
            adjustment = -np.pi if curr_yaw > 0 else np.pi
        curr_yaw = curr_yaw + adjustment
        return curr_yaw, adjustment

    def _infer_env_name_from_yaml(self, yaml_paths):
        """
        Inspect the first obstacle's half-length (xL/xl/l) in each YAML.
        If it matches ForestParamsLarge.large_size_l_range[0] -> 'obs_large_env'
        If it matches ForestParamsSmall.small_size_l_range[0] -> 'obs_small_env'
        All YAMLs in the batch must agree.
        """
        large_l = ForestParamsLarge().large_size_l_range[0]
        large_b = ForestParamsLarge().large_size_b_range[0]
        small_l = ForestParamsSmall().small_size_l_range[0]
        small_b = ForestParamsSmall().small_size_b_range[0]

        inferred = []
        for yp in yaml_paths:
            obs = load_obstacle_info_from_yaml(yp)
            if obs.shape[0] == 0:
                raise Exception(f"No obstacles in YAML {yp}")

            # Get first obstacle xL (fallback to xl/l if needed)
            xL = float(obs[0]["xL"])
            yL = float(obs[0]["yL"])
           
            if np.isclose(xL, large_l, atol=1e-6) and np.isclose(yL, large_b, atol=1e-6):
                inferred.append("obs_large_env")
            elif np.isclose(xL, small_l, atol=1e-6) and np.isclose(yL, small_b, atol=1e-6):
                inferred.append("obs_small_env")
            else:
                # unknown size, default to large with a warning
                raise Exception(f"Unrecognized obstacle size xL={xL} in {yp}; defaulting to large.")

        # ensure all match
        if len(set(inferred)) != 1:
            raise RuntimeError(f"Inconsistent forest types in batch: {inferred}")
        return inferred[0]

    def _destroy_env(self):
        """
        Best-effort cleanup of the current sim/viewer so a new env can be created
        in the same process without tripping PhysX/IsaacGym singletons.
        """
        try:
            if self.env_manager is None:
                return
            # Isaac Gym handles
            ige = getattr(self.env_manager, "IGE_env", None)
            if ige is not None and hasattr(ige, "gym"):
                gym = ige.gym
                # destroy viewer if present
                try:
                    viewer = getattr(ige, "viewer", None)
                    gym.destroy_viewer(viewer)
                except Exception:
                    pass
                # destroy sim if present
                try:
                    sim = getattr(ige, "sim", None)
                    if sim is not None:
                        gym.destroy_sim(sim)
                except Exception:
                    pass
        finally:
            self.env_manager = None

    def _ensure_env(self, env_name: str):
        """
        Build env if none, or rebuild if env_name changed (after clean destroy).
        """
        # If same env already built, reuse it
        if self.env_manager is not None and self._selected_env_name == env_name:
            return

        # If switching env types, destroy the previous sim cleanly
        if self.env_manager is not None and self._selected_env_name != env_name:
            self._destroy_env()

        self._selected_env_name = env_name
        self.env_manager = SimBuilder().build_env(
            sim_name="base_sim",
            env_name=env_name,
            robot_name="base_quadrotor_with_stereo_camera",
            controller_name="null_position_control",
            args=None,
            device=self.device,
            num_envs=self.num_envs,
            headless=self.headless,
            use_warp=self.use_warp,
        )

    def get_depth_for_csvs(self, subdirectory):
        step_stride = SAMPLING_STRIDE
        loaded = load_all_csvs_and_params(subdirectory)
        stems = sorted(loaded.keys())
        if not stems:
            logger.warning(f"No CSVs found under subdirectory '{subdirectory}'.")
            return

        # Build a list of (stem, yaml_path, csv_path, inferred_env)
        items = []
        for stem in stems:
            yaml_path = str(Path(PARAMS_DIR) / subdirectory / f"{stem}.yaml")
            csv_path = str(Path(CSV_DIR) / subdirectory / f"{stem}.csv")
            env_name = self._infer_env_name_from_yaml([yaml_path])  # 'obs_large_env' or 'obs_small_env'
            items.append((stem, yaml_path, csv_path, env_name))

        # Group by forest type (environment name)
        groups = {}
        for stem, yp, cp, env_name in items:
            groups.setdefault(env_name, []).append((stem, yp, cp))

        # Process each forest type separately so each run() sees a uniform type
        for env_name, group_items in groups.items():
            logger.info(f"\nProcessing {len(group_items)} trajectories for env '{env_name}'")
            logger.info(f"{group_items}")

            for i in range(0, len(group_items), self.num_envs):
                batch = group_items[i : i + self.num_envs]
                batch_stems = [b[0] for b in batch]
                yaml_paths = [b[1] for b in batch]
                csv_paths = [b[2] for b in batch]

                # Pad to num_envs by duplicating last if needed (run() expects equal lengths)
                if len(yaml_paths) < self.num_envs and yaml_paths:
                    yaml_paths += [yaml_paths[-1]] * (self.num_envs - len(yaml_paths))
                    csv_paths += [csv_paths[-1]] * (self.num_envs - len(csv_paths))

                # Run simulation and collect depth
                self.run(yaml_file_paths=yaml_paths, csv_file_paths=csv_paths,
                                       step_stride=step_stride, max_steps=10_000)


    def _process_items_batch(self, items, step_stride: int, subdirectory: str):
        """
        items: list of tuples (stem, yaml_path, csv_path)
        Infer env per item within this batch, group by env, run in mini-batches of self.num_envs, and save depth.
        """
        # Infer env for this batch only, then group
        groups = {}
        for stem, yp, cp in items:
            env_name = self._infer_env_name_from_yaml([str(yp)])
            groups.setdefault(env_name, []).append((stem, yp, cp))

        for env_name, group_items in groups.items():
            logger.info(f"\nProcessing {len(group_items)} trajectories for env '{env_name}'")
            logger.info(f"{group_items}")

            for i in range(0, len(group_items), self.num_envs):
                batch = group_items[i: i + self.num_envs]
                batch_stems = [b[0] for b in batch]
                batch_yaml_paths = [str(b[1]) for b in batch]
                batch_csv_paths = [str(b[2]) for b in batch]

                # Pad to num_envs by duplicating last if needed (run() expects equal lengths)
                yaml_paths = list(batch_yaml_paths)
                csv_paths = list(batch_csv_paths)
                if len(yaml_paths) < self.num_envs and yaml_paths:
                    pad_n = self.num_envs - len(yaml_paths)
                    yaml_paths += [yaml_paths[-1]] * pad_n
                    csv_paths += [csv_paths[-1]] * pad_n

                # Run simulation and collect depth
                self.run(
                    yaml_file_paths=yaml_paths,
                    csv_file_paths=csv_paths,
                    step_stride=step_stride,
                    max_steps=10_000,
                )

                # # Save only the non-padded entries
                # for env_idx, stem in enumerate(batch_stems):
                #     frames = depth_lists[env_idx] if env_idx < len(depth_lists) else []
                #     if not frames:
                #         logger.warning(f"No depth frames captured for '{stem}'. Skipping save.")
                #         continue
                #     depth_arr = np.stack(frames, axis=0)  # (T, H, W)

                #     # Load times from the corresponding CSV file
                #     data_row = load_csv(batch_csv_paths[env_idx])
                #     time_full = data_row[DatasetKeys.TIME]
                #     max_T = min(depth_arr.shape[0], int(np.ceil(len(time_full) / step_stride)))
                #     times_sub = np.asarray(time_full)[0: max_T * step_stride: step_stride]

                #     assert times_sub.shape[0] == depth_arr.shape[0], "You have undersampled the trajectory!"
                #     save_depth_data(stem, times_sub, depth_arr, subdirectory=subdirectory)

    def get_depth_for_csvs_sequential(self, subdirectory):
        step_stride = SAMPLING_STRIDE

        # 1) Get names only (csvs and yamls of the same length)
        csv_files, yaml_files = load_all_csvs_and_params_names(subdirectory)
        # select 10 random files from the extracted csv_files and yaml_files
        random.seed(123)
        # indices = random.sample(range(len(csv_files)), min(10, len(csv_files)))
        # csv_files = [csv_files[i] for i in indices]
        # yaml_files = [yaml_files[i] for i in indices]


        # csv_files = []
        # yaml_files = []        

        # for _ in range(10):
        #     csv_files.append("/home/mrunal/Documents/poly_fly/data/csvs/forests/forest_001_f0_s4106135923.csv")
        #     yaml_files.append("/home/mrunal/Documents/poly_fly/data/params/forests/forest_001_f0_s4106135923.yaml")
        # csv_files.append("/home/mrunal/Documents/poly_fly/data/csvs/forests/forest_030_f0_s3160967032.csv")
        # yaml_files.append("/home/mrunal/Documents/poly_fly/data/params/forests/forest_030_f0_s3160967032.yaml")
        if not csv_files:
            logger.warning(f"No CSVs found under subdirectory '{subdirectory}'.")
            return

        # Build list of (stem, yaml_path, csv_path) — do NOT infer env here
        items = []
        for csv_path, yaml_path in zip(csv_files, yaml_files):
            stem = Path(csv_path).stem
            items.append((stem, str(yaml_path), str(csv_path)))

        # 2) Process in batches (infer env per batch internally)
        BATCH_SIZE = 1
        for i in range(0, len(items), BATCH_SIZE):
            batch_items = items[i: i + BATCH_SIZE]
            self._process_items_batch(batch_items, step_stride, subdirectory)

    def get_next_desired_states(self, payload_pos, payload_vel, robot_pos, robot_vel,
                                quat_xyzw, robot_goal, depth_image):
        """
        Compute next desired states (mode 0) from current state + depth.
        Returns (desired_robot_positions, desired_payload_positions, desired_robot_quats, desired_robot_vel, desired_payload_vel)
        Shapes: (H,3), (H,3), (H,4), (H,3), (H,3)
        """
        if self.policy_model is None:
            raise RuntimeError("policy_model is not set. Load/assign a model before calling get_next_desired_states().")

        # --- Load & cache normalization stats (once) ---
        if not hasattr(self, "_norm_cache") or self._norm_cache is None:
            dict_mean, dict_std = load_normalization_stats()
            state_mean = np.concatenate(
                [
                    np.asarray(dict_mean[DatasetKeys.PAYLOAD_VEL]).reshape(-1),
                    np.asarray(dict_mean[DatasetKeys.ROBOT_VEL]).reshape(-1),
                    np.asarray(dict_mean[DatasetKeys.ROBOT_GOAL_RELATIVE]).reshape(-1),
                    np.asarray(dict_mean[DatasetKeys.ROBOT_QUAT]).reshape(-1),
                ],
                axis=0,
            ).astype(np.float32)
            state_std = np.concatenate(
                [
                    np.asarray(dict_std[DatasetKeys.PAYLOAD_VEL]).reshape(-1),
                    np.asarray(dict_std[DatasetKeys.ROBOT_VEL]).reshape(-1),
                    np.asarray(dict_std[DatasetKeys.ROBOT_GOAL_RELATIVE]).reshape(-1),
                    np.asarray(dict_std[DatasetKeys.ROBOT_QUAT]).reshape(-1),
                ],
                axis=0,
            ).astype(np.float32)
            self._norm_cache = {
                "state_mean": state_mean,
                "state_std": state_std,
                "depth_mean": float(np.asarray(dict_mean[DatasetKeys.DEPTH])),
                "depth_std": float(np.asarray(dict_std[DatasetKeys.DEPTH])),
                # Future stats for all components (per-horizon)
                "fut_robot_pos_mean": np.asarray(dict_mean[DatasetKeys.FUTURE_ROBOT_POS]).astype(np.float32),
                "fut_robot_pos_std":  np.asarray(dict_std[DatasetKeys.FUTURE_ROBOT_POS]).astype(np.float32),
                "fut_payload_pos_mean": np.asarray(dict_mean[DatasetKeys.FUTURE_PAYLOAD_POS]).astype(np.float32),
                "fut_payload_pos_std":  np.asarray(dict_std[DatasetKeys.FUTURE_PAYLOAD_POS]).astype(np.float32),
                "fut_quat_mean": np.asarray(dict_mean[DatasetKeys.FUTURE_QUATERNION]).astype(np.float32),
                "fut_quat_std":  np.asarray(dict_std[DatasetKeys.FUTURE_QUATERNION]).astype(np.float32),
                "fut_robot_vel_mean": np.asarray(dict_mean[DatasetKeys.FUTURE_ROBOT_VEL]).astype(np.float32),
                "fut_robot_vel_std":  np.asarray(dict_std[DatasetKeys.FUTURE_ROBOT_VEL]).astype(np.float32),
                "fut_payload_vel_mean": np.asarray(dict_mean[DatasetKeys.FUTURE_PAYLOAD_VEL]).astype(np.float32),
                "fut_payload_vel_std":  np.asarray(dict_std[DatasetKeys.FUTURE_PAYLOAD_VEL]).astype(np.float32),
            }
        norm = self._norm_cache

        # --- Build & normalize state vector ---
        robot_quat = np.asarray(quat_xyzw, dtype=np.float32)
        robot_goal_relative = np.asarray(robot_goal, dtype=np.float32) - np.asarray(robot_pos, dtype=np.float32)
        # robot_goal_relative[2] = 0.0  
        state_vec = np.concatenate([
            np.asarray(payload_vel, dtype=np.float32),
            np.asarray(robot_vel, dtype=np.float32),
            robot_goal_relative,
            robot_quat,
        ], axis=0)
        print(f"relative goal: {robot_goal_relative})")

        if state_vec.shape[0] != 13:
            raise ValueError(f"State vector has shape {state_vec.shape}, expected (13,)")

        state_norm = (state_vec - norm["state_mean"]) / (norm["state_std"] + 1e-6)
    
        depth_np = np.asarray(depth_image, dtype=np.float32)
        if depth_np.ndim != 2:
            raise ValueError(f"Depth image must be 2D, got shape {depth_np.shape}")
        depth_norm = (depth_np - norm["depth_mean"]) / (norm["depth_std"] + 1e-6)

        base_robot_pos = np.asarray(robot_pos, dtype=np.float32)

        # --- Inference ---
        self.policy_model.eval()
        with torch.no_grad():
            model_device = next(self.policy_model.parameters()).device
            state_tensor = torch.from_numpy(state_norm).to(model_device).unsqueeze(0)                # (1,13)
            depth_tensor = torch.from_numpy(depth_norm).to(model_device).unsqueeze(0).unsqueeze(0)  # (1,1,64,64)
            predicted_traj, mode_probs, *_ = self.policy_model(state_tensor, depth_tensor)          # expected (1,M,H,16)

        pred = predicted_traj.detach().cpu().numpy()
        if pred.ndim != 4:
            raise RuntimeError(f"Unexpected predicted_traj shape {pred.shape}, expected (1,M,H,16)")
        _, M, H, D = pred.shape
        if D < 16:
            raise RuntimeError(f"Model output feature dim D={D} < 16 (expected >=16)")

        # Select mode 0 horizon block
        hb = pred[0, 0]  # (H,D)

        # --- Slice normalized components (order must match model output) ---
        off = 0
        rel_robot_pos_norm = hb[:, off:off+3]; off += 3
        rel_payload_pos_norm = hb[:, off:off+3]; off += 3
        quat_norm = hb[:, off:off+4]; off += 4
        robot_vel_norm = hb[:, off:off+3]; off += 3
        payload_vel_norm = hb[:, off:off+3]; off += 3

        # --- Unnormalize each component using per-horizon stats ---
        rel_robot_pos = rel_robot_pos_norm * (norm["fut_robot_pos_std"] + 1e-6) + norm["fut_robot_pos_mean"]
        rel_payload_pos = rel_payload_pos_norm * (norm["fut_payload_pos_std"] + 1e-6) + norm["fut_payload_pos_mean"]
        quats = quat_norm * (norm["fut_quat_std"] + 1e-6) + norm["fut_quat_mean"]
        robot_vel = robot_vel_norm * (norm["fut_robot_vel_std"] + 1e-6) + norm["fut_robot_vel_mean"]
        payload_vel = payload_vel_norm * (norm["fut_payload_vel_std"] + 1e-6) + norm["fut_payload_vel_mean"]

        # Normalize quaternions robustly
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        quats = quats / norms

        # Set payload position based on quaternion direction & cable length
        cable_length = 0.561
        rot_mats = Rot.from_quat(quats).as_matrix()  # (H,3,3)
        payload_dirs = -rot_mats[:, :, 2]            # (H,3) -Z axis of robot body frame
        rel_payload_pos = cable_length * payload_dirs

        desired_positions = base_robot_pos[None, :] + rel_robot_pos.astype(np.float32)
        desired_payload_positions = desired_positions + rel_payload_pos.astype(np.float32)

        return (
            desired_positions.astype(np.float32),
            desired_payload_positions.astype(np.float32),
            robot_vel.astype(np.float32),
            payload_vel.astype(np.float32),
            quats.astype(np.float32),
        )

    def run_controller(self, desired_positions, desired_quats, desired_payload_positions, controller_state):
        """
        Run one MPC step.
        Inputs:
          desired_positions: (N,3) robot positions (xyzw frame)
          desired_quats: (N,4) robot quats in xyzw
          desired_payload_positions: (N,3)
          controller_state: current state vector (nx,)
        Returns:
          next_robot_pos (3,), next_quat (4, xyzw), next_dof_pos_list (3,), next_state (nx,)
        """
        N = desired_positions.shape[0]
        assert hasattr(self, "mpc"), "self.mpc not set"
        assert N == self.mpc.N, f"Horizon mismatch: got {N}, expected {self.mpc.N}"
        assert desired_quats.shape == (N, 4), f"desired_quats shape {desired_quats.shape} != {(N,4)}"
        assert desired_payload_positions.shape == (N, 3)

        # references rows: 0..N (N+1) with first row = current state
        references = np.zeros((self.mpc.N + 1, self.mpc.nx + 4), dtype=np.float64)

        # Finite-difference velocities
        desired_velocities = np.zeros_like(desired_positions)
        desired_velocities[:-1] = (desired_positions[1:] - desired_positions[:-1]) / self.mpc.dt
        desired_velocities[-1] = desired_velocities[-2]

        desired_payload_velocities = np.zeros_like(desired_payload_positions)
        desired_payload_velocities[:-1] = (desired_payload_positions[1:] - desired_payload_positions[:-1]) / self.mpc.dt
        desired_payload_velocities[-1] = desired_payload_velocities[-2]

        # Current (row 0)
        references[0, 0:3] = controller_state[0:3]      # payload pos
        references[0, 3:6] = controller_state[3:6]      # payload vel
        references[0, 6:9] = controller_state[6:9]      # robot pos
        references[0, 9:12] = controller_state[9:12]    # robot vel
        references[0, 12:16] = controller_state[12:16]  # robot quat (wxyz in state)

        # Fill predicted horizon (convert xyzw -> wxyz for MPC)
        for k in range(1, N+1):
            q_xyzw = desired_quats[k-1]
            # xyzw -> wxyz
            q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)

            references[k, 0:3] = desired_payload_positions[k-1]
            references[k, 3:6] = desired_payload_velocities[k-1]
            references[k, 6:9] = desired_positions[k-1]
            references[k, 9:12] = desired_velocities[k-1]
            references[k, 12:16] = q_wxyz

        u0, x_next = self.mpc.run_controller_step(controller_state, references)
        
        # Extract next state components
        next_robot_pos = x_next[6:9] + desired_positions[0, :3]
        next_payload_pos = x_next[0:3] + desired_positions[0, :3]
        next_quat_wxyz = x_next[12:16]  # stored as wxyz
        # wxyz -> xyzw
        next_quat = np.array([next_quat_wxyz[1], next_quat_wxyz[2], next_quat_wxyz[3], next_quat_wxyz[0]])

        # Orientation for cable frame
        rot_mat = Rot.from_quat(next_quat).as_matrix()
        payload_vector = (next_payload_pos - next_robot_pos) / 0.5  # cable length 0.5
        next_dof_pos_list = self.rpy_from_A_to_B(A=rot_mat[:, 2], B=-payload_vector)

        return next_robot_pos, next_quat, next_dof_pos_list, x_next

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


    # ----- run method (was __main__ logic) -----
    def run(self, yaml_file_paths=None, csv_file_paths=None, step_stride: int = 10, max_steps: int = 10_000):
        assert len(yaml_file_paths) == self.num_envs and len(csv_file_paths) == self.num_envs, "Provide num_envs YAML/CSV paths"

        # Infer env name from YAMLs and ensure env is built
        env_name = self._infer_env_name_from_yaml(yaml_file_paths)
        self._ensure_env(env_name)

        actions = torch.zeros((self.env_manager.num_envs, 4), device=self.env_manager.device)
        actions[:, 0] = 1.0

        self.env_manager.reset()
        self.print_env_asset_index_map()
        self.check_asset_names()

        # Place obstacles per env
        for env_id in range(self.num_envs):
            self.set_obstacles_from_yaml(env_id=env_id, yaml_path=yaml_file_paths[env_id])

        traj_data = []
        for env_id in range(self.num_envs):
            data_e = load_csv(csv_file_paths[env_id])
            times_e = data_e[DatasetKeys.TIME]
            sol_x_e = data_e[DatasetKeys.SOL_X]
            sol_u_e = data_e[DatasetKeys.SOL_U]
            sol_quad_x_e = data_e[DatasetKeys.SOL_QUAD_X]
            sol_robot_quat_e = data_e[DatasetKeys.SOL_QUAD_QUAT]
            sol_payload_rpy_e = data_e[DatasetKeys.SOL_PAYLOAD_RPY]
            traj_data.append({
                "times": times_e,
                "sol_quad_x": sol_quad_x_e,
                "sol_robot_quat": sol_robot_quat_e,
                "sol_payload_rpy": sol_payload_rpy_e,
                "sol_x": sol_x_e,
                "sol_u": sol_u_e,
                "robot_goal": sol_x_e[-1, 0:3],  # FIXME FIXME
            })

        im_plots = None
        depth_images = [[] for _ in range(self.num_envs)]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = "cpu"
        policy_model = load_model_from_checkpoint(
            "/home/mrunal/Documents/poly_fly/data/zarr/models/forests/policy_ep0500.pth",
            "/home/mrunal/Documents/poly_fly/src/poly_fly/deep_poly_fly/model/config.yaml",
            device,
        )
        self.policy_model = policy_model
        
        # init states 
        env_ids = []
        env_id = 0
        td = traj_data[env_id]
        robot_pos = td["sol_quad_x"][0, 0:3]
        payload_pos = td["sol_x"][0, 0:3]
        quat_xyzw = td["sol_robot_quat"][0, 0:4]
        payload_rpy = td["sol_payload_rpy"][0, :]
        payload_vel = td["sol_x"][0, 3:6]
        robot_vel = np.array([0, 0, 0])
        robot_goal = td["robot_goal"]
        dof_pos = payload_rpy.tolist() if payload_rpy.shape[0] >= 3 else payload_rpy[:2].tolist()
        
        # controller_state = np.zeros((self.mpc.nx,), dtype=np.float64)
        # controller_state[0:3] = payload_pos
        # controller_state[3:6] = payload_vel
        # controller_state[6:9] = robot_pos
        # controller_state[9:12] = robot_vel
        # controller_state[12:16] = np.array([1, 0, 0, 0], dtype=np.float64)

        positions = []
        quats = []
        dof_pos_list = []
        dof_vel_list = []

        env_ids.append(env_id)
        positions.append(robot_pos)
        quats.append(quat_xyzw)
        dof_pos_list.append(payload_rpy)
        dof_vel_list.append([0.0] * len(dof_pos))

        self.set_robot_pose(
            env_ids=env_ids,
            positions=positions,
            quat_xyzw_list=quats,
            dof_positions_list=dof_pos_list,
            dof_velocities_list=dof_vel_list,
            zero_vel=True,
        )

        self.env_manager.render(render_components="sensors")

        for i in range(1000):
            if i % 100 == 0:
                # init states 
                env_ids = []
                env_id = 0
                td = traj_data[env_id]
                robot_pos = td["sol_quad_x"][0, 0:3]
                payload_pos = td["sol_x"][0, 0:3]
                quat_xyzw = td["sol_robot_quat"][0, 0:4]
                payload_rpy = td["sol_payload_rpy"][0, :]
                payload_vel = td["sol_x"][0, 3:6]
                robot_vel = np.array([0, 0, 0])
                robot_goal = td["robot_goal"]
                dof_pos = payload_rpy.tolist() if payload_rpy.shape[0] >= 3 else payload_rpy[:2].tolist()
                
                # controller_state = np.zeros((self.mpc.nx,), dtype=np.float64)
                # controller_state[0:3] = payload_pos
                # controller_state[3:6] = payload_vel
                # controller_state[6:9] = robot_pos
                # controller_state[9:12] = robot_vel
                # controller_state[12:16] = np.array([1, 0, 0, 0], dtype=np.float64)

                positions = []
                quats = []
                dof_pos_list = []
                dof_vel_list = []

                env_ids.append(env_id)
                positions.append(robot_pos)
                quats.append(quat_xyzw)
                dof_pos_list.append(payload_rpy)
                dof_vel_list.append([0.0] * len(dof_pos))

                self.set_robot_pose(
                    env_ids=env_ids,
                    positions=positions,
                    quat_xyzw_list=quats,
                    dof_positions_list=dof_pos_list,
                    dof_velocities_list=dof_vel_list,
                    zero_vel=True,
                )

                self.env_manager.render(render_components="sensors")

            gtd = self.env_manager.global_tensor_dict
            depth_tensor = gtd["depth_range_pixels"]  # [env, cam, H, W] float
            for env_id in range(self.num_envs):
                # FIXME Need to scale by 10.0 to get meters from range [0.0, 1.0]
                frame = depth_tensor[env_id, 0] * DEPTH_SCALE_FACTOR
                frame = torch.nn.functional.interpolate(frame.unsqueeze(0).unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                frame = frame.detach().cpu().numpy()
                depth_images[env_id].append(frame)
        

            images = []
            for env_id in range(self.num_envs):
                img = (255.0 * depth_tensor[env_id, 0].detach().cpu().numpy()).astype(np.uint8)
                images.append(img)
            if im_plots is None:
                ncols = min(self.num_envs, 4)
                nrows = int(np.ceil(self.num_envs / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
                axes = np.atleast_1d(axes).ravel()
                im_plots = []
                for env_id in range(self.num_envs):
                    ax = axes[env_id]
                    im = ax.imshow(images[env_id], cmap='plasma', vmin=0, vmax=255)
                    ax.set_title(f"Depth Env{env_id} Cam0")
                    ax.axis('off')
                    im_plots.append(im)
                for ax in axes[self.num_envs:]:
                    ax.axis('off')
                plt.tight_layout()
            else:
                for env_id in range(self.num_envs):
                    im_plots[env_id].set_data(images[env_id])
                plt.pause(0.001)

            self.env_manager.step(actions=actions)

            desired_positions, desired_payload_positions, desired_vel, desired_payload_vel, desired_quats = self.get_next_desired_states(
                payload_pos, payload_vel, robot_pos, robot_vel,
                quat_xyzw, robot_goal, frame)

            use_controller = False
            if use_controller:
                next_robot_pos, next_quat, next_dof_pos_list, controller_state = self.run_controller(desired_positions, desired_quats, desired_payload_positions, controller_state)
            else:
                idx = 1
                next_robot_pos = desired_positions[idx, :]
                next_quat = desired_quats[idx, :]
                payload_vector = (desired_payload_positions[idx, :] - desired_positions[idx, :]) / 0.561
                next_dof_pos_list = self.rpy_from_A_to_B(A=Rot.from_quat(next_quat).as_matrix()[:, 2], B=-payload_vector)

                payload_pos = desired_payload_positions[idx, :]
                payload_vel = desired_payload_vel[idx, :]
                robot_pos = next_robot_pos
                robot_vel = desired_vel[idx, :]
                quat_xyzw = next_quat

                # import pdb; pdb.set_trace()
            self.set_robot_pose(
                env_ids=[0],
                positions=[next_robot_pos],
                quat_xyzw_list=[next_quat],
                dof_positions_list=[next_dof_pos_list],
                dof_velocities_list=[[0.0]*len(next_dof_pos_list)],
                zero_vel=True,
            )

            # print next robot pos 
            print(f"Step {i}: robot pos {next_robot_pos}, quat {next_quat}, dof {next_dof_pos_list}")
            self.env_manager.render(render_components="sensors")
            self.env_manager.reset_terminated_and_truncated_envs()


            time.sleep(0.1)

        # # check if length of depth images matches length of input trajectory states
        # for i in range(self.num_envs):
        #     assert len(depth_images[i]) == traj_data[i]["sol_quad_x"].shape[0], "You have undersampled the trajectory!"

        return None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect depth data with MotionCollector")
    parser.add_argument("--viz", action="store_true", help="Enable depth visualization")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    # forest type no longer required; inferred from YAML
    args = parser.parse_args()

    mc = MotionCollector(
        num_envs=args.num_envs,
        device="cuda:0",
        seed=0,
        headless=False,
        use_warp=True,
        viz_depth=args.viz
    )

    mc.get_depth_for_csvs_sequential("forests")