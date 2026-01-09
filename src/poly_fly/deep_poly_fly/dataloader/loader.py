import numpy as np
import torch
from torch.utils.data import Dataset
from poly_fly.data_io.utils import NORMALIZATION_STATS_DIR 
import os
from poly_fly.data_io.utils import save_normalization_stats, load_normalization_stats
from poly_fly.data_io.enums import DatasetKeys as DK

class DepthAndStateDataset(Dataset):
    """
    Dataset over all depth frames aggregated from load_zarr_folder.
    Each item returns:
      - depth image tensor: (1, H, W), normalized by dataset-wide depth mean/std
      - concatenated state tensor: 1D float tensor of [payload_pos, payload_vel, robot_pos, robot_vel, robot_quat]
      - future_robot_pos tensor: 1D float tensor
      - future_payload_pos tensor: 1D float tensor

      Note: state and future vectors are normalized per-dimension (vector-wise mean/std).
    """
    def __init__(self, datasets, normalize_scale: float= None, use_rotation_mat: bool = True):
        super().__init__()
        self.datasets = datasets
        self.min_depth_range = 0.02
        self.max_depth_range = 1.0
        self.use_rotation_mat = use_rotation_mat

        # Required keys
        self.required_state_keys = [
            DK.PAYLOAD_VEL,
            DK.ROBOT_VEL,
            DK.ROBOT_GOAL_RELATIVE,
            DK.ROT_MAT,
            DK.ROBOT_QUAT
        ]
        self.required_future_keys = [
            DK.FUTURE_ROBOT_POS,
            DK.FUTURE_PAYLOAD_POS,
            DK.FUTURE_ROBOT_VEL,
            DK.FUTURE_PAYLOAD_VEL,
            DK.FUTURE_ROT_MAT,
            DK.FUTURE_QUATERNION
        ]
        self.required_keys = [DK.DEPTH] + self.required_state_keys + self.required_future_keys

        self.validate_dataset(self.datasets, self.required_keys)
        self.index = []
        for di, ds in enumerate(self.datasets):
            T = int(ds.get(DK.DEPTH).shape[0])
            self.index.extend([(di, i) for i in range(T)])

        self.compute_data_stats()
        
    def save_normalization_stats(self):
        # Save normalization stats to NPZ
        dict_mean = {DK.DEPTH: self.depth_mean}
        dict_std = {DK.DEPTH: self.depth_std}
        for k in self.required_state_keys:
            dict_mean[k] = self.state_stats[k]["mean"]
            dict_std[k] = self.state_stats[k]["std"]
        for k in self.required_future_keys:
            dict_mean[k] = self.future_stats[k]["mean"]
            dict_std[k] = self.future_stats[k]["std"]
        save_normalization_stats(dict_mean, dict_std)

        loaded_mean, loaded_std = load_normalization_stats()
        required_keys = self.required_keys.copy()
        for k in required_keys:
            a_m = np.asarray(dict_mean[k])
            b_m = np.asarray(loaded_mean[k])
            if not np.allclose(a_m, b_m, rtol=1e-6, atol=1e-6):
                raise AssertionError(f"Mean mismatch for '{k}' (max abs diff {np.max(np.abs(a_m - b_m))})")
            a_s = np.asarray(dict_std[k])
            b_s = np.asarray(loaded_std[k])
            if not np.allclose(a_s, b_s, rtol=1e-6, atol=1e-6):
                raise AssertionError(f"Std mismatch for '{k}' (max abs diff {np.max(np.abs(a_s - b_s))})")
        print("Normalization stats round-trip verified.")

    def compute_data_stats(self):
        print("Computing dataset-wide stats over {} frames...".format(len(self.index)))

        n_mean_computation = 200
        if len(self.index) > n_mean_computation:
            sampled_indices = np.random.choice(len(self.index), size=n_mean_computation, replace=False)
            self.index_mean_computation = [self.index[i] for i in sampled_indices]
        else:
            self.index_mean_computation = list(self.index)

        acc_sum = 0.0
        acc_sq = 0.0
        n_pix = 0
        for di, ti in self.index_mean_computation:
            f = np.asarray(self.datasets[di][DK.DEPTH][ti], dtype=np.float32)
            acc_sum += float(f.sum())
            acc_sq += float((f * f).sum())
            n_pix += int(f.size)
        depth_mean = acc_sum / max(n_pix, 1)
        var = max(acc_sq / max(n_pix, 1) - depth_mean * depth_mean, 0.0)
        depth_std = float(np.sqrt(var)) if var > 0 else 1.0
        self.depth_mean = float(depth_mean)
        self.depth_std = float(depth_std if depth_std > 0 else 1.0)

        # State stats per variable (per-dimension mean/std across samples)
        self.state_stats = {}
        for key in self.required_state_keys:
            # Pre-initialize accumulators using the first sample's flattened shape
            first = np.asarray(self.datasets[0][key][0], dtype=np.float32).reshape(-1)
            flat_shape = first.shape
            sum_vec = np.zeros(flat_shape, dtype=np.float32)
            sumsq_vec = np.zeros(flat_shape, dtype=np.float32)
            n_samples = 0
            for di, ti in self.index_mean_computation:
                a = np.asarray(self.datasets[di][key][ti], dtype=np.float32).reshape(-1)
                if a.shape != flat_shape:
                    raise ValueError(f"Inconsistent shape for key '{key}': expected {flat_shape}, got {a.shape}")
             
                a64 = a.astype(np.float32, copy=False)
                sum_vec += a64
                sumsq_vec += a64 * a64
                n_samples += 1

            assert n_samples > 0
            mean_vec = (sum_vec / n_samples).astype(np.float32)
            var_vec = (sumsq_vec / n_samples) - (mean_vec.astype(np.float32) ** 2)
            var_vec = np.maximum(var_vec, 0.0)
            std_vec = np.sqrt(var_vec).astype(np.float32)
            std_vec[std_vec == 0.0] = 1.0
            self.state_stats[key] = {"mean": mean_vec, "std": std_vec}

        # Future stats per variable (per-dimension mean/std across samples)
        self.future_stats = {}
        for key in self.required_future_keys:
            # Capture original (H, A, ...) shape before flattening
            first_full = np.asarray(self.datasets[0][key][0], dtype=np.float32)
            orig_shape = first_full.shape
            first = first_full.reshape(-1)
            flat_shape = first.shape

            sum_vec = np.zeros(flat_shape, dtype=np.float32)
            sumsq_vec = np.zeros(flat_shape, dtype=np.float32)
            n_samples = 0
            for di, ti in self.index_mean_computation:
                a = np.asarray(self.datasets[di][key][ti], dtype=np.float32).reshape(-1)
                if a.shape != flat_shape:
                    raise ValueError(f"Inconsistent shape for key '{key}': expected {flat_shape}, got {a.shape}")

                a64 = a.astype(np.float32, copy=False)
                sum_vec += a64
                sumsq_vec += a64 * a64
                n_samples += 1

            assert  n_samples > 0
            mean_vec = (sum_vec / n_samples).astype(np.float32)
            var_vec = (sumsq_vec / n_samples) - (mean_vec.astype(np.float32) ** 2)
            var_vec = np.maximum(var_vec, 0.0)
            std_vec = np.sqrt(var_vec).astype(np.float32)
            std_vec[std_vec == 0.0] = 1.0

            # Reshape back to original (H, A, ...) shape before saving
            mean_arr = mean_vec.reshape(orig_shape)
            std_arr  = std_vec.reshape(orig_shape)
            self.future_stats[key] = {"mean": mean_arr, "std": std_arr}

        print(f"depth approx mean={self.depth_mean:.6f}  std={self.depth_std:.6f}  n_frames={len(self.index_mean_computation)}")
        for k in self.required_state_keys:
            ms = self.state_stats[k]
            print(f"{k:<14} mean_shape={ms['mean'].shape}  std_shape={ms['std'].shape}")
        for k in self.required_future_keys:
            ms = self.future_stats[k]
            print(f"{k:<14} mean_shape={ms['mean'].shape}  std_shape={ms['std'].shape}")
        
        # Print normalization stats for record-keeping
        print("\nNormalization stats (mean and std) for record-keeping:")
        print(f"Depth mean: {self.depth_mean:.6f}, std: {self.depth_std:.6f}")
        for k in self.required_state_keys:
            ms = self.state_stats[k]
            print(f"State '{k}': mean shape={ms['mean']}, std shape={ms['std']}")
        for k in self.required_future_keys:
            ms = self.future_stats[k]
            print(f"Future '{k}': mean shape={ms['mean']}, std shape={ms['std']}")

        self.save_normalization_stats()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        di, ti = self.index[idx]
        ds = self.datasets[di]

        # Depth image -> normalized
        depth = np.asarray(ds[DK.DEPTH][ti], dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(f"Expected depth frame to be 2D (H, W); got shape={depth.shape}")
        depth = (depth - self.depth_mean) / (self.depth_std + 1e-6)  # use renamed stats
        depth_t = torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)

        state_parts_keys = [
            DK.PAYLOAD_VEL,
            DK.ROBOT_VEL,
            DK.ROBOT_GOAL_RELATIVE
        ]
        if self.use_rotation_mat:
            state_parts_keys.append(DK.ROT_MAT)
        else:
            state_parts_keys.append(DK.ROBOT_QUAT)

        state_parts = []
        for k in state_parts_keys:
            arr = np.asarray(ds[k][ti], dtype=np.float32).reshape(-1)
            ms = self.state_stats[k]
            arr = (arr - ms["mean"]) / (ms["std"] + 1e-6)
            state_parts.append(torch.from_numpy(arr))
        state_concat = torch.cat(state_parts, dim=0).to(torch.float32)  # (S,)

        # Futures (normalize in original shape using reshaped stats, then flatten)
        fr = np.asarray(ds[DK.FUTURE_ROBOT_POS][ti], dtype=np.float32)  # keep original shape
        fpr = np.asarray(ds[DK.FUTURE_PAYLOAD_POS][ti], dtype=np.float32)  # keep original shape
        fq = np.asarray(ds[DK.FUTURE_QUATERNION][ti], dtype=np.float32)
        frot = np.asarray(ds[DK.FUTURE_ROT_MAT][ti], dtype=np.float32)  # flatten rotation matrix
        frv = np.asarray(ds[DK.FUTURE_ROBOT_VEL][ti], dtype=np.float32)
        fpv = np.asarray(ds[DK.FUTURE_PAYLOAD_VEL][ti], dtype=np.float32)

        fr_ms = self.future_stats[DK.FUTURE_ROBOT_POS]
        fpr_ms = self.future_stats[DK.FUTURE_PAYLOAD_POS]
        fq_ms = self.future_stats[DK.FUTURE_QUATERNION]
        frv_ms = self.future_stats[DK.FUTURE_ROBOT_VEL]
        fpv_ms = self.future_stats[DK.FUTURE_PAYLOAD_VEL]
        frot_ms = self.future_stats[DK.FUTURE_ROT_MAT]

        fr = (fr - fr_ms["mean"]) / (fr_ms["std"] + 1e-6)
        frp = (fpr - fpr_ms["mean"]) / (fpr_ms["std"] + 1e-6)
        fq = (fq - fq_ms["mean"]) / (fq_ms["std"] + 1e-6)
        frot = (frot - frot_ms["mean"]) / (frot_ms["std"] + 1e-6)
        frv = (frv - frv_ms["mean"]) / (frv_ms["std"] + 1e-6)
        fpv = (fpv - fpv_ms["mean"]) / (fpv_ms["std"] + 1e-6)

        # Flatten to 1D for concatenation
        future_robot_pos = torch.from_numpy(fr).to(torch.float32)
        future_payload_pos = torch.from_numpy(frp).to(torch.float32)
        future_quaternion = torch.from_numpy(fq).to(torch.float32)
        future_rot_mat = torch.from_numpy(frot).to(torch.float32)
        future_robot_vel = torch.from_numpy(frv).to(torch.float32)
        future_payload_vel = torch.from_numpy(fpv).to(torch.float32)

        if self.use_rotation_mat:
            future_trajectory = torch.cat([
                future_robot_pos, future_payload_pos, future_rot_mat, 
                future_robot_vel, future_payload_vel], dim=-1)
        else:
            future_trajectory = torch.cat([
                future_robot_pos, future_payload_pos, future_quaternion, 
                future_robot_vel, future_payload_vel], dim=-1)
        
        return depth_t, state_concat, future_trajectory

    def get_state_part_keys(self):
        # TODO 
        pass 
    
    def unnormalize_data(self, x):
        """
        Invert normalization for a concatenated future trajectory tensor.

        Expects x to be constructed in the same order as in __getitem__:
          - if use_rotation_mat:
              [FUTURE_ROBOT_POS, FUTURE_PAYLOAD_POS, FUTURE_ROT_MAT,
               FUTURE_ROBOT_VEL, FUTURE_PAYLOAD_VEL]
            else:
              [FUTURE_ROBOT_POS, FUTURE_PAYLOAD_POS, FUTURE_QUATERNION,
               FUTURE_ROBOT_VEL, FUTURE_PAYLOAD_VEL]

        x can be any shape with the last dimension being the concatenated feature dim.
        """
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)

        # Order must match how __getitem__ concatenates futures
        ordered_keys = [
            DK.FUTURE_ROBOT_POS,
            DK.FUTURE_PAYLOAD_POS,
            DK.FUTURE_ROT_MAT if self.use_rotation_mat else DK.FUTURE_QUATERNION,
            DK.FUTURE_ROBOT_VEL,
            DK.FUTURE_PAYLOAD_VEL,
        ]

        # Determine split sizes from the stats (assumes features are along the last axis)
        try:
            split_sizes = [int(np.asarray(self.future_stats[k]["mean"]).shape[-1]) for k in ordered_keys]
        except KeyError as e:
            raise KeyError(f"Missing future_stats for key: {e}")

        total_expected = int(sum(split_sizes))
        if x.shape[-1] != total_expected:
            raise ValueError(f"Input last dim {x.shape[-1]} != expected concat size {total_expected}")

        parts = list(torch.split(x, split_sizes, dim=-1))

        # Unnormalize each part and re-concatenate
        unnorm_parts = []
        for part, k in zip(parts, ordered_keys):
            ms = self.future_stats[k]
            mean_t = torch.as_tensor(ms["mean"], dtype=part.dtype, device=part.device)
            std_t = torch.as_tensor(ms["std"], dtype=part.dtype, device=part.device)

            # Ensure broadcasting with potential extra leading dims in part
            while mean_t.ndim < part.ndim:
                mean_t = mean_t.unsqueeze(0)
                std_t = std_t.unsqueeze(0)

            unnorm_part = part * (std_t + 1e-6) + mean_t
            unnorm_parts.append(unnorm_part)

        return torch.cat(unnorm_parts, dim=-1)

    def validate_dataset(self, datasets, required_keys):
        for di, ds in enumerate(datasets):
            for k in required_keys:
                if k not in ds:
                    print(f"Available keys in dataset[{di}]: {list(ds.keys())}")
                    raise KeyError(f"Dataset at index {di} is missing required key '{k}'")
            
            # Validate time dimension consistency for all required arrays
            lens = {}
            for k in required_keys:
                arr = ds.get(k)
                try:
                    T = int(arr.shape[0])
                except Exception as e:
                    raise ValueError(f"Dataset[{di}]['{k}'] must be time-major with shape (T, ...). Error: {e}")
                lens[k] = T
            if len(set(lens.values())) != 1:
                raise ValueError(f"Dataset[{di}] has mismatched time lengths among required keys: {lens}")
