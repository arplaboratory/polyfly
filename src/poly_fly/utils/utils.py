from pathlib import Path
from casadi.casadi import horzcat, vertcat
import numpy as np
import string
import yaml
import inspect
import os
import sys
from casadi import MX
import casadi as ca
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation as Rot


@dataclass
class MPC:
    dt: float
    mass_quad: float  # payload mass
    mass_load: float
    cable_length: float
    payload_radius: float
    robot_radius: float
    robot_height: float
    cable_radius: float
    initial_state: list
    end_state: list
    obstacles: list
    payload_pos_init: list
    payload_vel_init: list
    payload_input_init: list
    Qi: list
    Qu: float
    Q_dt: float
    horizon: int
    margin: float
    state_max: list
    state_min: list
    input_max: list
    input_min: list
    max_dt: float
    min_dt: float
    min_time: bool
    use_global_planner: bool
    use_robot_rotation: bool
    constraint_min_dist: float
    tube_distance: float
    global_planner_step_size: float
    global_planner_robot_radius: float


def dictToClass(clss, data):
    return clss(
        **{
            key: (data[key] if val.default == val.empty else data.get(key, val.default))
            for key, val in inspect.signature(clss).parameters.items()
        }
    )


# reads yaml file and creates a dictionary
def yamlToDict(path_to_yaml):
    base_path = os.path.join(os.path.dirname(path_to_yaml), "base.yaml")
    with open(base_path, 'r') as base_stream:
        base_yaml = yaml.safe_load(base_stream)

    with open(path_to_yaml, 'r') as stream:
        specific_yaml = yaml.safe_load(stream)

    # Merge base and specific YAMLs
    merged_yaml = {**base_yaml, **specific_yaml}
    return merged_yaml


# Transfer a 3 dimensional vector to a matrix
def vec2asym(vec):
    if type(vec) is np.ndarray:
        if len(vec.shape) == 1:
            row = vec.shape[0]
            col = 0
        elif len(vec.shape) == 2:
            row = vec.shape[0]
            col = vec.shape[1]
    elif type(vec) is list:
        row = len(vec)
        col = 0
    else:
        raise Exception("The vector type not list or numpy array")

    if row == 3:
        if col == 0:
            mat = np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])
        elif col == 1:
            mat = np.array(
                [[0, -vec[2][0], vec[1][0]], [vec[2][0], 0, -vec[0][0]], [-vec[1][0], vec[0][0], 0]]
            )
    else:
        raise Exception("The vector shape is not 3")

    return mat


def quaternion_multiplication_casadi():
    # Function that enables the rotation of a vector using quaternions

    # Creation of the symbolic variables for the quaternion and the vector
    quat_aux_1 = ca.MX.sym('quat_aux_1', 4, 1)
    quat_aux_2 = ca.MX.sym('quat_aux_2', 4, 1)

    quat = quat_aux_1

    H_plus_q = ca.vertcat(
        ca.horzcat(quat[0, 0], -quat[1, 0], -quat[2, 0], -quat[3, 0]),
        ca.horzcat(quat[1, 0], quat[0, 0], -quat[3, 0], quat[2, 0]),
        ca.horzcat(quat[2, 0], quat[3, 0], quat[0, 0], -quat[1, 0]),
        ca.horzcat(quat[3, 0], -quat[2, 0], quat[1, 0], quat[0, 0]),
    )

    # Computing the first multiplication
    quaternion_result = H_plus_q @ quat_aux_2

    # Create function
    f_multi = ca.Function('f_multi', [quat_aux_1, quat_aux_2], [quaternion_result[0:4, 0]])
    return f_multi


def rotation_inverse_casadi():
    # Creation of the symbolic variables for the quaternion and the vector
    quat_aux_1 = ca.MX.sym('quat_aux_1', 4, 1)
    vector_aux_1 = ca.MX.sym('vector_aux_1', 3, 1)

    # Auxiliary pure quaternion based on the information of the vector
    vector = ca.vertcat(0.0, vector_aux_1)

    # Quaternion
    quat = quat_aux_1

    # Quaternion conjugate
    quat_c = ca.vertcat(quat[0, 0], -quat[1, 0], -quat[2, 0], -quat[3, 0])
    # v' = q* x v x q
    # Rotation to the body Frame

    # QUaternion Multiplication vector form
    H_plus_q_c = ca.vertcat(
        ca.horzcat(quat_c[0, 0], -quat_c[1, 0], -quat_c[2, 0], -quat_c[3, 0]),
        ca.horzcat(quat_c[1, 0], quat_c[0, 0], -quat_c[3, 0], quat_c[2, 0]),
        ca.horzcat(quat_c[2, 0], quat_c[3, 0], quat_c[0, 0], -quat_c[1, 0]),
        ca.horzcat(quat_c[3, 0], -quat_c[2, 0], quat_c[1, 0], quat_c[0, 0]),
    )

    # First Multiplication
    aux_value = H_plus_q_c @ vector

    # Quaternion multiplication second element
    H_plus_aux = ca.vertcat(
        ca.horzcat(aux_value[0, 0], -aux_value[1, 0], -aux_value[2, 0], -aux_value[3, 0]),
        ca.horzcat(aux_value[1, 0], aux_value[0, 0], -aux_value[3, 0], aux_value[2, 0]),
        ca.horzcat(aux_value[2, 0], aux_value[3, 0], aux_value[0, 0], -aux_value[1, 0]),
        ca.horzcat(aux_value[3, 0], -aux_value[2, 0], aux_value[1, 0], aux_value[0, 0]),
    )

    # Rotated vector repected to the body frame
    vector_b = H_plus_aux @ quat

    # Defining function using casadi
    f_rot_inv = ca.Function('f_rot_inv', [quat_aux_1, vector_aux_1], [vector_b[1:4, 0]])
    return f_rot_inv


def mpc_from_yaml(path_to_yaml):
    with open(path_to_yaml, 'r') as f:
        data = yaml.safe_load(f)
    return dictToClass(MPC, data)


def load_params_sidecar_for_csv(csv_path):
    base, _ = os.path.splitext(csv_path)
    sidecar = base + ".params.yaml"
    if os.path.exists(sidecar):
        return mpc_from_yaml(sidecar)
    raise FileNotFoundError(f"No params sidecar found next to CSV: {sidecar}")

def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric cross-product matrix [v]_x."""
    x, y, z = v
    # Builds [v]_x = [[0,-z, y],[z,0,-x],[-y,x,0]]  ← matches your figure
    return np.array([[ 0.0, -z,   y ],
                     [  z,  0.0, -x ],
                     [ -y,  x,   0.0]])

def rotation_matrix_from_a_to_b(a, b, eps: float = 1e-8) -> np.ndarray:
    """
    Return R (3x3) that rotates vector a onto vector b.
    a, b may be any nonzero 3D vectors; they will be normalized internally.

    Implements: R = I + [v]_x + [v]_x^2 * ((1 - c) / s^2),
    with v = a x b, s = ||v||, c = a · b.
    Handles c≈1 (identity) and c≈-1 (180°) cases explicitly.
    """
    a = np.asarray(a, dtype=float)   # accept list/tuple; ensure float ndarray
    b = np.asarray(b, dtype=float)

    na = np.linalg.norm(a)           # ||a||
    nb = np.linalg.norm(b)           # ||b||
    if na < eps or nb < eps:
        raise ValueError("Input vectors must be nonzero.")

    a = a / na                       # normalize (your note assumes unit a, b)
    b = b / nb

    v = np.cross(a, b)               # ν = a × b           ← from screenshot
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))  # c = a · b, clamped to [-1,1]
    s = np.linalg.norm(v)            # s = ||ν|| = sin(angle)

    # Degeneracies when s≈0:
    # - If c>0: a and b are (almost) identical → R = I
    # - If c<0: a and b are opposite → rotate π around any axis ⟂ a
    if s < eps:
        if c > 0.0:                  # parallel case (θ≈0)
            # print("PARALLEL")
            return np.eye(3)
        else:
            # print("ANTI PARALLEL")
            # antiparallel (θ≈π): choose an arbitrary axis u ⟂ a
            tmp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            u = np.cross(a, tmp)
            u /= np.linalg.norm(u)
            # 180° rotation about u: R = 2uu^T - I (equivalent to Rodrigues at θ=π)
            return 2.0 * np.outer(u, u) - np.eye(3)

    print("\n\nGOOD\n\n")
    K = _skew(v)                     # K = [ν]_x
    # Rodrigues: R = I + K + K^2 * ((1-c)/s^2)
    # (Your note shows you can also use ((1-c)/s^2) = 1/(1+c), c≠-1)
    R = np.eye(3) + K + (K @ K) * ((1.0 - c) / (s * s))
    return R

def rpy_from_a_to_b(a, b):
    R = rotation_matrix_from_a_to_b(a, b)
    r, p, y = Rot.from_matrix(R).as_euler('zyx', degrees=False)
    return np.array([r, p, y])

def vector_in_bodyframe(v, q):
    """
    Rotate vector v (N, 3,) from world frame to body frame using quaternion q (N, 4,) (x,y,z,w).
    Returns vector in body frame (N, 3,).
    """
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("v must have shape (N, 3)")
    if q.ndim != 2 or q.shape[1] != 4 or q.shape[0] != v.shape[0]:
        raise ValueError("q must have shape (N, 4) matching v first dimension")

    N = v.shape[0]
    R_list = Rot.from_quat(q[:, :]).as_matrix()  # (N, 3, 3) (x,y,z,w)
    v_body = np.zeros_like(v)

    for i in range(N):
        R_i = R_list[i, :, :].T   # world->body is R^T
        v_body[i, :] = R_i @ v[i, :]
    return v_body

def get_yaw_along_trajectory(pos, vel, quats, times, max_yaw_rate=2.0*0.785398, zero_initial_yaw=False):
    """
    Inputs column-major:
        pos (3,N), vel (3,N), quats (4,N) (x,y,z,w), times (N,)
    Forces initial quaternion yaw to zero by applying a corrective rotation
    before generating the yaw trajectory.
    Returns (yaws (N,), new_quats (4,N) (w,x,y,z)).
    """
    pos   = np.asarray(pos, dtype=float)
    vel   = np.asarray(vel, dtype=float)
    quats = np.asarray(quats, dtype=float)
    times = np.asarray(times, dtype=float)

    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos must have shape (3,N)")
    if vel.shape != pos.shape:
        raise ValueError("vel must have shape (3,N)")
    if quats.ndim != 2 or quats.shape[1] != 4 or quats.shape[0] != pos.shape[0]:
        raise ValueError("quats must have shape (4,N) matching pos second dimension")
    if times.ndim != 1 or times.shape[0] != pos.shape[0]:
        raise ValueError("times must have shape (N,) with N matching pos.shape[1]")
    if not np.all(np.diff(times) > 0):
        raise ValueError("times must be strictly increasing")

    N = pos.shape[0]
    if N == 0:
        return np.zeros((0,)), np.zeros((4,0))

    if np.linalg.norm(vel[0, :]) > 1e-6 and np.abs(np.arctan2(vel[0,1], vel[0,0])) > 1e-2:
        raise ValueError("Initial velocity must be near zero.")

    # Initial orientation: extract yaw, pitch, roll (zyx gives yaw, pitch, roll)
    q0 = quats[0, :]
    r0_obj = Rot.from_quat([q0[0], q0[1], q0[2], q0[3]])  # (x,y,z,w)
    yaw0, pitch0, roll0 = r0_obj.as_euler('zyx', degrees=False)

    # Keep roll/pitch small requirement; yaw will be corrected
    if abs(roll0) > 1e-3 or abs(pitch0) > 1e-3:
        raise ValueError(f"Initial roll/pitch must be ~0. Got roll={roll0}, pitch={pitch0}")

    if zero_initial_yaw:
        # Force initial yaw to zero: apply rotation Δψ = (desired 0) - current yaw = -yaw0
        rot_fix = Rot.from_euler('z', -yaw0, degrees=False)
        r0_fixed = rot_fix * r0_obj
    else:
        r0_fixed = r0_obj
    q0_fixed_xyzw = r0_fixed.as_quat()  # (x,y,z,w)
    quats[0, :] = [q0_fixed_xyzw[0], q0_fixed_xyzw[1], q0_fixed_xyzw[2], q0_fixed_xyzw[3]]  # overwrite with yaw=0


    yaws = np.zeros(N)  # yaws[0] already zero after correction
    speed_eps = 1e-6

    for k in range(1, N):
        dt = times[k] - times[k-1]
        v = vel[k, :]
        speed = np.linalg.norm(v[:2])
        if speed > speed_eps:
            desired = np.arctan2(v[1], v[0])
        else:
            desired = yaws[k-1]
        delta = desired - yaws[k-1]
        delta = (delta + np.pi) % (2 * np.pi) - np.pi
        max_delta = max_yaw_rate * dt
        if delta > max_delta:
            delta = max_delta
        elif delta < -max_delta:
            delta = -max_delta
        yaws[k] = yaws[k-1] + delta

    # Post-process to avoid jumps at wrap boundary (spec: handle transitions near ±pi)
    zero_thr = 0.01     # radians: consider this "zero"
    pi_thr   = 0.1     # radians: closeness to ±pi
    # If initial yaw drifted numerically near ±pi, force to 0
    if abs(abs(yaws[0]) - np.pi) < pi_thr:
        yaws[0] = 0.0
    # Fix flips: previous ~0 and current ~±pi
    for k in range(1, N):
        if abs(yaws[k-1]) < zero_thr and (np.pi - abs(yaws[k])) < pi_thr:
            if yaws[k] > 0:
                yaws[k] -= np.pi  # 0 -> +pi  => subtract pi
            else:
                yaws[k] += np.pi  # 0 -> -pi => add pi

    # Apply yaw rotations to (possibly corrected) original quaternions
    new_quats = np.zeros_like(quats)
    for k in range(N):
        q_orig = quats[k, :]
        rot_orig = Rot.from_quat([q_orig[0], q_orig[1], q_orig[2], q_orig[3]])
        rot_yaw = Rot.from_euler('z', yaws[k], degrees=False)
        rot_new = rot_yaw * rot_orig
        new_quats[k, :] = rot_new.as_quat()

    return yaws, new_quats
