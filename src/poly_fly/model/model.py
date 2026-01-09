from acados_template import AcadosModel
from casadi import MX, MX, external, vertcat
from ctypes import *
import copy
import casadi as cs
import numpy as np
import os
from acados_template import AcadosSim, AcadosSimSolver

# from nmpc_control.utils import *
from casadi import tanh, SX, Sparsity, hessian, if_else, horzcat, DM, blockcat
from acados_template import AcadosOcp, AcadosOcpSolver
from scipy.spatial.transform import Rotation


class ACADOS_PARAMS_IDX:
    LOAD_POS = 0
    LOAD_VEL = 3
    POS = 6
    VEL = 9
    ROT = 12
    ANG_VEL = 16
    THRUST = 19
    INPUTS = 20
    TAUT = 24
    N = 25


CABLE_L = 0.5
ACADOS_HORIZON_N = 20
ACADOS_HORIZON_STEP = 0.1


def export_model(
    model_name="quadrotor_payload", g_val=9.81, return_params=False, cross_model=True, use_hybrid_mode=False
):
    mass_payload = 0.163  # 0.150
    mass_quad = 0.720

    hover_estimate = np.sqrt(((mass_payload + mass_quad) * g_val))
    print(f"hover rpm estimate {hover_estimate}")

    # set up states
    xL = MX.sym('xL')
    yL = MX.sym('yL')
    zL = MX.sym('zL')
    xLdot = MX.sym('xLdot')
    yLdot = MX.sym('yLdot')
    zLdot = MX.sym('zLdot')
    x = MX.sym('x')
    y = MX.sym('y')
    z = MX.sym('z')
    xdot = MX.sym('xdot')
    ydot = MX.sym('ydot')
    zdot = MX.sym('zdot')
    qw = MX.sym('qw')
    qx = MX.sym('qx')
    qy = MX.sym('qy')
    qz = MX.sym('qz')

    X = vertcat(
        xL,
        yL,
        zL,  # 0:3
        xLdot,
        yLdot,
        zLdot,  # 3:6
        x,
        y,
        z,  # 6:9
        xdot,
        ydot,
        zdot,  # 9:12
        qw,
        qx,
        qy,
        qz,  # 12:16
    )
    nx = X.shape[0]

    # parameters
    # associated time stamp
    ext_param = MX.sym('references', ACADOS_PARAMS_IDX.N, 1)

    # set up controls
    # motor rpms
    f = MX.sym('f')
    wx = MX.sym('wx')
    wy = MX.sym('wy')
    wz = MX.sym('wz')
    u = vertcat(wx, wy, wz, f)
    F = f

    norm_xi = cs.norm_2(cs.vertcat(xL - x, yL - y, zL - z))
    xi = cs.vertcat((xL - x) / norm_xi, (yL - y) / norm_xi, (zL - z) / norm_xi)
    xidot = cs.vertcat((xLdot - xdot) / CABLE_L, (yLdot - ydot) / CABLE_L, (zLdot - zdot) / CABLE_L)
    xi_omega = cs.cross(xi, xidot)

    # Evaluate forces
    e3 = np.array([[0.0], [0.0], [1.0]])
    g = g_val * e3

    quat = cs.vertcat(qw, qx, qy, qz)
    quad_force_vector = F * as_matrix(quat) @ e3
    quad_centrifugal_f = mass_quad * CABLE_L * (cs.dot(xi_omega, xi_omega))

    if use_hybrid_mode:
        temp_tension_vector = (
            (mass_payload / (mass_payload + mass_quad))
            * (-cs.dot(xi, quad_force_vector) + quad_centrifugal_f)
            * xi
        )
        # tension_vector = if_else(norm_xi < (CABLE_L - 0.002), cs.vertcat(0, 0, 0), temp_tension_vector)
        tension_vector = temp_tension_vector
    else:
        tension_vector = (
            (mass_payload / (mass_payload + mass_quad))
            * (-cs.dot(xi, quad_force_vector) + quad_centrifugal_f)
            * xi
        )

    accL = -tension_vector / mass_payload - g
    accQ = (quad_force_vector + tension_vector) / mass_quad - g

    p = wx
    q = wy
    r = wz
    qdot_helper = cs.vertcat(
        cs.horzcat(0, -p, -q, -r),
        cs.horzcat(p, 0, r, -q),
        cs.horzcat(q, -r, 0, p),
        cs.horzcat(r, q, -p, 0),
    )
    K_quat = 2.0
    quaterr = 1.0 - cs.norm_2(quat)
    qdot = 0.5 * qdot_helper @ quat + K_quat * quaterr * quat

    f_expl_rpm = vertcat(
        xLdot,
        yLdot,
        zLdot,
        accL[0],
        accL[1],
        accL[2],
        xdot,
        ydot,
        zdot,
        accQ[0],
        accQ[1],
        accQ[2],
        qdot[0],
        qdot[1],
        qdot[2],
        qdot[3]
    )

    f_expl = f_expl_rpm
    Xdot = MX.sym('Xdot', nx, 1)

    p = ext_param
    zz = []

    model = AcadosModel()
    model.f_impl_expr = Xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = X
    model.u = u
    model.xdot = Xdot
    model.z = zz
    model.p = p
    model.name = model_name

    f_expl_function = cs.Function('f_expl', [X, u], [f_expl_rpm])

    return model, f_expl_function, X, u, f_expl


def get_model(g_val=9.81, model_name="quadrotor_payload"):
    model, f_expl, X, u, f_exp = export_model(g_val=g_val, model_name=model_name)
    return model 


def get_acados_integrator(model, Tf=0.01, generate=False, build=False):
    sim = AcadosSim()
    sim.model = model

    # create integrator
    nx = model.x.size()[0]
    nu = model.u.size()[0]

    sim.solver_options.T = Tf
    sim.solver_options.integrator_type = 'IRK'
    sim.solver_options.num_stages = 2
    sim.solver_options.num_steps = 1
    sim.solver_options.newton_iter = 2  # for implicit integrator
    sim.solver_options.collocation_type = "GAUSS_RADAU_IIA"
    sim.parameter_values = np.zeros(model.p.shape)

    ws = os.getenv('POLYFLY_DIR')
    ws = os.path.join(ws, 'src', 'model')
    sim.code_export_directory = os.path.join(ws, 'c_generated_code')
    json_file_name = os.path.join(ws, "acados_sim.json")

    if generate is False and build is False:
        acados_integrator = AcadosSimSolver(
            sim, generate=generate, build=build, json_file=json_file_name
        )
    else:
        acados_integrator = AcadosSimSolver(
            sim, generate=generate, build=build, json_file=json_file_name
        )

    return acados_integrator

def as_matrix(quat):
    # expect w, x, y, z
    assert quat.shape == (4, 1)
    return cs.MX.eye(3) + 2 * quat[0] * cs.skew(quat[1:4]) + 2 * cs.mpower(cs.skew(quat[1:4]), 2)


if __name__ == "__main__":
    print(f"building {__file__}")
    print("Only run this script with pwd = model")
    get_acados_integrator(get_model(model_name="quadrotor_payload"), generate=True, build=True)
