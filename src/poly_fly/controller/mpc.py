from acados_template import AcadosModel
from casadi import MX, MX, external, vertcat
from ctypes import *
import copy
import casadi as cs
import numpy as  np
import os
from casadi import MX
import casadi as ca
from acados_template import AcadosSim, AcadosSimSolver
from casadi import tanh, SX, Sparsity, hessian, if_else, horzcat, DM, blockcat
from acados_template import AcadosSim, AcadosSimSolver
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import matplotlib.pyplot as plt  # (new) for plotting

EPS = 1e-4

def as_matrix(quat):
    # expect w, x, y, z
    assert quat.shape == (4, 1)
    return (
        cs.MX.eye(3)
        + 2 * quat[0] * cs.skew(quat[1:4])
        + 2 * cs.mpower(cs.skew(quat[1:4]), 2)
    )

def quatError(qk, qd, weight):
  q_aux = MX.zeros(4, 1)
  q_aux[0] = qd[0] * qk[0] + qd[1] * qk[1] + qd[2] * qk[2] + qd[3] * qk[3]
  q_aux[1] = -qd[1] * qk[0] + qd[0] * qk[1] + qd[3] * qk[2] - qd[2] * qk[3]
  q_aux[2] = -qd[2] * qk[0] - qd[3] * qk[1] + qd[0] * qk[2] + qd[1] * qk[3]
  q_aux[3] = -qd[3] * qk[0] + qd[2] * qk[1] - qd[1] * qk[2] + qd[0] * qk[3]
  q_att_denom = ca.sqrt(q_aux[0] * q_aux[0] + q_aux[3] * q_aux[3] + 1e-3)
  q_att = ca.vertcat(q_aux[0] * q_aux[1] - q_aux[2] * q_aux[3],
        q_aux[0] * q_aux[2] + q_aux[1] * q_aux[3],
        q_aux[3]) / q_att_denom
  return ca.transpose(q_att) @ weight @ q_att

def calcAttCost(Rdes, Rcurr, err_weight):
  att_err_mat = 0.5 * (mtimes(Rdes.T, Rcurr) - mtimes(Rcurr.T, Rdes))
  att_err = MX.sym('att_err', 3, 1)
  att_err[0] = err_weight[0] * (att_err_mat[2, 1] ** 2)
  att_err[1] = err_weight[1] * (att_err_mat[0, 2] ** 2)
  att_err[2] = err_weight[2] * (att_err_mat[1, 0] ** 2)
  att_err_cost = sum1(att_err)
  return att_err_cost

def calcSquareCost(udes_, ucurr_, sqr_weight):
  cost_square = 0
  row_ucurr_ = np.shape(ucurr_)
  for i in range(row_ucurr_[0]):
    cost_square += ((ucurr_[i] - udes_[i]) ** 2) * sqr_weight[i]
  return cost_square


def export_model(g_val=9.81, return_params=False):
    model_name = "quadrotor"
    mass_payload = 0.163 # 0.150
    cable_l = 0.56
    arm_length = 0.17
    prop_radius = 0.099
    inertia = np.zeros((3, 3))

    # voxl2
    inertia[0, 0] = 0.002404
    inertia[1, 1] = 0.00238
    inertia[2, 2] = 0.0028
    mass_quad = 0.720
    kf =  0.88e-08
    km = 1.34e-10
    km_kf = km/kf 
    invI = np.linalg.inv(inertia)

    hover_estimate = np.sqrt(((mass_payload + mass_quad)*g_val)/(kf*4.0))
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
    omega_x = MX.sym('omega_x')
    omega_y = MX.sym('omega_y')
    omega_z = MX.sym('omega_z')

    X = vertcat(
        xL, yL, zL,                 # 0:3
        xLdot, yLdot, zLdot,        # 3:6
        x, y, z,                    # 6:9
        xdot, ydot, zdot,           # 9:12
        qw, qx, qy, qz,             # 12:16
        omega_x, omega_y, omega_z,  # 16:19
    )
    nx = X.shape[0]

    # parameters
    # states + inputs
    ext_param = MX.sym('references', nx + 4, 1)

    # set up controls
    # motor rpms
    u1 = MX.sym('u1')
    u2 = MX.sym('u2')
    u3 = MX.sym('u3')
    u4 = MX.sym('u4')
    u = vertcat(u1, u2, u3, u4)

    dx = 0.21
    dy = 0.16
    ratio_x = 0.50
    ratio_y = 0.49
    dx_left = dx * ratio_x
    dx_right = dx * (1.0 - ratio_x)
    dy_front = dy * ratio_y
    dy_back = dy * (1.0 - ratio_y)

    
    moment_x = (dx_left * u1 + dx_left * u2
                - dx_right * u3 - dx_right * u4)

    moment_y = (- dy_front * u1 + dy_back * u2
                + dy_back * u3 - dy_front * u4)

    moment_z = km_kf * (u1 - u2 + u3 - u4)


    F = (u1 + u2 + u3 + u4)
    M = cs.vertcat(moment_x, moment_y, moment_z)

    norm_xi = cs.norm_2(cs.vertcat(xL - x, yL - y, zL - z))
    xi = cs.vertcat((xL - x)/norm_xi, (yL - y)/norm_xi, (zL - z)/norm_xi)
    xidot = cs.vertcat((xLdot - xdot)/cable_l, (yLdot - ydot)/cable_l, (zLdot - zdot)/cable_l)
    xi_omega = cs.cross(xi, xidot)

    # Evaluate forces
    e3=np.array([[0.0],[0.0],[1.0]])
    g = g_val * e3

    quat = cs.vertcat(qw, qx, qy, qz)
    quad_force_vector = F * as_matrix(quat) @ e3
    quad_centrifugal_f = mass_quad * cable_l * (cs.dot(xi_omega, xi_omega))

    tension_vector = (mass_payload/(mass_payload + mass_quad)) * (-cs.dot(xi, quad_force_vector) + quad_centrifugal_f) * xi

    accL = -tension_vector / mass_payload - g
    accQ = (quad_force_vector + tension_vector) / mass_quad - g

    p = omega_x
    q = omega_y
    r = omega_z
    qdot_helper = cs.vertcat(
                    cs.horzcat(0, -p, -q, -r),
                    cs.horzcat(p, 0, r, -q),
                    cs.horzcat(q, -r, 0, p),
                    cs.horzcat(r, q, -p, 0)
                    )
    K_quat = 2.0
    quaterr = 1.0 - cs.norm_2(quat)
    qdot = 0.5 * qdot_helper @ quat + K_quat * quaterr * quat

    omega = cs.vertcat(omega_x, omega_y, omega_z)
    pqrdot = invI @ (M - cs.cross(omega, inertia @ omega))


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
        qdot[3],
        pqrdot[0],
        pqrdot[1],
        pqrdot[2],
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

    if False:
        if not cross_model:
            evaluate_var("quad_centrifugal_f", quad_centrifugal_f, qw, qx, qy, qz, omega_x, omega_y, omega_z,  u1, u2, u3, u4)
            evaluate_var("tension_vector", tension_vector, qw, qx, qy, qz, omega_x, omega_y, omega_z,  u1, u2, u3, u4)
            evaluate_var("accL", accL, qw, qx, qy, qz, omega_x, omega_y, omega_z,  u1, u2, u3, u4)
            evaluate_var("accQ", accQ, qw, qx, qy, qz, omega_x, omega_y, omega_z,  u1, u2, u3, u4)
            evaluate_var("qdot", qdot, qw, qx, qy, qz, omega_x, omega_y, omega_z,  u1, u2, u3, u4)
            evaluate_var("pqrdot", pqrdot, qw, qx, qy, qz, omega_x, omega_y, omega_z,  u1, u2, u3, u4)
            evaluate_var_final("f_expl_rpm", f_expl_rpm, qw, qx, qy, qz, omega_x, omega_y, omega_z,  u1, u2, u3, u4,
                xLdot, yLdot, zLdot, xdot, ydot, zdot)
        else:
            evaluate_var_cross("F", F, qw, qx, qy, qz, omega_x, omega_y, omega_z,  u1, u2, u3, u4, x, y, z, xdot, ydot, zdot, xL, yL, zL, xLdot, yLdot, zLdot, u1dot, u2dot, u3dot, u4dot)
            evaluate_var_cross("M", M, qw, qx, qy, qz, omega_x, omega_y, omega_z,  u1, u2, u3, u4, x, y, z, xdot, ydot, zdot, xL, yL, zL, xLdot, yLdot, zLdot, u1dot, u2dot, u3dot, u4dot)
            evaluate_var_cross("accL", accL, qw, qx, qy, qz, omega_x, omega_y, omega_z,  u1, u2, u3, u4, x, y, z, xdot, ydot, zdot, xL, yL, zL, xLdot, yLdot, zLdot, u1dot, u2dot, u3dot, u4dot)
            evaluate_var_cross("accQ", accL, qw, qx, qy, qz, omega_x, omega_y, omega_z,  u1, u2, u3, u4, x, y, z, xdot, ydot, zdot, xL, yL, zL, xLdot, yLdot, zLdot, u1dot, u2dot, u3dot, u4dot)
            evaluate_var_cross("f_expl_rpm", f_expl_rpm, qw, qx, qy, qz, omega_x, omega_y, omega_z,  u1, u2, u3, u4, x, y, z, xdot, ydot, zdot, xL, yL, zL, xLdot, yLdot, zLdot, u1dot, u2dot, u3dot, u4dot)


    if return_params:
        return model, [xL, yL, zL, x, y, z, qw, qx, qy, qz, omega_x, omega_y, omega_z,  u1, u2, u3, u4, u1dot, u2dot, u3dot, u4dot,
        xLdot, yLdot, zLdot, xdot, ydot, zdot]

    return model

def get_model(g_val=9.81):
    return export_model(g_val=g_val)

def get_acados_integrator(model, ocp):
    sim = AcadosSim()
    sim.model = model

    # create integrator
    Tf = ocp.solver_options.tf/ocp.dims.N
    nx = model.x.size()[0]
    nu = model.u.size()[0]

    sim.solver_options.T = Tf
    sim.solver_options.integrator_type = 'IRK'
    sim.solver_options.num_stages = 3
    sim.solver_options.num_steps = 3
    sim.solver_options.newton_iter = 3 # for implicit integrator
    sim.solver_options.collocation_type = "GAUSS_RADAU_IIA"

    sim.parameter_values = np.zeros(model.p.shape)

    print("SAVING")
    acados_integrator = AcadosSimSolver(sim, json_file='acados_sim_solver_nmpc.json')

    return acados_integrator


def create_solver(model, verbose=True, save=False):
    N = 10
    Tf = 1.0
    Ts = Tf / N

    ocp = AcadosOcp()
    ocp.model = model

    nx = model.x.size()[0]
    nu = model.u.size()[0]

    ocp.dims.N = N
    ocp.dims.nx = nx
    ocp.dims.nbx = nx
    ocp.dims.nbu = nu
    ocp.dims.nbx_e = nx
    ocp.dims.nu = model.u.size()[0]
    ocp.dims.np = model.p.size()[0]
    ocp.dims.nbxe_0 = nx

    x = ocp.model.x
    u = ocp.model.u
    param = ocp.model.p

    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'

    quat = vertcat(x[12], x[13], x[14], x[15])
    quat_references = param[12:16]
    quat_costs = np.eye(3)
    quat_costs[0, 0] = 50
    quat_costs[1, 1] = 50
    quat_costs[2, 2] = 100

    payload_states = vertcat(x[0], x[1], x[2], x[3], x[4], x[5])  # payload position, velocity
    payload_state_references = vertcat(param[0:6])  # desired position and velocity
    payload_state_costs = 1.0 * np.array([100, 100, 100, 20, 20, 20]) # FOR N=10 # Q cost values for pos and vel

    quad_pos_states = vertcat(x[6], x[7], x[8])  # quad velocity
    quad_pos_state_references = vertcat(param[6:9])
    quad_pos_state_costs = 100.0 * np.ones((3))

    quad_vel_states = vertcat(x[9], x[10], x[11])  # quad velocity
    quad_vel_state_references = vertcat(param[9:12])
    quad_vel_state_costs = 5.0 * np.ones((3))

    # without small cost on zero angular velocity, quad can get unstable
    quad_angular_states = vertcat(x[16], x[17], x[18])  # quad angular velocity
    quad_angular_state_references = vertcat(param[16:19])
    quad_angular_state_costs = 0.1 * np.array([2.0, 2.0, 2.0])

    R = 1.5 * np.array([1, 1, 1, 1])
    rpm_reference = param[19:23]

    lower_bx = -1.0 * np.array([
        100, 100, 100,
        30, 30, 30,
        100, 100, 100,
        30, 30, 30,
        50, 50, 50, 50,
        4.0, 4.0, 4.0,  # angular vel
        ])

    upper_bx = np.array([
        100, 100, 100,
        30, 30, 30,
        100, 100, 100,
        30, 30, 30,
        50, 50, 50, 50,
        4.0, 4.0, 4.0,
        ])
    
    upper_bu = 5.5 * np.ones((4))
    lower_bu = 0.2 * np.ones((4))

    # initial reference values
    params_0 = np.zeros((nx) + 4)
    params_0[8] = 0.5
    params_0[12] = 1
    params_0[19:23] = 2.0
    ocp.parameter_values = np.array(params_0)
    
    ocp.model.cost_expr_ext_cost_0 = calcSquareCost(payload_state_references, payload_states, payload_state_costs) + \
                                    calcSquareCost(quad_pos_state_references, quad_pos_states, quad_pos_state_costs) + \
                                    calcSquareCost(quad_vel_state_references, quad_vel_states, quad_vel_state_costs) + \
                                    calcSquareCost(quad_angular_state_references, quad_angular_states, quad_angular_state_costs) + \
                                    quatError(quat_references, quat, quat_costs) + \
                                    calcSquareCost(rpm_reference, u, R) 

    # external cost
    ocp.model.cost_expr_ext_cost = calcSquareCost(payload_state_references, payload_states, payload_state_costs) + \
                                    calcSquareCost(quad_pos_state_references, quad_pos_states, quad_pos_state_costs) + \
                                    calcSquareCost(quad_vel_state_references, quad_vel_states, quad_vel_state_costs) + \
                                    calcSquareCost(quad_angular_state_references, quad_angular_states, quad_angular_state_costs) + \
                                    quatError(quat_references, quat, quat_costs) + \
                                    calcSquareCost(rpm_reference, u, R)                                    

    # external terminal cost
    ocp.model.cost_expr_ext_cost_e = calcSquareCost(payload_state_references, payload_states, payload_state_costs) + \
                                    calcSquareCost(quad_pos_state_references, quad_pos_states, quad_pos_state_costs) + \
                                    calcSquareCost(quad_vel_state_references, quad_vel_states, quad_vel_state_costs) + \
                                    param[-1] * calcSquareCost(quad_angular_state_references, quad_angular_states, quad_angular_state_costs) + \
                                    quatError(quat_references, quat, quat_costs)



    ocp.constraints.ubx = upper_bx
    ocp.constraints.lbx = lower_bx
    ocp.constraints.ubx_0 = upper_bx
    ocp.constraints.lbx_0 = lower_bx
    ocp.constraints.ubx_e = upper_bx
    ocp.constraints.lbx_e = lower_bx
    ocp.constraints.ubu = upper_bu
    ocp.constraints.lbu = lower_bu
    ocp.constraints.idxbx = np.array([i for i in range(nx)])
    ocp.constraints.idxbx_0 = np.array([i for i in range(nx)])
    ocp.constraints.idxbx_e = np.array([i for i in range(nx)])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = np.array(params_0[0:nx])

    sim = False
    if sim:
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'   # "PARTIAL_CONDENSING_HPIPM", "FULL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_type = 'SQP'              # "SQP", "SQP_RTI"
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'          # "GAUSS_NEWTON", "EXACT"
        ocp.solver_options.integrator_type = 'IRK'                  # "ERK", "IRK", "GNSF"
        ocp.solver_options.regularize_method = 'CONVEXIFY'        # ‘NO_REGULARIZE’, ‘MIRROR’, ‘PROJECT’, ‘PROJECT_REDUC_HESS’, ‘CONVEXIFY’
        ocp.solver_options.tf = Tf
        ocp.solver_options.levenberg_marquardt = 0.05
        ocp.solver_options.nlp_solver_max_iter = 5 #50  # corresponds to number of SQP iterations/number of times lienearized
        ocp.solver_options.nlp_solver_tol_stat = 1e-2
        ocp.solver_options.qp_solver_iter_max = 50  # this corresponds to qp iterations within each SQP
        ocp.solver_options.print_level = 0
    else:
        print("REAL TIME ITERATION")
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'   # "PARTIAL_CONDENSING_HPIPM", "FULL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'              # "SQP", "SQP_RTI"
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'          # "GAUSS_NEWTON", "EXACT"
        ocp.solver_options.integrator_type = 'IRK'                  # "ERK", "IRK", "GNSF"
        ocp.solver_options.regularize_method = 'CONVEXIFY'        # ‘NO_REGULARIZE’, ‘MIRROR’, ‘PROJECT’, ‘PROJECT_REDUC_HESS’, ‘CONVEXIFY’
        ocp.solver_options.tf = Tf
        ocp.solver_options.levenberg_marquardt = 0.05
        ocp.solver_options.nlp_solver_tol_stat = 1e-2
        ocp.solver_options.qp_solver_iter_max = 50  # this corresponds to qp iterations within each SQP
        ocp.solver_options.print_level = 0

    if save:
        acados_solver = AcadosOcpSolver(ocp, verbose=True, json_file='acados_ocp_nmpc.json')
    else:
        acados_solver = AcadosOcpSolver(ocp, verbose=True)

    print("payload controller created")

    return acados_solver, ocp


def print_state(x):
    print(f"payload x:  {x[0]:.3f}, y:          {x[1]:.3f},  z:         {x[2]:.3f}")
    print(f"payload vx: {x[3]:.3f}, vy:         {x[4]:.3f},  vz:        {x[5]:.3f}")
    print(f"robot x:    {x[6]:.3f}, y:          {x[7]:.3f},  z:         {x[8]:.3f}")
    print(f"robot vx:   {x[9]:.3f}, vy:         {x[10]:.3f}, vz:        {x[11]:.3f}")

    print(f"qw:         {x[12]:.3f}, qx:        {x[13]:.3f}, qy:        {x[14]:.3f}, qz: {x[15]:.3f}")
    print(f"omega_x:    {x[16]:.3f}, omega_y:   {x[17]:.3f}, omega_z:   {x[18]:.3f}")
    print(f"motor rpms {x[19:23]}")

class MPC():
    def __init__(self):
        self.model = get_model()
        self.ocp_solver, self.ocp = create_solver(self.model)
        self.integrator = get_acados_integrator(self.model, self.ocp)
        self.dt = self.ocp.solver_options.tf/self.ocp.dims.N
        assert self.dt == 0.1
        print(f"MPC dt {self.dt}")

        self.nx = self.integrator.acados_sim.model.x.shape[0]
        self.nu = self.integrator.acados_sim.model.u.shape[0]
        self.N = self.ocp.dims.N

        # internal state & last control (new)
        self.current_state = np.zeros(self.nx)
        self.last_u = np.zeros(self.nu)


        self.ocp_solver.set(0,'u', 2*np.ones((self.nu,)))

        print("PayloadSimulator DONE")
    
    def test(self):
        total_N = 2000
        references = np.zeros((total_N, self.nx))

        init_payload_z = 0
        des_payload_z = 0.0
        init_quad_z = init_payload_z + 0.5
        print(f"Desired payload Z: {des_payload_z}")

        for i in range(total_N):
            references[i][2] = init_payload_z
            references[i][8] = init_quad_z

            references[i][12] = 1.0 # qw
            # yaw of 90 deg
            # references[i][12:16] = 0.7316, 0, 0, 0.6816

            # references[i][19:23] = [37.65404 39.20034 39.20034 37.65404] # rpm
            # references[i][19:23] = 78.83
            # references[i][19:23] = 38.44

            # if i > 1:
            #     references[i][0] = 0.5
            #     references[i][2] = des_payload_z
            #     references[i][8] = des_payload_z + 0.5

            # if i == 0:
            #     references[i][0:3] = [0, 0, 0.2]
            #     references[i][3:6] = [0, 0, .0]
            #     references[i][6:9] = [0, 0, 0.5]
            #     references[i][9:12] = [0, 0, .0]

        cost = self.run_controller(references)
        return cost

  
    def test_quad_only(self):
        total_N = 2000
        references = np.zeros((total_N, self.nx))

        for i in range(total_N):
            references[i][:3] = [0, 0, 0.0]
            references[i][3:6] = [0, 0, 0.0]
            references[i][6:9] = [0, 0, 0.5]
            references[i][9:12] = [0, 0, 0]
            references[i][12:16] = [1, 0, 0, 0]
            references[i][16:19] = [0, 0, 0]
            references[i][19:23] = [130, 130, 130, 130]

        cost = self.run_controller(references)
        return cost

    def print_state(self, x):
        print(f"payload x:  {x[0]:.3f}, y:          {x[1]:.3f},  z:         {x[2]:.3f}")
        print(f"payload vx: {x[3]:.3f}, vy:         {x[4]:.3f},  vz:        {x[5]:.3f}")
        print(f"robot x:    {x[6]:.3f}, y:          {x[7]:.3f},  z:         {x[8]:.3f}")
        print(f"robot vx:   {x[9]:.3f}, vy:         {x[10]:.3f}, vz:        {x[11]:.3f}")
        print(f"qw:         {x[12]:.3f}, qx:        {x[13]:.3f}, qy:        {x[14]:.3f}, qz: {x[15]:.3f}")

        cable = np.linalg.norm(x[0:3] - x[6:9])
        print(f"cable length {cable}")
        print(f"motor rpms {x[19:23]}")

    def extract_state(self, x_current):
        payload_pos = x_current[0:3]
        payload_vel = x_current[3:6]
        quad_pos = x_current[6:9]
        quad_vel = x_current[9:12]
        quat_xyzw = np.array([x_current[13], x_current[14], x_current[15], x_current[12]])
        quad_angular_vel = x_current[16:19]
        
        state = {
            'payload_pos': payload_pos,
            'payload_vel': payload_vel,
            'quad_pos': quad_pos,
            'quad_vel': quad_vel,
            'quat_xyzw': quat_xyzw,
            'quad_angular_vel': quad_angular_vel
        }
        return state

    def run_controller_step(self, current_state, references):
        """
        Perform one MPC step.
        references: np.ndarray shape (self.N + 1, nx,) desired state (same fields used previously in run_controller).
                   Motor rpm references are set to zero (can be extended later).
        Returns: (u0, x_next)
        """
        assert references.shape[1] == self.nx + 4, f"reference must have shape ({self.nx  + 4},)"
        assert references.shape[0] == self.N + 1, f"references must have shape ({self.N + 1}, {self.nx + 4})"

        # set initial state
        self.ocp_solver.set(0, 'x', current_state)

        zeros_array = np.zeros((4,))
        # self.ocp_solver.set(0,'p', np.concatenate((current_state, zeros_array)))
        for j in range(0, self.N+1):
            self.ocp_solver.set(j, 'p', references[j, :])

    
        print("\n")
        # print current state 
        self.print_state(current_state)
        
        # print reference states 
        for i in range(self.N + 1):
            print(f"reference state at step {i}:")
            self.print_state(references[i, :])
        try:
            _ = self.ocp_solver.solve_for_x0(current_state)
        except Exception:
            raise Exception(f"acados solver failed")

        # sim for 5 steps 

        u0 = self.ocp_solver.get(0, 'u')
        x_next = self.simulate_system_acados(current_state, u0)

        x_next = self.ocp_solver.get(5, 'x')  # use predicted state from solver instead of integrator
        # import pdb; pdb.set_trace()
        print(f"end state")
        self.print_state(x_next)
        return u0, x_next

    def run_controller(self, references, plot=True):
        total_N = references.shape[0]

        xcurrent = np.zeros((self.nx))
        xcurrent[:] = references[0][:]
        simX = np.zeros((total_N + 1, self.nx))
        simX[0, :] = xcurrent
        simU = np.zeros((total_N, self.nu))
        for i, delta in enumerate(np.linspace(0, 1, total_N)):
            print(f"\nITERATION {i}")
            collision = False
            np.set_printoptions(precision=5, suppress=True)
            print(f"state before")
            self.print_state(xcurrent)

            self.ocp_solver.set(0,'x', xcurrent)
            if i == 0:
                self.ocp_solver.set(0,'u', 2*np.ones((self.nu,)))
            # else:
            #     self.ocp_solver.set(0,'u', np.zeros((self.nu,)))

            # set desired trajectory
            zeros_array = np.zeros((4,))
            if i < (total_N - self.N):
                # mpc window is fully contained within trajectory
                for j in range(0, self.N + 1):
                    ref = references[i + j]
                    ref = np.concatenate((references[i + j], zeros_array))
                    self.ocp_solver.set(j, 'p', ref)
            else:
                for j in range(0, total_N - i):
                    assert j < self.N
                    ref = references[i + j]
                    ref = np.concatenate((references[i + j], zeros_array))
                    self.ocp_solver.set(j, 'p', ref)

                for j in range(total_N - i, self.N):
                    assert j < self.N
                    ref = references[-1]
                    ref = np.concatenate((references[-1], zeros_array))
                    self.ocp_solver.set(j, 'p', ref)


            failed = False
            try:
                _ = self.ocp_solver.solve_for_x0(xcurrent)
            except Exception:
                failed = True
                print("FAILED")
                raise Exception(f"acados solver failed")

            if False:
                self.plot_solver_solution(self.ocp_solver, total_N)

            if not failed:
                simU[i, :] = self.ocp_solver.get(0, "u")
            else:
                simU[i, :] = [0, 0, 0, 0]

            # simulate system
            xcurrent = self.simulate_system_acados(xcurrent, simU[i, :])
            simX[i+1,:] = xcurrent

            print(f"Input: {simU[i, :]}")
            print(f"state after")
            self.print_state(xcurrent)

        # (updated) plotting after loop: now also plot reference trajectories
        if plot:
            dt = self.ocp.solver_options.tf / self.N
            t = np.arange(simX.shape[0]) * dt          # simulated state timestamps (N+1)
            t_ref = np.arange(references.shape[0]) * dt  # reference timestamps (N)
            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8,6))
            # Payload position (states 0:3)
            axs[0].plot(t,     simX[:,0], label='payload x')
            axs[0].plot(t,     simX[:,1], label='payload y')
            axs[0].plot(t,     simX[:,2], label='payload z')
            axs[0].plot(t_ref, references[:,0], '--', label='ref payload x')
            axs[0].plot(t_ref, references[:,1], '--', label='ref payload y')
            axs[0].plot(t_ref, references[:,2], '--', label='ref payload z')
            axs[0].set_ylabel('Payload (m)')
            axs[0].legend()
            axs[0].grid(True)
            # Robot position (states 6:9)
            axs[1].plot(t,     simX[:,6], label='robot x')
            axs[1].plot(t,     simX[:,7], label='robot y')
            axs[1].plot(t,     simX[:,8], label='robot z')
            axs[1].plot(t_ref, references[:,6], '--', label='ref robot x')
            axs[1].plot(t_ref, references[:,7], '--', label='ref robot y')
            axs[1].plot(t_ref, references[:,8], '--', label='ref robot z')
            axs[1].set_ylabel('Robot (m)')
            axs[1].set_xlabel('Time (s)')
            axs[1].legend()
            axs[1].grid(True)
            plt.tight_layout()
            plt.show()

    def generate_initial_guess(self, xcurrent, log=False):
        # generate guess for state and inputs
        u = np.zeros((self.N, 4))
        x = np.zeros((self.N + 1, xcurrent.shape[0]))
        x[0, :] = xcurrent

        for i in range(self.N):
            u[i, :] = [0, 0, 0, 0]
            next_x = self.simulate_system_acados(x[i, :], np.array(u[i, :]))
            x[i+1, :] = next_x
            last_rpm = next_x[19:23]

            if log:
                self.print_state(x[i])
                print(f"input {u[i]}")

        return x, u

    def simulate_system_acados(self, xcurrent, U):
        self.integrator.set("x", xcurrent)
        self.integrator.set("u", U)
        status = self.integrator.simulate(xcurrent, U)

        # update state
        xcurrent = self.integrator.get("x")

        return xcurrent

if __name__ == "__main__":
    mpc = MPC()
    mpc.test()