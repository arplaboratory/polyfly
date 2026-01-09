def single_payload_geometric_controller(self, state, qd_params, pl_params):
    # DESCRIPTION:
    # Controller for rigid link connected payload and MAV(s) 

    # INPUTS:
    # state          - a dictionary containing state of the payload and MAV combined, specifically
    #               Key             Type            Size            Description              
    #               'pos'           ndarray         3 by 1          payload position
    #               'vel'           ndarray         3 by 1          payload velocity
    #               'qd_pos'        ndarray         3 by 1          MAV position
    #               'qd_vel'        ndarray         3 by 1          MAV velocity
    #               'qd_quat'       ndarray         4 by 1          MAV orientation as unit quaternion
    #               'qd_omega'      ndarray         3 by 1          MAV angular velocity
    #               'qd_rot'        ndarray         3 by 3          MAV orientation as rotation matrix
    #               'pos_des'       ndarray         3 by 1          desired payload position
    #               'vel_des'       ndarray         3 by 1          desired payload velocity
    #               'acc_des'       ndarray         3 by 1          desired payload acceleration
    #               'jrk_des'       ndarray         3 by 1          desired payload jerk
    #               'quat_des'      ndarray         4 by 1          desired payload orientation as unit quaterion
    #                                                               set to [[1.], [0.], [0.], [0.]] currently
    #               'omega_des'     ndarray         3 by 1          desired payload angular velocity
    #                                                               set to [[0., 0., 0.]] currently
    #               'qd_yaw_des'    float           NA              desired MAV yaw, set to 0.0 current
    #               'qd_yawdot_des' float           NA              time derivative of desired MAV yaw, set to 0.0 currently
    # pl_params   - a read_params class object containing payload parameters
    # qd_params   - a read_params class objects containing all MAV parameters
    
    # OUTPUTS:
    # F               - a 1 by 1 ndarray, denoting the thrust force
    # M               - a list of size 3, containing three 1d ndarray of size 1, denoting the moment
    #                   M = [[array([Mx])]
    #                        [array([My])]
    #                        [array([Mz])]]                
    ## Parameter Initialization
    self.icnt = 0
    g = 9.81
    e3 = np.array([[0],[0],[1]])

    quad_m = 0.72
    pl_m = 0.2
    l = 0.5

    ## State Initialization
    quad_load_rel_pos = state["qd_pos"]-state["pos"]
    quad_load_rel_vel = state["qd_vel"]-state["vel"]
    quad_load_distance = np.linalg.norm(quad_load_rel_pos)
    xi_ = -quad_load_rel_pos/quad_load_distance
    xixiT_ = xi_ @ np.transpose(xi_)
    xidot_ = -quad_load_rel_vel/quad_load_distance
    xi_asym_ = vec2asym(xi_)
    w_ = np.cross(xi_, xidot_, axisa=0, axisb=0).T
    Rot_worldtobody = state["qd_rot"]

    ## Payload Position control
    #Position error
    ep = state["pos_des"]-state["pos"]
    #Velocity error
    ed = state["vel_des"]-state["vel"]

    # Desired acceleration This equation drives the errors of trajectory to zero.
    acceleration_des = state["acc_des"] + g*e3 + pl_params.Kp @ ep + pl_params.Kd @ ed

    # Desired yaw and yawdot
    yaw_des = state["qd_yaw_des"] # This can remain for Quad
    yawdot_des = state["qd_yawdot_des"]

    ## Cable Direction Control
    # Desired cable direction

    mu_des_ = (quad_m + pl_m) * acceleration_des + quad_m * l * (np.transpose(xidot_) @ xidot_) * xi_
    xi_des_ = -mu_des_ / np.linalg.norm(mu_des_)
    xi_des_dot_ = np.zeros((3, 1), dtype=float)
    w_des_ = np.cross(xi_des_, xi_des_dot_, axisa=0, axisb=0).T
    w_des_dot_ = np.zeros((3, 1), dtype=float)
    mu_ = xixiT_ @ mu_des_

    e_xi = np.cross(xi_des_, xi_, axisa=0, axisb=0).T
    e_w = w_ + xi_asym_ @ xi_asym_ @ w_des_
    Force = mu_ - quad_m*l*np.cross(xi_, qd_params[0].Kxi @ e_xi + qd_params[0].Kw @ e_w+ (xi_.T @ w_des_) * xidot_ + xi_asym_ @ xi_asym_ @ w_des_dot_, axisa=0, axisb=0).T
    F = np.transpose(Force) @ Rot_worldtobody @ e3

    # Attitude Control        
    Rot_des = np.zeros((3,3), dtype=float)
    Z_body_in_world = Force/np.linalg.norm(Force)
    Rot_des[:,2:3] = Z_body_in_world
    X_unit = np.array([[np.cos(yaw_des)], [np.sin(yaw_des)], [0]])
    Y_body_in_world = np.cross(Z_body_in_world, X_unit, axisa=0, axisb=0).T
    Y_body_in_world = Y_body_in_world/np.linalg.norm(Y_body_in_world)
    Rot_des[:,1:2] = Y_body_in_world
    X_body_in_world = np.cross(Y_body_in_world,Z_body_in_world, axisa=0, axisb=0).T
    Rot_des[:,0:1] = X_body_in_world

    # Errors of anlges and angular velocities
    e_Rot = np.transpose(Rot_des) @ Rot_worldtobody - Rot_worldtobody.T @ Rot_des
    e_angle = vee(e_Rot)/2

    p_des = 0.0
    q_des = 0.0
    r_des = yawdot_des*Z_body_in_world[2]
    e_omega = state["qd_omega"] - Rot_worldtobody.T @ Rot_des @ np.array([[p_des], [q_des], [r_des]])

    # Moment
    # Missing the angular acceleration term but in general it is neglectable.
    M = - qd_params[0].Kpe @ e_angle - qd_params[0].Kde @ e_omega + np.cross(state["qd_omega"],qd_params[0].I @ state["qd_omega"], axisa=0, axisb=0).T
    return F, M