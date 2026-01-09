import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from nav_msgs.msg import Odometry
from rclpy.qos import QoSPresetProfiles
from quadrotor_msgs.msg import MotorSpeed, PositionCommand, OutputData
from geometry_msgs.msg import PoseStamped
from pathlib import Path
from rclpy.time import Time
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import matplotlib.animation as animation
from matplotlib import ticker
from matplotlib.lines import Line2D

# register message type
from rosbags.typesys import get_types_from_idl, get_types_from_msg, register_types

trajp_text = Path(
    '/home/mrunal/arpl_ws/src/arpl_quadrotor_control/quadrotor_msgs/msg_ros2/control_cmds/TrajectoryPoint.msg'
).read_text()

pos_command_text = Path(
    '/home/mrunal/arpl_ws/src/arpl_quadrotor_control/quadrotor_msgs/msg_ros2/control_cmds/PositionCommand.msg'
).read_text()
output_data_text = Path(
    '/home/mrunal/arpl_ws/src/arpl_quadrotor_control/quadrotor_msgs/msg_ros2/OutputData.msg'
).read_text()
stamped_int_text = Path(
    '/home/mrunal/arpl_ws/src/arpl_quadrotor_control/quadrotor_msgs/msg_ros2/StampedInt.msg'
).read_text()

# plain dictionary to hold message definitions
add_types = {}

# add definition from one msg file
add_types.update(get_types_from_msg(trajp_text, 'quadrotor_msgs/msg/TrajectoryPoint'))
add_types.update(get_types_from_msg(pos_command_text, 'quadrotor_msgs/msg/PositionCommand'))
add_types.update(get_types_from_msg(output_data_text, 'quadrotor_msgs/msg/OutputData'))
add_types.update(get_types_from_msg(stamped_int_text, 'quadrotor_msgs/msg/StampedInt'))

# make types available to rosbags serializers/deserializers
register_types(add_types)

PLOT_BORDER_WIDTH = 1
PKG_PATH = "/home/mrunal/arpl_ws/src/arpl_quadrotor_control/manager/mav_manager_test/"

# Enable LaTeX rendering
plt.rcParams.update(
    {"text.usetex": True, "font.family": "sans", "font.size": 30, "font.weight": "bold"}
)

colors = [(0, 1, 0), (1, 1, 0), (1, 0, 0)]  # Green -> Yellow -> Red
colors = [
    (0.6470588235294118, 0.0, 0.14901960784313725),
    (0.84313725490196079, 0.18823529411764706, 0.15294117647058825),
    (0.95686274509803926, 0.42745098039215684, 0.2627450980392157),
    (0.99215686274509807, 0.68235294117647061, 0.38039215686274508),
    # (0.99607843137254903, 0.8784313725490196  , 0.54509803921568623),
    # (1.0                , 1.0                 , 0.74901960784313726),
    # (0.85098039215686272, 0.93725490196078431 , 0.54509803921568623),
    (0.65098039215686276, 0.85098039215686272, 0.41568627450980394),
    (0.4, 0.74117647058823533, 0.38823529411764707),
    (0.10196078431372549, 0.59607843137254901, 0.31372549019607843),
    (0.0, 0.40784313725490196, 0.21568627450980393),
]
colors.reverse()

n_bins = 100  # Number of bins for the colormap
cmap_name = 'HnYlRd'
custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)


def load_bag(base_path, file_name):
    quad_odom = []
    payload_odom = []
    position_cmd = []
    taut_mode = []

    # create reader instance and open for reading
    path = base_path + file_name
    with Reader(path) as reader:
        # topic and msgtype information is available on .connections list
        for connection in reader.connections:
            print(connection.topic, connection.msgtype)

        # iterate over messages
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == '/raxl5/ros2_control_odom':
                msg = deserialize_cdr(rawdata, connection.msgtype)
                quad_odom.append(msg)

            elif connection.topic == '/raxl5/payload/odom':
                msg = deserialize_cdr(rawdata, connection.msgtype)
                payload_odom.append(msg)

            elif connection.topic == '/raxl5/position_cmd':
                msg = deserialize_cdr(rawdata, connection.msgtype)
                position_cmd.append(msg)

    data = {
        "quad_odom": quad_odom,
        "payload_odom": payload_odom,
        "position_cmd": position_cmd,
    }

    return data


def load_bag_ball_vector(base_path, file_name):
    quad_odom = []
    payload_odom = []
    position_cmd = []
    ball_vector = []

    # create reader instance and open for reading
    path = base_path + file_name
    with Reader(path) as reader:
        # topic and msgtype information is available on .connections list
        for connection in reader.connections:
            print(connection.topic, connection.msgtype)

        # iterate over messages
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == '/raxl0/ros2_control_odom':
                msg = deserialize_cdr(rawdata, connection.msgtype)
                quad_odom.append(msg)

            elif connection.topic == '/raxl0/payload/odom':
                msg = deserialize_cdr(rawdata, connection.msgtype)
                payload_odom.append(msg)

            elif connection.topic == '/raxl0/position_cmd':
                msg = deserialize_cdr(rawdata, connection.msgtype)
                position_cmd.append(msg)

            elif connection.topic == '/raxl0/payload_ball_vector':
                msg = deserialize_cdr(rawdata, connection.msgtype)
                ball_vector.append(msg)

    data = {
        "quad_odom": quad_odom,
        "payload_odom": payload_odom,
        "position_cmd": position_cmd,
        "ball_vector": ball_vector,
    }

    return data


def get_payload_odom_data(data, t_start=None):
    # get quad odom data
    odom_data = data["payload_odom"]
    odom = np.zeros((len(odom_data), 6))
    odom_t_np = np.zeros((len(odom_data)))
    for i, msg in enumerate(odom_data):
        odom[i, :] = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        ]

        if t_start is None:
            t_start = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            odom_t_np[i] = 0
        else:
            odom_t_np[i] = (
                float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9 - t_start
            )

    return odom, odom_t_np, t_start


def get_payload_desired_data(data, t_start=None):
    pos_data = data["position_cmd"]
    pos = np.zeros((len(pos_data), 6))
    ts = np.zeros((len(pos_data)))

    for i, msg in enumerate(pos_data):
        pos[i, :] = [
            msg.points[0].position.x,
            msg.points[0].position.y,
            msg.points[0].position.z,
            msg.points[0].velocity.x,
            msg.points[0].velocity.y,
            msg.points[0].velocity.z,
        ]

        if t_start is None:
            t_start = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            ts[i] = 0
        else:
            ts[i] = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9 - t_start

    return pos, ts, t_start


def get_quad_odom_data(data, t_start=None):
    # get quad odom data
    odom_data = data["quad_odom"]
    odom = np.zeros((len(odom_data), 6))
    q_odom_t_np = np.zeros((len(odom_data)))
    for i, msg in enumerate(odom_data):
        odom[i, :] = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        ]

        if t_start is None:
            t_start = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            q_odom_t_np[i] = 0
        else:
            q_odom_t_np[i] = (
                float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9 - t_start
            )

    return odom, q_odom_t_np, t_start


def get_quad_ang_vel_odom_data(data, t_start=None):
    # get quad odom data
    odom_data = data["quad_odom"]
    odom = np.zeros((len(odom_data), 3))
    q_odom_t_np = np.zeros((len(odom_data)))
    for i, msg in enumerate(odom_data):
        odom[i, :] = [
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        ]

        if t_start is None:
            t_start = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            q_odom_t_np[i] = 0
        else:
            q_odom_t_np[i] = (
                float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9 - t_start
            )

    return odom, q_odom_t_np, t_start


def get_quad_desired_data(data, t_start=None):
    pos_data = data["position_cmd"]
    pos = np.zeros((len(pos_data), 6))
    ts = np.zeros((len(pos_data)))

    for i, msg in enumerate(pos_data):
        pos[i, :] = [
            msg.points[0].position_quad.x,
            msg.points[0].position_quad.y,
            msg.points[0].position_quad.z,
            msg.points[0].velocity_quad.x,
            msg.points[0].velocity_quad.y,
            msg.points[0].velocity_quad.z,
        ]

        if t_start is None:
            t_start = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            ts[i] = 0
        else:
            ts[i] = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9 - t_start

    # import pdb; pdb.set_trace()

    return pos, ts, t_start


def find_tracking_error(odom, t, desired_odom, desired_t):
    interp_odom = np.zeros((t.shape[0], 3))
    interp_odom[:, :] = odom[:, :3]

    # interp desired
    interp_desired_odom = np.zeros((t.shape[0], 3))
    interp_desired_odom[:, 0] = np.interp(t, desired_t, desired_odom[:, 0])
    interp_desired_odom[:, 1] = np.interp(t, desired_t, desired_odom[:, 1])
    interp_desired_odom[:, 2] = np.interp(t, desired_t, desired_odom[:, 2])

    diff = np.square(np.subtract(interp_odom, interp_desired_odom))
    square_err = np.sum(diff, axis=1)
    err = np.sqrt(square_err)
    mean_err = np.mean(err)
    print(f"Total error {mean_err * 100.0}")

    print(f"\nX")
    diff = np.abs(np.subtract(interp_odom[:, 0], interp_desired_odom[:, 0]))
    err_x = np.mean(diff)
    print(f"Mean: {err_x * 100.0}")
    print(f"Std: {np.std(diff) * 100.0}")

    print(f"\nY")
    diff = np.abs(np.subtract(interp_odom[:, 1], interp_desired_odom[:, 1]))
    err_x = np.mean(diff)
    print(f"Mean: {err_x * 100.0}")
    print(f"Std: {np.std(diff) * 100.0}")

    print(f"\nZ")
    diff = np.abs(np.subtract(interp_odom[:, 2], interp_desired_odom[:, 2]))
    err_x = np.mean(diff)
    print(f"Mean: {err_x * 100.0}")
    print(f"Std: {np.std(diff) * 100.0}")

    return mean_err


def plot_quad(
    data, coordinate, coordinate_name, save_fig_name, figsize, legend_1_loc, legend_2_loc, filename
):
    q_odom_np, q_odom_t_np, t_start = get_quad_odom_data(data)
    q_desired_np, q_desired_t_np, _ = get_quad_desired_data(data, t_start=t_start)
    payload_odom_np, payload_odom_t_np, _ = get_payload_odom_data(data, t_start=t_start)
    payload_desired_np, payload_desired_t_np, _ = get_payload_desired_data(data, t_start=t_start)

    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(q_odom_t_np, q_odom_np[:, 0], color='#DE8C2A', label=f"Quadrotor x", lw=5, alpha=0.9)
    ax1.plot(
        q_desired_t_np,
        q_desired_np[:, 0],
        linestyle='dashed',
        color='#DE8C2A',
        label=f"Desired Quadrotor x",
        lw=2,
    )

    ax1.plot(q_odom_t_np, q_odom_np[:, 1], color='#DE8C2A', label=f'Quadrotor y', lw=5, alpha=0.9)
    ax1.plot(
        q_desired_t_np,
        q_desired_np[:, 1],
        linestyle='dashed',
        color='#DE8C2A',
        label=f'Desired Quadrotor y',
        lw=2,
    )

    ax1.tick_params('y')
    ax1.set_xlabel('Time (sec)', fontweight='bold')
    ax1.set_ylabel('Distance (meters)', fontweight='bold')
    ax1.legend(loc=legend_1_loc)

    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.4))

    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.5))

    plt.tight_layout()

    plt.show()
    plt.close()


def plot_load(
    data, coordinate, coordinate_name, save_fig_name, figsize, legend_1_loc, legend_2_loc, filename
):
    q_odom_np, q_odom_t_np, t_start = get_quad_odom_data(data)
    q_desired_np, q_desired_t_np, _ = get_quad_desired_data(data, t_start=t_start)
    payload_odom_np, payload_odom_t_np, _ = get_payload_odom_data(data, t_start=t_start)
    payload_desired_np, payload_desired_t_np, _ = get_payload_desired_data(data, t_start=t_start)

    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(
        payload_odom_t_np,
        payload_odom_np[:, 0],
        color='#485000',
        label=f"Payload x",
        lw=5,
        alpha=0.9,
    )
    ax1.plot(
        payload_desired_t_np,
        payload_desired_np[:, 0],
        linestyle='dashed',
        color='#485000',
        label=f"Desired Payload x",
        lw=2,
    )

    ax1.plot(
        payload_odom_t_np,
        payload_odom_np[:, 1],
        color='#485000',
        label=f'Payload y',
        lw=5,
        alpha=0.9,
    )
    ax1.plot(
        payload_desired_t_np,
        payload_desired_np[:, 1],
        linestyle='dashed',
        color='#485000',
        label=f'Desired Payload y',
        lw=2,
    )

    ax1.tick_params('y')
    ax1.set_xlabel('Time (sec)', fontweight='bold')
    ax1.set_ylabel('Distance (meters)', fontweight='bold')
    ax1.legend(loc=legend_1_loc)

    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.4))

    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.5))

    plt.tight_layout()

    plt.show()
    plt.close()


def plot_position_info(data, filename):
    # if os.path.isdir(PKG_PATH + "/plots/" + filename):
    #     print(f"{PKG_PATH}/plots/{filename} already exists")
    #     return

    # os.mkdir(PKG_PATH + "/plots/" + filename)

    quad_odom = data["quad_odom"]
    payload_odom = data["payload_odom"]
    position_cmd = data["position_cmd"]

    plot_quad(data, 0, 'x', "data_x", (15, 12), 'upper left', 'upper right', filename)
    plot_load(data, 0, 'x', "data_x", (15, 12), 'upper left', 'upper right', filename)


def clip_bag(data, t_clip_start=0, t_clip_end=999999):
    clipped_data = {}
    for key in data.keys():
        print(key)
        clipped_data[key] = []

        msg = data[key][0]

        for i in range(len(data[key])):
            msg = data[key][i]
            t = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            if t >= t_clip_start and t <= t_clip_end:
                clipped_data[key].append(msg)

            if t > t_clip_end:
                break

    return clipped_data


def plot_quad_and_load_grid_xy(
    data, figsize=(18, 6), filename=None, quad_color='#a56a6b', load_color='#08747c'
):
    """
    2 × 2 figure (x-row, y-row).
    Left column  : quadrotor  (current & desired)
    Right column : payload    (current & desired)
    Y-axis limits: [min(data)−0.2 , max(data)+0.2] for each row.
    """
    # ------------- fetch data --------------------------------------------------
    q_odom, q_t, t0 = get_quad_odom_data(data)
    q_des, q_td, _ = get_quad_desired_data(data, t_start=t0)
    L_odom, L_t, _ = get_payload_odom_data(data, t_start=t0)
    L_des, L_td, _ = get_payload_desired_data(data, t_start=t0)

    mean_err = find_tracking_error(L_odom, L_t, L_des, L_td)

    # ------------- layout ------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    axes[0, 0].set_title('Quadrotor', fontweight='bold')
    axes[0, 1].set_title('Payload', fontweight='bold')

    cur = dict(lw=3, alpha=0.9)
    des = dict(lw=2, ls='--')

    # Helper to compute y-limits
    def y_limits(row_idx):
        data_stack = np.hstack(
            (q_odom[:, row_idx], q_des[:, row_idx], L_odom[:, row_idx], L_des[:, row_idx])
        )
        return data_stack.min() - 0.2, data_stack.max() + 0.2

    # ---- row 0 : x ------------------------------------------------------------
    ymin, ymax = y_limits(0)
    axes[0, 0].plot(q_t, q_odom[:, 0], color=quad_color, **cur)
    axes[0, 0].plot(q_td, q_des[:, 0], color=quad_color, **des)
    axes[0, 1].plot(L_t, L_odom[:, 0], color=load_color, **cur)
    axes[0, 1].plot(L_td, L_des[:, 0], color=load_color, **des)
    axes[0, 0].set_ylabel('x [m]')
    axes[0, 0].set_ylim(ymin, ymax)
    axes[0, 1].set_ylim(ymin, ymax)

    # ---- row 1 : y ------------------------------------------------------------
    ymin, ymax = y_limits(1)
    axes[1, 0].plot(q_t, q_odom[:, 1], color=quad_color, **cur)
    axes[1, 0].plot(q_td, q_des[:, 1], color=quad_color, **des)
    axes[1, 1].plot(L_t, L_odom[:, 1], color=load_color, **cur)
    axes[1, 1].plot(L_td, L_des[:, 1], color=load_color, **des)
    axes[1, 0].set_ylabel('y [m]')
    axes[1, 0].set_ylim(ymin, ymax)
    axes[1, 1].set_ylim(ymin, ymax)

    # ------------- ticks & grids ----------------------------------------------
    for row in axes:
        for ax in row:
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax.grid(True, linestyle='--', alpha=0.5, color='darkgray')
            ax.tick_params(which='minor', labelbottom=False, labelleft=False)

    axes[1, 0].set_xlabel('time [s]')
    axes[1, 1].set_xlabel('time [s]')

    # ------------- unified legend ---------------------------------------------
    handles = [
        Line2D([], [], color=quad_color, lw=3, label='quad current'),
        Line2D([], [], color=quad_color, lw=2, ls='--', label='quad desired'),
        Line2D([], [], color=load_color, lw=3, label='payload current'),
        Line2D([], [], color=load_color, lw=2, ls='--', label='payload desired'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_quad_and_load_grid_xyz(
    data,
    figsize=(18, 9),
    filename=None,
    quad_color='#DE8C2A',
    load_color='#485000',
    include_title=True,
):
    """
    3 × 2 figure (x-row, y-row, z-row).
    Left column  : quadrotor  (current & desired)
    Right column : payload    (current & desired)
    Y-axis limits: [min(data)−0.2 , max(data)+0.2] for each row.
    """
    # ------------- fetch data --------------------------------------------------
    q_odom, q_t, t0 = get_quad_odom_data(data)
    q_des, q_td, _ = get_quad_desired_data(data, t_start=t0)
    L_odom, L_t, _ = get_payload_odom_data(data, t_start=t0)
    L_des, L_td, _ = get_payload_desired_data(data, t_start=t0)

    q_des[:, 2] -= 0.1

    print("Payload tracking error")
    mean_err = find_tracking_error(L_odom, L_t, L_des, L_td)

    print("Quadrotor tracking error")
    mean_err = find_tracking_error(q_odom, q_t, q_des, q_td)

    # ------------- layout ------------------------------------------------------
    fig, axes = plt.subplots(
        3, 2, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]}
    )

    if include_title:
        axes[0, 0].set_title('Quadrotor', fontweight='bold')
        axes[0, 1].set_title('Payload', fontweight='bold')

    cur = dict(lw=4, alpha=0.5)
    des = dict(lw=2, ls='--')

    # Helper to compute y-limits for a given state index
    def y_limits(idx):
        data_stack = np.hstack((q_odom[:, idx], q_des[:, idx], L_odom[:, idx], L_des[:, idx]))
        return data_stack.min() - 0.2, data_stack.max() + 0.2

    # ---- row 0 : x ------------------------------------------------------------
    ymin, ymax = y_limits(0)
    axes[0, 0].plot(q_t, q_odom[:, 0], color=quad_color, **cur)
    axes[0, 0].plot(q_td, q_des[:, 0], color=quad_color, **des)
    axes[0, 1].plot(L_t, L_odom[:, 0], color=load_color, **cur)
    axes[0, 1].plot(L_td, L_des[:, 0], color=load_color, **des)
    axes[0, 0].set_ylabel('x [m]')
    axes[0, 0].set_ylim(ymin, ymax)
    axes[0, 1].set_ylim(ymin, ymax)

    # ---- row 1 : y ------------------------------------------------------------
    ymin, ymax = y_limits(1)
    axes[1, 0].plot(q_t, q_odom[:, 1], color=quad_color, **cur)
    axes[1, 0].plot(q_td, q_des[:, 1], color=quad_color, **des)
    axes[1, 1].plot(L_t, L_odom[:, 1], color=load_color, **cur)
    axes[1, 1].plot(L_td, L_des[:, 1], color=load_color, **des)
    axes[1, 0].set_ylabel('y [m]')
    axes[1, 0].set_ylim(ymin, ymax)
    axes[1, 1].set_ylim(ymin, ymax)

    # ---- row 2 : z ------------------------------------------------------------
    ymin, ymax = y_limits(2)
    axes[2, 0].plot(q_t, q_odom[:, 2], color=quad_color, **cur)
    axes[2, 0].plot(q_td, q_des[:, 2], color=quad_color, **des)
    axes[2, 1].plot(L_t, L_odom[:, 2], color=load_color, **cur)
    axes[2, 1].plot(L_td, L_des[:, 2], color=load_color, **des)
    axes[2, 0].set_ylabel('z [m]')
    axes[2, 0].set_ylim(ymin, ymax)
    axes[2, 1].set_ylim(ymin, ymax)

    # ------------- ticks & grids ----------------------------------------------
    for row_idx, row in enumerate(axes):
        for ax in row:
            if row_idx in [0, 1]:
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
                # ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                # ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
                ax.grid(True, linestyle='--', alpha=0.8, color='darkgray', linewidth=1.2)
                ax.tick_params(
                    which='minor',
                    labelbottom=False,
                    labelleft=False,
                    width=1.0,
                )
            else:
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
                # ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                # ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
                ax.grid(True, linestyle='--', alpha=0.8, color='darkgray', linewidth=1.2)
                ax.tick_params(
                    which='minor',
                    labelbottom=False,
                    labelleft=False,
                    width=1.0,
                )

    # x-labels only on bottom row
    axes[2, 0].set_xlabel('time [s]')
    axes[2, 1].set_xlabel('time [s]')
    for ax in fig.axes:  # loops over every Axes in this Figure
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)  # set your preferred width here
    # ------------- unified legend ---------------------------------------------
    handles = [
        Line2D([], [], color=quad_color, lw=4, label='quad current'),
        Line2D([], [], color=quad_color, lw=2, ls='--', label='quad desired'),
        Line2D([], [], color=load_color, lw=4, label='payload current'),
        Line2D([], [], color=load_color, lw=2, ls='--', label='payload desired'),
    ]

    if include_title:
        fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # ------------- save or display --------------------------------------------
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_quad_and_load_grid_vel(
    data, figsize=(18, 6), filename=None, quad_color='#a56a6b', load_color='#08747c'
):
    """
    2 × 2 figure (x-row, y-row).
    Left column  : quadrotor  (current & desired)
    Right column : payload    (current & desired)
    Y-axis limits: [min(data)−0.2 , max(data)+0.2] for each row.
    """
    # ------------- fetch data --------------------------------------------------
    q_odom, q_t, t0 = get_quad_odom_data(data)
    q_des, q_td, _ = get_quad_desired_data(data, t_start=t0)
    L_odom, L_t, _ = get_payload_odom_data(data, t_start=t0)
    L_des, L_td, _ = get_payload_desired_data(data, t_start=t0)

    print("Payload tracking error")
    mean_err = find_tracking_error(L_odom, L_t, L_des, L_td)

    # ------------- layout ------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    axes[0, 0].set_title('Quadrotor', fontweight='bold')
    axes[0, 1].set_title('Payload', fontweight='bold')

    cur = dict(lw=3, alpha=0.9)
    des = dict(lw=2, ls='--')

    # Helper to compute y-limits
    def y_limits(row_idx):
        data_stack = np.hstack(
            (q_odom[:, row_idx], q_des[:, row_idx], L_odom[:, row_idx], L_des[:, row_idx])
        )
        return data_stack.min() - 0.2, data_stack.max() + 0.2

    # ---- row 0 : x ------------------------------------------------------------
    ymin, ymax = y_limits(0)
    axes[0, 0].plot(q_t, q_odom[:, 3], color=quad_color, **cur)
    axes[0, 0].plot(q_td, q_des[:, 3], color=quad_color, **des)
    axes[0, 1].plot(L_t, L_odom[:, 3], color=load_color, **cur)
    axes[0, 1].plot(L_td, L_des[:, 3], color=load_color, **des)
    axes[0, 0].set_ylabel('x [m]')
    axes[0, 0].set_ylim(ymin, ymax)
    axes[0, 1].set_ylim(ymin, ymax)

    # ---- row 1 : y ------------------------------------------------------------
    ymin, ymax = y_limits(1)
    axes[1, 0].plot(q_t, q_odom[:, 4], color=quad_color, **cur)
    axes[1, 0].plot(q_td, q_des[:, 4], color=quad_color, **des)
    axes[1, 1].plot(L_t, L_odom[:, 4], color=load_color, **cur)
    axes[1, 1].plot(L_td, L_des[:, 4], color=load_color, **des)
    axes[1, 0].set_ylabel('y [m]')
    axes[1, 0].set_ylim(ymin, ymax)
    axes[1, 1].set_ylim(ymin, ymax)

    # ------------- ticks & grids ----------------------------------------------
    for row in axes:
        for ax in row:
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax.grid(True, linestyle='--', alpha=0.5, color='darkgray')
            ax.tick_params(which='minor', labelbottom=False, labelleft=False)

    axes[1, 0].set_xlabel('time [s]')
    axes[1, 1].set_xlabel('time [s]')

    # ------------- unified legend ---------------------------------------------
    handles = [
        Line2D([], [], color=quad_color, lw=3, label='quad current'),
        Line2D([], [], color=quad_color, lw=2, ls='--', label='quad desired'),
        Line2D([], [], color=load_color, lw=3, label='payload current'),
        Line2D([], [], color=load_color, lw=2, ls='--', label='payload desired'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    data = load_bag("/home/mrunal/Documents/TOPCAT/bags/", "june_30_test_25_maze_1")
    data = clip_bag(data, t_clip_start=1751325432.508, t_clip_end=1751325437.0)
    plot_quad_and_load_grid_xyz(data)

    data = load_bag("/home/mrunal/Documents/TOPCAT/bags/", "july_1_test_2_maze_3_v1")
    data = clip_bag(data, t_clip_start=1751385846.276, t_clip_end=1751385851.333)
    plot_quad_and_load_grid_xyz(data, include_title=False)

    data = load_bag("/home/mrunal/Documents/TOPCAT/bags/", "july_1_test_21_maze_7")
    data = clip_bag(data, t_clip_start=1751404553.062, t_clip_end=1751404556.068)
    plot_quad_and_load_grid_xyz(data, include_title=False)
