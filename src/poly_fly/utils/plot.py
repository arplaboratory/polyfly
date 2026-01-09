import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from scipy.spatial import ConvexHull
from pathlib import Path
from poly_fly.optimal_planner.polytopes import Obs, SquarePayload, Cable, Quadrotor


def plot_interpolated_positions_and_obstacles(params, interpolated_positions):
    """Plots the obstacles and the interpolated positions."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each obstacle as a filled 3D polytope
    for key in params.obstacles.keys():
        obs = params.obstacles[key]
        x, y, z = obs['x'], obs['y'], obs['z']
        l, b, h = obs['l'], obs['b'], obs['h']

        obstacle = Obs(obs['x'], obs['y'], obs['z'], obs['l'], obs['b'], obs['h'])
        vertices = obstacle.get_vertices()

        # Compute the convex hull of the vertices
        hull = ConvexHull(vertices)
        for simplex in hull.simplices:
            face = [vertices[i] for i in simplex]
            ax.add_collection3d(Poly3DCollection([face], alpha=0.8, facecolors='darkgray'))

    # Plot the interpolated positions
    ax.plot(
        interpolated_positions[0, :],
        interpolated_positions[1, :],
        interpolated_positions[2, :],
        color='blue',
        label='Interpolated Positions',
    )
    ax.scatter(
        interpolated_positions[0, :],
        interpolated_positions[1, :],
        interpolated_positions[2, :],
        color='blue',
        s=50,
    )

    # Set plot limits and labels
    ax.set_xlim([-1, 4])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-1, 5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Obstacles and Interpolated Positions")
    plt.legend()
    plt.show()


def plot_result(
    params,
    sol,
    jerk,
    differential_flatness,
    get_quadrotor_rotation_matrix,
    ax=None,
    show=True,
    title="3D Trajectory Visualization with Obstacles",
    delta_plot=1,
    ortho_view=False,
    save=False,
    save_filename=None,
):
    """Plots the 3D trajectory visualization with obstacles."""
    axis_given = True
    if ax is None:
        fig = plt.figure(figsize=(10, 8), dpi=600)  # Increased DPI for higher resolution
        ax = fig.add_subplot(111, projection='3d')
        axis_given = False

    # --- Color Palette ---
    obstacle_color = '#708090'  # Light beige
    # obstacle_color = '#7a95c4'# blue arxiv
    quadrotor_color = '#fac585'
    quadrotor_color = '#DE8C2A'
    payload_color = '#485000'  # Medium blue
    cable_color = '#7F7F7F'  # Light blue
    cable_color = 'black'

    # --- Extract data ---
    payload_positions = sol[:, :3]
    payload_velocities = sol[:, 3:6]
    payload_accelerations = sol[:, 6:9]
    quad_positions = np.zeros_like(payload_positions)

    for i in range(payload_positions.shape[0]):
        x = payload_positions[i, :]
        v = payload_velocities[i, :]
        a = payload_accelerations[i, :]
        jrk = jerk[i, :] if i < jerk.shape[0] else np.zeros(3)

        result = differential_flatness(x, v, a, jrk, params)
        quad_positions[i, :] = result["pos_quad"]

    payload_alpha = 0.8
    quadrotor_alpha = 0.7
    cable_alpha = 0.5
    obstacle_alpha = 0.2

    # narrow gap viz
    # payload_alpha = 0.2
    # quadrotor_alpha = 0.2
    # cable_alpha = 1.0
    # obstacle_alpha = 1.0

    # --- Plot Payload Trajectory ---
    for i in range(0, payload_positions.shape[0], delta_plot):
        x, y, z = payload_positions[i, :]

        # Get the vertices of the payload polytope
        payload_polytope = SquarePayload(params)
        vertices = np.array(payload_polytope.get_vertices())

        # Offset the vertices by the payload position
        vertices += np.array([x, y, z])

        # Compute the convex hull of the vertices
        hull = ConvexHull(vertices)
        for simplex in hull.simplices:
            face = [vertices[j] for j in simplex]
            ax.add_collection3d(
                Poly3DCollection([face], alpha=payload_alpha, facecolors=payload_color)
            )

    # --- Plot Quadrotor Trajectory ---
    for i in range(0, quad_positions.shape[0], delta_plot):
        x, y, z = quad_positions[i, :]

        # Generate vertices using the Quadrotor class
        quadrotor = Quadrotor(params)
        vertices = np.array(quadrotor.get_vertices())

        # Apply the rotation matrix
        rotation_matrix = get_quadrotor_rotation_matrix(
            payload_accelerations[i, :], params, use_robot_rotation=True
        )
        rotated_vertices = np.dot(vertices, rotation_matrix.T)

        # Offset the vertices by the quadrotor position
        translated_vertices = rotated_vertices + np.array([x, y, z])

        # Compute the convex hull of the vertices
        hull = ConvexHull(translated_vertices)
        for simplex in hull.simplices:
            face = [translated_vertices[j] for j in simplex]
            ax.add_collection3d(
                Poly3DCollection([face], alpha=quadrotor_alpha, facecolors=quadrotor_color)
            )

    # --- Plot Cable ---
    for i in range(0, payload_positions.shape[0], delta_plot):
        start = payload_positions[i, :]
        end = quad_positions[i, :]
        cable_length = np.linalg.norm(end - start)
        cable_direction = (end - start) / cable_length
        cable_width = params.cable_radius / 2  # Adjust width for cable thickness

        # Create two perpendicular vectors to the cable direction
        up_vector = np.array([0, 0, 1])
        if np.abs(np.dot(cable_direction, up_vector)) > 0.9:
            up_vector = np.array([0, 1, 0])
        perp_vector1 = np.cross(cable_direction, up_vector)
        perp_vector1 /= np.linalg.norm(perp_vector1)
        perp_vector2 = np.cross(cable_direction, perp_vector1)

        # Define the 8 vertices of the cuboid
        vertices = [
            start + (cable_width / 2) * perp_vector1 + (cable_width / 2) * perp_vector2,
            start + (cable_width / 2) * perp_vector1 - (cable_width / 2) * perp_vector2,
            start - (cable_width / 2) * perp_vector1 - (cable_width / 2) * perp_vector2,
            start - (cable_width / 2) * perp_vector1 + (cable_width / 2) * perp_vector2,
            end + (cable_width / 2) * perp_vector1 + (cable_width / 2) * perp_vector2,
            end + (cable_width / 2) * perp_vector1 - (cable_width / 2) * perp_vector2,
            end - (cable_width / 2) * perp_vector1 - (cable_width / 2) * perp_vector2,
            end - (cable_width / 2) * perp_vector1 + (cable_width / 2) * perp_vector2,
        ]

        # Define the 6 faces of the cuboid
        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front face
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back face
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left face
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right face
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
        ]

        # Add the cuboid to the plot
        ax.add_collection3d(Poly3DCollection(faces, facecolors=cable_color, alpha=cable_alpha))

    # --- Plot Obstacles ---
    for key in params.obstacles.keys():
        obs = params.obstacles[key]
        x, y, z = obs['x'], obs['y'], obs['z']
        l, b, h = obs['l'], obs['b'], obs['h']
        obstacle = Obs(obs['x'], obs['y'], obs['z'], obs['l'], obs['b'], obs['h'])
        vertices = obstacle.get_vertices()

        # Compute the convex hull of the vertices
        hull = ConvexHull(vertices)
        for simplex in hull.simplices:
            face = [vertices[i] for i in simplex]
            ax.add_collection3d(
                Poly3DCollection([face], alpha=obstacle_alpha, facecolors=obstacle_color)
            )

    # # --- Plot Trajectory Lines ---
    # ax.plot(
    #     payload_positions[0, :],
    #     payload_positions[1, :],
    #     payload_positions[2, :],
    #     color=payload_color,
    #     linewidth=0.8,
    #     label='Payload Trajectory',
    # )
    # ax.plot(
    #     quad_positions[0, :],
    #     quad_positions[1, :],
    #     quad_positions[2, :],
    #     color=quadrotor_color,
    #     linewidth=0.8,
    #     label='Quadrotor Trajectory',
    # )

    # --- Plot Markers ---
    # ax.scatter(
    #     payload_positions[0, ::5],
    #     payload_positions[1, ::5],
    #     payload_positions[2, ::5],
    #     color=payload_color,
    #     marker='o',
    #     s=5,
    #     label='Payload Points',
    # )
    # ax.scatter(
    #     quad_positions[0, ::5],
    #     quad_positions[1, ::5],
    #     quad_positions[2, ::5],
    #     color=quadrotor_color,
    #     marker='o',
    #     s=5,
    #     label='Quadrotor Points',
    # )

    # --- Plot Settings ---
    ax.w_xaxis.line.set_color((0, 0, 0, 0))  # X-axis line invisible
    ax.w_yaxis.line.set_color((0, 0, 0, 0))  # Y-axis line invisible
    ax.w_zaxis.line.set_color((0, 0, 0, 0))  # Z-axis line invisible

    # remove tick marks and their numeric labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # (optional) make sure no stray tick lines are drawn
    ax.tick_params(axis='both', which='both', length=0)

    xlims = [-3, 7]
    xlims[0] = min(xlims[0], np.min(payload_positions[:, 0]) - 1)
    xlims[1] = max(xlims[0], np.max(payload_positions[:, 0]) + 1)
    ylims = [-5, 5]
    ylims[0] = min(ylims[0], np.min(payload_positions[:, 1]) - 1)
    ylims[1] = -ylims[0]
    zlims = [-5, 5]
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_zlim(zlims)

    set_box_equal(ax)
    ax.grid(False)  # start with no default grid

    # Hide the two vertical wall panes so no grid appears on them
    ax.xaxis.pane.set_visible(False)  # hide the YZ wall
    ax.yaxis.pane.set_visible(False)  # hide the XZ wall

    # Keep the floor (XY) pane visible
    ax.zaxis.pane.set_visible(False)

    # Draw a custom light-gray grid on the floor
    z_floor = 0  # bottom z-limit for the floor
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    grid_spacing = 0.5  # adjust spacing to taste

    xs = np.arange(np.floor(x_min), np.ceil(x_max) + grid_spacing, grid_spacing)
    ys = np.arange(np.floor(y_min), np.ceil(y_max) + grid_spacing, grid_spacing)

    # vertical grid lines (parallel to Y axis)
    for x in xs:
        ax.plot(
            [x, x], [y_min, y_max], [z_floor, z_floor], color='lightgray', linewidth=0.5, zorder=0
        )

    # horizontal grid lines (parallel to X axis)
    for y in ys:
        ax.plot(
            [x_min, x_max], [y, y], [z_floor, z_floor], color='lightgray', linewidth=0.5, zorder=0
        )

    if show:
        ax.view_init(elev=90, azim=-180)  # elev = 90° ⇢ look from +Z toward XY
        # azim sets the left-right rotation; -90°
        # points X to the right, Y toward you
        # Optional: remove perspective distortion so it appears perfectly “top-down”
        ax.set_proj_type('ortho')  # requires Matplotlib ≥3.3
        plt.show()

    if save:
        if ortho_view:
            ax.view_init(elev=90, azim=-180)  # elev = 90° ⇢ look from +Z toward XY
            # azim sets the left-right rotation; -90°
            # points X to the right, Y toward you
            # Optional: remove perspective distortion so it appears perfectly “top-down”
            ax.set_proj_type('ortho')  # requires Matplotlib ≥3.3

        fig.savefig(Path(save_filename), dpi=600)
        print(f"Figure saved to {save_filename}")

    if not axis_given:
        plt.close(fig)  # Close the figure to free memory


def set_box_equal(ax):
    """
    Emulate ax.set_box_aspect((1,1,1)) for Matplotlib < 3.4
    (keeps x, y, z units roughly equal in screen space).
    """
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    max_range = (
        max(
            abs(xlim[1] - xlim[0]),
            abs(ylim[1] - ylim[0]),
            abs(zlim[1] - zlim[0]),
        )
        / 2.0
    )

    mid_x = np.mean(xlim)
    mid_y = np.mean(ylim)
    mid_z = np.mean(zlim)

    ax.set_xlim3d(mid_x - max_range, mid_x + max_range)
    ax.set_ylim3d(mid_y - max_range, mid_y + max_range)
    ax.set_zlim3d(mid_z - max_range, mid_z + max_range)


def plot_results_2d(params, time, sol, jerk, differential_flatness):
    """
    Quick-look time-series:
        • payload  : pos / vel / acc
        • quadrotor: pos / vel / acc   (via differential flatness)
        • inputs   : x,y,z jerks
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # ── payload state slices ───────────────────────────────────────────
    payload_pos = sol[:, 0:3]  # xL, yL, zL
    payload_vel = sol[:, 3:6]
    payload_acc = sol[:, 6:9]

    # ── derive the quadrotor state via flatness ------------------------
    quad_pos = np.zeros_like(payload_pos)
    quad_vel = np.zeros_like(payload_vel)
    quad_acc = np.zeros_like(payload_acc)

    for k in range(payload_pos.shape[0]):
        x = payload_pos[k, :]
        v = payload_vel[k, :]
        a = payload_acc[k, :]
        jrk = jerk[k, :] if k < jerk.shape[0] else np.zeros(3)
        flat = differential_flatness(x, v, a, jrk, params)
        quad_pos[k, :] = flat["pos_quad"]
        quad_vel[k, :] = flat["vel_quad"]
        quad_acc[k, :] = flat["acc_quad"]

    # ── figure & axes ---------------------------------------------------
    fig, axs = plt.subplots(3, 3, figsize=(15, 12), sharex='col')
    fig.subplots_adjust(hspace=0.35)

    # helper to keep code short
    def _plot(ax, y, title, ylabel, labels):
        for i, lbl in enumerate(labels):
            ax.plot(time, y[:, i], label=lbl)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend(fontsize="small")

    # row 0 ─────────────────────────────────────────────────────────────
    _plot(axs[0, 0], payload_pos, "Payload Position", "m", ["x", "y", "z"])
    _plot(axs[0, 1], quad_pos, "Quadrotor Position", "m", ["x", "y", "z"])
    _plot(axs[0, 2], jerk, "Input Jerks", "m/s³", ["jx", "jy", "jz"])

    # row 1 ─────────────────────────────────────────────────────────────
    _plot(axs[1, 0], payload_vel, "Payload Velocity", "m/s", ["ẋ", "ẏ", "ż"])
    _plot(axs[1, 1], quad_vel, "Quadrotor Velocity", "m/s", ["ẋ", "ẏ", "ż"])
    axs[1, 2].axis("off")  # empty slot

    # row 2 ─────────────────────────────────────────────────────────────
    _plot(axs[2, 0], payload_acc, "Payload Acceleration", "m/s²", ["ẍ", "ÿ", "ż̈"])
    _plot(axs[2, 1], quad_acc, "Quadrotor Acceleration", "m/s²", ["ẍ", "ÿ", "ż̈"])
    axs[2, 2].axis("off")  # empty slot

    # common X-label
    for ax in axs[-1, :2]:
        ax.set_xlabel("Time [s]")

    plt.tight_layout()
    plt.show()


def plot_obstacles(params):
    """Plots the obstacles defined in the YAML file."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each obstacle as a filled 3D box with dark gray color
    for key in params.obstacles.keys():
        obs = params.obstacles[key]
        x, y, z = obs['x'], obs['y'], obs['z']
        l, b, h = obs['l'], obs['b'], obs['h']

        # Define the vertices of the box
        vertices = [
            [x - l / 2, y - b / 2, z - h / 2],
            [x - l / 2, y + b / 2, z - h / 2],
            [x + l / 2, y + b / 2, z - h / 2],
            [x + l / 2, y - b / 2, z - h / 2],
            [x - l / 2, y - b / 2, z + h / 2],
            [x - l / 2, y + b / 2, z + h / 2],
            [x + l / 2, y + b / 2, z + h / 2],
            [x + l / 2, y - b / 2, z + h / 2],
        ]

        # Define the 6 faces of the box
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front face
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back face
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right face
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left face
        ]

        # Add the box to the plot
        ax.add_collection3d(Poly3DCollection(faces, alpha=0.5, facecolors='darkgray'))

    # Set plot limits and labels
    ax.set_xlim([-1, 4])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-1, 5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Obstacle Visualization")
    plt.show()


def plot_obstacles_and_initial_positions(params):
    """Plots the obstacles and the initialized positions."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each obstacle as a filled 3D box with dark gray color
    for key in params.obstacles.keys():
        obs = params.obstacles[key]
        x, y, z = obs['x'], obs['y'], obs['z']
        l, b, h = obs['l'], obs['b'], obs['h']

        # Define the vertices of the box
        vertices = [
            [x - l / 2, y - b / 2, z - h / 2],
            [x - l / 2, y + b / 2, z - h / 2],
            [x + l / 2, y + b / 2, z - h / 2],
            [x + l / 2, y - b / 2, z - h / 2],
            [x - l / 2, y - b / 2, z + h / 2],
            [x - l / 2, y + b / 2, z + h / 2],
            [x + l / 2, y + b / 2, z + h / 2],
            [x + l / 2, y - b / 2, z + h / 2],
        ]

        # Define the 6 faces of the box
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front face
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back face
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right face
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left face
        ]

        # Add the box to the plot
        ax.add_collection3d(Poly3DCollection(faces, alpha=0.5, facecolors='darkgray'))

    # Plot the initialized positions
    for pos in params.payload_initialization:
        ax.scatter(pos[0], pos[1], pos[2], color='blue', label='Initialized Position', s=50)

    # Avoid duplicate labels in the legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    # Set plot limits and labels
    ax.set_xlim([-1, 4])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-1, 5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Obstacles and Initialized Positions")
    plt.show()


def plot_obstacles_and_initialized_positions_using_opt(params, opti, variables):
    """Plots the obstacles and the initialized positions retrieved from opti."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each obstacle as a filled 3D box with dark gray color
    for key in params.obstacles.keys():
        obs = params.obstacles[key]
        x, y, z = obs['x'], obs['y'], obs['z']
        l, b, h = obs['l'], obs['b'], obs['h']

        # Define the vertices of the box
        vertices = [
            [x - l / 2, y - b / 2, z - h / 2],
            [x - l / 2, y + b / 2, z - h / 2],
            [x + l / 2, y + b / 2, z - h / 2],
            [x + l / 2, y - b / 2, z - h / 2],
            [x - l / 2, y - b / 2, z + h / 2],
            [x - l / 2, y + b / 2, z + h / 2],
            [x + l / 2, y + b / 2, z + h / 2],
            [x + l / 2, y - b / 2, z + h / 2],
        ]

        # Define the 6 faces of the box
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front face
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back face
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right face
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left face
        ]

        # Add the box to the plot
        ax.add_collection3d(Poly3DCollection(faces, alpha=0.5, facecolors='darkgray'))

    # Retrieve and plot the initialized positions from opti
    for j in range(params.horizon + 1):
        x = opti.value(variables["x"][j, 0])  # x position
        y = opti.value(variables["x"][j, 1])  # y position
        z = opti.value(variables["x"][j, 2])  # z position
        ax.scatter(x, y, z, color='blue', label='Initialized Position' if j == 0 else "", s=50)

    # Avoid duplicate labels in the legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    # Set plot limits and labels
    ax.set_xlim([-1, 4])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-1, 5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Obstacles and Initialized Positions (from opti)")
    plt.show()


def plot_resuls_times(planner, params, sol_values):
    """Plots the time intervals (dt values) for each step in the horizon."""

    if not planner.params.min_time:
        print("Min time optimization is not enabled. No time intervals to plot.")
        return

    # Extract dt values
    dt_values = sol_values[f"t"]

    # Plot the dt values
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(dt_values) + 1), dt_values, marker='o', label='Time Intervals (dt)')
    plt.xlabel("Horizon Step")
    plt.ylabel("Time Interval (s)")
    plt.title("Time Intervals (dt) Across Horizon")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
