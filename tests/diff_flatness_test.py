import numpy as np
import matplotlib.pyplot as plt
from optimal_planner import Planner
from poly_fly.utils.utils import MPC
from utils import vec2asym, dictToClass, yamlToDict


def test_compute_quadrotor_rotation_matrix():
    """
    Test the compute_quadrotor_rotation_matrix function by plotting a rotated cube.
    """
    path_to_yaml = "params/narrow_gap_for_cable.yaml"
    params = dictToClass(MPC, yamlToDict(path_to_yaml))
    init_state = params.initial_state

    planner = Planner(params)

    # Call the method with all arguments set to 0
    R = planner.compute_quadrotor_rotation_matrix(
        a=[0, 0, 1], j=[0, 0, 0], yaw_des=0, yaw_dot_des=0, mQ=0, mL=0
    )
    R = planner.compute_quadrotor_rotation_matrix_no_jrk([0, 0, 1], params)
    # Define a cube centered at the origin
    cube_vertices = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ]
    )

    # Apply the rotation matrix to the cube vertices
    rotated_vertices = np.dot(R, cube_vertices.T).T

    # Define the edges of the cube
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # Bottom face
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # Top face
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # Vertical edges
    ]

    # Plot the rotated cube
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for edge in edges:
        start, end = edge
        ax.plot(
            [rotated_vertices[start, 0], rotated_vertices[end, 0]],
            [rotated_vertices[start, 1], rotated_vertices[end, 1]],
            [rotated_vertices[start, 2], rotated_vertices[end, 2]],
            color='blue',
        )

    # Set axis labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # Set plot limits for better visualization
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    plt.title("Rotated Cube Visualization")
    plt.show()


test_compute_quadrotor_rotation_matrix()
