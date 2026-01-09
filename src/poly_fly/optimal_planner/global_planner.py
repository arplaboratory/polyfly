import math
import yaml
from poly_fly.utils.utils import yamlToDict
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')  # Use TkAgg backend to avoid QtCore.QTimer issues


class OpenSetEmptyException(Exception):
    pass


class AStarPlanner:
    def __init__(
        self, ox, oy, resolution, rr, show_animation=False, x_max=5, y_max=5, x_min=-5, y_min=-5
    ):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.rr = rr
        self.max_x = x_max
        self.max_y = y_max
        self.min_x = x_min
        self.min_y = y_min
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)
        self.show_animation = show_animation
        self.explored_nodes = []

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return (
                str(self.x)
                + ","
                + str(self.y)
                + ","
                + str(self.cost)
                + ","
                + str(self.parent_index)
            )

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(
            self.calc_xy_index(sx, self.min_x), self.calc_xy_index(sy, self.min_y), 0.0, -1
        )
        goal_node = self.Node(
            self.calc_xy_index(gx, self.min_x), self.calc_xy_index(gy, self.min_y), 0.0, -1
        )

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                print("Open set is empty..")
                raise OpenSetEmptyException()

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o]),
            )
            current = open_set[c_id]

            # Add current node to explored nodes in world coordinates
            world_x = self.calc_grid_position(current.x, self.min_x)
            world_y = self.calc_grid_position(current.y, self.min_y)
            self.explored_nodes.append([world_x, world_y])

            # show graph
            if self.show_animation:  # pragma: no cover
                plt.plot(world_x, world_y, "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    'key_release_event', lambda event: [exit(0) if event.key == 'escape' else None]
                )
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(
                    current.x + self.motion[i][0],
                    current.y + self.motion[i][1],
                    current.cost + self.motion[i][2],
                    c_id,
                )
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)
        ]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.5  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):
        # self.min_x = round(min(ox))
        # self.min_y = round(min(oy))
        # self.max_x = round(max(ox))
        # self.max_y = round(max(oy))

        # print("min_x:", self.min_x)
        # print("min_y:", self.min_y)
        # print("max_x:", self.max_x)
        # print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        # print("x_width:", self.x_width)
        # print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)] for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [
            [1, 0, 1],
            [0, 1, 1],
            [-1, 0, 1],
            [0, -1, 1],
            [-1, -1, math.sqrt(2)],
            [-1, 1, math.sqrt(2)],
            [1, -1, math.sqrt(2)],
            [1, 1, math.sqrt(2)],
        ]

        return motion

    def get_explored_nodes(self):
        return self.explored_nodes


def solve_with_polytopes(data, robot_radius=0.2, show_animation=False):
    """
    Solve the planning problem using A* with obstacles defined in the YAML file.

    :param data: MPC dataclass containing planning parameters
    :param robot_radius: Radius to consider for robot collision checking
    :param show_animation: Flag to show animation of planning
    """
    # Extract 2D obstacles from the YAML file
    ox, oy = [], []
    for key in data.obstacles.keys():
        obstacle = data.obstacles[key]
        x, y = obstacle["x"], obstacle["y"]
        l, b = obstacle["l"], obstacle["b"]
        # Calculate the edges of the rectangular obstacle based on its center
        x_min = x - l / 2
        x_max = x + l / 2
        y_min = y - b / 2
        y_max = y + b / 2

        # Trace the edges of the rectangular obstacle
        for i in range(int(l * 10) + 1):  # Increase resolution by scaling
            ox.append(x_min + i * 0.1)
            oy.append(y_min)
            ox.append(x_min + i * 0.1)
            oy.append(y_max)
        for j in range(int(b * 10) + 1):
            ox.append(x_min)
            oy.append(y_min + j * 0.1)
            ox.append(x_max)
            oy.append(y_min + j * 0.1)

    # Start and goal positions
    sx, sy = data.initial_state[0], data.initial_state[1]  # Start position
    gx, gy = data.end_state[0], data.end_state[1]  # Goal position
    x_max = data.state_max[0]
    y_max = data.state_max[1]
    x_min = data.state_min[0]
    y_min = data.state_min[1]
    grid_size = data.global_planner_step_size  # [m]
    robot_radius = 2 * data.global_planner_robot_radius

    # Visualize obstacles, start, and goal
    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    # Plan the path using A* algorithm
    a_star = AStarPlanner(
        ox,
        oy,
        grid_size,
        robot_radius,
        show_animation=show_animation,
        x_max=x_max,
        y_max=y_max,
        x_min=x_min,
        y_min=y_min,
    )
    rx, ry = a_star.planning(sx, sy, gx, gy)

    # Get explored nodes from A* planner
    explored_nodes = a_star.get_explored_nodes()

    # Visualize the resulting path
    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()

    return rx, ry, explored_nodes


def main():
    solve_with_polytopes("params/maze_2.yaml")


if __name__ == '__main__':
    main()
