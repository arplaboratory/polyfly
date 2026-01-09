import numpy as np
from scipy.spatial import HalfspaceIntersection
from scipy.spatial import ConvexHull


class Polytope:
    def __init__(self):
        pass

    def get_vertices(self):
        A, B = self.get_convex_rep()

        # Objective function (minimize zero to find a feasible point)
        interior_point = self.get_interior_point()

        # Create half-space representation
        halfspaces = np.hstack((A, -B))

        # Compute vertices
        hs = HalfspaceIntersection(halfspaces, interior_point)
        vertices = hs.intersections

        # Filter vertices to ensure they are part of the convex hull
        hull = ConvexHull(vertices)
        vertices = vertices[hull.vertices]

        return vertices


class Obs(Polytope):
    def __init__(
        self, origin_x, origin_y, origin_z, length, breadth, height, theta_x=0, theta_y=0, theta_z=0
    ):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.origin_z = origin_z
        self.length = length
        self.breadth = breadth
        self.height = height

    def get_convex_rep(self):
        A = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
        B = np.array(
            [
                [self.origin_z + self.height / 2],
                [-self.origin_z + self.height / 2],
                [self.origin_x + self.length / 2],
                [-self.origin_x + self.length / 2],
                [self.origin_y + self.breadth / 2],
                [-self.origin_y + self.breadth / 2],
            ]
        )
        # print(f'B = {B}')
        return A, B

    def get_interior_point(self):
        return np.array((self.origin_x, self.origin_y, self.origin_z))

    def get_center(self):
        return np.array([self.origin_x, self.origin_y, self.origin_z])


class SquarePayload(Polytope):
    def __init__(self, params):
        self.radius = params.payload_radius

    def get_convex_rep(self):
        radius = self.radius
        # a abd B matrices for a square payload of width and length of 5cm and height of 5cm
        A = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
        B = np.array([[radius], [radius], [radius], [radius], [radius], [radius]])
        return A, B

    def get_interior_point(self):
        return np.array((0, 0, 0))


class TrianglePayload(Polytope):
    def __init__(self, params):
        self.radius = params.payload_radius

    def get_convex_rep(self):
        radius = self.radius
        # a abd B matrices for a square payload of width and length of 5cm and height of 5cm
        A = np.array([[0, 0, 1], [0, 0, -1], [1, 1, 0], [-1, 1, 0], [0, -1, 0]])
        B = np.array([[radius], [radius], [radius * 3.0], [radius * 3.0], [0]])
        return A, B

    def get_interior_point(self):
        return np.array((0, radius / 2.0, 0))


class Cable(Polytope):
    def __init__(self, params):
        self.l = params.cable_length
        self.radius = params.cable_radius

    def get_convex_rep(self):
        # a abd B matrices for a square payload of width and length of 5cm and height of 5cm
        A = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
        B = np.array(
            [[self.l / 2], [self.l / 2], [self.radius], [self.radius], [self.radius], [self.radius]]
        )
        return A, B

    def get_interior_point(self):
        return np.array((0, 0, 0))


class Quadrotor(Polytope):
    def __init__(self, params):
        self.radius = params.robot_radius
        self.height = params.robot_height

    def get_convex_rep(self):
        # a abd B matrices for a square payload of width and length of 5cm and height of 5cm
        A = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
        B = np.array(
            [
                [self.height / 2],
                [self.height / 2],
                [self.radius],
                [self.radius],
                [self.radius],
                [self.radius],
            ]
        )
        return A, B

    def get_interior_point(self):
        return np.array((0, 0, 0))
