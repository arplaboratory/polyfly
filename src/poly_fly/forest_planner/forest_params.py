import numpy as np  # needed for goal_y_candidates


class ForestParamsLarge:
    def __init__(self):
        # obstacle half-size ranges
        self.large_size_l_range = (0.8, 0.8)
        self.large_size_b_range = (2.5, 2.5)
        self.small_size_l_range = (0.5, 0.5)
        self.small_size_b_range = (0.5, 0.5)

        self.n_large_obs = 8
        self.n_small_obs = 0

        # obstacle geometry
        self.central_trunk = True
        self.trunk_height = 4.0  # full height

        # sampling distributions
        # allowed: "uniform" or "gauss_y" (Gaussian truncated to valid bounds)
        self.dist_mode_x = "uniform"
        self.dist_mode_y = "gauss"

        # Gaussian sigmas for x/y sampling (used when dist_mode_* == "gauss_y")
        self.gauss_sigma_x = 2.0
        self.gauss_sigma_y = 2.0

        # Workspace bounds
        self.x_range = (0.0, 16.0)  # [m] where trunks may appear (x)
        self.y_range = (-6.0, 6.0)  # [m] where trunks may appear (y)
        self.x_obs_range = (self.x_range[0] + 2, self.x_range[1] - 2)  # [m] trunks (x)
        self.y_obs_range = (-7, 7)  # [m] trunks (y)

        # Goal & planning defaults
        self.goal_x = 15.0
        self.goal_y = 0.0
        self.goal_z = 0.0
        self.goal_y_candidates = np.linspace(-2.0, 2.0, 17)


class ForestParamsSmall:
    def __init__(self):
        # obstacle half-size ranges
        self.large_size_l_range = (0.8, 0.8)
        self.large_size_b_range = (2.5, 2.5)
        self.small_size_l_range = (1.0, 1.0)
        self.small_size_b_range = (1.0, 1.0)

        self.n_large_obs = 0
        self.n_small_obs = 15

        # obstacle geometry
        self.central_trunk = False

        self.trunk_height = 4.0  # full height

        # sampling distributions
        # allowed: "uniform" or "gauss_y" (Gaussian truncated to valid bounds)
        self.dist_mode_x = "uniform"
        self.dist_mode_y = "gauss"

        # Gaussian sigmas for x/y sampling (used when dist_mode_* == "gauss_y")
        self.gauss_sigma_x = 2.0
        self.gauss_sigma_y = 4.0

        # Workspace bounds
        self.x_range = (0.0, 16.0)  # [m] where trunks may appear (x)
        self.y_range = (-6.0, 6.0)  # [m] where trunks may appear (y)
        self.x_obs_range = (self.x_range[0] + 1.5, self.x_range[1] - 3)  # [m] trunks (x)
        self.y_obs_range = (-7, 7)  # [m] trunks (y)

        # Goal & planning defaults
        self.goal_x = 15.0
        self.goal_y = 0.0
        self.goal_z = 0.0
        self.goal_y_candidates = np.linspace(-2.0, 2.0, 17)


# Constants exported for selection (ids used elsewhere across the codebase)
FOREST_SMALL_OBS = {"id": 0, "class": ForestParamsSmall}
FOREST_LARGE_OBS = {"id": 2, "class": ForestParamsLarge}


def get_forest_params(forest_type: int):
    """
    Return a forest-params instance based on type id.
    Defaults to ForestParamsLarge if unknown.
    """
    mapping = {
        FOREST_SMALL_OBS["id"]: ForestParamsSmall,
        FOREST_LARGE_OBS["id"]: ForestParamsLarge,
    }
    return mapping.get(forest_type, ForestParamsLarge)()
