from poly_fly.optimal_planner.planner import Planner, dictToClass, yamlToDict, interpolate_distance
from poly_fly.utils.utils import MPC
import poly_fly.utils.plot as plotter

# load any template YAML (must include dt, mass, etc. but start/goal may be dummy)
file_dir = "/home/mrunal/Documents/optimal_payload_planning/planner/params/long_range/base.yaml"
base_params = dictToClass(MPC, yamlToDict(file_dir))
planner = Planner(base_params, plot=True)

# start clean
planner.clear_obstacles()

planner.add_obstacle_box(x=2.0, y=0.0, z=0.0, xl=0.5, yl=0.5, zl=3.0)  # obs1
planner.add_obstacle_box(x=2.0, y=-2.0, z=0.0, xl=0.5, yl=0.5, zl=3.0)  # obs2
planner.add_obstacle_box(x=2.0, y=2.0, z=0.0, xl=0.5, yl=0.5, zl=3.0)  # obs3

planner.add_obstacle_box(x=4.0, y=1.0, z=0.0, xl=0.5, yl=1.0, zl=3.0)  # obs4
planner.add_obstacle_box(x=4.0, y=-1.0, z=0.0, xl=0.5, yl=1.0, zl=3.0)  # obs5
planner.add_obstacle_box(x=4.0, y=-3.0, z=0.0, xl=0.5, yl=1.0, zl=3.0)  # obs6

planner.add_obstacle_box(x=6.0, y=0.0, z=0.0, xl=0.5, yl=1.0, zl=3.0)  # obs7
planner.add_obstacle_box(x=6.0, y=-2.0, z=0.0, xl=0.5, yl=1.0, zl=3.0)  # obs8
planner.add_obstacle_box(x=6.0, y=2.0, z=0.0, xl=0.5, yl=1.0, zl=3.0)  # obs9

planner.set_start_state((0.0, 0.0, 0.0))
planner.set_end_state((8.0, 0.0, 0.0))

planner.setup()
sol_values, *_ = planner.optimize()

params = planner.params

interpolated_time, interpolated_x, interpolated_u = interpolate_distance(params, sol_values)
plotter.plot_result(
    params,
    interpolated_x,
    interpolated_u,
    planner.differential_flatness,
    planner.compute_quadrotor_rotation_matrix_no_jrk,
)
