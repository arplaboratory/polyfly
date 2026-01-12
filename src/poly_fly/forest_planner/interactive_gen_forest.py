#!/usr/bin/env python3
"""
Interactive forest generator + optimiser
----------------------------------------
Press keys *inside* the plot window:

    r   regenerate obstacles
    c   continue & solve
    q   quit

Requires:  generate_forest.py  (your helper module with the new counts /
           size ranges / central trunk logic)
"""

import sys
import os
import random
import io, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import openai

from poly_fly.forest_planner.generate_forest import ForestGenerator, ObstacleGenerationException
from poly_fly.forest_planner.forest_params import (
    ForestParamsSmall,
)  # NEW: use smaller obstacles for interactive regen
from poly_fly.optimal_planner.planner import (
    interpolate_distance,
)
import poly_fly.utils.plot as plotter

seed = 1
random.seed(seed)
np.random.seed(seed)


# ───────────────────────── helper: draw footprint ────────────────────────
def draw_forest(ax, forest, goal):
    """Top-down footprint with filled polygons matching plot.py style."""
    ax.clear()

    # Use the same color as in plot.py
    obstacle_color = '#708090'  # Light slate gray
    obstacle_alpha = 0.5

    for obs in forest:
        # Get half-dimensions (ForestGenerator._sx/_sy already return half-sizes)
        w = ForestGenerator._sx(obs) / 2.0
        d = ForestGenerator._sy(obs) / 2.0

        # Create a rectangular polygon for each obstacle
        rect_x = np.array([-w, w, w, -w]) + obs["x"]
        rect_y = np.array([-d, -d, d, d]) + obs["y"]

        # Draw filled polygon instead of outline
        poly = plt.Polygon(
            np.column_stack([rect_x, rect_y]),
            facecolor=obstacle_color,
            alpha=obstacle_alpha,
            edgecolor='black',
            linewidth=0.8,
        )
        ax.add_patch(poly)

    # Add start and goal markers with distinctive colors
    start_color = '#485000'  # Dark green
    goal_color = '#DE8C2A'  # Orange

    ax.plot(0, 0, "o", color=start_color, ms=8, label="start")
    ax.plot(goal[0], goal[1], "x", color=goal_color, ms=8, label="goal")

    # Set plot properties
    ax.set_aspect("equal", "box")
    ax.set_xlim(0, 16)
    ax.set_ylim(-6, 6)  # widened to match typical workspace and avoid exaggerated appearance

    # Add grid and floor shading similar to plot.py
    ax.grid(True, linestyle='--', alpha=0.5, color='lightgray')

    # Add axis labels
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')

    ax.legend()
    ax.set_title(
        "Click here, then press:  r – regenerate   |   c – solve   |   q – quit",
        fontsize=9,
    )


# ───────────────────────── interactive figure class ──────────────────────
class ForestChooser:
    def __init__(self, dist_mode=None, forest=None, goal=None):
        # dist_mode kept for backward compatibility; ForestGenerator now uses ForestParams
        self.choice = None  # 'c', 'q', None
        self.fg = ForestGenerator(seed=1)
        # Use smaller obstacle params for interactive regeneration
        small_params = ForestParamsSmall()

        self.fg.set_forest_params(small_params)
        if forest is not None:
            self.forest = forest
            self.fg.set_forest(self.forest)
            # If caller provided a goal, use it; else pick a collision-free one
            self.goal = goal if goal is not None else self.fg.pick_goal(self.forest)
            self.fg.set_goal(*self.goal)
        else:
            while True:
                try:
                    self._regen()
                except ObstacleGenerationException:
                    continue
                break

    def mk_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(7, 4))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        if self.forest is not None:
            draw_forest(self.ax, self.forest, self.goal)

        # ─── NEW: show instructions in the terminal *before* blocking ───
        print(
            "\nControls(click inside the plot window first)\n\n"
            "r:   regenerate obstacles\n"
            "c:   continue & solve\n"
            "q:   quit without solving\n"
        )

        plt.show()  # blocks until the figure is closed

    # -------------------------------------------------
    def _on_key(self, event):
        if event.key == "r":
            try:
                self._regen()
            except ObstacleGenerationException:
                print("failed to generate a new forest, try again")
                return

            draw_forest(self.ax, self.forest, self.goal)
            self.fig.canvas.draw_idle()
        elif event.key in ("c", "q", "escape"):
            self.choice = event.key[0]  # 'c' or 'q'
            plt.close(self.fig)  # this returns control to caller

    # -------------------------------------------------
    def _regen(self):
        # Use the (small) params configured in __init__
        self.forest = self.fg.generate_forest()
        self.fg.set_forest(self.forest)
        self.goal = self.fg.pick_goal(self.forest)
        self.fg.set_goal(*self.goal)

    # -------------------------------------------------
    def get_result(self):
        return self.choice

    def solve_and_plot(self):
        self.fg.set_margin(0.2)
        result = self.fg.solve()
        if result is None:
            print("Optimization Failed")
            return
        sol_values, sol_opt, success, iterations, opt_time = result
        if not success:
            print("Optimization Failed")
            return

        print_trajectory_info(sol_values, sol_opt, self.fg.planner)

        t, x, u = interpolate_distance(
            self.fg.planner.params, sol_values, ds=0.25, append_zero=False
        )
        plotter.plot_result(
            self.fg.planner.params,
            x,
            u,
            self.fg.planner.differential_flatness,
            self.fg.planner.compute_quadrotor_rotation_matrix_no_jrk,
        )


# ───────────────────────── run planner ───────────────────────────────────
def print_trajectory_info(sol_values, sol_opt, planner):
    total_time = 1000
    if planner.params.min_time:
        total_time = 0
        for i in range(planner.params.horizon):
            total_time += sol_opt.value(planner.variables["t"][i])
        print(f"Total Time = {total_time}")

    # time-major: x (N+1, M); take position columns [:3] and diff over time axis
    pos = sol_values["x"][:, :3]
    deltas = np.diff(pos, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    path_length = segment_lengths.sum()
    print(f"Path Length = {path_length}")


def csv_to_forest(csv_text: str, *, trunk_half_height=3.0):
    """
    Convert each line:  x,y,width,height  (width/height are full extents)
    into the planner’s dict format, where xl/yl are half-sizes and zL is half-height.
    """
    forest = []
    reader = csv.reader(io.StringIO(csv_text.strip()))
    for x, y, w, h in reader:
        forest.append(
            dict(
                x=float(x),  # centre-x
                y=float(y),  # centre-y
                z=0.0,
                xl=float(w),  # width
                yl=float(h),  # depth
                zl=trunk_half_height,  # height
            )
        )
    return forest


def compose_gpt_prompt(X_LENGTH=13, Y_WIDTH=3, START_X=0, START_Y=0, GOAL_X=12, GOAL_Y=0):
    return f"""
        You are a path-planning assistant.   
        Your job is to turn plain-English layout instructions into a list of
        axis-aligned rectangular obstacles (“boxes”) for a 2-D workspace.

        • The workspace is (0, X_LENGTH) metres in X-dimension and (-Y_WIDTH, Y_WIDTH) metres in the Y-dimension,  with its origin (0,0) in the
        lower-left corner and +X to the right, +Y upwards.
        • The robot's **start** is at (START_X, START_Y).
        • The robot's **goal**  is at (GOAL_X,  GOAL_Y).

        Box sizes:
        - “small box”  = SMALL_W x SMALL_H  (HALF-SIZES; half-width x half-height)
        - “large box”  = LARGE_W x LARGE_H

        RULES for your reply
        --------------------
        1. Output exactly one CSV record *per line*:
            x, y, width, height
        where (x,y) is the **center** of the box and width,height are **HALF-SIZES**.

        2. Use only the two canonical box sizes above
        (pick one per obstacle as the instruction implies).

        3. Keep every box completely inside the workspace and
        **at least CLEARANCE metres away** from both the start and goal points.
        Additionally, ensure any corridor/gap between any two boxes and between a box and the workspace boundary is at least MIN_GAP metres wide.

        4. Do **not** add narrative text, Markdown, or column headers - only raw CSV lines.

        If the natural-language request is impossible without violating a rule,
        respond with a single line:
        ERROR, reason


        -------------
        Workspace: X_LENGTH={X_LENGTH},  Y_WIDTH={Y_WIDTH}
        Start:  START_X={START_X}, START_Y={START_Y}
        Goal:   GOAL_X={GOAL_X},  GOAL_Y={GOAL_Y}

        Box sizes:
        small  = 0.3 x 0.3
        large  = 0.5 x 0.5
        CLEARANCE = 0.6
        MIN_GAP = 1.5
        """


def get_chatgpt_response(prompt, X_LENGTH=13, Y_WIDTH=3, START_X=0, START_Y=0, GOAL_X=12, GOAL_Y=0):
    full_prompt = compose_gpt_prompt(X_LENGTH, Y_WIDTH, START_X, START_Y, GOAL_X, GOAL_Y)
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    print("Getting response from GPT...")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    print("Successfully got response from GPT")
    return response.choices[0].message.content


# ───────────────────────── main entry point ──────────────────────────────
if __name__ == "__main__":
    CSV_TEXT = """\
0.6,-1.8,1.0,1.0
1.2,-0.5,1.0,1.0
2.4,1.4,1.0,1.0
3.2,0.5,1.0,1.0
4.4,-1.8,1.0,1.0
5.2,-0.5,1.0,1.0
6.4,1.4,1.0,1.0
7.2,0.5,1.0,1.0
8.4,-1.8,1.0,1.0
9.2,-0.5,1.0,1.0
10.4,1.4,1.0,1.0
11.2,0.5,1.0,1.0
    """
    custom_forest = csv_to_forest(CSV_TEXT)
    custom_goal = (15, 0.0, 0)
    chooser = ForestChooser(
        forest=custom_forest,
        goal=custom_goal,
    )

    while True:
        chooser.mk_figure()
        choice = chooser.get_result()

        if choice == "c":  # user accepted – solve it
            chooser.solve_and_plot()
            print("\nReturning to forest generator …\n")

        elif choice == "q":  # user quit
            print("Good-bye!")
            break

        else:  # closed window without a key
            print("No choice made; exiting.")
            break
