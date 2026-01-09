import numpy as np
from typing import Any, Dict, Iterable, Tuple
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from poly_fly.data_io.utils import find_csvs, load_csv


class MotionProfiles:
    @staticmethod
    def extract_motion_profile(
        pos: Any,
        times: Iterable[float],
        bc_type = "natural",
        num_samples: int = 200,
        show: bool = True,
    ) -> Dict[str, Any]:
        """
        Fit cubic splines to pos_x and pos_y over time and overlay the originals.

        Args:
            pos: container with attributes/keys 'pos_x' and 'pos_y' (1-D arrays)
            times: 1-D iterable of timestamps (must be strictly increasing)
            bc_type: CubicSpline boundary condition ('natural' by default)
            num_samples: number of dense samples for plotting/evaluation
            show: whether to plot the original data and spline overlays

        Returns:
            dict with:
              - spline_x, spline_y: CubicSpline objects
              - t_dense, x_dense, y_dense: dense evaluation for plotting
        """
        # Extract arrays from dict-like or attribute-like container
        def _get(arr_like, key):
            if isinstance(arr_like, dict):
                return arr_like[key]
            return getattr(arr_like, key)

        t = np.asarray(times, dtype=float).reshape(-1)
        x = np.asarray(_get(pos, "pos_x"), dtype=float).reshape(-1)
        y = np.asarray(_get(pos, "pos_y"), dtype=float).reshape(-1)

        if not (t.size == x.size == y.size):
            raise ValueError(f"times, pos_x, pos_y must have same length; got {t.size}, {x.size}, {y.size}")

        # Ensure strictly increasing time by sorting (and checking duplicates)
        order = np.argsort(t)
        t_sorted = t[order]
        x_sorted = x[order]
        y_sorted = y[order]

        t_sorted = np.array(([t[0], t[-1]]))
        x_sorted = np.array(([x_sorted[0], x_sorted[-1]]))
        y_sorted = np.array(([y_sorted[0], y_sorted[-1]]))

        # import pdb; pdb.set_trace()
        if np.any(np.diff(t_sorted) <= 0):
            raise ValueError("times must be strictly increasing and unique.")

        # Fit cubic splines
        spline_x = CubicSpline(t_sorted, x_sorted, bc_type=bc_type)
        spline_y = CubicSpline(t_sorted, y_sorted, bc_type=bc_type)

        # Evaluate on a dense grid
        t_dense = np.linspace(t_sorted[0], t_sorted[-1], int(num_samples))
        x_dense = spline_x(t_dense)
        y_dense = spline_y(t_dense)

        # Plot overlays
        if show:
            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
            axs[0].plot(t_sorted, x_sorted, "o", label="x data", alpha=0.8)
            axs[0].plot(t_dense, x_dense, "-", label="x spline")
            axs[0].set_ylabel("x")
            axs[0].grid(True)
            axs[0].legend()

            axs[1].plot(t_sorted, y_sorted, "o", label="y data", alpha=0.8)
            axs[1].plot(t_dense, y_dense, "-", label="y spline")
            axs[1].set_xlabel("time")
            axs[1].set_ylabel("y")
            axs[1].grid(True)
            axs[1].legend()

            plt.tight_layout()
            plt.show()

        return {
            "spline_x": spline_x,
            "spline_y": spline_y,
            "t_dense": t_dense,
            "x_dense": x_dense,
            "y_dense": y_dense,
        }

    @staticmethod
    def extract_motion_profile_fit(
        pos: Any,
        times: Iterable[float],
        bc_type="natural",
        num_samples: int = 200,
        show: bool = True,
    ) -> Dict[str, Any]:
        """
        Fit 3rd-degree polynomials to pos_x and pos_y over time and overlay the originals.

        Args:
            pos: container with attributes/keys 'pos_x' and 'pos_y' (1-D arrays)
            times: 1-D iterable of timestamps (must be strictly increasing)
            bc_type: unused (kept for signature compatibility)
            num_samples: number of dense samples for plotting/evaluation
            show: whether to plot the original data and polynomial overlays

        Returns:
            dict with:
              - spline_x, spline_y: np.poly1d objects (callables)
              - t_dense, x_dense, y_dense: dense evaluation for plotting
        """
        # Extract arrays from dict-like or attribute-like container
        def _get(arr_like, key):
            if isinstance(arr_like, dict):
                return arr_like[key]
            return getattr(arr_like, key)

        t = np.asarray(times, dtype=float).reshape(-1)
        x = np.asarray(_get(pos, "pos_x"), dtype=float).reshape(-1)
        y = np.asarray(_get(pos, "pos_y"), dtype=float).reshape(-1)

        if not (t.size == x.size == y.size):
            raise ValueError(f"times, pos_x, pos_y must have same length; got {t.size}, {x.size}, {y.size}")

        # Sort by time and ensure strictly increasing timestamps
        order = np.argsort(t)
        t_sorted = t[order]
        x_sorted = x[order]
        y_sorted = y[order]

        if np.any(np.diff(t_sorted) <= 0):
            raise ValueError("times must be strictly increasing and unique.")

        # Degree-3 polynomial fit
        rcond = 2*1e-7
        coeff_x = np.polyfit(t_sorted, x_sorted, deg=2, rcond=rcond)
        coeff_y = np.polyfit(t_sorted, y_sorted, deg=2, rcond=rcond)
        poly_x = np.poly1d(coeff_x)
        poly_y = np.poly1d(coeff_y)

        print(coeff_x)
        print(coeff_y)
        print("\n")
        # Evaluate on a dense grid
        t_dense = np.linspace(t_sorted[0], t_sorted[-1], int(num_samples))
        x_dense = poly_x(t_dense)
        y_dense = poly_y(t_dense)

        # Plot overlays
        if show:
            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
            axs[0].plot(t_sorted, x_sorted, "o", label="x data", alpha=0.8)
            axs[0].plot(t_dense, x_dense, "-", label="x poly deg-3")
            axs[0].set_ylabel("x")
            axs[0].grid(True)
            axs[0].legend()

            axs[1].plot(t_sorted, y_sorted, "o", label="y data", alpha=0.8)
            axs[1].plot(t_dense, y_dense, "-", label="y poly deg-3")
            axs[1].set_xlabel("time")
            axs[1].set_ylabel("y")
            axs[1].grid(True)
            axs[1].legend()

            plt.tight_layout()
            plt.show()

        return {
            "spline_x": poly_x,  # callable np.poly1d
            "spline_y": poly_y,  # callable np.poly1d
            "t_dense": t_dense,
            "x_dense": x_dense,
            "y_dense": y_dense,
        }

    @staticmethod
    def extract_profile_for_trajectory(
        sol_x,
        time, 
        step: float = 0.2,
        segment_s: float = 1.0,
        show: bool = True,
    ) -> Dict[str, Any]:
        """
        Sliding-window spline fit:
          - Fit a cubic spline to each 1-second (segment_s) window,
            advancing the start by `step` seconds each time.
          - Return same fields as extract_profile_for_trajectory_non_overlapp
            plus `indexes` (start indices for each window in the input arrays).
        """
        x = sol_x[:, 0]
        y = sol_x[:, 1]

        if time.size < 2:
            raise ValueError("CSV must contain at least two time samples.")
        if segment_s <= 0:
            raise ValueError("segment_s must be positive.")
        if step <= 0:
            raise ValueError("step must be positive.")

        t0 = float(time[0])
        t_end = float(time[-1])
        t_step = float(time[1]) - float(time[0])
        
        # Robust integer lengths from seconds
        step_n = max(1, int(round(step / t_step)))
        win_n = max(2, int(round(segment_s / t_step)) + 1)  # +1 to include ~segment_s duration

        segments = []
        indexes = []

        seg_start_idx = 0
        k = 0

        while seg_start_idx < len(time) - 1:
            end_excl = min(len(time), seg_start_idx + win_n)
            idxs = np.arange(seg_start_idx, end_excl, dtype=int)
            if idxs.size < 2:
                break

            t_seg = time[idxs]
            x_seg = x[idxs]
            y_seg = y[idxs]

            indexes.append(seg_start_idx)

            res = MotionProfiles.extract_motion_profile_fit(
                {"pos_x": x_seg, "pos_y": y_seg},
                t_seg,
                show=False,  # suppress per-segment plots
            )
            segments.append(
                {
                    "k": k,
                    "start": float(t_seg[0]),
                    "end": float(t_seg[-1]),
                    "indices": idxs,
                    "spline_x": res["spline_x"],
                    "spline_y": res["spline_y"],
                    "t_dense": res["t_dense"],
                    "x_dense": res["x_dense"],
                    "y_dense": res["y_dense"],
                }
            )

            seg_start_idx += step_n
            k += 1

        # Plot overlays: x(t), y(t) with segment spline predictions
        if show:
            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
            axs[0].plot(time, x, "k-", lw=3.0, label="x data", alpha=0.5)
            axs[0].set_ylabel("x [m]")
            axs[0].grid(True)

            axs[1].plot(time, y, "k-", lw=3.0, label="y data", alpha=0.5)
            axs[1].set_xlabel("time [s]")
            axs[1].set_ylabel("y [m]")
            axs[1].grid(True)

            stride = 10  # thin dense samples for readability
            cmap = plt.get_cmap("tab20")
            for i, seg in enumerate(segments):
                c = cmap(i % cmap.N)
                axs[0].scatter(seg["t_dense"][::stride], seg["x_dense"][::stride], s=30, alpha=1, label=None, color=c)
                axs[1].scatter(seg["t_dense"][::stride], seg["y_dense"][::stride], s=30, alpha=1, label=None, color=c)

            fig.suptitle(f"Spline fit: {segment_s:.1f}s window, {step:.2f}s step")
            plt.tight_layout()

            # XY overlay: original vs spline-predicted
            plt.figure(figsize=(6.5, 6))
            plt.plot(x, y, "k.", ms=2, label="data (x,y)")
            for seg in segments:
                plt.plot(seg["x_dense"], seg["y_dense"], "-", lw=1.2, alpha=1.0, label=None)
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.title("XY: original vs spline (sliding windows)")
            plt.axis("equal")
            plt.grid(True)
            plt.legend(loc="best")
            plt.show()

        # Match the non-overlapping method's return signature, plus 'indexes'
        return {
            "segments": segments,
            "time": time,
            "x": x,
            "y": y,
            "indexes": indexes,
        }

    @staticmethod
    def extract_profile_for_trajectory_non_overlapp(
        csv: Any = None,
        segment_s: float = 1.0,
        show: bool = True,
    ) -> Dict[str, Any]:
        """
        For every `segment_s`-second segment in the CSV, fit a cubic spline using extract_motion_profile.
        Then overlay the original CSV positions with the spline-predicted values.

        Args:
            csv: path to a CSV; if None, a random CSV is chosen via find_csvs(1)
            segment_s: segment length in seconds (default: 1.0)
            bc_type: boundary condition for CubicSpline
            num_samples_per_segment: dense evaluation per segment for plotting
            show: display plots

        Returns:
            dict containing:
              - csv_path
              - segments: list of dicts with per-segment splines and dense samples
              - time, x, y: original series
        """
        if csv is None:
            _, csvs = find_csvs(1)
            csv_path = csvs[0]
        else:
            csv_path = csv

        time, sol_x, _, _, _, _ = load_csv(csv_path)
        x = sol_x[0, :]
        y = sol_x[1, :]

        if time.size < 2:
            raise ValueError("CSV must contain at least two time samples.")

        t0 = float(time[0])
        t_end = float(time[-1])
        if segment_s <= 0:
            raise ValueError("segment_s must be positive.")

        # Build non-overlapping segments [start, end) except last which is inclusive
        n_segments = int(np.ceil((t_end - t0) / segment_s))
        segments = []

        for k in range(n_segments):
            seg_start = t0 + k * segment_s
            seg_end = min(seg_start + segment_s, t_end)
            if k < n_segments - 1:
                mask = (time >= seg_start) & (time < seg_end)
            else:
                mask = (time >= seg_start) & (time <= seg_end)

            if not np.any(mask):
                continue

            t_seg = time[mask]
            x_seg = x[mask]
            y_seg = y[mask]

            # Need at least two samples to fit a spline
            if t_seg.size < 2:
                continue

            res = MotionProfiles.extract_motion_profile(
                {"pos_x": x_seg, "pos_y": y_seg},
                t_seg,
                show=False,  # suppress per-segment plots
            )
            segments.append(
                {
                    "k": k,
                    "start": seg_start,
                    "end": seg_end,
                    "indices": np.nonzero(mask)[0],
                    "spline_x": res["spline_x"],
                    "spline_y": res["spline_y"],
                    "t_dense": res["t_dense"],
                    "x_dense": res["x_dense"],
                    "y_dense": res["y_dense"],
                }
            )

        # Plot overlays: x(t), y(t) with segment spline predictions
        if show:
            # Time-domain overlays
            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
            axs[0].plot(time, x, "k-", lw=2.0, label="x data", alpha=0.5)
            axs[0].set_ylabel("x [m]")
            axs[0].grid(True)

            axs[1].plot(time, y, "k-", lw=2.0, label="y data", alpha=0.5)
            axs[1].set_xlabel("time [s]")
            axs[1].set_ylabel("y [m]")
            axs[1].grid(True)

            stride = 10
            for seg in segments:
                axs[0].scatter(seg["t_dense"][::stride], seg["x_dense"][::stride], alpha=0.9, label=None)
                axs[1].scatter(seg["t_dense"][::stride], seg["y_dense"][::stride] , alpha=0.9, label=None)

            fig.suptitle(f"Spline fit per {segment_s:.1f}s segment — {csv_path}")
            plt.tight_layout()

            # XY overlay: original vs spline-predicted
            plt.figure(figsize=(6.5, 6))
            plt.plot(x, y, "k.", ms=2, label="data (x,y)")
            for seg in segments:
                plt.plot(seg["x_dense"], seg["y_dense"], "-", lw=1.5, alpha=0.9, label=None)
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.title("XY: original vs spline per segment")
            plt.axis("equal")
            plt.grid(True)
            plt.legend(loc="best")
            plt.show()

        return {
            "csv_path": csv_path,
            "segments": segments,
            "time": time,
            "x": x,
            "y": y,
        }

    @staticmethod
    def plot_csv(csv_path: Any = None, show: bool = True) -> Dict[str, Any]:
        """
        Plot x(t) and y(t) from a CSV for simple inspection.

        Args:
            csv_path: path to a CSV. If None, a random one is chosen via find_csvs(1).
            show: call plt.show() if True.

        Returns:
            dict with: csv_path, time, x, y
        """
        if csv_path is None:
            _, csvs = find_csvs(1)
            csv_path = csvs[1]

        time, sol_x, _, _, _, _ = load_csv(csv_path)
        x = sol_x[0, :]
        y = sol_x[1, :]

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
        axs[0].plot(time, x, "-", label="x(t)")
        axs[0].set_ylabel("x [m]")
        axs[0].grid(True)
        axs[0].legend()

        axs[1].plot(time, y, "-", label="y(t)")
        axs[1].set_xlabel("time [s]")
        axs[1].set_ylabel("y [m]")
        axs[1].grid(True)
        axs[1].legend()

        fig.suptitle(f"CSV: {csv_path}")
        plt.tight_layout()
        if show:
            plt.show()

        return {"csv_path": csv_path, "time": time, "x": x, "y": y}

if "__main__" == __name__:
    _, csvs = find_csvs(1)
    time, sol_x, _, _, _, _ = load_csv(csvs[0])

    result = MotionProfiles.extract_profile_for_trajectory(sol_x, time, segment_s=1.5, step=1.7, show=True)
    # MotionProfiles.extract_profile_for_trajectory_no_overlap()