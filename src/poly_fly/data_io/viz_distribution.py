from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib
# matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as mcolors

from poly_fly.data_io.zarr_io import load_zarr_folder, ZARR_DIR
from poly_fly.data_io.enums import DatasetKeys as DK


def resolve_zarr_path(arg: str | None) -> Path:
    """
    If arg endswith .zarr -> treat as full path (under ZARR_DIR unless absolute).
    Else treat as dataset subdirectory name and resolve as ZARR_DIR/<arg>.zarr.
    """
    p = Path(arg) if arg else None
    if p and p.suffix == ".zarr":
        return p if p.is_absolute() else (Path(ZARR_DIR) / p)
    if not arg:
        raise ValueError("dataset name or .zarr path is required")
    return Path(ZARR_DIR) / f"{arg}.zarr"


def _scan_min_max(datasets, key: str):
    xmin, ymin = np.inf, np.inf
    xmax, ymax = -np.inf, -np.inf
    for ds in datasets:
        arr = np.asarray(ds[key][:], dtype=np.float32)  # (T, 3)
        if arr.size == 0:
            continue
        xy = arr[:, :2]
        mn = xy.min(axis=0)
        mx = xy.max(axis=0)
        xmin = min(xmin, float(mn[0])); ymin = min(ymin, float(mn[1]))
        xmax = max(xmax, float(mx[0])); ymax = max(ymax, float(mx[1]))
    if not np.isfinite([xmin, ymin, xmax, ymax]).all():
        raise RuntimeError(f"No finite values found for key={key}")
    return xmin, xmax, ymin, ymax


def _build_edges(vmin: float, vmax: float, bin_size: float):
    start = np.floor(vmin / bin_size) * bin_size
    end = np.ceil(vmax / bin_size) * bin_size
    if end <= start:
        end = start + bin_size
    return np.arange(start, end + bin_size * 0.5, bin_size, dtype=float)


def _accumulate_hist2d(datasets, key: str, xedges: np.ndarray, yedges: np.ndarray) -> np.ndarray:
    H = np.zeros((len(xedges) - 1, len(yedges) - 1), dtype=np.int64)
    for ds in datasets:
        xy = np.asarray(ds[key][:], dtype=np.float32)[:, :2]
        if xy.size == 0:
            continue
        counts, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(xedges, yedges))
        H += counts.astype(np.int64)
    return H


def _plot_heatmap(H: np.ndarray, xedges: np.ndarray, yedges: np.ndarray, title: str, out_path: Path,
                  vmin: float = 0.0, vmax: float = 200.0, over_color: str = "#8B0000"):
    """
    Plot a seaborn heatmap of 2D counts with fixed color range [vmin, vmax].
    Values > vmax are shown using 'over_color' via colormap.set_over and cbar extend='max'.
    Also shows an on-hover tooltip with (x,y,count).
    """
    sns.set(style="white", context="talk")
    # centers for labeling
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])

    # Flip vertically to make origin bottom-left
    H_plot = H.T[::-1, :]

    # Build a colormap that uses a specific color for values above vmax
    base = plt.get_cmap("viridis", 256)
    cmap = matplotlib.colors.ListedColormap(base(np.linspace(0, 1, 256)))
    cmap.set_over(mcolors.to_rgba(over_color))

    fig, ax = plt.subplots(figsize=(8, 7))
    hm = sns.heatmap(
        H_plot,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        cbar_kws={"label": "count", "extend": "max"},
        square=True
    )

    # Ticks at ~10 positions max on each axis
    def tick_idx(n):
        step = max(1, n // 10)
        return np.arange(0, n, step, dtype=int)

    xti = tick_idx(len(xcenters))
    yti = tick_idx(len(ycenters))
    ax.set_xticks(xti + 0.5)  # heatmap cell centers
    ax.set_yticks(yti + 0.5)
    ax.set_xticklabels([f"{xcenters[i]:.1f}" for i in xti], rotation=0)
    # y is flipped, so map index to reversed centers
    ycenters_rev = ycenters[::-1]
    ax.set_yticklabels([f"{ycenters_rev[i]:.1f}" for i in yti], rotation=0)

    # --- Add interactive hover tooltip ---
    ny, nx = H_plot.shape  # rows (y), cols (x)
    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="0.3"),
        fontsize=9,
    )
    annot.set_visible(False)

    def update_annot(i, j, val):
        # Place at cell center
        annot.xy = (i + 0.5, j + 0.5)
        # Map j (row in H_plot) to physical y via reversed centers
        x_m = xcenters[i]
        y_m = ycenters_rev[j]
        annot.set_text(f"x={x_m:.2f} m\ny={y_m:.2f} m\ncount={int(val)}")
        # Color text differently if value exceeds vmax
        annot.get_bbox_patch().set_facecolor("w")
        annot.get_bbox_patch().set_edgecolor("0.4")

    def hover(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()
            return

        i = int(np.floor(event.xdata))
        j = int(np.floor(event.ydata))
        if 0 <= i < nx and 0 <= j < ny:
            val = H_plot[j, i]
            update_annot(i, j, val)
            if not annot.get_visible():
                annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect("motion_notify_event", hover)
    # --- end hover ---

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    fig.tight_layout()
    # fig.savefig(out_path, dpi=160)
    plt.show()
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Visualize XY occupancy distribution of robot and payload.")
    ap.add_argument("--dataset", type=str, required=True, help="Zarr dataset name (under ZARR_DIR) or path to .zarr")
    ap.add_argument("--bin-size", type=float, default=0.10, help="Bin size in meters (default: 0.10 m)")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on number of trajectories")
    ap.add_argument("--out-dir", type=str, default=None, help="Output directory for figures")
    ap.add_argument("--cbar-min", type=float, default=0.0, help="Colorbar minimum (default: 0)")
    ap.add_argument("--cbar-max", type=float, default=200.0, help="Colorbar maximum (default: 200)")
    ap.add_argument("--over-color", type=str, default="#8B0000", help="Color for values above cbar-max (e.g., '#8B0000')")
    args = ap.parse_args()

    zarr_path = resolve_zarr_path(args.dataset)
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr dataset not found: {zarr_path}")

    out_dir = Path(args.out_dir) if args.out_dir else (zarr_path.parent / "viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load lazily (depth-only flag enables lazy path; still includes positions we need)
    lazy_args = argparse.Namespace(lazy_depth_only=True, lazy=True, combined=False)
    datasets = load_zarr_folder(zarr_path, lazy_args, limit=args.limit, random_sample=False)
    if not datasets:
        raise RuntimeError("No trajectories found in dataset")

    bin_size = float(args.bin_size)

    # Robot histogram
    rx_min, rx_max, ry_min, ry_max = _scan_min_max(datasets, DK.ROBOT_POS)
    rx_edges = _build_edges(rx_min, rx_max, bin_size)
    ry_edges = _build_edges(ry_min, ry_max, bin_size)
    H_robot = _accumulate_hist2d(datasets, DK.ROBOT_POS, rx_edges, ry_edges)
    _plot_heatmap(
        H_robot, rx_edges, ry_edges,
        title=f"Robot XY occupancy (bin={bin_size:.2f} m)",
        out_path=out_dir / f"{zarr_path.stem}_robot_xy_hist.png",
        vmin=args.cbar_min, vmax=args.cbar_max, over_color=args.over_color
    )

    # Payload histogram
    px_min, px_max, py_min, py_max = _scan_min_max(datasets, DK.PAYLOAD_POS)
    px_edges = _build_edges(px_min, px_max, bin_size)
    py_edges = _build_edges(py_min, py_max, bin_size)
    H_payload = _accumulate_hist2d(datasets, DK.PAYLOAD_POS, px_edges, py_edges)
    _plot_heatmap(
        H_payload, px_edges, py_edges,
        title=f"Payload XY occupancy (bin={bin_size:.2f} m)",
        out_path=out_dir / f"{zarr_path.stem}_payload_xy_hist.png",
        vmin=args.cbar_min, vmax=args.cbar_max, over_color=args.over_color
    )


if __name__ == "__main__":
    main()
