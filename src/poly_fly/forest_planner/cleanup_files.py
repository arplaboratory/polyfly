#!/usr/bin/env python3
"""
delete_matching_ext_polyfly.py

Delete all files with a given extension in a chosen PolyFly subdirectory
(CSV_DIR/specific_folder OR PARAMS_DIR/specific_folder OR DEPTH_DIR/specific_folder), except ONE file to keep.

Before deleting, the script prints exactly which files will be removed and waits
for a single key:
  - Press Enter to proceed
  - Press Esc to cancel

Usage examples:
  python delete_matching_ext_polyfly.py --base csv  forests  --ext .test --keep keep_me.test
  python delete_matching_ext_polyfly.py --base params configs --ext test  --keep /abs/path/to/keep_me.test
  python delete_matching_ext_polyfly.py --base depth datasets --ext .npz --keep keep_me.npz

Notes:
  - Non-recursive: only deletes files in the chosen folder (no subdirectories).
  - Extension matching is case-insensitive (".TEST" matches ".test").
  - If the keep file is given by name, it’s matched within the chosen folder.
  - For DEPTH_DIR, the typical extension is '.npz'.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import shutil

from poly_fly.data_io.utils import save_csv_arrays, save_params, BASE_DIR, PARAMS_DIR, \
    GIFS_DIR, CSV_DIR, IMG_DIR, DEPTH_DIR, ZARR_DIR


def read_single_key() -> bytes:
    """
    Read a single keypress without requiring Enter.
    Returns a single byte string (e.g., b'\\n' or b'\\r' for Enter, b'\\x1b' for Esc).
    Works on Windows, macOS, and Linux using only the standard library.
    """
    try:
        # Windows
        import msvcrt  # type: ignore
        while True:
            ch = msvcrt.getch()
            # Skip prefix bytes for function keys
            if ch in (b'\x00', b'\xe0'):
                _ = msvcrt.getch()
                continue
            return ch
    except ImportError:
        # POSIX (macOS/Linux)
        import termios  # type: ignore
        import tty  # type: ignore

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            return ch.encode()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def normalize_ext(ext: str) -> str:
    """Normalize an extension to start with a single dot; treat empty as error."""
    ext = ext.strip()
    if not ext:
        raise ValueError("Extension cannot be empty.")
    if not ext.startswith("."):
        ext = "." + ext
    return ext


def main():
    parser = argparse.ArgumentParser(
        description="Delete all files with a given extension in CSV_DIR/<folder>, PARAMS_DIR/<folder>, or DEPTH_DIR/<folder>, except one to keep."
    )
    parser.add_argument(
        "--base", choices=["csv", "params", "depth", "zarr"], required=True,
        help="Which base directory to use: CSV_DIR, PARAMS_DIR, DEPTH_DIR, or ZARR_DIR."
    )
    parser.add_argument(
        "specific_folder", type=str,
        help="Subfolder under the chosen base directory (e.g., 'forests', 'configs')."
    )
    parser.add_argument(
        "-e", "--ext", required=True, type=str,
        help="File extension to target (e.g., '.test' or 'test')."
    )
    parser.add_argument(
        "-k", "--keep", required=False, type=str,
        help="The single file to keep (basename or absolute/relative path). Required for non-.zarr."
    )
    args = parser.parse_args()

    # Normalize extension (case-insensitive)
    try:
        ext = normalize_ext(args.ext)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    ext_lc = ext.lower()

    # Special handling for Zarr datasets: ZARR_DIR/<specific_folder>.zarr
    if ext_lc == ".zarr":
        zarr_root = Path(ZARR_DIR).expanduser().resolve()
        zarr_dir = (zarr_root / f"{args.specific_folder}{ext}").resolve()
        if not zarr_dir.exists() or not zarr_dir.is_dir():
            print(f"No Zarr dataset directory found: {zarr_dir}. Nothing to do.")
            return
        items = sorted(zarr_dir.iterdir())
        if not items:
            print(f"Zarr dataset exists but is already empty: {zarr_dir}")
            return
        print(f"Operating in Zarr dataset: {zarr_dir}")
        print("\nThe following entries inside the dataset WILL be deleted:\n")
        for p in items:
            print(f"  {p}")
        print("\nPress Enter to DELETE these entries, or press Esc to CANCEL...")
        while True:
            key = read_single_key()
            if key in (b"\n", b"\r"):  # Enter
                break
            if key == b"\x1b":  # Esc
                print("\nDeletion canceled.")
                return
        failures = []
        for p in items:
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
            except Exception as e:
                failures.append((p, str(e)))
        deleted_count = len(items) - len(failures)
        print(f"\nDeleted {deleted_count} entrie(s) from {zarr_dir}.")
        if failures:
            print("\nSome entries could not be deleted:")
            for p, err in failures:
                print(f"  {p} -> {err}")
            sys.exit(2)
        print("Done.")
        return

    # For non-.zarr operations, --keep is required
    if not args.keep:
        print("Error: --keep is required for non-.zarr extensions.", file=sys.stderr)
        sys.exit(1)

    # Determine base path from --base (non-.zarr only)
    base_map = {"csv": CSV_DIR, "params": PARAMS_DIR, "depth": DEPTH_DIR, "zarr": ZARR_DIR}
    base_path_str = base_map[args.base]
    base_path = Path(base_path_str).expanduser().resolve()

    target_dir = (base_path / args.specific_folder).resolve()
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"Error: target directory does not exist or is not a directory: {target_dir}", file=sys.stderr)
        sys.exit(1)

    # Keep file handling
    keep_input_path = Path(args.keep).expanduser()
    keep_name = keep_input_path.name  # always compare by basename within target_dir

    # Also try resolving absolute path for safety (if user passed an absolute path)
    try:
        keep_resolved = keep_input_path.resolve(strict=False)
    except Exception:
        keep_resolved = None

    print(f"Operating in: {target_dir}")
    print(f"Target extension: '{ext}' (case-insensitive)")
    print(f"File to keep: {keep_name}")

    # Collect matching files (non-recursive)
    all_matching = [
        p for p in target_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ext_lc
    ]

    # Exclude the keep file (by basename match in this folder, or absolute path equality)
    to_delete = []
    for p in all_matching:
        same_name = (p.name == keep_name)
        same_abs = False
        try:
            if keep_resolved is not None:
                same_abs = (p.resolve() == keep_resolved)
        except Exception:
            pass
        if same_name or same_abs:
            continue
        to_delete.append(p)

    if not all_matching:
        print(f"No '{ext}' files found in {target_dir}. Nothing to do.")
        return

    if not to_delete:
        print(f"Only the keep file was found among '{ext}' files; nothing to delete.")
        print(f"Keeping: {keep_name}")
        return

    print("\nThe following files WILL be deleted:\n")
    for p in sorted(to_delete):
        print(f"  {p}")

    if keep_name not in [p.name for p in all_matching]:
        print(f"\n[Warning] The keep file '{keep_name}' was not found among '{ext}' files in {target_dir}.")

    print("\nPress Enter to DELETE these files, or press Esc to CANCEL...")

    # Wait for either Enter or Esc
    while True:
        key = read_single_key()
        if key in (b"\n", b"\r"):  # Enter
            break
        if key == b"\x1b":  # Esc
            print("\nDeletion canceled.")
            return

    # Proceed with deletion
    failures = []
    for p in to_delete:
        try:
            p.unlink()
        except Exception as e:
            failures.append((p, str(e)))

    deleted_count = len(to_delete) - len(failures)
    print(f"\nDeleted {deleted_count} file(s).")

    if failures:
        print("\nSome files could not be deleted:")
        for p, err in failures:
            print(f"  {p} -> {err}")
        sys.exit(2)

    print("Done.")


if __name__ == "__main__":
    # Example usage 
    # python cleanup_files.py --base params forests --ext .yaml --keep base.yaml
    # python cleanup_files.py --base csv forests --ext .csv --keep base.csv
    # python cleanup_files.py --base depth forests --ext .npz --keep base.npz
    # python cleanup_files.py --base zarr forests --ext .zarr
    main()
