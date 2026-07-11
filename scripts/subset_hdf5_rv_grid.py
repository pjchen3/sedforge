#!/usr/bin/env python3
"""Create a lighter HDF5 Rv grid by selecting existing Rv chunks."""

import argparse
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import yaml


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _selected_rv_indices(rvs, rv_min, rv_max, rv_step, atol=1e-5):
    wanted = np.arange(rv_min, rv_max + 0.5 * rv_step, rv_step)
    wanted = np.round(wanted, 8)
    indices = []
    for value in wanted:
        idx = int(np.argmin(np.abs(rvs - value)))
        if not np.isclose(rvs[idx], value, rtol=0.0, atol=atol):
            raise ValueError(f"Could not find Rv={value:g} in source grid.")
        indices.append(idx)
    return np.asarray(indices, dtype=int)


def _link_chunk(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        os.symlink(os.path.relpath(src, start=dst.parent), dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown link mode: {mode}")


def _copy_group_datasets(src_group, dst_group):
    for name in src_group:
        src = src_group[name]
        if isinstance(src, h5py.Dataset):
            dst_group.create_dataset(name, data=src[:])
        elif isinstance(src, h5py.Group):
            child = dst_group.create_group(name)
            _copy_group_datasets(src, child)


def _write_virtual_grid(src_h5, dst_path, grid_name, selected, dst_chunk_files):
    nspec, _nrv, nav, nfilter = src_h5["flux"].shape
    layout = h5py.VirtualLayout(
        shape=(nspec, len(selected), nav, nfilter),
        dtype=src_h5["flux"].dtype,
    )
    for out_i, chunk_file in enumerate(dst_chunk_files):
        source = h5py.VirtualSource(
            os.path.relpath(chunk_file, start=dst_path.parent),
            "flux",
            shape=(nspec, nav, nfilter),
        )
        layout[:, out_i, :, :] = source

    str_dtype = h5py.string_dtype(encoding="utf-8")
    if dst_path.exists():
        dst_path.unlink()

    with h5py.File(dst_path, "w", libver="latest") as dst:
        dst.create_virtual_dataset("flux", layout, fillvalue=np.nan)

        axes = dst.create_group("axes")
        for name in src_h5["axes"]:
            if name == "rv":
                axes.create_dataset("rv", data=np.asarray(src_h5["axes/rv"][:][selected], dtype=np.float32))
            elif name == "photband":
                values = [_decode(value) for value in src_h5["axes/photband"][:]]
                axes.create_dataset("photband", data=np.asarray(values, dtype=object), dtype=str_dtype)
            else:
                axes.create_dataset(name, data=src_h5[f"axes/{name}"][:])

        spectra = dst.create_group("spectra")
        _copy_group_datasets(src_h5["spectra"], spectra)

        dst.create_dataset(
            "chunk_files",
            data=np.asarray(
                [os.path.relpath(path, start=dst_path.parent) for path in dst_chunk_files],
                dtype=object,
            ),
            dtype=str_dtype,
        )

        for key, value in src_h5.attrs.items():
            dst.attrs[key] = value
        dst.attrs["grid"] = str(grid_name)
        dst.attrs["rv_subsampled"] = True
        dst.attrs["rv_step"] = float(np.median(np.diff(src_h5["axes/rv"][:][selected])))
        dst.attrs["source_grid_file"] = os.path.relpath(src_h5.filename, start=dst_path.parent)


def _update_grid_description(model_dir, source_name, grid_name, grid_path):
    desc_path = Path(model_dir) / "grid_description.yaml"
    if desc_path.is_file():
        with desc_path.open() as handle:
            desc = yaml.safe_load(handle) or {}
    else:
        desc = {}

    source_desc = dict(desc.get(source_name, {}))
    source_desc.update({
        "filename": grid_name,
        "integrated_path": os.path.relpath(grid_path, start=Path(model_dir)),
        "integrated_format": "hdf5",
        "axes": ["teff", "logg", "feh", "rv", "av"],
        "supports_feh": True,
        "supports_rv": True,
        "info": (
            source_desc.get("info", "HDF5 integrated grid")
            + " Subsampled to Rv step=0.05."
        ),
    })
    desc[grid_name] = source_desc
    with desc_path.open("w") as handle:
        yaml.safe_dump(desc, handle, sort_keys=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-grid", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-name", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--rv-min", type=float, default=None)
    parser.add_argument("--rv-max", type=float, default=None)
    parser.add_argument("--rv-step", type=float, default=0.05)
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
    )
    parser.add_argument("--update-grid-description", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = args.output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.source_grid, "r") as src:
        rvs = np.asarray(src["axes/rv"][:], dtype=float)
        rv_min = float(rvs[0] if args.rv_min is None else args.rv_min)
        rv_max = float(rvs[-1] if args.rv_max is None else args.rv_max)
        selected = _selected_rv_indices(rvs, rv_min, rv_max, args.rv_step)
        source_chunks = [_decode(value) for value in src["chunk_files"][:]]

        dst_chunk_files = []
        for src_i in selected:
            src_chunk = (args.source_grid.parent / source_chunks[src_i]).resolve()
            dst_chunk = chunks_dir / src_chunk.name
            _link_chunk(src_chunk, dst_chunk, args.link_mode)
            dst_chunk_files.append(dst_chunk)

        grid_path = args.output_dir / f"{args.grid_name}_grid.h5"
        _write_virtual_grid(src, grid_path, args.grid_name, selected, dst_chunk_files)

    if args.update_grid_description:
        _update_grid_description(
            args.model_dir,
            args.source_name,
            args.grid_name,
            grid_path,
        )

    print(f"Wrote {grid_path}")
    print(f"Selected {len(selected)} Rv slices: {rv_min:g}..{rv_max:g} step {args.rv_step:g}")
    print(f"Link mode: {args.link_mode}")


if __name__ == "__main__":
    main()
