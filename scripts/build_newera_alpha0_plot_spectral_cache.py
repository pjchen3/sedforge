#!/usr/bin/env python
"""Build the PHOENIX NewEra alpha=0 spectral cache used for plotting."""

import argparse
import os
from pathlib import Path

import numpy as np
import yaml

from sedforge import spectral_cache
from build_newera_alpha0_integrated import (
    expected_alpha0_count,
    iter_alpha0_spectra,
    wavelength_from_header,
)


def _ensure_capacity(flux, params, used_rows, add_rows, n_wave):
    if flux is None:
        n_rows = max(4096, add_rows * 16)
        return np.empty((n_rows, n_wave), dtype=np.float32), params
    if used_rows + add_rows <= len(flux):
        return flux, params

    n_rows = max(used_rows + add_rows, int(len(flux) * 1.5))
    new_flux = np.empty((n_rows, n_wave), dtype=np.float32)
    new_flux[:used_rows] = flux[:used_rows]
    return new_flux, params


def _target_wave(wave, step, wave_min=None, wave_max=None):
    start = float(np.nanmin(wave) if wave_min is None else max(wave_min, np.nanmin(wave)))
    stop = float(np.nanmax(wave) if wave_max is None else min(wave_max, np.nanmax(wave)))
    return np.arange(start, stop + 0.5 * step, step, dtype=np.float32)


def _update_grid_description(model_dir, grid_name, cache_name):
    path = Path(model_dir) / "grid_description.yaml"
    if path.is_file():
        with path.open() as handle:
            desc = yaml.safe_load(handle) or {}
    else:
        desc = {}
    desc.setdefault(grid_name, {})
    desc[grid_name]["spectral_cache"] = cache_name
    with path.open("w") as handle:
        yaml.safe_dump(desc, handle, sort_keys=False)
    print(f"Updated {path}", flush=True)


def build(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / args.outfile
    outfile.parent.mkdir(parents=True, exist_ok=True)

    expected = expected_alpha0_count(args.newera_dir)
    params = {"teff": [], "logg": [], "feh": []}
    flux_grid = None
    target_wave = None
    row = 0
    seen = set()

    for header, mh, flux in iter_alpha0_spectra(args.newera_dir):
        key = (
            round(header["teff"], 6),
            round(header["logg"], 6),
            round(mh, 6),
        )
        if key in seen:
            continue
        seen.add(key)

        wave = wavelength_from_header(header)
        if target_wave is None:
            target_wave = _target_wave(
                wave,
                args.wave_step,
                wave_min=args.wave_min,
                wave_max=args.wave_max,
            )
            n_rows = int(expected or 0)
            if n_rows > 0:
                flux_grid = np.empty((n_rows, len(target_wave)), dtype=np.float32)

        flux_grid, params = _ensure_capacity(
            flux_grid, params, row, 1, len(target_wave)
        )
        flux_grid[row] = np.interp(target_wave, wave, flux, left=np.nan, right=np.nan)
        params["teff"].append(header["teff"])
        params["logg"].append(header["logg"])
        params["feh"].append(mh)
        row += 1

        if row % args.progress_every == 0:
            print(f"Cached {row} spectra", flush=True)

        if args.max_spectra is not None and row >= args.max_spectra:
            break

    if row == 0:
        raise RuntimeError("No alpha=0 NewEra spectra were cached.")

    print(
        f"Writing {outfile} ({row} spectra x {len(target_wave)} wavelengths)",
        flush=True,
    )
    spectral_cache.write_cache(
        outfile,
        {name: values[:row] for name, values in params.items()},
        target_wave,
        flux_grid[:row],
        metadata={
            "grid": args.grid_name,
            "source": "NewEraV3",
            "alpha": 0.0,
            "step": args.wave_step,
        },
        overwrite=args.overwrite,
    )

    if args.update_grid_description:
        cache_name = str(outfile.relative_to(output_dir))
        _update_grid_description(args.output_dir, args.grid_name, cache_name)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--newera-dir", default=os.environ.get("NEWERA_DIR", "NewEra"))
    parser.add_argument("--output-dir", default=os.environ.get("SEDFORGE_MODELS", "sed_models"))
    parser.add_argument("--grid-name", default="newera_alpha0")
    parser.add_argument("--outfile", default="spectral_cache/newera_alpha0_plot_spectra.fits")
    parser.add_argument("--wave-step", type=float, default=2.0)
    parser.add_argument("--wave-min", type=float, default=2500.0)
    parser.add_argument("--wave-max", type=float, default=25000.0)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--max-spectra", type=int, default=None)
    parser.add_argument("--update-grid-description", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
