#!/usr/bin/env python
"""
Build uniform random-access spectral caches for plotting.

The integrated grids remain the fitting products. These caches are only used to
draw continuous model spectra from the nearest raw model point.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits

from sedforge import spectral_cache


CK_MEMBERS = [
    ("ckm25", "raw/ck/ck03_m25.fits", {"feh": -2.5}),
    ("ckm20", "raw/ck/ck03_m20.fits", {"feh": -2.0}),
    ("ckm15", "raw/ck/ck03_m15.fits", {"feh": -1.5}),
    ("ckm10", "raw/ck/ck03_m10.fits", {"feh": -1.0}),
    ("ckm05", "raw/ck/ck03_m05.fits", {"feh": -0.5}),
    ("ckp00", "raw/ck/ck03_p00.fits", {"feh": 0.0}),
    ("ckp02", "raw/ck/ck03_p02.fits", {"feh": 0.2}),
    ("ckp05", "raw/ck/ck03_p05.fits", {"feh": 0.5}),
]

TLUSTY_MEMBERS = [
    ("tlusty01", "raw/tlusty/tlusty_z0.10.fits", {"zfac": 0.1, "feh": -1.0}),
    ("tlusty02", "raw/tlusty/tlusty_z0.20.fits", {"zfac": 0.2, "feh": np.log10(0.2)}),
    ("tlusty05", "raw/tlusty/tlusty_z0.50.fits", {"zfac": 0.5, "feh": np.log10(0.5)}),
    ("tlusty10", "raw/tlusty/tlusty_z1.00.fits", {"zfac": 1.0, "feh": 0.0}),
    ("tlusty20", "raw/tlusty/tlusty_z2.00.fits", {"zfac": 2.0, "feh": np.log10(2.0)}),
]


def _read_wave(path):
    with fits.open(path, memmap=True) as hdul:
        return np.asarray(hdul[1].data["wavelength"], dtype=float)


def _target_wave(path, wave_step=None, wave_min=None, wave_max=None):
    wave = _read_wave(path)
    if wave_step is None:
        if wave_min is None and wave_max is None:
            return wave
        keep = np.ones(len(wave), dtype=bool)
        if wave_min is not None:
            keep &= wave >= wave_min
        if wave_max is not None:
            keep &= wave <= wave_max
        return wave[keep]

    start = float(np.nanmin(wave) if wave_min is None else max(wave_min, np.nanmin(wave)))
    stop = float(np.nanmax(wave) if wave_max is None else min(wave_max, np.nanmax(wave)))
    return np.arange(start, stop + 0.5 * wave_step, wave_step, dtype=np.float32)


def _header_value(header, name, defaults):
    if name == "teff":
        return header.get("TEFF", defaults.get(name, np.nan))
    if name == "logg":
        return header.get("LOGG", defaults.get(name, np.nan))
    if name == "feh":
        for key in ("FEH", "M_H", "MH", "Z"):
            if key in header:
                return header[key]
        return defaults.get(name, np.nan)
    if name == "he_mass":
        for key in ("HEMASS", "HE_MASS", "HE", "YHE"):
            if key in header:
                return header[key]
        return defaults.get(name, np.nan)
    return defaults.get(name, np.nan)


def _count_spectra(files):
    total = 0
    for path, _defaults in files:
        with fits.open(path, memmap=True) as hdul:
            total += len(hdul) - 1
    return total


def _build_from_legacy(files, outfile, wave, overwrite=False):
    n_spectra = _count_spectra(files)
    flux = np.empty((n_spectra, len(wave)), dtype=np.float32)
    params = {name: [] for name in spectral_cache.PARAMETER_COLUMNS}

    row = 0
    for path, defaults in files:
        print(f"Reading {path}", flush=True)
        with fits.open(path, memmap=True) as hdul:
            for ext in hdul[1:]:
                header = ext.header
                data = ext.data
                for name in spectral_cache.PARAMETER_COLUMNS:
                    params[name].append(_header_value(header, name, defaults))

                spec_wave = np.asarray(data["wavelength"], dtype=float)
                spec_flux = np.asarray(data["flux"], dtype=float)
                if len(spec_wave) == len(wave) and np.allclose(spec_wave, wave):
                    flux[row] = spec_flux
                else:
                    flux[row] = np.interp(wave, spec_wave, spec_flux, left=np.nan, right=np.nan)
                row += 1

    keep_params = {
        name: values for name, values in params.items()
        if np.any(np.isfinite(np.asarray(values, dtype=float)))
    }
    print(f"Writing {outfile} ({n_spectra} spectra x {len(wave)} wavelengths)", flush=True)
    spectral_cache.write_cache(
        outfile,
        keep_params,
        wave,
        flux,
        metadata={"ninput": len(files)},
        overwrite=overwrite,
    )


def _cache_name(grid):
    return f"spectral_cache/{grid}_plot_spectra.fits"


def _build_one(model_dir, grid, filename, defaults, wave_step=None,
               wave_min=None, wave_max=None, overwrite=False):
    path = model_dir / filename
    if not path.is_file():
        print(f"Skipping {grid}: missing {path}", flush=True)
        return None
    outfile = model_dir / _cache_name(grid)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    if outfile.exists() and not overwrite:
        print(f"Keeping existing {outfile}", flush=True)
        return str(outfile.relative_to(model_dir))

    wave = _target_wave(path, wave_step=wave_step, wave_min=wave_min, wave_max=wave_max)
    _build_from_legacy([(path, defaults)], outfile, wave, overwrite=overwrite)
    return str(outfile.relative_to(model_dir))


def _build_stack(model_dir, grid, members, wave_step=None,
                 wave_min=None, wave_max=None, overwrite=False):
    files = []
    for _name, filename, defaults in members:
        path = model_dir / filename
        if not path.is_file():
            print(f"Skipping member {filename}: missing {path}", flush=True)
            continue
        files.append((path, defaults))
    if not files:
        print(f"Skipping {grid}: no member files found", flush=True)
        return None

    outfile = model_dir / _cache_name(grid)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    if outfile.exists() and not overwrite:
        print(f"Keeping existing {outfile}", flush=True)
        return str(outfile.relative_to(model_dir))

    wave = _target_wave(files[0][0], wave_step=wave_step,
                        wave_min=wave_min, wave_max=wave_max)
    _build_from_legacy(files, outfile, wave, overwrite=overwrite)
    return str(outfile.relative_to(model_dir))


def _load_description(model_dir):
    path = model_dir / "grid_description.yaml"
    if not path.is_file():
        return {}, path
    with path.open() as handle:
        return yaml.safe_load(handle) or {}, path


def _save_description(path, desc):
    with path.open("w") as handle:
        yaml.safe_dump(desc, handle, sort_keys=False)


def _set_cache(desc, grid, cache_name):
    if cache_name is None:
        return
    desc.setdefault(grid, {})
    desc[grid]["spectral_cache"] = cache_name


def build(args):
    model_dir = Path(args.model_dir).expanduser().resolve()
    desc, desc_path = _load_description(model_dir)
    wave_min = args.wave_min
    wave_max = args.wave_max

    ck_all_cache = _build_stack(
        model_dir, "ck_all", CK_MEMBERS, overwrite=args.overwrite,
    )
    _set_cache(desc, "ck_all", ck_all_cache)
    for grid, _filename, _defaults in CK_MEMBERS:
        _set_cache(desc, grid, ck_all_cache)

    _set_cache(desc, "blackbody", _build_one(
        model_dir, "blackbody", "raw/blackbody/blackbody_discint.fits", {},
        overwrite=args.overwrite,
    ))
    _set_cache(desc, "koester2", _build_one(
        model_dir, "koester2", "raw/koester2/koester2.fits", {},
        overwrite=args.overwrite,
    ))

    _set_cache(desc, "tlusty00", _build_one(
        model_dir, "tlusty00", "raw/tlusty/tlusty_z0.00.fits",
        {"zfac": 0.0}, wave_step=args.highres_step,
        wave_min=wave_min, wave_max=wave_max,
        overwrite=args.overwrite,
    ))
    tlusty_all_cache = _build_stack(
        model_dir, "tlusty_all", TLUSTY_MEMBERS,
        wave_step=args.highres_step, wave_min=wave_min, wave_max=wave_max,
        overwrite=args.overwrite,
    )
    _set_cache(desc, "tlusty_all", tlusty_all_cache)
    for grid, _filename, _defaults in TLUSTY_MEMBERS:
        _set_cache(desc, grid, tlusty_all_cache)

    for bad_index in range(1, 10):
        desc.pop(f"tmap_he{bad_index:03d}", None)

    tmap_cache = _build_one(
        model_dir, "tmap", "raw/tmap/tmap.fits", {},
        wave_step=args.highres_step, wave_min=wave_min, wave_max=wave_max,
        overwrite=args.overwrite,
    )
    for he_index in range(0, 101, 10):
        _set_cache(desc, f"tmap_he{he_index:03d}", tmap_cache)

    _save_description(desc_path, desc)
    print(f"Updated {desc_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("SEDFORGE_MODELS", "sed_models"),
    )
    parser.add_argument("--highres-step", type=float, default=2.0)
    parser.add_argument("--wave-min", type=float, default=1000.0)
    parser.add_argument("--wave-max", type=float, default=285500.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
