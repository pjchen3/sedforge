"""Uniform random-access cache for model spectra used by plotting."""

import os

import numpy as np
from astropy.io import fits
from astropy.table import Table


CACHE_VERSION = "SEDSPEC1"
PARAMETER_COLUMNS = ("teff", "logg", "feh", "he_mass")
PC_TO_RSOL = 44365810.04823812


def _models_directory(directory=None):
    return directory or os.environ.get("SEDFORGE_MODELS")


def cache_file(grid, grid_description=None, directory=None):
    """Return the configured spectral-cache file for a grid, or ``None``."""
    if grid_description is None or grid not in grid_description:
        return None

    desc = grid_description[grid]
    filename = desc.get("spectral_cache", desc.get("raw_cache"))
    if filename is None:
        return None

    if os.path.isabs(filename):
        return filename

    models_dir = _models_directory(directory)
    if models_dir is None:
        return None
    return os.path.join(models_dir, filename)


def has_cache(grid, grid_description=None, directory=None):
    path = cache_file(grid, grid_description=grid_description, directory=directory)
    return path is not None and os.path.isfile(path)


def _normalise_parameter_table(params):
    table = Table(params)
    for name in table.colnames:
        table[name] = np.asarray(table[name], dtype=np.float32)
    return table


def write_cache(outfile, params, wave, flux, metadata=None, overwrite=False):
    """
    Write a spectral cache.

    The format is:

    - ``PARAMS``: one row per model spectrum.
    - ``WAVE``: common wavelength grid in Angstrom.
    - ``FLUX``: ``(n_spectra, n_wave)`` array in Flambda units.
    """
    params = _normalise_parameter_table(params)
    wave = np.asarray(wave, dtype=np.float32)
    flux = np.asarray(flux, dtype=np.float32)

    if flux.ndim != 2:
        raise ValueError("flux must have shape (n_spectra, n_wave)")
    if flux.shape[0] != len(params):
        raise ValueError("flux row count must match PARAMS length")
    if flux.shape[1] != len(wave):
        raise ValueError("flux wavelength dimension must match WAVE length")

    primary = fits.PrimaryHDU()
    primary.header["FORMAT"] = CACHE_VERSION
    primary.header["NWAVE"] = len(wave)
    primary.header["NSPEC"] = len(params)
    primary.header["WAVEUNIT"] = "Angstrom"
    primary.header["FLUXTYPE"] = "Flambda"
    if metadata:
        for key, value in metadata.items():
            if value is not None:
                primary.header[str(key).upper()[:8]] = value

    hdul = fits.HDUList([
        primary,
        fits.BinTableHDU(params, name="PARAMS"),
        fits.ImageHDU(wave, name="WAVE"),
        fits.ImageHDU(flux, name="FLUX"),
    ])
    hdul.writeto(outfile, overwrite=overwrite)


def _parameter_columns(params, requested):
    names = []
    for name in PARAMETER_COLUMNS:
        if name in params.names and requested.get(name) is not None:
            names.append(name)
    return names


def _nearest_index(params, requested):
    names = _parameter_columns(params, requested)
    if not names:
        return 0

    score = np.zeros(len(params), dtype=float)
    for name in names:
        values = np.asarray(params[name], dtype=float)
        target = float(requested[name])
        span = np.nanmax(values) - np.nanmin(values)
        if not np.isfinite(span) or span == 0:
            span = 1.0
        score += ((values - target) / span) ** 2
    return int(np.nanargmin(score))


def read_spectrum(path, teff=None, logg=None, feh=None, he_mass=None,
                  rad=None, distance=None, dist=None, d=None):
    """
    Return the nearest cached spectrum without loading the full cache.

    ``rad`` and ``distance`` apply the same radius/distance scaling used by the
    integrated-grid model calls. Reddening is intentionally left to the caller.
    """
    requested = {
        "teff": teff,
        "logg": logg,
        "feh": feh,
        "he_mass": he_mass,
    }

    with fits.open(path, memmap=True) as hdul:
        if hdul[0].header.get("FORMAT") != CACHE_VERSION:
            raise ValueError(f"{path} is not a {CACHE_VERSION} spectral cache")
        params = hdul["PARAMS"].data
        index = _nearest_index(params, requested)
        wave = np.array(hdul["WAVE"].data, dtype=float)
        flux = np.array(hdul["FLUX"].data[index], dtype=float)

    if rad is not None:
        flux = flux * np.asarray(rad, dtype=float) ** 2

    if distance is None:
        distance = dist
    if distance is not None:
        flux = flux / (np.asarray(distance, dtype=float) * PC_TO_RSOL) ** 2

    if d is not None:
        flux = flux / np.asarray(d, dtype=float) ** 2

    return wave, flux


def _resample_flux(wave, flux, target_wave):
    return np.interp(target_wave, wave, flux, left=np.nan, right=np.nan)


def build_cache_from_legacy_fits(raw_gridfile, outfile, parameter_defaults=None,
                                 wave=None, wave_step=None, overwrite=False):
    """
    Convert the legacy raw-grid FITS layout to the uniform spectral cache.

    Legacy raw grids store one model spectrum per FITS extension, with TEFF/LOGG
    in the header and ``wavelength``/``flux`` table columns. If ``wave_step`` is
    given, the spectra are resampled to a common uniform grid for compact
    plotting caches.
    """
    parameter_defaults = parameter_defaults or {}
    params = {name: [] for name in PARAMETER_COLUMNS}
    rows = []

    with fits.open(raw_gridfile, memmap=True) as hdul:
        if len(hdul) < 2:
            raise ValueError(f"{raw_gridfile} does not contain model spectra")

        first_wave = np.asarray(hdul[1].data["wavelength"], dtype=float)
        if wave is None:
            if wave_step is None:
                wave = first_wave
            else:
                wave = np.arange(
                    np.nanmin(first_wave),
                    np.nanmax(first_wave) + 0.5 * wave_step,
                    wave_step,
                    dtype=float,
                )
        wave = np.asarray(wave, dtype=float)

        for ext in hdul[1:]:
            header = ext.header
            data = ext.data
            for name in PARAMETER_COLUMNS:
                if name == "teff":
                    value = header.get("TEFF", parameter_defaults.get(name, np.nan))
                elif name == "logg":
                    value = header.get("LOGG", parameter_defaults.get(name, np.nan))
                else:
                    value = header.get(name.upper(), parameter_defaults.get(name, np.nan))
                params[name].append(value)

            spec_wave = np.asarray(data["wavelength"], dtype=float)
            spec_flux = np.asarray(data["flux"], dtype=float)
            if len(spec_wave) == len(wave) and np.allclose(spec_wave, wave):
                rows.append(spec_flux.astype(np.float32))
            else:
                rows.append(_resample_flux(spec_wave, spec_flux, wave).astype(np.float32))

    flux = np.vstack(rows)
    keep_params = {
        name: values for name, values in params.items()
        if np.any(np.isfinite(np.asarray(values, dtype=float)))
    }
    write_cache(
        outfile,
        keep_params,
        wave,
        flux,
        metadata={"source": os.path.basename(raw_gridfile)},
        overwrite=overwrite,
    )
