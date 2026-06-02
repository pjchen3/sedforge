#!/usr/bin/env python
"""
Run a binary-star synthetic recovery test with two local ck_all components.

The generated photometry uses the magnitude project format:
    photband mag mag_err system mag_type mag_zp_offset
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np
from astropy.io import ascii
from astropy.table import Table


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'sed_models'
OUTDIR = ROOT / 'sed_fit_tests' / 'synthetic_recovery_binary_ck_all'
os.environ['SEDFORGE_MODELS'] = str(MODELS)

from sedforge import catalog_photometry, main, model  # noqa: E402
from sedforge.setup_format import setup_to_readable_yaml  # noqa: E402


PHOTBANDS = [
    'GALEX_FUV',
    'GALEX_NUV',
    'HST_WFC3_F275W',
    'HST_WFC3_F336W',
    'GAIA3E_BP',
    'GAIA3E_G',
    'GAIA3E_RP',
    'PS1_g',
    'PS1_r',
    'PS1_i',
    'PS1_z',
    'PS1_y',
    '2MASS_J',
    '2MASS_H',
    '2MASS_Ks',
    'WISE_RSR_W1',
    'WISE_RSR_W2',
]


def _system_for_band(photband):
    try:
        return catalog_photometry.default_magnitude_system(photband)
    except ValueError:
        return 'ab'


def _mag_type_for_band(photband):
    return catalog_photometry.default_magnitude_type(photband)


def _mag_offset_for_band(photband):
    return catalog_photometry.default_mag_zp_offset(photband)


TRUTH = {
    'teff': 12000.0,
    'logg': 4.0,
    'feh': -0.5,
    'rad': 2.0,
    'teff2': 4500.0,
    'logg2': 2.5,
    'rad2': 8.0,
    'distance': 1000.0,
    'av': 0.186,
}


def _write_setup(path, photometry_path):
    setup = {
        'objectname': 'synthetic_binary_ck_all',
        'photometryfile': str(photometry_path),
        'photband_exclude': [],
        'pnames': [
            'teff',
            'feh',
            'rad',
            'teff2',
            'rad2',
            'distance',
            'av',
        ],
        'limits': [
            [10000.0, 14000.0],
            [-0.8, -0.2],
            [1.0, 3.0],
            [4000.0, 5500.0],
            [5.0, 11.0],
            [850.0, 1150.0],
            [0.06, 0.31],
        ],
        # Broadband binary SEDs strongly constrain radius/distance ratios; a
        # distance prior keeps the two radii separately identifiable.
        'fixed': {
            'logg': TRUTH['logg'],
            'logg2': TRUTH['logg2'],
        },
        'priors': {
            'distance': [TRUTH['distance'], 50.0],
        },
        'grids': ['ck_all', 'ck_all'],
        'reddening_law': 'WC2019',
        'reddening_Rv': 3.1,
        'reddening_case1': 1,
        'nwalkers': 96,
        'nsteps': 2000,
        'nrelax': 500,
        'a': 2,
        'percentiles': [16, 50, 84],
        'resultfile': str(OUTDIR / 'synthetic_binary_ck_all_results.csv'),
        'datafile': str(OUTDIR / 'synthetic_binary_ck_all_samples.fits'),
        'plot1': {
            'type': 'sed_fit',
            'result': 'pc',
            'path': str(OUTDIR / 'synthetic_binary_ck_all_sed.png'),
        },
        'plot2': {
            'type': 'distribution',
            'show_best': True,
            'path': str(OUTDIR / 'synthetic_binary_ck_all_distribution.png'),
            'parameters': [
                'teff',
                'feh',
                'rad',
                'teff2',
                'rad2',
                'distance',
                'av',
            ],
        },
    }
    path.write_text(setup_to_readable_yaml(setup))
    return setup


def _make_photometry(path):
    flux, _ = model.get_itable(
        grid=['ck_all', 'ck_all'],
        photbands=PHOTBANDS,
        teff=TRUTH['teff'],
        logg=TRUTH['logg'],
        feh=TRUTH['feh'],
        rad=TRUTH['rad'],
        teff2=TRUTH['teff2'],
        logg2=TRUTH['logg2'],
        rad2=TRUTH['rad2'],
        distance=TRUTH['distance'],
        av=TRUTH['av'],
    )

    frac_err = 0.02
    flux_err = frac_err * flux
    observed = flux.copy()

    systems = [_system_for_band(photband) for photband in PHOTBANDS]
    mag_types = [_mag_type_for_band(photband) for photband in PHOTBANDS]
    mag_offsets = [_mag_offset_for_band(photband) for photband in PHOTBANDS]
    mags, mag_errs = [], []
    for photband, value, error, system, mag_type, offset in zip(
            PHOTBANDS, observed, flux_err, systems, mag_types, mag_offsets):
        mag, mag_err = catalog_photometry.flux_to_mag(
            photband,
            value,
            error,
            system,
            mag_type=mag_type,
            mag_zp_offset=offset,
        )
        mags.append(mag)
        mag_errs.append(mag_err)

    table = Table(
        [PHOTBANDS, mags, mag_errs, systems, mag_types, mag_offsets, observed, flux_err, flux],
        names=[
            'photband',
            'mag',
            'mag_err',
            'system',
            'mag_type',
            'mag_zp_offset',
            'flux',
            'flux_err',
            'true_flux',
        ],
    )
    ascii.write(
        table['photband', 'mag', 'mag_err', 'system', 'mag_type', 'mag_zp_offset'],
        path,
        overwrite=True,
    )
    ascii.write(table, OUTDIR / 'synthetic_binary_ck_all_photometry_with_truth.dat', overwrite=True)
    return table


def _write_recovery_summary(path, results):
    rows = []
    for name in [
        'teff',
        'logg',
        'feh',
        'rad',
        'teff2',
        'logg2',
        'rad2',
        'distance',
        'av',
    ]:
        best, median, err_lo, err_hi = results[name]
        truth = TRUTH[name]
        sigma = 0.5 * (err_lo + err_hi)
        rows.append((
            name,
            truth,
            best,
            median,
            err_lo,
            err_hi,
            median - truth,
            (median - truth) / sigma if sigma > 0 else np.nan,
        ))

    table = Table(
        rows=rows,
        names=[
            'parameter',
            'truth',
            'best',
            'median',
            'err_lo',
            'err_hi',
            'median_minus_truth',
            'pull_sigma',
        ],
    )
    ascii.write(table, path, overwrite=True)
    return table


def main_cli():
    np.random.seed(20260526)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    photometry_path = OUTDIR / 'synthetic_binary_ck_all.phot'
    setup_path = OUTDIR / 'synthetic_binary_ck_all_setup.yaml'

    _make_photometry(photometry_path)
    setup = _write_setup(setup_path, photometry_path)

    main.validate_setup(setup)
    photbands, obs, obs_err = main.get_observations(setup)
    results, samples, priors, gridnames = main.fit_sed(setup, photbands, obs, obs_err)
    main.write_results(setup, results, samples, obs, obs_err, photbands)
    main.plot_results(setup, results, samples, priors, gridnames, obs, obs_err, photbands)

    summary = _write_recovery_summary(
        OUTDIR / 'synthetic_binary_ck_all_recovery_summary.dat',
        results,
    )
    print(summary)
    print(f'Wrote outputs to {OUTDIR}')


if __name__ == '__main__':
    main_cli()
