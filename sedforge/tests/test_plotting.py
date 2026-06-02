import matplotlib
matplotlib.use('Agg')

import numpy as np
import pylab as pl
import pytest
from astropy.table import Table

from sedforge import plotting


def test_photband_system_and_label_use_current_filter_names():
    assert plotting._photband_system('GAIA3E_G') == 'GAIA3E'
    assert plotting._photband_system('2MASS_Ks') == '2MASS'
    assert plotting._photband_system('HST_ACS_WFC_F814W') == 'HST_ACS_WFC'
    assert plotting._photband_label('HST_ACS_WFC_F814W') == 'F814W'


def test_sample_for_corner_downsamples_deterministically():
    data = np.zeros(100, dtype=[('x', float)])
    data['x'] = np.arange(100)

    subset1 = plotting.sample_for_corner(data, max_samples=10, random_seed=7)
    subset2 = plotting.sample_for_corner(data, max_samples=10, random_seed=7)

    assert len(subset1) == 10
    assert np.array_equal(subset1, subset2)


def test_corner_labels_include_default_units():
    labels = plotting.corner_labels(['teff', 'logg', 'feh', 'rad', 'distance', 'av'])

    assert labels == [
        r'$T_{\rm eff}$ (K)',
        r'$\log g$ (dex)',
        r'$\mathrm{[Fe/H]}$ (dex)',
        r'$R$ ($R_\odot$)',
        r'$d$ (pc)',
        r'$A_V$ (mag)',
    ]


def test_corner_labels_include_group_jitter_parameter():
    labels = plotting.corner_labels(['jitter_GAIA3E'])

    assert labels == [r'$f_{\rm GAIA3E}$']


def test_corner_labels_keep_component_suffixes():
    labels = plotting.corner_labels(['teff2', 'rad2', 'mass2'])

    assert labels == [
        r'${T_{\rm eff}}_{2}$ (K)',
        r'${R}_{2}$ ($R_\odot$)',
        r'${M}_{2}$ ($M_\odot$)',
    ]


def test_corner_labels_can_omit_units_for_titles():
    labels = plotting.corner_labels(['teff', 'distance', 'av', 'rv'], include_units=False)

    assert labels == [r'$T_{\rm eff}$', r'$d$', r'$A_V$', r'$R_V$']


def test_corner_labels_units_override_without_duplication():
    labels = plotting.corner_labels(['teff', 'rad'], units={'teff': 'kK', 'rad': r'$R_\odot$'})

    assert labels == [r'$T_{\rm eff}$ (kK)', r'$R$ ($R_\odot$)']


def test_corner_labels_explicit_units_are_not_repeated():
    labels = plotting.corner_labels(
        ['teff'],
        labels=[r'$T_{\rm eff}$ (K)'],
        units={'teff': 'K'},
    )

    assert labels == [r'$T_{\rm eff}$ (K)']


def test_auto_residual_ylim_uses_residuals_and_errors():
    low, high = plotting._auto_residual_ylim(
        np.array([-0.02, 0.03]),
        np.array([0.01, 0.02]),
    )

    assert low < -0.03
    assert high > 0.05
    assert abs(low) == high


def test_auto_sed_xlim_uses_half_bandwidth():
    low, high = plotting._auto_sed_xlim(
        np.array([5000.0, 10000.0]),
        np.array([1000.0, 4000.0]),
        pad_fraction=0.0,
        minimum_pad=0.0,
    )

    assert low == pytest.approx(4500.0)
    assert high == pytest.approx(12000.0)


def test_plot_fit_uses_physical_distance_and_writes_filter_names(monkeypatch, tmp_path):
    captured = {}

    def fake_get_info(photbands):
        return Table(
            [
                np.array(photbands, dtype=str),
                np.array([5000.0, 22000.0]),
                np.array([1000.0, 3000.0]),
            ],
            names=['photband', 'eff_wave', 'bandwidth'],
        )

    def fake_get_itable(**kwargs):
        captured.update(kwargs)
        return np.array([[1.0], [2.0]]), {'L': np.array([1.0])}

    monkeypatch.setattr(plotting.filters, 'get_info', fake_get_info)
    monkeypatch.setattr(plotting.model, 'get_itable', fake_get_itable)

    observations_path = tmp_path / 'observations.txt'
    pl.figure()
    plotting.plot_fit(
        np.array([1.0, 2.0]),
        np.array([0.1, 0.2]),
        np.array(['GAIA3E_G', 'WISE_RSR_W1']),
        pars={
            'teff': [8000.0, 7900.0, 100.0, 100.0],
            'logg': [4.0, 4.0, 0.0, 0.0],
            'rad': [2.0, 2.0, 0.0, 0.0],
            'distance': [800.0, 805.0, 20.0, 20.0],
            'd': [800.0, 805.0, 20.0, 20.0],
            'av': [0.155, 0.155, 0.0, 0.0],
            'jitter_GAIA3E': [0.1, 0.1, 0.0, 0.0],
            'jitter_WISE_RSR': [0.2, 0.2, 0.0, 0.0],
            'chi2': [1.2, 1.2, 0.0, 0.0],
        },
        grids=[],
        gridnames=[],
        observations_path=str(observations_path),
        error_model={
            'type': 'fitted_fraction_by_group',
            'band_groups': {
                'GAIA3E': ['GAIA3E_G'],
                'WISE_RSR': ['WISE_RSR_W1'],
            },
            'group_parameters': {
                'GAIA3E': 'jitter_GAIA3E',
                'WISE_RSR': 'jitter_WISE_RSR',
            },
            'fixed_fractions': {},
            'include_log_norm': True,
        },
    )
    fig = pl.gcf()
    assert len(fig.axes) == 2
    assert tuple(fig.get_size_inches()) == (9.0, 7.0)
    assert fig.axes[0].get_ylabel() == r'$\lambda F_\lambda$ (erg cm$^{-2}$ s$^{-1}$)'
    assert fig.axes[1].get_ylabel() == r'O-C (mag)'
    pl.close('all')

    assert 'distance' in captured
    assert 'd' not in captured
    text = observations_path.read_text()
    assert 'photband' in text
    assert 'GAIA3E_G' in text
    assert 'WISE_RSR_W1' in text
    table = Table.read(observations_path, format='ascii')
    assert 'raw_error' in table.colnames
    assert 'total_error' in table.colnames
    assert table['raw_error'][0] == pytest.approx(0.1)
    assert table['total_error'][0] == pytest.approx(np.sqrt(0.1 ** 2 + (0.1 * 1.0) ** 2))
    assert table['total_error'][1] == pytest.approx(np.sqrt(0.2 ** 2 + (0.2 * 2.0) ** 2))
