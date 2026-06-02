import pytest

import numpy as np

from sedforge import statfunc


class TestMCMC:

    def test_derived_properties(self):
        theta = (26000, 5.8, 0.1429, 5771, 4.438, 1, 0.02)
        pnames = ('teff', 'logg', 'rad', 'teff2', 'logg2', 'rad2', 'av')

        derived_properties = statfunc.get_derived_properties(**dict(zip(pnames, theta)))

        assert derived_properties['mass'] == pytest.approx(0.47, 0.01), \
            "sdB mass wrongly calculated: {}".format(derived_properties['mass'])

        assert derived_properties['mass2'] == pytest.approx(1.00, 0.01), \
            "Solar mass wrongly calculated: {}".format(derived_properties['mass2'])

        assert derived_properties['q'] == pytest.approx(0.47, 0.001), \
            "q wrongly calculated: {}".format(derived_properties['q'])

    def test_derived_properties_include_third_component(self):
        pars = {
            'teff': 6000.0,
            'logg': 4.0,
            'rad': 1.0,
            'teff2': 5000.0,
            'logg2': 4.5,
            'rad2': 0.8,
            'teff3': 3500.0,
            'logg3': 5.0,
            'rad3': 0.3,
        }

        derived_properties = statfunc.get_derived_properties(**pars)

        assert 'mass3' in derived_properties
        assert 'L3' in derived_properties
        assert 'q3' in derived_properties
        assert 'lr3' in derived_properties
        assert 'rr3' in derived_properties
        assert derived_properties['rr3'] == pytest.approx(pars['rad'] / pars['rad3'])

    def test_stat_chi2(self):
        meas = np.array([1.0, 2.0, 3.0])
        e_meas = np.array([0.25, 0.30, 0.15])
        syn = np.array([1.5, 1.89, 3.6])

        # -- fluxes are physically normalised; no profile scale is fit
        chi2 = statfunc.stat_chi2(meas, e_meas, syn)
        assert chi2 == pytest.approx(20.1344, 0.0001), \
            "Chi2 for absolute fluxes not correct ({} != {})".format(chi2, 20.1344)

    def test_group_level_jitter_adds_fractional_variance(self):
        meas = np.array([1.0, 2.0, 3.0])
        e_meas = np.array([0.25, 0.30, 0.15])
        syn = np.array([1.5, 1.89, 3.6])
        photbands = np.array(['GAIA3E_G', 'GAIA3E_BP', '2MASS_Ks'])
        error_model = {
            'type': 'fitted_fraction_by_group',
            'band_groups': {
                'GAIA3E': ['GAIA3E_G', 'GAIA3E_BP'],
                '2MASS': ['2MASS_Ks'],
            },
            'group_parameters': {
                'GAIA3E': 'jitter_GAIA3E',
                '2MASS': 'jitter_2MASS',
            },
            'fixed_fractions': {},
            'include_log_norm': True,
        }
        pars = {'jitter_GAIA3E': 0.1, 'jitter_2MASS': 0.2}

        chi2, deviance = statfunc.stat_chi2(
            meas,
            e_meas,
            syn,
            pars=pars,
            photbands=photbands,
            error_model=error_model,
        )

        variance = e_meas ** 2 + np.array([
            (0.1 * meas[0]) ** 2,
            (0.1 * meas[1]) ** 2,
            (0.2 * meas[2]) ** 2,
        ])
        errors = statfunc.effective_errors(
            meas,
            e_meas,
            pars=pars,
            photbands=photbands,
            error_model=error_model,
        )
        expected_chi2 = np.sum((syn - meas) ** 2 / variance)
        expected_deviance = expected_chi2 + np.sum(np.log(2 * np.pi * variance))
        assert errors == pytest.approx(np.sqrt(variance))
        assert chi2 == pytest.approx(expected_chi2)
        assert deviance == pytest.approx(expected_deviance)
