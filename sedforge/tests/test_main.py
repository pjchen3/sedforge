import pytest
from argparse import Namespace

from sedforge import main


def _setup(path):
    return {
        'photometryfile': str(path),
        'pnames': ['teff', 'logg', 'rad', 'distance', 'av'],
        'limits': [
            [3500, 8000],
            [3.5, 5.0],
            [0.1, 10.0],
            [10.0, 10000.0],
            [0.0, 1.0],
        ],
        'priors': {},
        'grids': ['ck_all'],
    }


def test_create_photometry_rejects_gaia_id_with_radec(tmp_path):
    args = Namespace(
        gaia_id='123',
        ra=1.0,
        dec=2.0,
        coord=None,
        catalog_config=None,
        output=str(tmp_path / 'target.phot'),
        radius=3.0,
        default_mag_error=0.03,
        timeout=60,
        metadata_output=None,
    )

    with pytest.raises(ValueError, match='--gaia-id or --ra/--dec'):
        main.create_photometry(args)


def test_get_observations_reads_magnitude_photometry(monkeypatch, tmp_path):
    def fake_mag_to_flux(
            photband,
            mag,
            mag_err,
            system=None,
            mag_type=None,
            mag_zp_offset=0.0):
        assert system in {'vega', 'ab'}
        return mag * 1.0e-12, mag_err * 1.0e-12

    monkeypatch.setattr(main.catalog_photometry, 'mag_to_flux', fake_mag_to_flux)

    photfile = tmp_path / 'target.phot'
    photfile.write_text(
        'photband mag mag_err system\n'
        'GAIA3E_G 10.0 0.01 vega\n'
        'PS1_g 20.0 0.02 ab\n'
    )

    photbands, obs, obs_err = main.get_observations(_setup(photfile))

    assert list(photbands) == ['GAIA3E_G', 'PS1_g']
    assert list(obs) == [10.0e-12, 20.0e-12]
    assert list(obs_err) == [0.01e-12, 0.02e-12]


def test_get_observations_infers_common_magnitude_systems(monkeypatch, tmp_path):
    seen = []

    def fake_mag_to_flux(
            photband,
            mag,
            mag_err,
            system=None,
            mag_type=None,
            mag_zp_offset=0.0):
        seen.append((photband, system))
        return mag, mag_err

    monkeypatch.setattr(main.catalog_photometry, 'mag_to_flux', fake_mag_to_flux)

    photfile = tmp_path / 'target.phot'
    photfile.write_text(
        'photband mag mag_err\n'
        'GAIA3E_G 10.0 0.01\n'
        'PS1_g 20.0 0.02\n'
    )

    main.get_observations(_setup(photfile))

    assert seen == [('GAIA3E_G', 'vega'), ('PS1_g', 'ab')]


def test_get_observations_defaults_sdss_to_asinh_and_ab_offset(monkeypatch, tmp_path):
    seen = []

    def fake_mag_to_flux(
            photband,
            mag,
            mag_err,
            system=None,
            mag_type=None,
            mag_zp_offset=0.0):
        seen.append((photband, system, mag_type, mag_zp_offset))
        return mag, mag_err

    monkeypatch.setattr(main.catalog_photometry, 'mag_to_flux', fake_mag_to_flux)

    photfile = tmp_path / 'sdss.phot'
    photfile.write_text(
        'photband mag mag_err\n'
        'SDSS_u 20.0 0.02\n'
        'SDSS_g 20.0 0.02\n'
        'SDSS_z 20.0 0.02\n'
    )

    main.get_observations(_setup(photfile))

    assert seen == [
        ('SDSS_u', 'ab', 'asinh', -0.04),
        ('SDSS_g', 'ab', 'asinh', 0.0),
        ('SDSS_z', 'ab', 'asinh', 0.02),
    ]


def test_get_observations_rejects_ambiguous_magnitude_system(tmp_path):
    photfile = tmp_path / 'hst.phot'
    photfile.write_text(
        'photband mag mag_err\n'
        'HST_WFC3_F814W 20.0 0.01\n'
    )

    with pytest.raises(ValueError, match='No default magnitude system'):
        main.get_observations(_setup(photfile))


def test_get_observations_prefers_mag_over_check_flux(monkeypatch, tmp_path):
    monkeypatch.setattr(
        main.catalog_photometry,
        'mag_to_flux',
        lambda photband, mag, mag_err, system=None, mag_type=None, mag_zp_offset=0.0: (3.0, 0.3),
    )

    photfile = tmp_path / 'target_with_mag.phot'
    photfile.write_text(
        'photband mag mag_err system flux flux_err\n'
        'GAIA3E_G 12.3 0.01 vega 1.0e-12 1.0e-14\n'
    )

    photbands, obs, obs_err = main.get_observations(_setup(photfile))

    assert list(photbands) == ['GAIA3E_G']
    assert list(obs) == [3.0]
    assert list(obs_err) == [0.3]


def test_get_observations_reads_advanced_flux_photometry(tmp_path):
    photfile = tmp_path / 'target_flux.phot'
    photfile.write_text(
        'photband flux flux_err\n'
        'GAIA3E_G 1.0e-12 1.0e-14\n'
        '2MASS_Ks 2.0e-13 2.0e-15\n'
    )

    photbands, obs, obs_err = main.get_observations(_setup(photfile))

    assert list(photbands) == ['GAIA3E_G', '2MASS_Ks']
    assert list(obs) == [1.0e-12, 2.0e-13]
    assert list(obs_err) == [1.0e-14, 2.0e-15]


def test_get_observations_rejects_legacy_photometry_columns(tmp_path):
    photfile = tmp_path / 'legacy.phot'
    photfile.write_text(
        'band flux eflux\n'
        'GAIA3E_G 1.0e-12 1.0e-14\n'
    )

    with pytest.raises(ValueError, match='photband mag mag_err'):
        main.get_observations(_setup(photfile))


def test_validate_setup_rejects_legacy_column_index_keys(tmp_path):
    photfile = tmp_path / 'target.phot'
    photfile.write_text(
        'photband flux flux_err\n'
        'GAIA3E_G 1.0e-12 1.0e-14\n'
    )
    setup = _setup(photfile)
    setup['photband_index'] = 'band'
    setup['obs_index'] = 'flux'
    setup['err_index'] = 'eflux'

    with pytest.raises(ValueError, match='Legacy photometry column-index'):
        main.validate_setup(setup)


def test_validate_setup_rejects_non_scalar_fixed_parameter(tmp_path):
    photfile = tmp_path / 'target.phot'
    photfile.write_text(
        'photband flux flux_err\n'
        'GAIA3E_G 1.0e-12 1.0e-14\n'
    )
    setup = _setup(photfile)
    setup['fixed'] = {'feh': [0.0, 0.1]}

    with pytest.raises(ValueError, match='single numeric value'):
        main.validate_setup(setup)


def test_validate_setup_rejects_retired_constraints_key(tmp_path):
    photfile = tmp_path / 'target.phot'
    photfile.write_text(
        'photband flux flux_err\n'
        'GAIA3E_G 1.0e-12 1.0e-14\n'
    )
    setup = _setup(photfile)
    setup['constraints'] = {'distance': [1000.0, 50.0]}

    with pytest.raises(ValueError, match="'constraints' is retired"):
        main.validate_setup(setup)


@pytest.mark.parametrize(
    'mutate',
    [
        lambda setup: setup['pnames'].__setitem__(-1, 'ebv'),
        lambda setup: setup.update({'fixed': {'ebv': 0.05}}),
        lambda setup: setup.update({'priors': {'ebv': [0.05, 0.01]}}),
        lambda setup: setup.update({'grid_variables': ['teff', 'logg', 'ebv']}),
        lambda setup: setup.update({
            'plot2': {'type': 'distribution', 'parameters': ['teff', 'ebv']}
        }),
    ],
)
def test_validate_setup_rejects_legacy_ebv_parameter(tmp_path, mutate):
    photfile = tmp_path / 'target.phot'
    photfile.write_text(
        'photband flux flux_err\n'
        'GAIA3E_G 1.0e-12 1.0e-14\n'
    )
    setup = _setup(photfile)
    mutate(setup)

    with pytest.raises(ValueError, match="Use 'av'"):
        main.validate_setup(setup)


def test_required_parameter_must_be_fitted_or_fixed(tmp_path):
    photfile = tmp_path / 'target.phot'
    photfile.write_text(
        'photband flux flux_err\n'
        'GAIA3E_G 1.0e-12 1.0e-14\n'
    )
    setup = _setup(photfile)
    setup['pnames'].remove('logg')
    setup['limits'].pop(1)
    setup['fixed'] = {'feh': 0.0}

    photbands, obs, obs_err = main.get_observations(setup)
    with pytest.raises(ValueError, match='Missing required model parameter'):
        main.fit_sed(setup, photbands, obs, obs_err)


def test_zero_width_limits_are_not_fixed_parameters(tmp_path):
    photfile = tmp_path / 'target.phot'
    photfile.write_text(
        'photband flux flux_err\n'
        'GAIA3E_G 1.0e-12 1.0e-14\n'
    )
    setup = _setup(photfile)
    setup['limits'][1] = [4.0, 4.0]
    setup['fixed'] = {'feh': 0.0}

    photbands, obs, obs_err = main.get_observations(setup)
    with pytest.raises(ValueError, match='Fixed parameters must be listed in fixed'):
        main.fit_sed(setup, photbands, obs, obs_err)


def test_metallicity_grid_requires_explicit_feh(tmp_path):
    photfile = tmp_path / 'target.phot'
    photfile.write_text(
        'photband flux flux_err\n'
        'GAIA3E_G 1.0e-12 1.0e-14\n'
    )
    setup = _setup(photfile)

    photbands, obs, obs_err = main.get_observations(setup)
    with pytest.raises(ValueError, match=r'requires an explicit \[Fe/H\] value'):
        main.fit_sed(setup, photbands, obs, obs_err)


def test_prior_must_target_sampled_parameter(tmp_path):
    photfile = tmp_path / 'target.phot'
    photfile.write_text(
        'photband flux flux_err\n'
        'GAIA3E_G 1.0e-12 1.0e-14\n'
    )
    setup = _setup(photfile)
    setup['fixed'] = {'feh': 0.0}
    setup['priors'] = {'L': [5.0, 0.1]}

    photbands, obs, obs_err = main.get_observations(setup)
    with pytest.raises(ValueError, match='not a fitted parameter'):
        main.fit_sed(setup, photbands, obs, obs_err)


def test_prior_cannot_target_fixed_parameter(tmp_path):
    photfile = tmp_path / 'target.phot'
    photfile.write_text(
        'photband flux flux_err\n'
        'GAIA3E_G 1.0e-12 1.0e-14\n'
    )
    setup = _setup(photfile)
    setup['fixed'] = {'feh': 0.0}
    setup['priors'] = {'feh': [0.0, 0.1]}

    photbands, obs, obs_err = main.get_observations(setup)
    with pytest.raises(ValueError, match='targets a fixed parameter'):
        main.fit_sed(setup, photbands, obs, obs_err)


def test_metallicity_grid_variables_must_include_feh(tmp_path):
    photfile = tmp_path / 'target.phot'
    photfile.write_text(
        'photband flux flux_err\n'
        'GAIA3E_G 1.0e-12 1.0e-14\n'
    )
    setup = _setup(photfile)
    setup['fixed'] = {'feh': 0.0}
    setup['grid_variables'] = ['teff', 'logg', 'av']

    photbands, obs, obs_err = main.get_observations(setup)
    with pytest.raises(ValueError, match="does not include 'feh'"):
        main.fit_sed(setup, photbands, obs, obs_err)


def test_rv_parameter_is_rejected_for_fixed_rv_grids(tmp_path):
    photfile = tmp_path / 'target.phot'
    photfile.write_text(
        'photband flux flux_err\n'
        'GAIA3E_G 1.0e-12 1.0e-14\n'
    )
    setup = _setup(photfile)
    setup['pnames'].append('rv')
    setup['limits'].append([2.0, 5.0])
    setup['fixed'] = {'feh': 0.0}

    photbands, obs, obs_err = main.get_observations(setup)
    with pytest.raises(ValueError, match='not used by the selected model'):
        main.fit_sed(setup, photbands, obs, obs_err)


def test_rv_axis_grid_requires_rv_parameter(monkeypatch, tmp_path):
    photfile = tmp_path / 'target.phot'
    photfile.write_text(
        'photband flux flux_err\n'
        'GAIA3E_G 1.0e-12 1.0e-14\n'
    )
    monkeypatch.setitem(
        main.model.grid_description,
        'rv_grid',
        {
            'filename': 'rv_grid/grid',
            'integrated_format': 'hdf5',
            'axes': ['teff', 'logg', 'feh', 'rv', 'av'],
            'supports_feh': True,
            'supports_rv': True,
        },
    )
    setup = _setup(photfile)
    setup['grids'] = ['rv_grid']
    setup['fixed'] = {'feh': 0.0}

    photbands, obs, obs_err = main.get_observations(setup)
    with pytest.raises(ValueError, match='Missing required model parameter'):
        main.fit_sed(setup, photbands, obs, obs_err)


def test_rv_axis_grid_rejects_reddening_rv_setup_key(monkeypatch, tmp_path):
    photfile = tmp_path / 'target.phot'
    photfile.write_text(
        'photband flux flux_err\n'
        'GAIA3E_G 1.0e-12 1.0e-14\n'
    )
    monkeypatch.setitem(
        main.model.grid_description,
        'rv_grid',
        {
            'filename': 'rv_grid/grid',
            'integrated_format': 'hdf5',
            'axes': ['teff', 'logg', 'feh', 'rv', 'av'],
            'supports_feh': True,
            'supports_rv': True,
        },
    )
    setup = _setup(photfile)
    setup['grids'] = ['rv_grid']
    setup['pnames'].append('rv')
    setup['limits'].append([2.0, 5.0])
    setup['fixed'] = {'feh': 0.0}
    setup['reddening_Rv'] = 3.1

    with pytest.raises(ValueError, match='Remove setup key'):
        main.validate_setup(setup)


def test_fixed_parameter_is_not_sampled_but_is_used_for_grid(monkeypatch, tmp_path):
    photfile = tmp_path / 'target.phot'
    photfile.write_text(
        'photband flux flux_err\n'
        'GAIA3E_G 1.0e-12 1.0e-14\n'
    )
    setup = _setup(photfile)
    setup['fixed'] = {'feh': 0.0}

    calls = {}

    def fake_load_grids(gridnames, pnames, limits, photbands, **kwargs):
        calls['grid_pnames'] = list(pnames)
        calls['grid_limits'] = main.np.asarray(limits, dtype=float)
        return ['prepared-grid']

    def fake_mcmc(obs, obs_err, photbands, pnames, limits, grids, **kwargs):
        calls['mcmc_pnames'] = list(pnames)
        calls['mcmc_limits'] = main.np.asarray(limits, dtype=float)
        calls['fixed_variables'] = dict(kwargs['fixed_variables'])
        samples = main.np.zeros(3, dtype=[(name, 'f8') for name in pnames] + [('chi2', 'f8')])
        for name in pnames:
            samples[name] = 1.0
        return {name: 1.0 for name in samples.dtype.names}, samples

    monkeypatch.setattr(main.model, 'load_grids', fake_load_grids)
    monkeypatch.setattr(main.mcmc, 'MCMC', fake_mcmc)

    photbands, obs, obs_err = main.get_observations(setup)
    results, samples, priors, gridnames = main.fit_sed(setup, photbands, obs, obs_err)

    assert 'feh' in calls['grid_pnames']
    feh_index = calls['grid_pnames'].index('feh')
    assert list(calls['grid_limits'][feh_index]) == [0.0, 0.0]
    assert 'feh' not in calls['mcmc_pnames']
    assert calls['fixed_variables']['feh'] == 0.0
    assert results['feh'] == [0.0, 0.0, 0, 0]
    assert 'feh' not in samples.dtype.names


def test_fitted_group_jitter_is_sampled_but_not_sent_to_grid(monkeypatch, tmp_path):
    photfile = tmp_path / 'target.phot'
    photfile.write_text(
        'photband flux flux_err\n'
        'GAIA3E_G 1.0e-12 1.0e-14\n'
        'GAIA3E_BP 1.1e-12 1.2e-14\n'
        '2MASS_Ks 2.0e-13 2.0e-15\n'
    )
    setup = _setup(photfile)
    setup['fixed'] = {'feh': 0.0}
    setup['error_model'] = {
        'type': 'fitted_fraction_by_group',
        'default_limits': [0.0, 0.2],
        'default_prior': [0.03, 0.01],
    }
    calls = {}

    def fake_load_grids(gridnames, pnames, limits, photbands, **kwargs):
        calls['grid_pnames'] = list(pnames)
        return ['prepared-grid']

    def fake_mcmc(obs, obs_err, photbands, pnames, limits, grids, **kwargs):
        calls['mcmc_pnames'] = list(pnames)
        calls['mcmc_limits'] = main.np.asarray(limits, dtype=float)
        calls['priors'] = dict(kwargs['priors'])
        calls['error_model'] = kwargs['error_model']
        samples = main.np.zeros(3, dtype=[(name, 'f8') for name in pnames] + [('chi2', 'f8')])
        for name in pnames:
            samples[name] = 1.0
        return {name: 1.0 for name in samples.dtype.names}, samples

    monkeypatch.setattr(main.model, 'load_grids', fake_load_grids)
    monkeypatch.setattr(main.mcmc, 'MCMC', fake_mcmc)

    photbands, obs, obs_err = main.get_observations(setup)
    results, samples, priors, gridnames = main.fit_sed(setup, photbands, obs, obs_err)

    assert 'jitter_GAIA3E' not in calls['grid_pnames']
    assert 'jitter_2MASS' not in calls['grid_pnames']
    assert 'jitter_GAIA3E' in calls['mcmc_pnames']
    assert 'jitter_2MASS' in calls['mcmc_pnames']
    assert calls['mcmc_limits'][-2:].tolist() == [[0.0, 0.2], [0.0, 0.2]]
    assert calls['priors']['jitter_GAIA3E'] == [0.03, 0.01, 0.01]
    assert calls['error_model']['group_parameters'] == {
        'GAIA3E': 'jitter_GAIA3E',
        '2MASS': 'jitter_2MASS',
    }
    assert calls['error_model']['band_groups']['GAIA3E'] == ['GAIA3E_G', 'GAIA3E_BP']
    assert 'jitter_GAIA3E' in results
    assert 'jitter_2MASS' in samples.dtype.names


def test_jitter_switch_enables_default_group_jitter(monkeypatch, tmp_path):
    photfile = tmp_path / 'target.phot'
    photfile.write_text(
        'photband flux flux_err\n'
        'GAIA3E_G 1.0e-12 1.0e-14\n'
        '2MASS_Ks 2.0e-13 2.0e-15\n'
    )
    setup = _setup(photfile)
    setup['fixed'] = {'feh': 0.0}
    setup['jitter'] = True
    calls = {}

    def fake_load_grids(gridnames, pnames, limits, photbands, **kwargs):
        calls['grid_pnames'] = list(pnames)
        return ['prepared-grid']

    def fake_mcmc(obs, obs_err, photbands, pnames, limits, grids, **kwargs):
        calls['mcmc_pnames'] = list(pnames)
        calls['priors'] = dict(kwargs['priors'])
        calls['error_model'] = kwargs['error_model']
        samples = main.np.zeros(3, dtype=[(name, 'f8') for name in pnames] + [('chi2', 'f8')])
        for name in pnames:
            samples[name] = 1.0
        return {name: 1.0 for name in samples.dtype.names}, samples

    monkeypatch.setattr(main.model, 'load_grids', fake_load_grids)
    monkeypatch.setattr(main.mcmc, 'MCMC', fake_mcmc)

    photbands, obs, obs_err = main.get_observations(setup)
    main.fit_sed(setup, photbands, obs, obs_err)

    assert 'jitter_GAIA3E' not in calls['grid_pnames']
    assert 'jitter_GAIA3E' in calls['mcmc_pnames']
    assert calls['priors']['jitter_GAIA3E'] == [0.03, 0.03, 0.03]
    assert calls['error_model']['type'] == 'fitted_fraction_by_group'


def test_jitter_switch_false_conflicts_with_error_model(tmp_path):
    photfile = tmp_path / 'target.phot'
    photfile.write_text(
        'photband flux flux_err\n'
        'GAIA3E_G 1.0e-12 1.0e-14\n'
    )
    setup = _setup(photfile)
    setup['fixed'] = {'feh': 0.0}
    setup['jitter'] = False
    setup['error_model'] = {'type': 'fitted_fraction_by_group'}

    photbands, obs, obs_err = main.get_observations(setup)
    with pytest.raises(ValueError, match='jitter: false conflicts'):
        main.fit_sed(setup, photbands, obs, obs_err)


def test_write_results_keeps_asymmetric_uncertainties(tmp_path):
    resultfile = tmp_path / 'results.csv'
    setup = {'resultfile': str(resultfile)}
    results = {
        'teff': [6100.0, 6000.0, 100.0, 250.0],
        'feh': [0.0, 0.0, 0.0, 0.0],
    }

    main.write_results(setup, results, None, None, None, None)

    import pandas as pd
    table = pd.read_csv(resultfile)

    assert list(table.columns) == [
        'teff', 'teff_err_minus', 'teff_err_plus',
        'feh', 'feh_err_minus', 'feh_err_plus',
    ]
    assert table.loc[0, 'teff'] == 6000.0
    assert table.loc[0, 'teff_err_minus'] == 100.0
    assert table.loc[0, 'teff_err_plus'] == 250.0


def test_binary_fixed_metallicities_are_available_to_mcmc(monkeypatch, tmp_path):
    photfile = tmp_path / 'target.phot'
    photfile.write_text(
        'photband flux flux_err\n'
        'GAIA3E_G 1.0e-12 1.0e-14\n'
    )
    setup = {
        'photometryfile': str(photfile),
        'pnames': ['teff', 'logg', 'rad', 'teff2', 'logg2', 'rad2', 'distance', 'av'],
        'limits': [
            [7000, 9000],
            [3.5, 4.5],
            [1.0, 3.0],
            [4000, 6000],
            [2.0, 3.0],
            [5.0, 10.0],
            [500, 1500],
            [0.0, 0.2],
        ],
        'fixed': {'feh': 0.0, 'feh2': -0.5},
        'priors': {},
        'grids': ['ck_all', 'ck_all'],
    }
    calls = {}

    def fake_load_grids(gridnames, pnames, limits, photbands, **kwargs):
        calls['grid_pnames'] = list(pnames)
        return ['primary-grid', 'secondary-grid']

    def fake_mcmc(obs, obs_err, photbands, pnames, limits, grids, **kwargs):
        calls['fixed_variables'] = dict(kwargs['fixed_variables'])
        samples = main.np.zeros(2, dtype=[(name, 'f8') for name in pnames])
        return {name: 1.0 for name in samples.dtype.names}, samples

    monkeypatch.setattr(main.model, 'load_grids', fake_load_grids)
    monkeypatch.setattr(main.mcmc, 'MCMC', fake_mcmc)

    photbands, obs, obs_err = main.get_observations(setup)
    main.fit_sed(setup, photbands, obs, obs_err)

    assert 'feh' in calls['grid_pnames']
    assert 'feh2' in calls['grid_pnames']
    assert calls['fixed_variables'] == {'feh': 0.0, 'feh2': -0.5}


def test_three_component_blue_red_mir_blackbody_reaches_mcmc(monkeypatch, tmp_path):
    photfile = tmp_path / 'target.phot'
    photfile.write_text(
        'photband flux flux_err\n'
        'GAIA3E_BP 1.1e-12 1.0e-14\n'
        'GAIA3E_G  1.0e-12 1.0e-14\n'
        'GAIA3E_RP 9.0e-13 1.0e-14\n'
        '2MASS_Ks  4.0e-13 1.0e-14\n'
        'WISE_RSR_W3 2.0e-13 1.0e-14\n'
        'WISE_RSR_W4 3.0e-13 1.0e-14\n'
    )
    setup = {
        'photometryfile': str(photfile),
        'pnames': [
            'teff', 'logg', 'rad',
            'teff2', 'logg2', 'rad2',
            'teff3', 'rad3',
            'distance', 'av',
        ],
        'limits': [
            [12000, 18000],  # blue CK component
            [3.5, 5.0],
            [1.0, 5.0],
            [3500, 5000],    # red CK component
            [0.5, 3.0],
            [20.0, 200.0],
            [300, 1500],     # cool blackbody, mostly mid-infrared
            [10.0, 5000.0],
            [500, 1500],
            [0.0, 0.2],
        ],
        'fixed': {'feh': 0.0, 'feh2': -0.5, 'logg3': 5.0},
        'priors': {},
        'grids': ['ck_all', 'ck_all', 'blackbody'],
    }
    calls = {}

    def fake_load_grids(gridnames, pnames, limits, photbands, **kwargs):
        calls['gridnames'] = list(gridnames)
        calls['grid_pnames'] = list(pnames)
        return ['primary-grid', 'secondary-grid', 'tertiary-grid']

    def fake_mcmc(obs, obs_err, photbands, pnames, limits, grids, **kwargs):
        calls['fixed_variables'] = dict(kwargs['fixed_variables'])
        samples = main.np.zeros(2, dtype=[(name, 'f8') for name in pnames])
        return {name: 1.0 for name in samples.dtype.names}, samples

    monkeypatch.setattr(main.model, 'load_grids', fake_load_grids)
    monkeypatch.setattr(main.mcmc, 'MCMC', fake_mcmc)

    photbands, obs, obs_err = main.get_observations(setup)
    main.fit_sed(setup, photbands, obs, obs_err)

    assert calls['gridnames'] == ['ck_all', 'ck_all', 'blackbody']
    assert 'feh' in calls['grid_pnames']
    assert 'feh2' in calls['grid_pnames']
    assert 'feh3' not in calls['grid_pnames']
    assert 'logg3' in calls['grid_pnames']
    assert calls['fixed_variables'] == {'feh': 0.0, 'feh2': -0.5, 'logg3': 5.0}


def test_component_count_must_match_grids(tmp_path):
    photfile = tmp_path / 'target.phot'
    photfile.write_text(
        'photband flux flux_err\n'
        'GAIA3E_G 1.0e-12 1.0e-14\n'
    )
    setup = {
        'photometryfile': str(photfile),
        'pnames': [
            'teff', 'logg', 'rad',
            'teff2', 'logg2', 'rad2',
            'teff3', 'logg3', 'rad3',
            'distance', 'av',
        ],
        'limits': [
            [7000, 9000],
            [3.5, 4.5],
            [1.0, 3.0],
            [4000, 6000],
            [2.0, 3.0],
            [5.0, 10.0],
            [3000, 4000],
            [4.0, 5.5],
            [0.1, 1.0],
            [500, 1500],
            [0.0, 0.2],
        ],
        'fixed': {'feh': 0.0, 'feh2': -0.5, 'feh3': 0.2},
        'priors': {},
        'grids': ['ck_all', 'ck_all'],
    }

    photbands, obs, obs_err = main.get_observations(setup)
    with pytest.raises(ValueError, match='3 model component'):
        main.fit_sed(setup, photbands, obs, obs_err)


def test_fixed_parameter_cannot_also_have_a_fit_range(tmp_path):
    photfile = tmp_path / 'target.phot'
    photfile.write_text(
        'photband flux flux_err\n'
        'GAIA3E_G 1.0e-12 1.0e-14\n'
    )
    setup = _setup(photfile)
    setup['pnames'].append('feh')
    setup['limits'].append([-0.5, 0.5])
    setup['fixed'] = {'feh': 0.0}

    photbands, obs, obs_err = main.get_observations(setup)
    with pytest.raises(ValueError, match='both fitted'):
        main.fit_sed(setup, photbands, obs, obs_err)


def test_photometry_selectors_match_underscore_filter_names():
    photbands = main.np.array(['GAIA3E_G', 'GAIA3E_BP', '2MASS_Ks'])
    obs = main.np.array([1.0, 2.0, 3.0])
    err = main.np.array([0.1, 0.2, 0.3])

    selected, selected_obs, _ = main.select_photometry(
        photbands, obs, err, include=['GAIA3E'], verbose=False
    )

    assert list(selected) == ['GAIA3E_G', 'GAIA3E_BP']
    assert list(selected_obs) == [1.0, 2.0]


def test_distribution_plot_uses_requested_corner_style(monkeypatch, tmp_path):
    calls = {}

    class FakeSpine:
        def set_linewidth(self, value):
            calls.setdefault('spine_widths', []).append(value)

        def set_edgecolor(self, value):
            calls.setdefault('spine_colors', []).append(value)

    class FakeAxis:
        def __init__(self):
            self.spines = {'left': FakeSpine(), 'right': FakeSpine()}
            self.xaxis = type('FakeLabelAxis', (), {})()
            self.yaxis = type('FakeLabelAxis', (), {})()

        def tick_params(self, **kwargs):
            calls.setdefault('tick_params', []).append(kwargs)

    class FakeFigure:
        def __init__(self):
            self.axes = [FakeAxis()]

        def get_axes(self):
            return self.axes

        def subplots_adjust(self, **kwargs):
            calls['subplots_adjust'] = kwargs

    def fake_corner(corner_data, **kwargs):
        calls['corner_data_shape'] = corner_data.shape
        calls['corner_kwargs'] = kwargs
        calls['fake_fig'] = FakeFigure()
        return calls['fake_fig']

    def fake_savefig(path, **kwargs):
        calls['savefig'] = (path, kwargs)

    monkeypatch.setattr(main.corner, 'corner', fake_corner)
    monkeypatch.setattr(main.pl, 'savefig', fake_savefig)

    samples = main.np.zeros(20, dtype=[('teff', float), ('rad', float)])
    samples['teff'] = main.np.linspace(7000, 9000, 20)
    samples['rad'] = main.np.linspace(1.0, 2.0, 20)
    results = {
        'teff': [8050.0, 8000.0, 100.0, 100.0],
        'rad': [1.4, 1.5, 0.1, 0.1],
    }
    setup = {
        'grids': ['ck_all'],
        'plot2': {
            'type': 'distribution',
            'path': str(tmp_path / 'corner.png'),
            'parameters': ['teff', 'rad'],
            'labels': ['Teff', 'Radius'],
            'titles': ['Teff title', 'Radius title'],
        },
    }

    main.plot_results(
        setup, results, samples, {}, ['ck_all'],
        main.np.array([1.0]), main.np.array([0.1]), main.np.array(['GAIA3E_G'])
    )

    kwargs = calls['corner_kwargs']
    assert calls['corner_data_shape'] == (20, 2)
    assert kwargs['labels'] == ['Teff', 'Radius']
    assert kwargs['quantiles'] == [0.16, 0.5, 0.84]
    assert kwargs['titles'] == ['Teff title', 'Radius title']
    assert kwargs['truths'] == [8050.0, 1.4]
    assert kwargs['truth_color'] == 'tab:red'
    assert kwargs['title_kwargs'] == {"fontsize": 20}
    assert kwargs['label_kwargs'] == {"fontsize": 20}
    assert kwargs['smooth'] == 0.5
    assert kwargs['labelpad'] == 0.12
    assert kwargs['max_n_ticks'] == 4
    assert kwargs['use_math_text'] is True
    assert 'fig' in kwargs
    assert calls['tick_params'][0] == {
        'axis': 'both',
        'labelsize': 18,
        'direction': 'out',
        'length': 4,
        'width': 1.2,
        'pad': 7,
    }
    axis = calls['fake_fig'].get_axes()[0]
    assert axis.xaxis.labelpad == 18
    assert axis.yaxis.labelpad == 18
    assert calls['subplots_adjust'] == {
        'left': 0.10,
        'bottom': 0.10,
        'right': 0.98,
        'top': 0.98,
        'wspace': 0.08,
        'hspace': 0.08,
    }
    assert calls['spine_widths'] == [1.4, 1.4]
    assert calls['spine_colors'] == ['black', 'black']
    assert calls['savefig'][1]['dpi'] == 200
