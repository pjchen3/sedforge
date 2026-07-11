import yaml

from sedforge.setup_format import setup_to_readable_yaml


def test_setup_to_readable_yaml_keeps_sections_and_round_trips():
    setup = {
        'objectname': 'target',
        'photometryfile': 'target.phot',
        'photband_exclude': [],
        'grids': ['ck_all'],
        'reddening_law': 'WC2019',
        'reddening_Rv': 3.1,
        'reddening_case1': 1,
        'pnames': ['teff', 'distance', 'av'],
        'limits': [[7000, 9000], [100, 1000], [0, 1]],
        'fixed': {'feh': 0.0},
        'priors': {'distance': [500, 25]},
        'nwalkers': 32,
        'nsteps': 100,
        'nrelax': 50,
        'a': 2,
        'percentiles': [16, 50, 84],
        'resultfile': 'target_results.csv',
        'plot1': {'type': 'sed_fit', 'path': 'target_sed.png'},
    }

    text = setup_to_readable_yaml(setup)
    loaded = yaml.safe_load(text)

    assert '# Fitted Parameters' in text
    assert '# Fixed Parameters' in text
    assert '# Priors' in text
    assert '- [7000, 9000]  # teff' in text
    assert loaded['pnames'] == setup['pnames']
    assert loaded['limits'] == setup['limits']
    assert loaded['fixed'] == setup['fixed']
    assert loaded['priors'] == setup['priors']


def test_setup_to_readable_yaml_keeps_grid_rescue_controls():
    setup = {
        'objectname': 'rescue',
        'photometryfile': 'rescue.phot',
        'grids': ['newera_alpha0_rv005'],
        'pnames': ['teff'],
        'limits': [[3000.0, 8000.0]],
        'fixed': {},
        'priors': {},
        'init_grid_rescue': True,
        'init_grid_rescue_chi2_threshold': 40.0,
        'init_grid_rescue_cache_max_gb': 2.0,
        'init_grid_rescue_maxiter': 80,
        'init_grid_rescue_popsize': 12,
    }

    loaded = yaml.safe_load(setup_to_readable_yaml(setup))

    for key in (
        'init_grid_rescue',
        'init_grid_rescue_chi2_threshold',
        'init_grid_rescue_cache_max_gb',
        'init_grid_rescue_maxiter',
        'init_grid_rescue_popsize',
    ):
        assert loaded[key] == setup[key]
