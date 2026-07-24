import pytest

import numpy as np

from sedforge import mcmc, statfunc


class TestMCMC:

    def test_grid_list_keeps_prepared_fits_grid_as_one_component(self):
        grid = [
            [np.array([3000.0, 4000.0])],
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array(['teff']),
        ]

        direct = mcmc._grid_list(grid)
        wrapped = mcmc._grid_list([grid])
        assert len(direct) == 1 and direct[0] is grid
        assert len(wrapped) == 1 and wrapped[0] is grid

    def test_vectorized_log_probability_reuses_derived_values_when_all_valid(self):
        calls = []

        def properties(**pars):
            calls.append(len(np.atleast_1d(pars['teff'])))
            return {'L': np.asarray(pars['teff'], dtype=float)}

        def model_func(**pars):
            values = np.atleast_1d(np.asarray(pars['teff'], dtype=float))
            return values[None, :], {'L': values}

        theta = np.array([[2.0], [3.0], [4.0]])
        results = mcmc.lnprob_vectorized(
            theta,
            np.array([3.0]),
            np.array([0.1]),
            np.array([[1.0, 5.0]]),
            pnames=['teff'],
            grid=[],
            fixed_variables={},
            priors={},
            photbands=['B1'],
            error_model={'type': 'none'},
            prop_func=properties,
            model_func=model_func,
        )

        assert len(results) == 3
        assert calls == [3]

    def test_vectorized_log_probability_matches_scalar_with_jitter(self):
        def model_func(**pars):
            scalar = np.asarray(pars['teff']).ndim == 0
            teff = np.atleast_1d(np.asarray(pars['teff'], dtype=float))
            rad = np.atleast_1d(np.asarray(pars['rad'], dtype=float))
            distance = np.atleast_1d(np.asarray(pars['distance'], dtype=float))
            scale = (rad / distance) ** 2
            flux = np.vstack((teff * scale, 2.0 * teff * scale))
            luminosity = teff * rad ** 2
            if scalar:
                return flux[:, 0], {'L': luminosity[:1]}
            return flux, {'L': luminosity}

        theta = np.array([
            [5.0, 2.0, 10.0, 0.01],
            [7.0, 1.5, 12.0, 0.03],
            [20.0, 2.0, 10.0, 0.02],
        ])
        limits = np.array([
            [1.0, 10.0],
            [0.5, 3.0],
            [5.0, 20.0],
            [0.0, 0.1],
        ])
        kwargs = {
            'pnames': ['teff', 'rad', 'distance', 'jitter_G'],
            'grid': [],
            'fixed_variables': {},
            'priors': {'distance': (11.0, 1.0, 2.0)},
            'photbands': ['B1', 'B2'],
            'error_model': {
                'type': 'fitted_fraction_by_group',
                'band_groups': {'G': [0, 1]},
                'group_parameters': {'G': 'jitter_G'},
                'fixed_fractions': {},
                'include_log_norm': True,
            },
            'prop_func': statfunc.get_derived_properties,
            'model_func': model_func,
        }
        obs = np.array([0.2, 0.4])
        obs_err = np.array([0.01, 0.02])

        vectorized = mcmc.lnprob_vectorized(theta, obs, obs_err, limits, **kwargs)
        scalar = [mcmc.lnprob(row, obs, obs_err, limits, **kwargs) for row in theta]

        for (vector_logp, vector_blob), (scalar_logp, scalar_blob) in zip(vectorized, scalar):
            assert vector_logp == pytest.approx(scalar_logp)
            for name in set(vector_blob) & set(scalar_blob):
                assert vector_blob[name] == pytest.approx(np.asarray(scalar_blob[name]).item())

        no_jitter_kwargs = dict(kwargs, error_model={'type': 'none'})
        vectorized = mcmc.lnprob_vectorized(
            theta, obs, obs_err, limits, **no_jitter_kwargs
        )
        scalar = [
            mcmc.lnprob(row, obs, obs_err, limits, **no_jitter_kwargs)
            for row in theta
        ]
        for (vector_logp, vector_blob), (scalar_logp, scalar_blob) in zip(vectorized, scalar):
            assert vector_logp == pytest.approx(scalar_logp)
            if np.isfinite(vector_logp):
                assert vector_blob['chi2'] == pytest.approx(scalar_blob['chi2'])

    def test_vectorized_log_probability_matches_prepared_fits_grid(self):
        grid = [[np.array([4000.0, 5000.0])], np.array([
            [0.0, 1.0, 2.0],
            [0.1, 1.1, 2.1],
        ]), np.array(['teff'])]
        theta = np.array([
            [4200.0, 1.2, 10.0],
            [4700.0, 2.0, 15.0],
            [5100.0, 1.0, 12.0],
        ])
        limits = np.array([[4000.0, 5000.0], [0.5, 3.0], [5.0, 20.0]])
        kwargs = {
            'pnames': ['teff', 'rad', 'distance'],
            'grid': grid,
            'fixed_variables': {},
            'priors': {'distance': (12.0, 2.0, 3.0)},
            'photbands': ['B1', 'B2'],
            'error_model': {'type': 'none'},
            'prop_func': statfunc.get_derived_properties,
        }
        obs = np.array([1.0e-2, 1.0e-1])
        obs_err = obs * 0.05
        vectorized = mcmc.lnprob_vectorized(theta, obs, obs_err, limits, **kwargs)
        scalar = [mcmc.lnprob(row, obs, obs_err, limits, **kwargs) for row in theta]

        for (vector_logp, vector_blob), (scalar_logp, scalar_blob) in zip(vectorized, scalar):
            assert vector_logp == pytest.approx(scalar_logp)
            if np.isfinite(vector_logp):
                assert vector_blob['chi2'] == pytest.approx(scalar_blob['chi2'])

    def test_grid_initializer_rescues_implausibly_bad_seed(self, monkeypatch):
        scale = (2.0 / (10.0 * mcmc.model.PC_TO_RSOL)) ** 2

        class FakeHDF5Grid:
            def profile_seed_candidates(self, *args, **kwargs):
                return [{'teff': 1.0, 'scale': scale, 'profile_chi2': 4900.0}]

            def preload_full_active_subgrid(self, max_gb):
                return True

            def profile_continuous_seed_candidate(self, *args, **kwargs):
                return {'teff': 8.0, 'scale': scale, 'profile_chi2': 0.0}

            def close(self):
                return None

        monkeypatch.setattr(mcmc.model, 'HDF5IntegratedGrid', FakeHDF5Grid)

        def model_func(**kwargs):
            return np.array([kwargs['teff']], dtype=float), {}

        grid = FakeHDF5Grid()
        obs = np.array([8.0])
        obs_err = np.array([0.1])
        pnames = ['teff', 'rad', 'distance']
        limits = np.array([
            [0.0, 10.0],
            [1.0, 3.0],
            [5.0, 20.0],
        ])
        kwargs = {
            'pnames': pnames,
            'grid': [grid],
            'fixed_variables': {},
            'priors': {'distance': (10.0, 0.5, 0.5)},
            'photbands': ['B1'],
            'error_model': {'type': 'none'},
            'prop_func': statfunc.get_derived_properties,
            'model_func': model_func,
        }

        positions = mcmc._grid_positions(
            obs,
            obs_err,
            limits,
            kwargs,
            nwalkers=6,
            maximum_modes=1,
            spread=1.0e-6,
        )

        assert positions.shape == (6, 3)
        assert np.allclose(positions[:, 0], 8.0, atol=1.0e-3)

    def test_grid_initializer_profiles_sparse_prepared_fits_grid(self):
        axes = [
            np.array([4000.0, 6000.0]),
            np.array([0.0, 1.0]),
        ]
        flux = np.array([
            [[1.0, 1.0, 1.0], [0.5, 0.8, 1.0]],
            [[1.0, 4.0, 2.0], [0.5, 3.2, 2.0]],
        ])
        grid = [
            axes,
            np.log10(flux),
            np.array(['teff', 'av']),
            {'non_rectangular': True},
        ]
        scale = (2.0 / (10.0 * mcmc.model.PC_TO_RSOL)) ** 2
        obs = np.array([1.0, 4.0]) * scale
        obs_err = obs * 0.01
        pnames = ['teff', 'rad', 'distance', 'av']
        limits = np.array([
            [4000.0, 6000.0],
            [1.0, 3.0],
            [5.0, 20.0],
            [0.0, 1.0],
        ])
        kwargs = {
            'pnames': pnames,
            'grid': [grid],
            'fixed_variables': {},
            'priors': {'distance': (10.0, 0.5, 0.5)},
            'photbands': ['B1', 'B2'],
            'error_model': {'type': 'none'},
            'prop_func': statfunc.get_derived_properties,
        }

        positions = mcmc._grid_positions(
            obs,
            obs_err,
            limits,
            kwargs,
            nwalkers=8,
            maximum_modes=1,
            spread=1.0e-8,
        )

        assert positions.shape == (8, 4)
        assert np.allclose(positions[:, 0], 6000.0, atol=1.0e-2)
        assert np.allclose(positions[:, 3], 0.0, atol=1.0e-6)
        assert all(np.isfinite(
            mcmc.lnprob(row, obs, obs_err, limits, **kwargs)[0]
        ) for row in positions)

    def test_representative_cache_positions_keep_separated_high_posterior_modes(self):
        positions = np.array([
            [0.10, 0.10],
            [0.11, 0.10],
            [0.80, 0.80],
            [0.79, 0.80],
        ])
        representatives = mcmc._representative_cache_positions(
            positions,
            np.array([10.0, 9.0, 8.0, 7.0]),
            np.array([[0.0, 1.0], [0.0, 1.0]]),
            maximum=2,
            min_separation=0.05,
        )

        assert representatives.shape == (2, 2)
        assert np.allclose(representatives[0], positions[0])
        assert np.allclose(representatives[1], positions[2])

    def test_chain_diagnostics_passes_well_mixed_chain(self):
        rng = np.random.default_rng(123)
        chain = rng.normal(size=(200, 12, 2))
        log_prob = rng.normal(size=(200, 12))

        diagnostics = mcmc.chain_diagnostics(
            chain,
            log_prob,
            ['teff', 'av'],
            np.full(12, 0.3),
        )

        assert diagnostics['passed']
        assert diagnostics['status'] == 'passed'
        assert diagnostics['max_split_rhat'] < 1.05
        assert diagnostics['min_bulk_ess'] > 100
        assert diagnostics['min_tail_ess'] > 100
        assert diagnostics['sampling_quality']['passed']

    def test_chain_diagnostics_flags_trapped_walker(self):
        rng = np.random.default_rng(456)
        chain = rng.normal(size=(200, 12, 2))
        chain[:, -1, :] = np.array([8.0, -5.0])
        log_prob = rng.normal(size=(200, 12))

        diagnostics = mcmc.chain_diagnostics(
            chain,
            log_prob,
            ['teff', 'av'],
            np.r_[np.full(11, 0.3), 0.0],
        )

        assert not diagnostics['passed']
        assert diagnostics['max_split_rhat'] > 1.05

    def test_chain_diagnostics_separates_low_ess_from_identifiability(self):
        rng = np.random.default_rng(789)
        innovations = rng.normal(size=(500, 12, 1))
        chain = np.zeros_like(innovations)
        for step in range(1, len(chain)):
            chain[step] = 0.95 * chain[step - 1] + innovations[step]
        diagnostics = mcmc.chain_diagnostics(
            chain,
            rng.normal(size=(500, 12)),
            ['teff'],
            np.full(12, 0.3),
            rhat_threshold=2.0,
            min_bulk_ess=500.0,
            min_tail_ess=500.0,
            limits=np.array([[-20.0, 20.0]]),
        )

        assert not diagnostics['passed']
        assert diagnostics['status'] == 'low_effective_sample_size'
        assert diagnostics['min_bulk_ess'] < 500
        identifiability = diagnostics['identifiability']['teff']
        assert 0 < identifiability['posterior_90_to_limit_width'] < 1

    def test_map_initialization_rejects_prepared_hdf5_grid(self, monkeypatch):
        monkeypatch.setattr(
            mcmc.model,
            'grid_has_nonrectangular_coverage',
            lambda grids: True,
        )

        with pytest.raises(ValueError, match='MAP initialization.*HDF5'):
            mcmc.MCMC(
                np.array([1.0]),
                np.array([0.1]),
                ['GAIA3E_G'],
                ['teff'],
                np.array([[3000.0, 8000.0]]),
                ['prepared-hdf5-grid'],
                init_method='map',
            )

    def test_map_initialization_rejects_sparse_prepared_fits_grid(self):
        grid = [
            [np.array([3000.0, 4000.0])],
            np.array([[0.0, 0.0], [np.inf, np.inf]]),
            np.array(['teff']),
        ]

        with pytest.raises(ValueError, match='non-rectangular'):
            mcmc.MCMC(
                np.array([1.0]),
                np.array([0.1]),
                ['GAIA3E_G'],
                ['teff'],
                np.array([[3000.0, 4000.0]]),
                [grid],
                init_method='map',
            )

    # def setUp(self):

    # self.evolution_model = 'mist'
    # self.variables = ['log_L', 'log_Teff', 'log_g', 'M_H']
    # self.limits = [(0.2, 1.1), (-1.0, 0.25), (5.0, 9.0)]
    # self.obs = np.array([-0.55, 3.67, 4.50, -0.35])
    # self.obs_err = np.array([0.15, 0.05, 0.50, 0.40])

    def test_prior(self):
        theta = (26000, 5.8, 0.1429, 5771, 4.438, 1, 0.02)
        pnames = ('teff', 'logg', 'rad', 'teff2', 'logg2', 'rad2', 'av')

        limits = np.array([(20000, 40000), (5.0, 6.5), (0.05, 0.25),
                           (4000, 8000), (4.0, 5.0), (0.50, 2.10),
                           (0.00, 0.05)])

        derived_properties = statfunc.get_derived_properties(**dict(zip(pnames, theta)))

        prior = mcmc.lnprior(theta, derived_properties, limits, pnames=pnames)

        assert prior == 0, "theta within limits, expected priod = 0, was {}".format(prior)

        theta = (15000, 5.8, 0.1429, 5771, 4.438, 1, 0.02)
        derived_properties = statfunc.get_derived_properties(**dict(zip(pnames, theta)))

        prior = mcmc.lnprior(theta, derived_properties, limits, pnames=pnames)

        assert prior == -np.inf, "theta out of limits, expected prior = -inf, was {}".format(prior)

        theta = (26000, 5.8, 0.1429, 5771, 4.438, 1, 0.02)
        derived_properties = statfunc.get_derived_properties(**dict(zip(pnames, theta)))

        prior = mcmc.lnprior(theta, derived_properties, limits,
                             pnames=pnames, priors={'av': (0.01, 0.01)})

        assert prior == pytest.approx(-0.5)
