import pytest

import numpy as np

from sedforge import mcmc, statfunc


class TestMCMC:

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
