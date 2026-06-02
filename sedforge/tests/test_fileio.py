import numpy as np

from sedforge import fileio

default = """
# photometry file. Required columns: photband mag mag_err system
photometryfile: path/to/file.dat
# parameters to fit and the limits on them in same order as parameters
pnames: [teff, logg, rad, teff2, logg2, rad2, av]
limits:
- [3500, 6000]
- [3.5, 5.0]
- [0.7, 1.5]
- [20000, 40000]
- [4.5, 6.5]
- [0.05, 0.3]
- [0, 0.02]
# Gaussian priors on sampled parameters
priors:
  distance: [600, 50] # in parsec
# path to the model grids with integrated photometry
grids:
- path/to/grid/1.fits
- path/to/grid/2.fits
# setup for the MCMC algorithm
nwalkers: 100    # total number of walkers
nsteps: 2000     # steps taken by each walker (not including burn-in)
nrelax: 500      # burn-in steps taken by each walker
a: 10            # relative size of the steps taken
# output options
datafile: none   # filepath to write results of all walkers
"""


class TestFileIO:

    def test_write2fits(self, tmp_path):

        data = np.array([(1, 2), (2, 3)], dtype=[('a', 'f8'), ('b', 'f8')])
        filename = tmp_path / 'testfile.fits'

        fileio.write2fits(data, filename, setup=default)

        samples, setup = fileio.read_fits(filename)

        assert setup == default

        for name in data.dtype.names:
            for v1, v2 in zip(data[name], samples[name]):
                assert v1 == v2

    def test_write2fits_accepts_setup_dict(self, tmp_path):
        data = np.array([(1, 2), (2, 3)], dtype=[('a', 'f8'), ('b', 'f8')])
        setup = {'photometryfile': 'target.phot', 'pnames': ['teff', 'distance']}
        filename = tmp_path / 'testfile_dict_setup.fits'

        fileio.write2fits(data, filename, setup=setup)
        samples, setup_text = fileio.read_fits(filename)

        assert 'photometryfile: target.phot' in setup_text
        for name in data.dtype.names:
            for v1, v2 in zip(data[name], samples[name]):
                assert v1 == v2
