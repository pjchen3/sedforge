import numpy as np

from sedforge import reddening


def test_wc2019_returns_positive_angstrom_curve():
    wave, awav = reddening.wc2019(Rv=3.1, num=500)

    assert np.all(np.diff(wave) > 0)
    assert np.isclose(wave[0], 1000.0)
    assert np.isclose(wave[-1], 300000.0)
    assert np.all(np.isfinite(awav))
    assert np.all(awav > 0)


def test_wc2019_can_redden_flux_on_requested_wavelengths():
    wave = np.array([1500.0, 5500.0, 20000.0])
    flux = np.ones_like(wave)

    reddened = reddening.redden(flux, wave=wave, av=0.31,
                                law='WC2019', Rv=3.1)

    assert np.all(np.isfinite(reddened))
    assert np.all(reddened < flux)


def test_av_matches_legacy_ebv_conversion():
    wave = np.array([1500.0, 5500.0, 20000.0])
    flux = np.ones_like(wave)

    by_av = reddening.redden(flux, wave=wave, av=0.31,
                             law='WC2019', Rv=3.1)
    by_ebv = reddening.redden(flux, wave=wave, ebv=0.1,
                              law='WC2019', Rv=3.1)

    assert np.allclose(by_av, by_ebv)


def test_rvcodeg_alias_still_uses_wc2019():
    wave, awav = reddening.rvcodeg(Rv=3.1, num=100)
    wave_wc, awav_wc = reddening.wc2019(Rv=3.1, num=100)

    assert np.allclose(wave, wave_wc)
    assert np.allclose(awav, awav_wc)
