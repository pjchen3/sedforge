import numpy as np
import pytest

from sedforge import filters, integrate_grid
from sedforge._compat import trapezoid


def test_list_response_uses_local_transmission_curves():
    responses = filters.list_response('GAIA3E')

    assert set(responses) == {'GAIA3E_BP', 'GAIA3E_G', 'GAIA3E_RP'}


@pytest.mark.parametrize(
    'photband',
    ['GAIA3E_G', 'GAIA3E.G', '2MASS_Ks', '2MASS.KS', 'WISE_RSR_W1', 'WISE.W1'],
)
def test_get_response_accepts_current_names_and_legacy_aliases(photband):
    wave, trans = filters.get_response(photband)

    assert wave.ndim == 1
    assert trans.ndim == 1
    assert len(wave) == len(trans)
    assert len(wave) > 2
    assert np.all(np.diff(wave) > 0)
    assert np.all(np.isfinite(wave))
    assert np.all(np.isfinite(trans))
    assert np.nanmax(trans) > 0


def test_filter_info_eff_wave_matches_vega_weighted_response_curve():
    wave, trans = filters.get_response('GAIA3E_G')
    vega_wave, vega_flux = filters._load_vega()
    vega = np.interp(wave, vega_wave, vega_flux)
    weight = vega * trans * wave
    expected = trapezoid(wave * weight, x=wave) / trapezoid(weight, x=wave)
    info = filters._load_filter_info()
    row = info[info['photband'] == 'GAIA3E_G'][0]

    assert np.isclose(row['eff_wave'], expected)


def test_eff_wave_and_bandwidth_prefer_filter_info(monkeypatch):
    info = filters._load_filter_info()
    row = info[info['photband'] == 'GAIA3E_G'][0]

    def fail_if_called(*args, **kwargs):
        raise AssertionError('filter_info.dat should be used before recalculating')

    monkeypatch.setattr(filters, '_vega_effective_wave', fail_if_called)
    monkeypatch.setattr(filters, '_response_bandwidth', fail_if_called)

    assert filters.eff_wave('GAIA3E_G') == pytest.approx(row['eff_wave'])
    assert filters.bandwidth('GAIA3E_G') == pytest.approx(row['bandwidth'])


def test_bandwidth_and_get_info_use_filter_metadata():
    info = filters.get_info(['GAIA3E_G', '2MASS_Ks'])

    assert list(info['photband']) == ['GAIA3E_G', '2MASS_Ks']
    assert np.all(np.isfinite(info['eff_wave']))
    assert np.all(np.isfinite(info['bandwidth']))
    assert np.all(info['bandwidth'] > 0)
    assert np.isclose(info['bandwidth'][0], filters.bandwidth('GAIA3E_G'))


def test_synthetic_flux_with_new_filter_name():
    wave = np.linspace(2500.0, 12000.0, 5000)
    flux = np.ones_like(wave) * 2.5

    synflux = filters.synthetic_flux(wave, flux, ['GAIA3E_G'])

    assert np.all(np.isfinite(synflux))
    assert np.isclose(synflux[0], 2.5)


@pytest.mark.parametrize('photband', ['SPITZER_IRAC_36', 'WISE_RSR_W1'])
def test_synthetic_flux_uses_energy_weight_for_energy_responses(photband):
    wave = np.linspace(28000.0, 43000.0, 5000)
    if photband == 'WISE_RSR_W1':
        wave = np.linspace(25000.0, 65000.0, 8000)
    flux = (wave / 35000.0) ** -2
    waver, transr = filters.get_response(photband)
    flux_i = np.interp(waver, wave, flux)
    expected = trapezoid(flux_i * transr, x=waver) / trapezoid(transr, x=waver)

    synflux = filters.synthetic_flux(wave, flux, [photband])

    assert filters.response_type(photband) == 'energy'
    assert np.all(np.isfinite(synflux))
    assert np.isclose(synflux[0], expected)


def test_explicit_hst_acs_response_selection_is_kept():
    responses = integrate_grid.get_responses(
        ['HST_ACS_WFC'], wave=np.array([1000.0, 30000.0])
    )

    assert 'HST_ACS_WFC_F435W' in responses
    assert 'HST_ACS_WFC_F814W' in responses


def test_fast_integrated_fluxes_match_synthetic_flux_loop():
    wave = np.linspace(2500.0, 12000.0, 5000)
    flux = (wave / 5500.0) ** -1.3
    avs = np.array([0.0, 0.31, 2.17, 6.2])
    photbands = ['GAIA3E_G', 'HST_WFC3_F814W', '2MASS_J']

    slow = []
    for av in avs:
        reddened = integrate_grid.reddening.redden(
            flux, wave=wave, av=av, law='WC2019', Rv=3.1, case1=1
        )
        slow.append(np.r_[av, filters.synthetic_flux(wave, reddened, photbands)])
    slow = np.vstack(slow)

    fast = integrate_grid._integrated_fluxes_fast(
        wave, flux, avs, photbands, 'WC2019', 3.1, 1, {}, {}
    )

    assert np.allclose(fast, slow, rtol=1e-12, atol=1e-12)
