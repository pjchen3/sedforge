import os
from importlib import reload

import numpy as np
import pytest
from astropy.io import fits

from sedforge import integrate_grid, mcmc, model, spectral_cache, statfunc


def test_axis_bounds_accepts_decimal_endpoint_stored_as_float32():
    # The reader exposes float64 values even when the file stored float32.
    axis = np.array([0.0, 6.2], dtype=np.float32).astype(float)
    assert model._axis_bounds(axis, 6.2) == [(1, 1.0)]
    with pytest.raises(ValueError, match="outside grid axis"):
        model._axis_bounds(axis, 6.2001)


def test_get_itable_single_accepts_fixed_fits_grid_name(monkeypatch):
    axis_values = [
        np.array([5000.0]),
        np.array([4.0]),
        np.array([0.0]),
    ]
    pixelgrid = np.empty((1, 1, 1, 2), dtype=float)
    grid_names = np.array(['teff', 'logg', 'av'])
    monkeypatch.setattr(model, '_grid_integrated_format', lambda grid: 'fits')
    monkeypatch.setattr(
        model,
        'prepare_grid',
        lambda *args, **kwargs: (
            axis_values,
            np.array([[5000.0, 4.0, 0.0]]),
            pixelgrid,
            grid_names,
        ),
    )
    monkeypatch.setattr(
        model.interpol,
        'interpolate',
        lambda parameters, axes, pixels: np.log10(np.array([[2.0], [3.0]])),
    )

    flux, luminosity = model.get_itable_single(
        teff=5000.0,
        logg=4.0,
        av=0.0,
        grid='newera_alpha0',
        photbands=['GAIA3E_G'],
    )

    assert np.allclose(flux, [2.0])
    assert np.isclose(luminosity, 3.0)


def test_get_itable_single_broadcasts_fixed_parameters_for_vectorized_walkers(monkeypatch):
    axis_values = [
        np.array([5000.0, 5250.0]),
        np.array([4.5, 5.0]),
        np.array([0.0, 1.0]),
        np.array([-1.5, -1.0]),
    ]
    pixelgrid = np.empty((2, 2, 2, 2, 2), dtype=float)
    grid_names = np.array(['teff', 'logg', 'av', 'feh'])
    captured = {}

    def fake_interpolate(parameters, axes, pixels):
        captured['parameters'] = parameters.copy()
        npoints = parameters.shape[1]
        return np.log10(np.vstack([np.full(npoints, 2.0), np.full(npoints, 3.0)]))

    monkeypatch.setattr(model.interpol, 'interpolate', fake_interpolate)
    av = np.array([0.1, 0.2, 0.3])
    radius = np.array([1.0, 1.1, 1.2])
    distance = np.array([100.0, 110.0, 120.0])
    flux, luminosity = model.get_itable_single(
        teff=5205.0,
        logg=4.7172,
        feh=-1.313,
        av=av,
        rad=radius,
        distance=distance,
        grid=[axis_values, pixelgrid, grid_names],
    )

    assert captured['parameters'].shape == (4, 3)
    assert np.allclose(captured['parameters'][0], 5205.0)
    assert np.allclose(captured['parameters'][1], 4.7172)
    assert np.allclose(captured['parameters'][2], av)
    assert np.allclose(captured['parameters'][3], -1.313)
    assert flux.shape == (1, 3)
    assert luminosity.shape == (3,)

grid_description_ex = """
ckp00:
    filename: 'ck03_p00'
"""


def test_builtin_rv_grid_metadata():
    ck_ranges = model.get_grid_ranges(grid='ck03_rv')
    assert ck_ranges['teff'] == (3500, 50000)
    assert ck_ranges['logg'] == (0.0, 5.0)
    assert ck_ranges['rad'] == (0.05, 500.0)
    assert ck_ranges['feh'] == (-2.5, 0.5)
    assert ck_ranges['rv'] == (2.0, 5.0)
    assert model.grid_has_axis('ck03_rv', 'rv')
    assert model.grid_requires_feh_value('ck03_rv')

    newera_ranges = model.get_grid_ranges(grid='newera_alpha0_rv')
    assert newera_ranges['teff'] == (2300, 12000)
    assert newera_ranges['logg'] == (0.0, 6.0)
    assert newera_ranges['feh'] == (-2.5, 0.5)
    assert newera_ranges['rv'] == (2.0, 5.0)
    assert model.grid_has_axis('newera_alpha0_rv', 'rv')
    assert model.grid_requires_feh_value('newera_alpha0_rv')


def test_binary_components_follow_grid_order_even_when_kwargs_are_unsorted():
    axis_values = [np.array([1.0])]
    grid_names = np.array(['teff'])
    primary = [axis_values, np.array([[0.0, 0.0]]), grid_names]
    secondary = [axis_values, np.array([[1.0, 1.0]]), grid_names]

    flux, labs = model.get_itable(
        teff2=1.0,
        teff=1.0,
        grid=[primary, secondary],
    )

    assert np.allclose(flux, [11.0])
    assert np.allclose(labs['L'], [1.0])
    assert np.allclose(labs['L2'], [10.0])


def test_three_components_follow_grid_order_and_return_l3():
    axis_values = [np.array([1.0])]
    grid_names = np.array(['teff'])
    primary = [axis_values, np.array([[0.0, 0.0]]), grid_names]
    secondary = [axis_values, np.array([[1.0, 1.0]]), grid_names]
    tertiary = [axis_values, np.array([[2.0, 2.0]]), grid_names]

    flux, labs = model.get_itable(
        teff3=1.0,
        teff=1.0,
        teff2=1.0,
        grid=[primary, secondary, tertiary],
    )

    assert np.allclose(flux, [111.0])
    assert np.allclose(labs['L'], [1.0])
    assert np.allclose(labs['L2'], [10.0])
    assert np.allclose(labs['L3'], [100.0])


def test_prepared_fits_grid_hdf5_check_does_not_walk_pixel_values():
    class PixelGrid:
        shape = (2, 2)

        def __iter__(self):
            raise AssertionError('pixel values must not be traversed')

    prepared = [[np.array([1.0])], PixelGrid(), np.array(['teff'])]
    assert not model.uses_hdf5_integrated_grid(prepared)


def test_newera_and_sparse_prepared_fits_are_nonrectangular():
    assert model.grid_has_nonrectangular_coverage('newera_alpha0')

    sparse = [
        [np.array([1.0, 2.0])],
        np.array([[0.0, 0.0], [np.inf, np.inf]]),
        np.array(['teff']),
    ]
    complete = [
        [np.array([1.0, 2.0])],
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array(['teff']),
    ]

    assert model.grid_has_nonrectangular_coverage(sparse)
    assert not model.grid_has_nonrectangular_coverage(complete)


def test_component_grid_count_mismatch_raises():
    axis_values = [np.array([1.0])]
    grid_names = np.array(['teff'])
    primary = [axis_values, np.array([[0.0, 0.0]]), grid_names]
    secondary = [axis_values, np.array([[1.0, 1.0]]), grid_names]

    with pytest.raises(ValueError, match='3 model component'):
        model.get_itable(
            teff=1.0,
            teff2=1.0,
            teff3=1.0,
            grid=[primary, secondary],
        )


class TestCheckGrids:

    @pytest.fixture(scope='class')
    def make_modeldir(self, tmpdir_factory):
        org_dir = os.environ.get('SEDFORGE_MODELS')

        models_dir = tmpdir_factory.mktemp('SED_models')
        models_dir.join('grid_description.yaml').write(grid_description_ex)
        models_dir.join('ck03_p00.fits').write('test')
        models_dir.join('ick03_p00_lawWC2019_Rv3.10.fits').write('test')

        yield models_dir

        if org_dir is None:
            os.environ.pop('SEDFORGE_MODELS', None)
        else:
            os.environ['SEDFORGE_MODELS'] = org_dir
        reload(model)

    def test_check_grids__model_dir_without_dash(self, make_modeldir):
        models_dir = make_modeldir

        os.environ['SEDFORGE_MODELS'] = str(models_dir)

        reload(model)

        assert model.defaults['directory'] == str(models_dir)

        assert 'ckp00' in model.grid_description
        assert 'filename' in model.grid_description['ckp00']
        assert model.grid_description['ckp00']['filename'] == 'ck03_p00'

    def test_check_grids__model_dir_with_dash(self, make_modeldir):
        models_dir = make_modeldir

        os.environ['SEDFORGE_MODELS'] = str(models_dir) + '/'

        reload(model)

        assert model.defaults['directory'] == str(models_dir) + '/'

        assert 'ckp00' in model.grid_description
        assert 'filename' in model.grid_description['ckp00']
        assert model.grid_description['ckp00']['filename'] == 'ck03_p00'

    def test_get_grid_file_keeps_default_wc2019_case1_filename_compact(self, make_modeldir):
        models_dir = make_modeldir
        os.environ['SEDFORGE_MODELS'] = str(models_dir)
        reload(model)

        path = model.get_grid_file(integrated=True, grid='ckp00',
                                   reddening_law='WC2019',
                                   reddening_Rv=3.1,
                                   reddening_case1=1)

        assert path.endswith('ick03_p00_lawWC2019_Rv3.10.fits')

    def test_get_grid_file_labels_non_default_wc2019_case1(self, make_modeldir):
        models_dir = make_modeldir
        os.environ['SEDFORGE_MODELS'] = str(models_dir)
        reload(model)

        path = model.get_grid_file(integrated=True, grid='ckp00',
                                   reddening_law='WC2019',
                                   reddening_Rv=3.1,
                                   reddening_case1=2)

        assert path.endswith('ick03_p00_lawWC2019_case2_Rv3.10.fits')

    def test_get_grid_file_uses_integrated_subdir(self, tmpdir):
        org_dir = os.environ.get('SEDFORGE_MODELS')
        models_dir = tmpdir.mkdir('SED_models')
        models_dir.join('grid_description.yaml').write("""
ck_all:
    filename: 'ck_all'
    integrated_subdir: 'integrated'
""")

        try:
            os.environ['SEDFORGE_MODELS'] = str(models_dir)
            reload(model)

            path = model.get_grid_file(integrated=True, grid='ck_all',
                                       reddening_law='WC2019',
                                       reddening_Rv=3.1)

            assert path.endswith(
                os.path.join('integrated', 'ick_all_lawWC2019_Rv3.10.fits')
            )
        finally:
            if org_dir is None:
                os.environ.pop('SEDFORGE_MODELS', None)
            else:
                os.environ['SEDFORGE_MODELS'] = org_dir
            reload(model)

    def test_feh_axis_is_only_added_for_metallicity_grids(self, tmpdir):
        org_dir = os.environ.get('SEDFORGE_MODELS')
        models_dir = tmpdir.mkdir('SED_models')
        models_dir.join('grid_description.yaml').write("""
ckm25:
    filename: 'ck03_m25'
    feh: -2.5
ck_all:
    filename: 'ck_all'
    members:
    - grid: ckm25
      feh: -2.5
koester2:
    filename: 'koester2'
blackbody:
    filename: 'blackbody_discint'
newera_alpha0:
    filename: 'newera_alpha0'
    supports_feh: true
""")

        try:
            os.environ['SEDFORGE_MODELS'] = str(models_dir)
            reload(model)
            pnames = ['teff', 'logg', 'feh', 'rad',
                      'teff2', 'logg2', 'rad2', 'distance', 'ebv']

            assert model._variables_for_component(
                None, 'ck_all', 0, pnames, ''
            ) == ['teff', 'logg', 'ebv', 'feh']
            assert model._variables_for_component(
                None, 'koester2', 1, pnames, '2'
            ) == ['teff', 'logg', 'ebv']
            assert model._variables_for_component(
                None, 'blackbody', 1, pnames, '2'
            ) == ['teff', 'logg', 'ebv']
            assert model._variables_for_component(
                None, 'newera_alpha0', 0, pnames, ''
            ) == ['teff', 'logg', 'ebv', 'feh']
        finally:
            if org_dir is None:
                os.environ.pop('SEDFORGE_MODELS', None)
            else:
                os.environ['SEDFORGE_MODELS'] = org_dir
            reload(model)

    def test_tmap_fixed_helium_grid_uses_raw_filename_and_no_fit_axis(self, tmpdir):
        org_dir = os.environ.get('SEDFORGE_MODELS')
        models_dir = tmpdir.mkdir('SED_models')
        models_dir.join('grid_description.yaml').write("""
tmap_he050:
    raw_filename: 'tmap'
    filename: 'tmap_he050'
    he_mass: 0.5
""")

        try:
            os.environ['SEDFORGE_MODELS'] = str(models_dir)
            reload(model)

            raw = model.get_grid_file(integrated=False, grid='tmap_he050')
            integrated = model.get_grid_file(integrated=True, grid='tmap_he050',
                                             reddening_law='WC2019')

            assert raw.endswith('tmap.fits')
            assert integrated.endswith('itmap_he050_lawWC2019_Rv3.10.fits')
            assert model._variables_for_component(
                None, 'tmap_he050', 0,
                ['teff', 'logg', 'he_mass', 'rad', 'distance', 'ebv'],
                '',
            ) == ['teff', 'logg', 'ebv']
        finally:
            if org_dir is None:
                os.environ.pop('SEDFORGE_MODELS', None)
            else:
                os.environ['SEDFORGE_MODELS'] = org_dir
            reload(model)


class TestModel:

    def _write_integrated_grid(self, path, feh):
        rows = []
        for teff in (4000.0, 5000.0):
            for logg in (4.0, 4.5):
                for ebv in (0.0, 0.1):
                    flux = 1.0 + teff / 10000.0 + logg / 10.0 + ebv + feh
                    rows.append((teff, logg, ebv, flux, flux * 10.0))
        cols = [
            fits.Column(name='teff', array=[r[0] for r in rows], format='D'),
            fits.Column(name='logg', array=[r[1] for r in rows], format='D'),
            fits.Column(name='ebv', array=[r[2] for r in rows], format='D'),
            fits.Column(name='TEST.BAND', array=[r[3] for r in rows], format='D'),
            fits.Column(name='Labs', array=[r[4] for r in rows], format='D'),
        ]
        fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(path)

    def test_absolute_fits_capabilities_are_read_once(self, tmpdir, monkeypatch):
        path = str(tmpdir.join('capabilities.fits'))
        columns = [
            fits.Column(name='teff', array=[4000.0, 5000.0], format='D'),
            fits.Column(name='logg', array=[4.0, 4.5], format='D'),
            fits.Column(name='feh', array=[-0.5, 0.0], format='D'),
            fits.Column(name='av', array=[0.0, 1.0], format='D'),
            fits.Column(name='TEST.BAND', array=[1.0, 2.0], format='D'),
            fits.Column(name='Labs', array=[3.0, 4.0], format='D'),
        ]
        fits.HDUList([
            fits.PrimaryHDU(),
            fits.BinTableHDU.from_columns(columns),
        ]).writeto(path)

        model.clear_grid_cache()
        original_open = model.fits.open
        calls = []

        def counted_open(*args, **kwargs):
            calls.append(args[0])
            return original_open(*args, **kwargs)

        monkeypatch.setattr(model.fits, 'open', counted_open)
        assert model.grid_has_axis(path, 'teff')
        assert model.grid_has_axis(path, 'feh')
        assert not model.grid_has_axis(path, 'rv')
        assert model.grid_requires_feh_value(path)
        assert len(calls) == 1

    def _write_hdf5_rv_grid(self, path, drop_index=None):
        import h5py

        teffs = np.array([4000.0, 5000.0])
        loggs = np.array([1.0, 2.0])
        fehs = np.array([-0.5, 0.0])
        rvs = np.array([2.0, 4.0])
        avs = np.array([0.0, 1.0])
        bands = np.array(['B1', 'B2'], dtype=object)

        rows = []
        for feh in fehs:
            for teff in teffs:
                for logg in loggs:
                    labs = 10.0 ** (2.0 + teff / 10000.0 + logg / 10.0 + feh)
                    rows.append((teff, logg, feh, labs))
        rows = np.asarray(rows, dtype=[
            ('teff', 'f8'),
            ('logg', 'f8'),
            ('feh', 'f8'),
            ('Labs', 'f8'),
        ])

        flux = np.empty((len(rows), len(rvs), len(avs), len(bands)), dtype='f4')
        for ispec, row in enumerate(rows):
            for irv, rv in enumerate(rvs):
                for iav, av in enumerate(avs):
                    for iband in range(len(bands)):
                        # Keep the reference node unchanged while giving the
                        # second band a parameter-dependent colour.  A free
                        # normalization can otherwise fit every grid node
                        # exactly when only two bands have a fixed ratio.
                        colour_term = (
                            0.01 * (row['teff'] - 5000.0) / 1000.0
                            + 0.02 * (row['logg'] - 2.0)
                            + 0.04 * row['feh']
                            + 0.08 * (rv - 4.0) / 2.0
                            + 0.16 * (av - 1.0)
                        )
                        exponent = (
                            1.0
                            + row['teff'] / 10000.0
                            + row['logg'] / 10.0
                            + row['feh']
                            + rv / 10.0
                            + av / 10.0
                            + iband
                            + iband * colour_term
                        )
                        flux[ispec, irv, iav, iband] = 10.0 ** exponent

        if drop_index is not None:
            rows = np.delete(rows, int(drop_index))
            flux = np.delete(flux, int(drop_index), axis=0)

        with h5py.File(path, 'w') as h5:
            axes = h5.create_group('axes')
            axes.create_dataset('teff', data=teffs)
            axes.create_dataset('logg', data=loggs)
            axes.create_dataset('feh', data=fehs)
            axes.create_dataset('rv', data=rvs)
            axes.create_dataset('av', data=avs)
            axes.create_dataset('photband', data=bands, dtype=h5py.string_dtype())
            spectra = h5.create_group('spectra')
            for name in rows.dtype.names:
                spectra.create_dataset(name, data=rows[name])
            h5.create_dataset('flux', data=flux)
            h5.attrs['flux_layout'] = 'spec,rv,av,filter'

    def _write_raw_grid(self, path, teff, logg, flux_level):
        cols = [
            fits.Column(name='wavelength', array=np.array([4000.0, 5000.0, 6000.0]), format='D'),
            fits.Column(name='flux', array=np.array([flux_level, flux_level, flux_level]), format='D'),
        ]
        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.header['TEFF'] = teff
        hdu.header['LOGG'] = logg
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)

    def test_integrated_grid_can_store_feh_axis(self, tmpdir, monkeypatch):
        org_dir = os.environ.get('SEDFORGE_MODELS')
        models_dir = tmpdir.mkdir('SED_models')
        models_dir.join('grid_description.yaml').write("""
minus:
    filename: 'minus'
solar:
    filename: 'solar'
""")
        self._write_raw_grid(str(models_dir.join('minus.fits')), 4500.0, 4.0, 1.0)
        self._write_raw_grid(str(models_dir.join('solar.fits')), 4500.0, 4.0, 2.0)

        monkeypatch.setattr(integrate_grid, 'get_responses',
                            lambda responses=None, wave=(0, np.inf): ['TEST.BAND'])
        monkeypatch.setattr(
            integrate_grid,
            '_integrated_fluxes_fast',
            lambda wave, flux, avs, responses, law, Rv, case1,
            weight_cache, reddening_cache:
                np.column_stack((avs, np.ones(len(avs)) * np.mean(flux))),
        )

        try:
            os.environ['SEDFORGE_MODELS'] = str(models_dir)
            reload(model)

            outfile = str(models_dir.join('ick_all_lawcustom_Rv3.10.fits'))
            integrate_grid.calc_integrated_grid(
                threads=1,
                avs=np.array([0.0, 0.31]),
                law='custom',
                Rv=3.1,
                responses=['TEST.BAND'],
                grid={
                    'members': [
                        {'grid': 'minus', 'feh': -0.5},
                        {'grid': 'solar', 'feh': 0.0},
                    ],
                },
                outfile=outfile,
            )

            with fits.open(outfile) as hdul:
                names = set(hdul[1].data.dtype.names)
                assert {'teff', 'logg', 'feh', 'av', 'TEST.BAND'}.issubset(names)
                assert set(np.round(hdul[1].data['feh'], 3)) == {-0.5, 0.0}
                assert set(np.round(hdul[1].data['av'], 3)) == {0.0, 0.31}
        finally:
            if org_dir is None:
                os.environ.pop('SEDFORGE_MODELS', None)
            else:
                os.environ['SEDFORGE_MODELS'] = org_dir
            reload(model)

    def test_get_table_reads_uniform_spectral_cache(self, tmpdir):
        org_dir = os.environ.get('SEDFORGE_MODELS')
        models_dir = tmpdir.mkdir('SED_models')
        models_dir.join('grid_description.yaml').write("""
cached_grid:
    filename: 'cached_grid'
    spectral_cache: 'cached_grid_spectra.fits'
    supports_feh: true
""")
        wave = np.array([4000.0, 5000.0, 6000.0])
        params = {
            'teff': [5000.0, 6000.0],
            'logg': [4.0, 4.5],
            'feh': [-0.5, 0.0],
        }
        flux = np.array([
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
        ])
        spectral_cache.write_cache(
            str(models_dir.join('cached_grid_spectra.fits')),
            params,
            wave,
            flux,
        )

        try:
            os.environ['SEDFORGE_MODELS'] = str(models_dir)
            reload(model)

            assert model.raw_spectrum_available('cached_grid')
            out_wave, out_flux = model.get_table(
                grid='cached_grid',
                teff=6100.0,
                logg=4.4,
                feh=0.1,
                rad=2.0,
            )

            assert np.allclose(out_wave, wave)
            assert np.allclose(out_flux, flux[1] * 4.0)
        finally:
            if org_dir is None:
                os.environ.pop('SEDFORGE_MODELS', None)
            else:
                os.environ['SEDFORGE_MODELS'] = org_dir
            reload(model)

    def test_hdf5_rv_grid_interpolates_and_scales_fluxes(self, tmpdir):
        org_dir = os.environ.get('SEDFORGE_MODELS')
        models_dir = tmpdir.mkdir('SED_models')
        models_dir.mkdir('rv_grid')
        models_dir.join('grid_description.yaml').write("""
rv_grid:
    filename: 'rv_grid/grid'
    integrated_path: 'rv_grid/grid.h5'
    integrated_format: hdf5
    axes: [teff, logg, feh, rv, av]
    supports_feh: true
    supports_rv: true
""")
        self._write_hdf5_rv_grid(str(models_dir.join('rv_grid').join('grid.h5')))

        try:
            os.environ['SEDFORGE_MODELS'] = str(models_dir)
            reload(model)

            assert model.get_grid_file(integrated=True, grid='rv_grid').endswith(
                os.path.join('rv_grid', 'grid.h5')
            )
            assert model.grid_has_axis('rv_grid', 'rv')

            grids = model.load_grids(
                ['rv_grid'],
                ['teff', 'logg', 'feh', 'rad', 'distance', 'av', 'rv'],
                np.array([
                    [4000.0, 5000.0],
                    [1.0, 2.0],
                    [-0.5, 0.0],
                    [1.0, 2.0],
                    [10.0, 20.0],
                    [0.0, 1.0],
                    [2.0, 4.0],
                ]),
                ['B1', 'B2'],
            )
            flux, extra = model.get_itable(
                grid=grids,
                teff=5000.0,
                logg=2.0,
                feh=0.0,
                rad=2.0,
                av=1.0,
                rv=4.0,
            )

            raw_b1 = 10.0 ** (1.0 + 0.5 + 0.2 + 0.0 + 0.4 + 0.1 + 0.0)
            raw_b2 = raw_b1 * 10.0
            scale = 2.0 ** 2
            assert np.allclose(
                flux,
                np.array([raw_b1, raw_b2]) * scale,
                rtol=1e-6,
                atol=0.0,
            )

            raw_labs = 10.0 ** (2.0 + 0.5 + 0.2 + 0.0)
            assert extra['L'][0] == pytest.approx(raw_labs * 2.0 ** 2)
        finally:
            if org_dir is None:
                os.environ.pop('SEDFORGE_MODELS', None)
            else:
                os.environ['SEDFORGE_MODELS'] = org_dir
            reload(model)

    def test_hdf5_profile_seed_search_recovers_a_discrete_node(self, tmpdir):
        org_dir = os.environ.get('SEDFORGE_MODELS')
        models_dir = tmpdir.mkdir('SED_models')
        models_dir.mkdir('rv_grid')
        models_dir.join('grid_description.yaml').write("""
rv_grid:
    filename: 'rv_grid/grid'
    integrated_path: 'rv_grid/grid.h5'
    integrated_format: hdf5
    axes: [teff, logg, feh, rv, av]
    supports_feh: true
    supports_rv: true
""")
        self._write_hdf5_rv_grid(str(models_dir.join('rv_grid').join('grid.h5')))

        try:
            os.environ['SEDFORGE_MODELS'] = str(models_dir)
            reload(model)
            grid = model.prepare_hdf5_grid(
                ['B1', 'B2'],
                'rv_grid',
                variables=['teff', 'logg', 'feh', 'rv', 'av'],
                ranges={
                    'teff': (4000.0, 5000.0),
                    'logg': (1.0, 2.0),
                    'feh': (-0.5, 0.0),
                    'rv': (2.0, 4.0),
                    'av': (0.0, 1.0),
                },
                preload=False,
            )
            raw_b1 = 10.0 ** (1.0 + 0.5 + 0.2 + 0.0 + 0.4 + 0.1)
            obs = np.array([raw_b1, raw_b1 * 10.0]) * 0.04
            candidates = grid.profile_seed_candidates(
                obs,
                obs * 0.01,
                coarse_rv_points=2,
                coarse_av_points=2,
            )

            weights = 1.0 / (obs * 0.01) ** 2
            spectrum_indices = grid._seed_spectrum_indices(12000)
            rv_indices = grid._seed_axis_indices('rv')
            av_indices = grid._seed_axis_indices('av')
            brute_candidates = grid._append_profile_candidates(
                [],
                spectrum_indices,
                grid._sample_axis_indices(grid.axes['rv'], rv_indices, 2),
                grid._sample_axis_indices(grid.axes['av'], av_indices, 2, logarithmic=True),
                obs,
                weights,
                keep_count=96,
            )
            refined_indices = np.array(
                list(dict.fromkeys(item['spec_index'] for item in brute_candidates))[:24],
                dtype=int,
            )
            brute_candidates = grid._append_profile_candidates(
                brute_candidates,
                refined_indices,
                rv_indices,
                av_indices,
                obs,
                weights,
                keep_count=128,
            )
            brute_candidates.sort(key=lambda item: item['profile_chi2'])
            grid.close()

            assert grid.preload_full_active_subgrid(max_gb=1.0)
            cached_candidates = grid.profile_seed_candidates(
                obs,
                obs * 0.01,
                coarse_rv_points=2,
                coarse_av_points=2,
            )
            assert grid._handle is None

            best = candidates[0]
            brute_best = brute_candidates[0]
            cached_best = cached_candidates[0]
            assert best['teff'] == pytest.approx(5000.0)
            assert best['logg'] == pytest.approx(2.0)
            assert best['feh'] == pytest.approx(0.0)
            assert best['rv'] == pytest.approx(4.0)
            assert best['av'] == pytest.approx(1.0)
            assert best['scale'] == pytest.approx(0.04)
            assert best['profile_chi2'] == pytest.approx(0.0)
            for name in ('teff', 'logg', 'feh', 'rv', 'av', 'scale'):
                assert best[name] == pytest.approx(brute_best[name])
                assert cached_best[name] == pytest.approx(best[name], rel=1.0e-6)
            assert cached_best['profile_chi2'] == pytest.approx(
                best['profile_chi2'], abs=1.0e-6
            )
        finally:
            if org_dir is None:
                os.environ.pop('SEDFORGE_MODELS', None)
            else:
                os.environ['SEDFORGE_MODELS'] = org_dir
            reload(model)

    def test_hdf5_walker_neighborhood_cache_matches_lazy_interpolation(self, tmpdir):
        org_dir = os.environ.get('SEDFORGE_MODELS')
        models_dir = tmpdir.mkdir('SED_models')
        models_dir.mkdir('rv_grid')
        models_dir.join('grid_description.yaml').write("""
rv_grid:
    filename: 'rv_grid/grid'
    integrated_path: 'rv_grid/grid.h5'
    integrated_format: hdf5
    axes: [teff, logg, feh, rv, av]
    supports_feh: true
    supports_rv: true
""")
        self._write_hdf5_rv_grid(str(models_dir.join('rv_grid').join('grid.h5')))

        try:
            os.environ['SEDFORGE_MODELS'] = str(models_dir)
            reload(model)
            grid = model.prepare_hdf5_grid(
                ['B1', 'B2'],
                'rv_grid',
                variables=['teff', 'logg', 'feh', 'rv', 'av'],
                ranges={
                    'teff': (4000.0, 5000.0),
                    'logg': (1.0, 2.0),
                    'feh': (-0.5, 0.0),
                    'rv': (2.0, 4.0),
                    'av': (0.0, 1.0),
                },
                preload=False,
            )
            values = {
                'teff': 5000.0,
                'logg': 2.0,
                'feh': 0.0,
                'rv': 4.0,
                'av': 1.0,
            }
            lazy_flux, lazy_labs = grid.evaluate(**values)
            assert grid.preload_neighborhood(values, padding=0)
            cached_flux, cached_labs = grid.evaluate(**values)
            assert np.allclose(cached_flux, lazy_flux)
            assert np.allclose(cached_labs, lazy_labs)
            assert grid.cache_diagnostics()['cached_points'] == 1

            outside = dict(values, teff=4000.0)
            fallback_flux, fallback_labs = grid.evaluate(**outside)
            assert grid.cache_diagnostics()['fallback_points'] == 1

            assert grid.preload_neighborhoods([outside], padding=0)
            assert grid.cache_diagnostics()['cache_count'] == 2
            second_cached_flux, second_cached_labs = grid.evaluate(**outside)
            assert np.allclose(second_cached_flux, fallback_flux)
            assert np.allclose(second_cached_labs, fallback_labs)

            budget = grid.cache_diagnostics()['estimate_gb']
            third_mode = dict(values, feh=-0.5)
            assert not grid.preload_neighborhoods(
                [third_mode],
                padding=0,
                max_total_gb=budget,
            )
            assert grid.cache_diagnostics()['cache_count'] == 2

            assert grid.preload_mode_envelope(
                [values, outside],
                padding=0,
                max_gb=1.0e-6,
            )
            assert grid.cache_diagnostics()['cache_count'] == 1
            envelope_flux, envelope_labs = grid.evaluate(**outside)
            assert np.allclose(envelope_flux, fallback_flux)
            assert np.allclose(envelope_labs, fallback_labs)

            assert grid.preload_full_active_subgrid(max_gb=1.0e-6)
            assert grid.cache_diagnostics()['cache_count'] == 1
            assert grid._active_cache['is_full_active_grid']
            grid.close()

            control = model.prepare_hdf5_grid(
                ['B1', 'B2'],
                'rv_grid',
                variables=['teff', 'logg', 'feh', 'rv', 'av'],
                ranges=grid.ranges,
                preload=False,
            )
            control_flux, control_labs = control.evaluate(**outside)
            control.close()
            assert np.allclose(fallback_flux, control_flux)
            assert np.allclose(fallback_labs, control_labs)
        finally:
            if org_dir is None:
                os.environ.pop('SEDFORGE_MODELS', None)
            else:
                os.environ['SEDFORGE_MODELS'] = org_dir
            reload(model)

    def test_hdf5_sparse_cache_stores_only_existing_spectra(self, tmpdir):
        org_dir = os.environ.get('SEDFORGE_MODELS')
        models_dir = tmpdir.mkdir('SED_models')
        models_dir.mkdir('rv_grid')
        models_dir.join('grid_description.yaml').write("""
rv_grid:
    filename: 'rv_grid/grid'
    integrated_path: 'rv_grid/grid.h5'
    integrated_format: hdf5
    axes: [teff, logg, feh, rv, av]
    supports_feh: true
    supports_rv: true
""")
        self._write_hdf5_rv_grid(
            str(models_dir.join('rv_grid').join('grid.h5')),
            drop_index=0,
        )

        try:
            os.environ['SEDFORGE_MODELS'] = str(models_dir)
            reload(model)
            grid = model.prepare_hdf5_grid(
                ['B1', 'B2'],
                'rv_grid',
                variables=['teff', 'logg', 'feh', 'rv', 'av'],
                ranges={
                    'teff': (4000.0, 5000.0),
                    'logg': (1.0, 2.0),
                    'feh': (-0.5, 0.0),
                    'rv': (2.0, 4.0),
                    'av': (0.0, 1.0),
                },
                preload=False,
            )
            assert grid.preload_full_active_subgrid(max_gb=1.0)
            cache = grid._active_cache
            assert cache['log_flux'].shape[0] == 7
            assert cache['spec_row_index'].size == 8
            assert np.count_nonzero(cache['spec_row_index'] >= 0) == 7

            missing = {
                'teff': 4000.0,
                'logg': 1.0,
                'feh': -0.5,
                'rv': 2.0,
                'av': 0.0,
            }
            missing_flux, missing_labs = grid.evaluate(**missing)
            assert np.all(~np.isfinite(missing_flux))
            assert np.all(~np.isfinite(missing_labs))
            assert grid.cache_diagnostics()['invalid_cached_points'] == 1
            assert grid.cache_diagnostics()['fallback_points'] == 0

            values = {
                'teff': 5000.0,
                'logg': 2.0,
                'feh': 0.0,
                'rv': 4.0,
                'av': 1.0,
            }
            cached_flux, cached_labs = grid.evaluate(**values)
            grid.close()

            control = model.prepare_hdf5_grid(
                ['B1', 'B2'],
                'rv_grid',
                variables=['teff', 'logg', 'feh', 'rv', 'av'],
                ranges=grid.ranges,
                preload=False,
            )
            lazy_flux, lazy_labs = control.evaluate(**values)
            control.close()
            assert np.allclose(cached_flux, lazy_flux)
            assert np.allclose(cached_labs, lazy_labs)
        finally:
            if org_dir is None:
                os.environ.pop('SEDFORGE_MODELS', None)
            else:
                os.environ['SEDFORGE_MODELS'] = org_dir
            reload(model)

    def test_hdf5_shared_union_cache_attaches_to_narrower_grid(self, tmpdir):
        org_dir = os.environ.get('SEDFORGE_MODELS')
        models_dir = tmpdir.mkdir('SED_models')
        models_dir.mkdir('rv_grid')
        models_dir.join('grid_description.yaml').write("""
rv_grid:
    filename: 'rv_grid/grid'
    integrated_path: 'rv_grid/grid.h5'
    integrated_format: hdf5
    axes: [teff, logg, feh, rv, av]
    supports_feh: true
    supports_rv: true
""")
        self._write_hdf5_rv_grid(str(models_dir.join('rv_grid').join('grid.h5')))

        try:
            os.environ['SEDFORGE_MODELS'] = str(models_dir)
            reload(model)
            union = model.prepare_hdf5_grid(
                ['B1', 'B2'],
                'rv_grid',
                variables=['teff', 'logg', 'feh', 'rv', 'av'],
                ranges={
                    'teff': (4000.0, 5000.0),
                    'logg': (1.0, 2.0),
                    'feh': (-0.5, 0.0),
                    'rv': (2.0, 4.0),
                    'av': (0.0, 1.0),
                },
                preload=False,
            )
            assert union.preload_full_active_subgrid(max_gb=1.0)
            assert union.register_active_cache_as_shared()

            narrow = model.prepare_hdf5_grid(
                ['B2', 'B1'],
                'rv_grid',
                variables=['teff', 'logg', 'feh', 'rv', 'av'],
                ranges={
                    'teff': (5000.0, 5000.0),
                    'logg': (1.0, 2.0),
                    'feh': (0.0, 0.0),
                    'rv': (2.0, 4.0),
                    'av': (0.0, 1.0),
                },
                preload=False,
            )
            assert narrow._active_cache['log_flux'] is union._active_cache['log_flux']
            values = {
                'teff': 5000.0,
                'logg': 1.5,
                'feh': 0.0,
                'rv': 3.0,
                'av': 0.5,
            }
            union_flux, union_labs = union.evaluate(**values)
            narrow_flux, narrow_labs = narrow.evaluate(**values)
            assert np.allclose(narrow_flux, union_flux[::-1])
            assert np.allclose(narrow_labs, union_labs)
            assert narrow._handle is None
        finally:
            if org_dir is None:
                os.environ.pop('SEDFORGE_MODELS', None)
            else:
                os.environ['SEDFORGE_MODELS'] = org_dir
            reload(model)

    def test_hdf5_persistent_runtime_cache_is_validated_and_memory_mapped(self, tmpdir):
        org_dir = os.environ.get('SEDFORGE_MODELS')
        models_dir = tmpdir.mkdir('SED_models')
        models_dir.mkdir('rv_grid')
        runtime_dir = tmpdir.mkdir('runtime_cache')
        models_dir.join('grid_description.yaml').write("""
rv_grid:
    filename: 'rv_grid/grid'
    integrated_path: 'rv_grid/grid.h5'
    integrated_format: hdf5
    axes: [teff, logg, feh, rv, av]
    supports_feh: true
    supports_rv: true
""")
        grid_path = str(models_dir.join('rv_grid').join('grid.h5'))
        self._write_hdf5_rv_grid(grid_path)
        ranges = {
            'teff': (4000.0, 5000.0),
            'logg': (1.0, 2.0),
            'feh': (-0.5, 0.0),
            'rv': (2.0, 4.0),
            'av': (0.0, 1.0),
        }
        values = {'teff': 5000.0, 'logg': 1.5, 'feh': 0.0, 'rv': 3.0, 'av': 0.5}

        try:
            os.environ['SEDFORGE_MODELS'] = str(models_dir)
            reload(model)
            first = model.prepare_hdf5_grid(
                ['B1', 'B2'], 'rv_grid',
                variables=['teff', 'logg', 'feh', 'rv', 'av'],
                ranges=ranges, preload=False,
                runtime_cache_dir=str(runtime_dir),
            )
            assert first.preload_full_active_subgrid(max_gb=1.0)
            expected_flux, expected_labs = first.evaluate(**values)
            assert not first.cache_diagnostics()['runtime_cache_hit']

            second = model.prepare_hdf5_grid(
                ['B1', 'B2'], 'rv_grid',
                variables=['teff', 'logg', 'feh', 'rv', 'av'],
                ranges=ranges, preload=False,
                runtime_cache_dir=str(runtime_dir),
            )
            assert second.preload_full_active_subgrid(max_gb=1.0)
            assert second.cache_diagnostics()['runtime_cache_hit']
            assert isinstance(second._active_cache['log_flux'], np.memmap)
            actual_flux, actual_labs = second.evaluate(**values)
            assert np.allclose(actual_flux, expected_flux)
            assert np.allclose(actual_labs, expected_labs)

            cache_path = second.cache_diagnostics()['runtime_cache_path']
            os.remove(os.path.join(cache_path, 'log_labs.npy'))
            rebuilt = model.prepare_hdf5_grid(
                ['B1', 'B2'], 'rv_grid',
                variables=['teff', 'logg', 'feh', 'rv', 'av'],
                ranges=ranges, preload=False,
                runtime_cache_dir=str(runtime_dir),
            )
            assert rebuilt.preload_full_active_subgrid(max_gb=1.0)
            assert not rebuilt.cache_diagnostics()['runtime_cache_hit']
            assert os.path.isfile(os.path.join(cache_path, 'log_labs.npy'))

            stat = os.stat(grid_path)
            os.utime(grid_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            third = model.prepare_hdf5_grid(
                ['B1', 'B2'], 'rv_grid',
                variables=['teff', 'logg', 'feh', 'rv', 'av'],
                ranges=ranges, preload=False,
                runtime_cache_dir=str(runtime_dir),
            )
            assert third.preload_full_active_subgrid(max_gb=1.0)
            assert not third.cache_diagnostics()['runtime_cache_hit']
        finally:
            if org_dir is None:
                os.environ.pop('SEDFORGE_MODELS', None)
            else:
                os.environ['SEDFORGE_MODELS'] = org_dir
            reload(model)

    def test_hdf5_grid_initializer_draws_only_high_posterior_walkers(self, tmpdir):
        org_dir = os.environ.get('SEDFORGE_MODELS')
        models_dir = tmpdir.mkdir('SED_models')
        models_dir.mkdir('rv_grid')
        models_dir.join('grid_description.yaml').write("""
rv_grid:
    filename: 'rv_grid/grid'
    integrated_path: 'rv_grid/grid.h5'
    integrated_format: hdf5
    axes: [teff, logg, feh, rv, av]
    supports_feh: true
    supports_rv: true
""")
        self._write_hdf5_rv_grid(str(models_dir.join('rv_grid').join('grid.h5')))

        try:
            os.environ['SEDFORGE_MODELS'] = str(models_dir)
            reload(model)
            grid = model.prepare_hdf5_grid(
                ['B1', 'B2'],
                'rv_grid',
                variables=['teff', 'logg', 'feh', 'rv', 'av'],
                ranges={
                    'teff': (4000.0, 5000.0),
                    'logg': (1.0, 2.0),
                    'feh': (-0.5, 0.0),
                    'rv': (2.0, 4.0),
                    'av': (0.0, 1.0),
                },
                preload=False,
            )
            distance = 10.0
            radius = 2.0
            scale = (radius / (distance * model.PC_TO_RSOL)) ** 2
            raw_b1 = 10.0 ** (1.0 + 0.5 + 0.2 + 0.0 + 0.4 + 0.1)
            obs = np.array([raw_b1, raw_b1 * 10.0]) * scale
            obs_err = obs * 0.01
            pnames = ['teff', 'logg', 'feh', 'rad', 'distance', 'av', 'rv']
            limits = np.array([
                [4000.0, 5000.0],
                [1.0, 2.0],
                [-0.5, 0.0],
                [1.0, 3.0],
                [5.0, 20.0],
                [0.0, 1.0],
                [2.0, 4.0],
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
                nwalkers=12,
                coarse_rv_points=2,
                coarse_av_points=2,
                spread=1.0e-5,
            )
            logprobs = [mcmc.lnprob(theta, obs, obs_err, limits, **kwargs)[0] for theta in positions]

            assert positions.shape == (12, 7)
            assert np.all(np.isfinite(logprobs))
            assert np.min(logprobs) >= np.max(logprobs) - 25.0
            grid.close()
        finally:
            if org_dir is None:
                os.environ.pop('SEDFORGE_MODELS', None)
            else:
                os.environ['SEDFORGE_MODELS'] = org_dir
            reload(model)

    def test_metallicity_stack_and_distance_parameter(self, tmpdir):
        org_dir = os.environ.get('SEDFORGE_MODELS')
        models_dir = tmpdir.mkdir('SED_models')
        models_dir.join('grid_description.yaml').write("""
minus:
    filename: 'minus'
solar:
    filename: 'solar'
""")
        self._write_integrated_grid(
            str(models_dir.join('iminus_lawcustom_Rv3.10.fits')), -0.5
        )
        self._write_integrated_grid(
            str(models_dir.join('isolar_lawcustom_Rv3.10.fits')), 0.0
        )

        try:
            os.environ['SEDFORGE_MODELS'] = str(models_dir)
            reload(model)
            grid_spec = {
                'members': [
                    {'grid': 'minus', 'feh': -0.5},
                    {'grid': 'solar', 'feh': 0.0},
                ]
            }
            axis_values, grid_pars, pixelgrid, grid_names = model.prepare_grid(
                ['TEST.BAND'], grid_spec,
                variables=['teff', 'logg', 'ebv', 'feh'],
                ranges={
                    'teff': (4000.0, 5000.0),
                    'logg': (4.0, 4.5),
                    'ebv': (0.0, 0.1),
                    'feh': (-0.5, 0.0),
                },
                reddening_law='custom',
                reddening_Rv=3.1,
            )
            grid = [axis_values, pixelgrid, grid_names]

            flux, labs = model.get_itable_single(
                teff=4500.0, logg=4.25, ebv=0.05, feh=-0.25, grid=grid
            )
            flux_d, labs_d = model.get_itable_single(
                teff=4500.0, logg=4.25, ebv=0.05, feh=-0.25,
                distance=10.0, grid=grid
            )

            assert np.isfinite(flux[0])
            assert flux_d[0] == pytest.approx(flux[0] / (10.0 * model.PC_TO_RSOL) ** 2)
            assert labs_d[0] == pytest.approx(labs[0])
        finally:
            if org_dir is None:
                os.environ.pop('SEDFORGE_MODELS', None)
            else:
                os.environ['SEDFORGE_MODELS'] = org_dir
            reload(model)

    def test_fits_runtime_cache_and_shared_union_preserve_interpolation(self, tmpdir):
        org_dir = os.environ.get('SEDFORGE_MODELS')
        models_dir = tmpdir.mkdir('SED_models')
        runtime_dir = tmpdir.mkdir('runtime_cache')
        models_dir.join('grid_description.yaml').write("""
minus:
    filename: 'minus'
solar:
    filename: 'solar'
""")
        self._write_integrated_grid(
            str(models_dir.join('iminus_lawcustom_Rv3.10.fits')), -0.5
        )
        self._write_integrated_grid(
            str(models_dir.join('isolar_lawcustom_Rv3.10.fits')), 0.0
        )
        grid_spec = {
            'members': [
                {'grid': 'minus', 'feh': -0.5},
                {'grid': 'solar', 'feh': 0.0},
            ]
        }
        variables = ['teff', 'logg', 'ebv', 'feh']
        ranges = {
            'teff': (4000.0, 5000.0),
            'logg': (4.0, 4.5),
            'ebv': (0.0, 0.1),
            'feh': (-0.5, 0.0),
        }

        try:
            os.environ['SEDFORGE_MODELS'] = str(models_dir)
            reload(model)
            axes, _pars, pixelgrid, names = model.prepare_grid(
                ['TEST.BAND'], grid_spec,
                variables=variables, ranges=ranges,
                reddening_law='custom', reddening_Rv=3.1,
                runtime_cache_dir=str(runtime_dir),
            )
            control = [axes, pixelgrid, names]
            control_flux, control_labs = model.get_itable_single(
                teff=4500.0, logg=4.25, ebv=0.05, feh=-0.25,
                grid=control,
            )

            axes2, pars2, pixelgrid2, names2 = model.prepare_grid(
                ['TEST.BAND'], grid_spec,
                variables=variables, ranges=ranges,
                reddening_law='custom', reddening_Rv=3.1,
                runtime_cache_dir=str(runtime_dir),
            )
            assert pars2 is None
            assert isinstance(pixelgrid2, np.memmap)
            assert np.allclose(pixelgrid2, pixelgrid)

            assert model.register_shared_fits_grid(
                grid_spec, variables, ['TEST.BAND'], axes2, pixelgrid2, names2,
                reddening_law='custom', reddening_rv=3.1,
            )
            shared = model._shared_fits_grid(
                grid_spec, variables,
                {'teff': (4400.0, 4600.0), 'logg': (4.1, 4.4),
                 'ebv': (0.02, 0.08), 'feh': (-0.4, -0.1)},
                ['TEST.BAND'],
                reddening_law='custom', reddening_rv=3.1,
            )
            assert shared is not None
            assert shared[1] is pixelgrid2
            shared_flux, shared_labs = model.get_itable_single(
                teff=4500.0, logg=4.25, ebv=0.05, feh=-0.25,
                grid=shared,
            )
            assert np.allclose(shared_flux, control_flux)
            assert np.allclose(shared_labs, control_labs)
        finally:
            if org_dir is None:
                os.environ.pop('SEDFORGE_MODELS', None)
            else:
                os.environ['SEDFORGE_MODELS'] = org_dir
            reload(model)
