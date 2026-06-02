import pytest

import numpy as np

from sedforge import integrate_grid, model, spectral_cache

from importlib import reload

import os
from astropy.io import fits

grid_description_ex = """
ckp00:
    filename: 'ck03_p00'
"""


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

    def _write_hdf5_rv_grid(self, path):
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
                        exponent = (
                            1.0
                            + row['teff'] / 10000.0
                            + row['logg'] / 10.0
                            + row['feh']
                            + rv / 10.0
                            + av / 10.0
                            + iband
                        )
                        flux[ispec, irv, iav, iband] = 10.0 ** exponent

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
