import numpy as np
from astropy.table import Table

from sedforge import catalog_photometry


def test_mag_to_flux_uses_band_zero_point(monkeypatch):
    monkeypatch.setattr(catalog_photometry, "zero_point_flux", lambda photband, system: 100.0)

    flux, flux_err = catalog_photometry.mag_to_flux("TEST_BAND", 2.5, 0.1, "ab")

    assert np.isclose(flux, 10.0)
    assert np.isclose(flux_err, 0.4 * np.log(10.0) * flux * 0.1)


def test_default_magnitude_systems_for_common_filters():
    assert catalog_photometry.default_magnitude_system("GAIA3E_G") == "vega"
    assert catalog_photometry.default_magnitude_system("2MASS_Ks") == "vega"
    assert catalog_photometry.default_magnitude_system("WISE_RSR_W1") == "vega"
    assert catalog_photometry.default_magnitude_system("SPITZER_IRAC_36") == "vega"
    assert catalog_photometry.default_magnitude_system("PS1_g") == "ab"
    assert catalog_photometry.default_magnitude_system("SDSS_r") == "ab"
    assert catalog_photometry.default_magnitude_system("SkyMapper_g") == "ab"
    assert catalog_photometry.default_magnitude_system("GALEX_NUV") == "ab"


def test_default_magnitude_type_uses_sdss_asinh_only():
    assert catalog_photometry.default_magnitude_type("SDSS_u") == "asinh"
    assert catalog_photometry.default_magnitude_type("SDSS_z") == "asinh"
    assert catalog_photometry.default_magnitude_type("PS1_g") == "pogson"
    assert catalog_photometry.default_magnitude_type("GAIA3E_G") == "pogson"


def test_hst_requires_explicit_magnitude_system():
    import pytest

    with pytest.raises(ValueError, match="No default magnitude system"):
        catalog_photometry.default_magnitude_system("HST_WFC3_F814W")


def test_configured_hst_band_requires_explicit_system():
    import pytest

    catalog = catalog_photometry.CatalogSpec(
        name="HST",
        vizier_id="local",
        bands=(
            catalog_photometry.BandSpec(
                photband="HST_WFC3_F814W",
                mag="mag",
                mag_err="mag_err",
            ),
        ),
    )
    table = Table({"mag": [20.0], "mag_err": [0.01]})

    with pytest.raises(ValueError, match="No default magnitude system"):
        catalog_photometry.extract_photometry_from_row(table[0], catalog)


def test_mag_to_flux_applies_optional_mag_zero_point_offset(monkeypatch):
    monkeypatch.setattr(catalog_photometry, "zero_point_flux", lambda photband, system: 100.0)

    flux, _ = catalog_photometry.mag_to_flux(
        "PS1_g",
        10.0,
        0.01,
        "ab",
        mag_zp_offset=0.1,
    )

    assert np.isclose(flux, 100.0 * 10.0 ** (-0.4 * 10.1))


def test_sdss_asinh_mag_to_flux_uses_luptitude_formula(monkeypatch):
    monkeypatch.setattr(catalog_photometry, "zero_point_flux", lambda photband, system: 1.0)

    mag = 22.0
    mag_err = 0.03
    flux, flux_err = catalog_photometry.mag_to_flux(
        "SDSS_u",
        mag,
        mag_err,
        "ab",
        mag_type="asinh",
        mag_zp_offset=-0.04,
    )

    b = catalog_photometry.SDSS_ASINH_SOFTENING["u"]
    arg = -mag * np.log(10.0) / 2.5 - np.log(b)
    offset_scale = 10.0 ** (-0.4 * -0.04)
    expected_flux = offset_scale * 2.0 * b * np.sinh(arg)
    expected_err = (
        offset_scale
        * 2.0
        * b
        * (np.log(10.0) / 2.5)
        * np.cosh(arg)
        * mag_err
    )

    assert np.isclose(flux, expected_flux)
    assert np.isclose(flux_err, expected_err)


def test_sdss_flux_to_asinh_mag_round_trips(monkeypatch):
    monkeypatch.setattr(catalog_photometry, "zero_point_flux", lambda photband, system: 1.0)

    mag = 20.5
    mag_err = 0.02
    flux, flux_err = catalog_photometry.mag_to_flux(
        "SDSS_z",
        mag,
        mag_err,
        "ab",
        mag_type="asinh",
        mag_zp_offset=0.02,
    )
    recovered_mag, recovered_err = catalog_photometry.flux_to_mag(
        "SDSS_z",
        flux,
        flux_err,
        "ab",
        mag_type="asinh",
        mag_zp_offset=0.02,
    )

    assert np.isclose(recovered_mag, mag)
    assert np.isclose(recovered_err, mag_err)


def test_extract_photometry_from_mag_row(monkeypatch):
    monkeypatch.setattr(catalog_photometry, "zero_point_flux", lambda photband, system: 1e-8)
    catalog = catalog_photometry.CatalogSpec(
        name="PS1",
        vizier_id="II/349/ps1",
        bands=(
            catalog_photometry.BandSpec(
                photband="PS1_g",
                mag="gmag",
                mag_err="e_gmag",
                system="ab",
                mag_zp_offset=0.1,
            ),
        ),
    )
    table = Table({"gmag": [20.0], "e_gmag": [0.02]})

    rows = catalog_photometry.extract_photometry_from_row(table[0], catalog)

    assert len(rows) == 1
    assert rows[0]["photband"] == "PS1_g"
    assert rows[0]["catalog"] == "PS1"
    assert rows[0]["flux"] > 0
    assert rows[0]["flux_err"] > 0
    assert rows[0]["mag"] == 20.0
    assert rows[0]["mag_err"] == 0.02
    assert rows[0]["system"] == "ab"
    assert rows[0]["mag_type"] == "pogson"
    assert rows[0]["mag_zp_offset"] == 0.1
    assert np.isclose(rows[0]["flux"], 1e-8 * 10.0 ** (-0.4 * 20.1))


def test_extract_photometry_uses_default_mag_error(monkeypatch):
    monkeypatch.setattr(catalog_photometry, "zero_point_flux", lambda photband, system: 1.0)
    catalog = catalog_photometry.CatalogSpec(
        name="Gaia",
        vizier_id="I/355/gaiadr3",
        bands=(
            catalog_photometry.BandSpec(
                photband="GAIA3E_G",
                mag="Gmag",
                mag_err="e_Gmag",
                system="vega",
            ),
        ),
    )
    table = Table({"Gmag": [10.0], "e_Gmag": [0.0]})

    rows = catalog_photometry.extract_photometry_from_row(
        table[0],
        catalog,
        default_mag_error=0.05,
    )

    expected_flux = 10.0 ** (-0.4 * 10.0)
    expected_err = 0.4 * np.log(10.0) * expected_flux * 0.05
    assert np.isclose(rows[0]["flux"], expected_flux)
    assert np.isclose(rows[0]["flux_err"], expected_err)
    assert np.isclose(rows[0]["mag"], 10.0)
    assert np.isclose(rows[0]["mag_err"], 0.05)


def test_convert_jansky_flux_to_flambda(monkeypatch):
    monkeypatch.setattr(catalog_photometry.filters, "eff_wave", lambda photband: 5000.0)

    flux, flux_err = catalog_photometry.convert_flux_unit(1.0, 0.1, "jy", "TEST_BAND")

    expected = 1e-23 * catalog_photometry.C_ANGSTROM_PER_S / 5000.0**2
    assert np.isclose(flux, expected)
    assert np.isclose(flux_err, 0.1 * expected)


def test_load_catalog_config(tmp_path):
    config = tmp_path / "catalogs.yaml"
    config.write_text(
        """
catalogs:
  - name: GaiaDR3
    vizier_id: I/355/gaiadr3
    source_id_column: Source
    bands:
      - photband: GAIA3E_G
        mag: Gmag
        mag_err: e_Gmag
        system: vega
""",
    )

    catalogs = catalog_photometry.load_catalog_config(config)

    assert len(catalogs) == 1
    assert catalogs[0].source_id_column == "Source"
    assert catalogs[0].bands[0].photband == "GAIA3E_G"


def test_load_builtin_catalogs():
    catalogs = catalog_photometry.load_catalog_config(
        catalog_photometry.default_catalog_config()
    )

    names = [catalog.name for catalog in catalogs]
    assert names == [
        "GaiaDR3",
        "2MASS",
        "AllWISE",
        "PanSTARRS1",
        "SDSSDR12",
        "GLIMPSE",
        "SkyMapperDR2",
        "GALEXAIS",
    ]


def test_builtin_sdss_offsets_are_configured():
    catalogs = catalog_photometry.load_catalog_config(
        catalog_photometry.default_catalog_config()
    )
    sdss = next(catalog for catalog in catalogs if catalog.name == "SDSSDR12")
    offsets = {
        band.photband: catalog_photometry.resolve_mag_zp_offset(
            band.photband,
            band.mag_zp_offset,
        )
        for band in sdss.bands
    }
    mag_types = {band.photband: band.mag_type for band in sdss.bands}

    assert offsets["SDSS_u"] == -0.04
    assert offsets["SDSS_g"] == 0.0
    assert offsets["SDSS_r"] == 0.0
    assert offsets["SDSS_i"] == 0.0
    assert offsets["SDSS_z"] == 0.02
    assert set(mag_types.values()) == {"asinh"}


def test_column_aliases_use_first_available_value():
    table = Table({"RA_ICRS": [12.3]})

    assert catalog_photometry._row_value(table[0], ("RAJ2000", "RA_ICRS")) == 12.3


def test_2mass_quality_flags_are_applied(monkeypatch):
    monkeypatch.setattr(catalog_photometry, "zero_point_flux", lambda photband, system: 1.0)
    catalog = catalog_photometry.CatalogSpec(
        name="2MASS",
        vizier_id="II/246/out",
        quality_checks=True,
        bands=(
            catalog_photometry.BandSpec(
                photband="2MASS_J",
                mag="Jmag",
                mag_err="e_Jmag",
                system="vega",
            ),
            catalog_photometry.BandSpec(
                photband="2MASS_H",
                mag="Hmag",
                mag_err="e_Hmag",
                system="vega",
            ),
        ),
    )
    table = Table(
        {
            "Jmag": [10.0],
            "e_Jmag": [0.02],
            "Hmag": [10.0],
            "e_Hmag": [0.02],
            "Qflg": ["AUU"],
            "Cflg": ["000"],
        }
    )

    rows = catalog_photometry.extract_photometry_from_row(table[0], catalog)

    assert [row["photband"] for row in rows] == ["2MASS_J"]


def test_allwise_extended_sources_are_rejected(monkeypatch):
    monkeypatch.setattr(catalog_photometry, "zero_point_flux", lambda photband, system: 1.0)
    catalog = catalog_photometry.CatalogSpec(
        name="AllWISE",
        vizier_id="II/328/allwise",
        quality_checks=True,
        bands=(
            catalog_photometry.BandSpec(
                photband="WISE_RSR_W1",
                mag="W1mag",
                mag_err="e_W1mag",
                system="vega",
            ),
        ),
    )
    table = Table(
        {
            "W1mag": [10.0],
            "e_W1mag": [0.02],
            "qph": ["A"],
            "ex": [1],
        }
    )

    rows = catalog_photometry.extract_photometry_from_row(table[0], catalog)

    assert rows == []


def test_select_best_row_uses_vizier_distance():
    table = Table({"_r": [2.0, 0.5], "Gmag": [11.0, 10.0]})
    catalog = catalog_photometry.CatalogSpec(
        name="Gaia",
        vizier_id="I/355/gaiadr3",
        bands=(),
    )

    row, separation = catalog_photometry.select_best_row(table, catalog)

    assert row["Gmag"] == 10.0
    assert separation == 0.5


def test_write_photometry_file(tmp_path):
    output = tmp_path / "target.phot"
    rows = [
        {
            "photband": "GAIA3E_G",
            "flux": 1.0e-12,
            "flux_err": 1.0e-14,
            "mag": 12.3,
            "mag_err": 0.01,
            "system": "vega",
        }
    ]

    table = catalog_photometry.write_photometry(rows, output)

    assert table.colnames == [
        "photband",
        "mag",
        "mag_err",
        "system",
        "mag_type",
        "mag_zp_offset",
        "flux",
        "flux_err",
    ]
    assert table["mag_type"][0] == "pogson"
    assert table["mag_zp_offset"][0] == 0.0
    assert "GAIA3E_G" in output.read_text()
    assert "12.3" in output.read_text()
    assert "vega" in output.read_text()


def test_write_photometry_respects_explicit_hst_system(tmp_path):
    output = tmp_path / "hst.phot"
    rows = [
        {
            "photband": "HST_WFC3_F814W",
            "flux": 1.0e-16,
            "flux_err": 1.0e-18,
            "mag": 22.0,
            "mag_err": 0.02,
            "system": "ab",
        }
    ]

    table = catalog_photometry.write_photometry(rows, output)

    assert table["system"][0] == "ab"
    assert "HST_WFC3_F814W" in output.read_text()
