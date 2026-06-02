"""Download VizieR photometry and write sedforge magnitude tables.

The required fitting columns remain deliberately small:

    photband mag mag_err system

Downloaded tables also keep internally converted fluxes for inspection:

    photband mag mag_err system mag_type mag_zp_offset flux flux_err

Catalog-specific details live in a YAML configuration file so new VizieR
catalogs can be added without changing the fitter.
"""

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
import numpy.ma as ma
import yaml
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astropy.table import Table
import astropy.units as u

from ._compat import trapezoid
from . import filters


C_ANGSTROM_PER_S = 2.99792458e18
AB_FNU_ZERO = 3.631e-20
SDSS_ASINH_SOFTENING = {
    "u": 1.4e-10,
    "g": 0.9e-10,
    "r": 1.2e-10,
    "i": 1.8e-10,
    "z": 7.4e-10,
}
SDSS_AB_MAG_OFFSETS = {
    "u": -0.04,
    "z": 0.02,
}

DEFAULT_MAGNITUDE_SYSTEM_PREFIXES = (
    ("GAIA3E_", "vega"),
    ("2MASS_", "vega"),
    ("WISE_RSR_", "vega"),
    ("SPITZER_IRAC_", "vega"),
    ("WFCAM_", "vega"),
    ("GALEX_", "ab"),
    ("PS1_", "ab"),
    ("SDSS_", "ab"),
    ("SkyMapper_", "ab"),
    ("ZTF_", "ab"),
)


@dataclass(frozen=True)
class BandSpec:
    photband: str
    mag: object = None
    mag_err: object = None
    system: str = None
    mag_type: str = None
    mag_zp_offset: float = None
    flux: object = None
    flux_err: object = None
    flux_unit: str = "flam"


@dataclass(frozen=True)
class CatalogSpec:
    name: str
    vizier_id: str
    bands: tuple[BandSpec, ...]
    source_id_column: object = None
    ra_column: object = None
    dec_column: object = None
    quality_checks: bool = False


def default_catalog_config():
    """Return the bundled VizieR catalog configuration."""
    return str(files("sedforge").joinpath("catalogs.default.yaml"))


def _column_aliases(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return str(value)


def _first_column_alias(value):
    if isinstance(value, tuple):
        return value[0] if value else None
    return value


def load_catalog_config(path):
    """Read catalog definitions from YAML."""
    with open(path, "r") as handle:
        config = yaml.safe_load(handle) or {}

    catalogs = []
    for index, raw_catalog in enumerate(config.get("catalogs", [])):
        vizier_id = raw_catalog.get("vizier_id", raw_catalog.get("vizier"))
        if not vizier_id:
            raise ValueError(f"Catalog entry {index} is missing vizier_id.")

        bands = []
        for raw_band in raw_catalog.get("bands", []):
            photband = raw_band.get("photband")
            if not photband:
                raise ValueError(f"Catalog {vizier_id} has a band without photband.")
            raw_offset = raw_band.get(
                "mag_zp_offset",
                raw_band.get("zp_offset", raw_band.get("mag_offset")),
            )
            bands.append(
                BandSpec(
                    photband=str(photband),
                    mag=_column_aliases(raw_band.get("mag", raw_band.get("mag_column"))),
                    mag_err=_column_aliases(raw_band.get("mag_err", raw_band.get("mag_err_column"))),
                    system=raw_band.get("system"),
                    mag_type=raw_band.get("mag_type", raw_band.get("magnitude_type")),
                    mag_zp_offset=None if raw_offset is None else float(raw_offset),
                    flux=_column_aliases(raw_band.get("flux", raw_band.get("flux_column"))),
                    flux_err=_column_aliases(raw_band.get("flux_err", raw_band.get("flux_err_column"))),
                    flux_unit=str(raw_band.get("flux_unit", "flam")).lower(),
                )
            )

        catalogs.append(
            CatalogSpec(
                name=str(raw_catalog.get("name", vizier_id)),
                vizier_id=str(vizier_id),
                bands=tuple(bands),
                source_id_column=_column_aliases(raw_catalog.get("source_id_column")),
                ra_column=_column_aliases(raw_catalog.get("ra_column")),
                dec_column=_column_aliases(raw_catalog.get("dec_column")),
                quality_checks=bool(raw_catalog.get("quality_checks", False)),
            )
        )

    if not catalogs:
        raise ValueError("Catalog config must define at least one catalog.")
    return catalogs


def _masked_or_missing(value):
    return value is None or value is ma.masked or ma.is_masked(value)


def _as_float(value):
    if _masked_or_missing(value):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _row_value(row, column):
    if column is None:
        return None
    if isinstance(column, (list, tuple)):
        for candidate in column:
            value = _row_value(row, candidate)
            if value is not None:
                return value
        return None
    try:
        return row[column]
    except (KeyError, ValueError):
        return None


def _catalog_key(catalog):
    return "".join(ch for ch in catalog.name.lower() if ch.isalnum())


def _flag_string(row, columns, default):
    value = _row_value(row, columns)
    if _masked_or_missing(value):
        return default
    return str(value)


def _int_value(row, columns):
    value = _as_float(_row_value(row, columns))
    if value is None:
        return None
    return int(value)


def _band_index(photband, names):
    try:
        return names.index(photband)
    except ValueError:
        return None


def _mag_passes_basic_qc(row, band, max_err=1.0):
    if not band.mag:
        return True
    mag = _as_float(_row_value(row, band.mag))
    if mag is None:
        return False
    mag_err = _as_float(_row_value(row, band.mag_err))
    return mag_err is None or mag_err <= max_err


def catalog_row_passes_qc(row, catalog):
    """Return False for catalog-level flags that make all bands unreliable."""
    if not catalog.quality_checks:
        return True

    key = _catalog_key(catalog)
    if key in ("sdss", "sdssdr12"):
        source_class = _int_value(row, "class")
        quality = _int_value(row, "Q")
        if source_class is None or quality is None:
            return False
        return source_class == 6 and quality in (2, 3)

    if key in ("panstarrs", "panstarrs1", "ps1"):
        quality = _int_value(row, "Qual")
        if quality is None:
            return False
        is_star = not (quality & 1 and quality & 2)
        is_good = (quality & 4 or quality & 16) and not (quality & 128)
        return is_star and bool(is_good)

    if key in ("allwise", "wise"):
        extended = _int_value(row, "ex")
        return extended is None or extended == 0

    if key in ("skymapper", "skymapperdr2"):
        flags = _int_value(row, ("flags", "Flags"))
        return flags is None or flags == 0

    return True


def band_passes_qc(row, catalog, band):
    """Return False for per-band catalog flags in the bundled query policy."""
    if not catalog.quality_checks:
        return True
    if not _mag_passes_basic_qc(row, band):
        return False

    key = _catalog_key(catalog)
    if key == "2mass":
        idx = _band_index(band.photband, ["2MASS_J", "2MASS_H", "2MASS_Ks"])
        if idx is None:
            return True
        qflg = _flag_string(row, "Qflg", "UUU")
        cflg = _flag_string(row, "Cflg", "999")
        if idx >= len(qflg) or idx >= len(cflg):
            return False
        return qflg[idx] in "ABCD" and cflg[idx] == "0"

    if key in ("allwise", "wise"):
        idx = _band_index(band.photband, ["WISE_RSR_W1", "WISE_RSR_W2"])
        if idx is None:
            return True
        qph = _flag_string(row, "qph", "UU")
        if idx >= len(qph):
            return False
        return qph[idx] in "ABC"

    if key in ("galex", "galexais"):
        if band.photband == "GALEX_FUV":
            for column in ("Fexf", "Fafl"):
                flag = _int_value(row, column)
                if flag is not None and flag > 0:
                    return False
        if band.photband == "GALEX_NUV":
            for column in ("Nexf", "Nafl"):
                flag = _int_value(row, column)
                if flag is not None and flag > 0:
                    return False

    return True


def _vega_flux_on_response(photband):
    wave, _trans = filters.get_response(photband)
    vega_wave, vega_flux = filters._load_vega()
    flux = np.interp(wave, vega_wave, vega_flux, left=np.nan, right=np.nan)
    good = np.isfinite(flux)
    if np.count_nonzero(good) < 2:
        raise ValueError(f"Vega spectrum does not cover {photband}.")
    return wave[good], flux[good]


def _band_averaged_flux(wave, flux, trans, photband):
    weight = filters.integration_weight(photband, wave)
    denom = trapezoid(trans * weight, x=wave)
    if denom <= 0:
        raise ValueError("Filter response has zero integrated throughput.")
    return trapezoid(flux * trans * weight, x=wave) / denom


def normalise_magnitude_system(system):
    """Return the canonical magnitude system name supported by sedforge."""
    if system is None:
        raise ValueError("Magnitude system is missing.")
    text = str(system).strip().lower().replace("_", "")
    aliases = {
        "ab": "ab",
        "abmag": "ab",
        "vega": "vega",
        "vegamag": "vega",
    }
    if text in aliases:
        return aliases[text]
    if text in {"st", "stmag"}:
        raise ValueError(
            "STMag input is not currently supported. Convert it to AB/Vega "
            "or provide a band-averaged flux explicitly."
        )
    raise ValueError(f"Unknown magnitude system '{system}'. Use 'ab' or 'vega'.")


def default_magnitude_system(photband):
    """
    Infer the default catalog magnitude system for common filter families.

    HST filters are deliberately not inferred because the same passband is
    commonly reported in VegaMag, ABMag, or STMag. Provide a ``system`` column
    for those filters.
    """
    try:
        canonical = filters._canonical_photband(photband)
    except Exception:
        canonical = str(photband)

    for prefix, system in DEFAULT_MAGNITUDE_SYSTEM_PREFIXES:
        if canonical.startswith(prefix):
            return system

    raise ValueError(
        "No default magnitude system is known for photband '{}'. Add a "
        "'system' column with 'ab' or 'vega'.".format(photband)
    )


def resolve_magnitude_system(photband, system=None):
    if system is None:
        return default_magnitude_system(photband)
    text = str(system).strip()
    if text == "" or text.lower() in {"none", "nan", "--"}:
        return default_magnitude_system(photband)
    return normalise_magnitude_system(text)


def _sdss_filter_key(photband):
    try:
        canonical = filters._canonical_photband(photband)
    except Exception:
        canonical = str(photband)
    if not canonical.startswith("SDSS_"):
        return None
    suffix = canonical.rsplit("_", 1)[-1].lower()
    return suffix if suffix in SDSS_ASINH_SOFTENING else None


def default_magnitude_type(photband):
    """Return the default magnitude definition for a photband."""
    return "asinh" if _sdss_filter_key(photband) is not None else "pogson"


def normalise_magnitude_type(mag_type):
    if mag_type is None:
        raise ValueError("Magnitude type is missing.")
    text = str(mag_type).strip().lower().replace("_", "")
    aliases = {
        "pogson": "pogson",
        "log": "pogson",
        "logarithmic": "pogson",
        "asinh": "asinh",
        "luptitude": "asinh",
        "luptitudes": "asinh",
    }
    if text in aliases:
        return aliases[text]
    raise ValueError(f"Unknown magnitude type '{mag_type}'. Use 'pogson' or 'asinh'.")


def resolve_magnitude_type(photband, mag_type=None):
    if mag_type is None:
        return default_magnitude_type(photband)
    text = str(mag_type).strip()
    if text == "" or text.lower() in {"none", "nan", "--"}:
        return default_magnitude_type(photband)
    return normalise_magnitude_type(text)


def default_mag_zp_offset(photband):
    """Return the default fixed magnitude offset needed before AB conversion."""
    key = _sdss_filter_key(photband)
    if key is None:
        return 0.0
    return SDSS_AB_MAG_OFFSETS.get(key, 0.0)


def resolve_mag_zp_offset(photband, mag_zp_offset=None):
    if mag_zp_offset is None:
        return default_mag_zp_offset(photband)
    text = str(mag_zp_offset).strip()
    if text == "" or text.lower() in {"none", "nan", "--"}:
        return default_mag_zp_offset(photband)
    return float(mag_zp_offset)


def zero_point_flux(photband, system):
    """Return the zero-magnitude band-averaged Flambda for one filter."""
    system = resolve_magnitude_system(photband, system)
    wave, trans = filters.get_response(photband)

    if system == "ab":
        flux = AB_FNU_ZERO * C_ANGSTROM_PER_S / wave**2
        return _band_averaged_flux(wave, flux, trans, photband)

    if system == "vega":
        vega_wave, vega_flux = _vega_flux_on_response(photband)
        trans_i = np.interp(vega_wave, wave, trans)
        return _band_averaged_flux(vega_wave, vega_flux, trans_i, photband)

    raise ValueError(f"Unknown magnitude system '{system}' for {photband}.")


def mag_to_flux(photband, mag, mag_err, system=None, mag_type=None, mag_zp_offset=None):
    """Convert magnitude to Flambda in erg/s/cm2/Angstrom."""
    flux0 = zero_point_flux(photband, system)
    mag = float(mag)
    mag_err = float(mag_err)
    mag_type = resolve_magnitude_type(photband, mag_type)
    mag_zp_offset = resolve_mag_zp_offset(photband, mag_zp_offset)

    if mag_type == "pogson":
        corrected_mag = mag + mag_zp_offset
        flux = flux0 * 10.0 ** (-0.4 * corrected_mag)
        flux_err = abs(0.4 * np.log(10.0) * flux * mag_err)
        return flux, flux_err

    key = _sdss_filter_key(photband)
    if key is None:
        raise ValueError("Asinh magnitudes are currently supported only for SDSS filters.")
    b = SDSS_ASINH_SOFTENING[key]
    arg = -mag * np.log(10.0) / 2.5 - np.log(b)
    flux_ratio = 2.0 * b * np.sinh(arg)
    ratio_err = abs(2.0 * b * (np.log(10.0) / 2.5) * np.cosh(arg) * mag_err)
    offset_scale = 10.0 ** (-0.4 * mag_zp_offset)
    flux = flux0 * offset_scale * flux_ratio
    flux_err = flux0 * offset_scale * ratio_err
    return flux, flux_err


def flux_to_mag(photband, flux, flux_err, system=None, mag_type=None, mag_zp_offset=None):
    """Convert a band-averaged Flambda to magnitude in the requested system."""
    flux = float(flux)
    flux_err = float(flux_err)
    if flux <= 0 or flux_err <= 0:
        raise ValueError("Flux and flux_err must be positive.")
    flux0 = zero_point_flux(photband, system)
    mag_type = resolve_magnitude_type(photband, mag_type)
    mag_zp_offset = resolve_mag_zp_offset(photband, mag_zp_offset)

    if mag_type == "pogson":
        mag = -2.5 * np.log10(flux / flux0) - mag_zp_offset
        mag_err = 2.5 / np.log(10.0) * flux_err / flux
        return float(mag), float(abs(mag_err))

    key = _sdss_filter_key(photband)
    if key is None:
        raise ValueError("Asinh magnitudes are currently supported only for SDSS filters.")
    b = SDSS_ASINH_SOFTENING[key]
    offset_scale = 10.0 ** (-0.4 * mag_zp_offset)
    flux_ratio = flux / (flux0 * offset_scale)
    arg = np.arcsinh(flux_ratio / (2.0 * b))
    mag = -(2.5 / np.log(10.0)) * (arg + np.log(b))
    ratio_err = flux_err / (flux0 * offset_scale)
    ratio_deriv = 2.0 * b * (np.log(10.0) / 2.5) * np.cosh(arg)
    mag_err = ratio_err / ratio_deriv
    return float(mag), float(abs(mag_err))


def fnu_to_flam(fnu, fnu_err, photband):
    """Convert Fnu to Flambda using the stored effective wavelength."""
    wave = filters.eff_wave(photband)
    scale = C_ANGSTROM_PER_S / wave**2
    return fnu * scale, fnu_err * scale


def convert_flux_unit(flux, flux_err, unit, photband):
    """Convert configured catalog flux units to package Flambda units."""
    unit = str(unit).lower()
    if unit in ("flam", "flambda", "erg/s/cm2/a", "erg/s/cm2/angstrom"):
        return flux, flux_err
    if unit == "fnu_cgs":
        return fnu_to_flam(flux, flux_err, photband)
    if unit == "jy":
        return fnu_to_flam(flux * 1e-23, flux_err * 1e-23, photband)
    if unit == "mjy":
        return fnu_to_flam(flux * 1e-26, flux_err * 1e-26, photband)
    if unit in ("ujy", "microjy"):
        return fnu_to_flam(flux * 1e-29, flux_err * 1e-29, photband)
    raise ValueError(f"Unknown flux unit '{unit}' for {photband}.")


def extract_band_from_row(row, band, default_mag_error=0.03):
    """Extract one configured band from one VizieR row."""
    system = resolve_magnitude_system(band.photband, band.system)
    mag_type = resolve_magnitude_type(band.photband, band.mag_type)
    mag_zp_offset = resolve_mag_zp_offset(band.photband, band.mag_zp_offset)
    if band.flux:
        flux = _as_float(_row_value(row, band.flux))
        if flux is None or flux <= 0:
            return None
        flux_err = _as_float(_row_value(row, band.flux_err))
        if flux_err is None or flux_err <= 0:
            return None
        flux, flux_err = convert_flux_unit(flux, flux_err, band.flux_unit, band.photband)
        mag = _as_float(_row_value(row, band.mag))
        mag_err = _as_float(_row_value(row, band.mag_err))
        return flux, flux_err, mag, mag_err, system, mag_type, mag_zp_offset

    if band.mag:
        mag = _as_float(_row_value(row, band.mag))
        if mag is None:
            return None
        mag_err = _as_float(_row_value(row, band.mag_err))
        if mag_err is None or mag_err <= 0:
            mag_err = float(default_mag_error)
        flux, flux_err = mag_to_flux(
            band.photband,
            mag,
            mag_err,
            system,
            mag_type=mag_type,
            mag_zp_offset=mag_zp_offset,
        )
        return flux, flux_err, mag, mag_err, system, mag_type, mag_zp_offset

    raise ValueError(f"Band {band.photband} needs either mag or flux column.")


def extract_photometry_from_row(row, catalog, default_mag_error=0.03):
    if not catalog_row_passes_qc(row, catalog):
        return []

    rows = []
    for band in catalog.bands:
        if not band_passes_qc(row, catalog, band):
            continue
        values = extract_band_from_row(row, band, default_mag_error=default_mag_error)
        if values is None:
            continue
        flux, flux_err, mag, mag_err, system, mag_type, mag_zp_offset = values
        if np.isfinite(flux) and np.isfinite(flux_err) and flux > 0 and flux_err > 0:
            rows.append(
                {
                    "photband": band.photband,
                    "flux": float(flux),
                    "flux_err": float(flux_err),
                    "mag": np.nan if mag is None else float(mag),
                    "mag_err": np.nan if mag_err is None else float(mag_err),
                    "system": system,
                    "mag_type": mag_type,
                    "mag_zp_offset": float(mag_zp_offset),
                    "catalog": catalog.name,
                }
            )
    return rows


def parse_coordinate(ra=None, dec=None, coord=None):
    """Parse CLI coordinate inputs."""
    if coord:
        try:
            return SkyCoord(coord)
        except ValueError:
            parts = str(coord).replace(",", " ").split()
            if len(parts) != 2:
                raise
            return SkyCoord(float(parts[0]) * u.deg, float(parts[1]) * u.deg)
    if ra is None or dec is None:
        raise ValueError("Provide either --coord or both --ra and --dec.")
    return SkyCoord(float(ra) * u.deg, float(dec) * u.deg)


def _first_table(result, catalog_id):
    if result is None or len(result) == 0:
        return None
    try:
        return result[catalog_id]
    except (KeyError, TypeError):
        return result[0]


def _table_coord(row, catalog):
    if catalog.ra_column is None or catalog.dec_column is None:
        return None
    ra = _as_float(_row_value(row, catalog.ra_column))
    dec = _as_float(_row_value(row, catalog.dec_column))
    if ra is None or dec is None:
        return None
    return SkyCoord(ra * u.deg, dec * u.deg)


def select_best_row(table, catalog, coord=None, max_separation_arcsec=None):
    """Select nearest acceptable catalog row."""
    if table is None or len(table) == 0:
        return None, None

    if "_r" in table.colnames:
        table = table.copy()
        table.sort("_r")

    best_row = None
    best_sep = None
    for row in table:
        sep = None
        if "_r" in table.colnames:
            sep = _as_float(row["_r"])
        elif coord is not None:
            row_coord = _table_coord(row, catalog)
            if row_coord is not None:
                sep = coord.separation(row_coord).arcsec

        if max_separation_arcsec is not None and sep is not None:
            if sep > max_separation_arcsec:
                continue

        best_row = row
        best_sep = sep
        break

    return best_row, best_sep


def _new_vizier(timeout=60):
    try:
        from astroquery.vizier import Vizier
    except ImportError as exc:
        raise ImportError(
            "Downloading catalog photometry requires astroquery. "
            "Install it with `pip install astroquery`."
        ) from exc
    return Vizier(row_limit=-1, columns=["all", "+_r"], timeout=timeout)


def resolve_gaia_id(
    gaia_id,
    catalog=None,
    gaia_catalog=None,
    source_column=None,
    ra_column=None,
    dec_column=None,
    timeout=60,
):
    """Resolve a Gaia DR3 source id to coordinates through VizieR."""
    if catalog is not None:
        gaia_catalog = catalog.vizier_id
        source_column = catalog.source_id_column
        ra_column = catalog.ra_column
        dec_column = catalog.dec_column
    if not all([gaia_catalog, source_column, ra_column, dec_column]):
        raise ValueError(
            "Resolving --gaia-id needs a configured catalog with "
            "vizier_id, source_id_column, ra_column, and dec_column."
        )

    vizier = _new_vizier(timeout=timeout)
    source_column = _first_column_alias(source_column)
    result = vizier.query_constraints(
        catalog=gaia_catalog,
        **{source_column: str(gaia_id)},
    )
    table = _first_table(result, gaia_catalog)
    if table is None or len(table) == 0:
        raise ValueError(f"Gaia source {gaia_id} was not found in {gaia_catalog}.")
    row = table[0]
    ra = _as_float(_row_value(row, ra_column))
    dec = _as_float(_row_value(row, dec_column))
    if ra is None or dec is None:
        raise ValueError(
            f"Gaia catalog {gaia_catalog} did not provide {ra_column}/{dec_column}."
        )
    return SkyCoord(ra * u.deg, dec * u.deg)


def _catalog_for_source_id_resolution(catalogs):
    for catalog in catalogs:
        if catalog.source_id_column and catalog.ra_column and catalog.dec_column:
            return catalog
    return None


def query_vizier_photometry(
    catalogs,
    coord=None,
    gaia_id=None,
    radius_arcsec=3.0,
    default_mag_error=0.03,
    timeout=60,
):
    """Query configured VizieR catalogs and return photometry rows."""
    if coord is None and gaia_id is None:
        raise ValueError("Provide coordinates or a Gaia source id.")

    if coord is None:
        source_catalog = _catalog_for_source_id_resolution(catalogs)
        if source_catalog is None:
            raise ValueError(
                "Using --gaia-id requires at least one configured VizieR catalog "
                "with source_id_column, ra_column, and dec_column."
            )
        coord = resolve_gaia_id(gaia_id, catalog=source_catalog, timeout=timeout)

    vizier = _new_vizier(timeout=timeout)
    rows = []
    metadata = []
    seen = set()
    radius = float(radius_arcsec) * u.arcsec

    for catalog in catalogs:
        table = None
        if gaia_id is not None and catalog.source_id_column:
            source_column = _first_column_alias(catalog.source_id_column)
            result = vizier.query_constraints(
                catalog=catalog.vizier_id,
                **{source_column: str(gaia_id)},
            )
            table = _first_table(result, catalog.vizier_id)

        if table is None or len(table) == 0:
            result = vizier.query_region(coord, radius=radius, catalog=catalog.vizier_id)
            table = _first_table(result, catalog.vizier_id)

        row, separation = select_best_row(
            table,
            catalog,
            coord=coord,
            max_separation_arcsec=float(radius_arcsec),
        )
        metadata.append(
            {
                "catalog": catalog.name,
                "vizier_id": catalog.vizier_id,
                "n_rows": 0 if table is None else len(table),
                "selected_separation_arcsec": np.nan if separation is None else separation,
            }
        )
        if row is None:
            continue

        for item in extract_photometry_from_row(
            row,
            catalog,
            default_mag_error=default_mag_error,
        ):
            if item["photband"] in seen:
                continue
            seen.add(item["photband"])
            rows.append(item)

    return rows, metadata, coord


def photometry_table(rows):
    """Build the magnitude photometry table from row dictionaries."""
    systems = [
        resolve_magnitude_system(row["photband"], row.get("system", None))
        for row in rows
    ]
    mag_types = [
        resolve_magnitude_type(row["photband"], row.get("mag_type", None))
        for row in rows
    ]
    return Table(
        [
            [row["photband"] for row in rows],
            [row.get("mag", np.nan) for row in rows],
            [row.get("mag_err", np.nan) for row in rows],
            systems,
            mag_types,
            [
                resolve_mag_zp_offset(row["photband"], row.get("mag_zp_offset", None))
                for row in rows
            ],
            [row["flux"] for row in rows],
            [row["flux_err"] for row in rows],
        ],
        names=[
            "photband",
            "mag",
            "mag_err",
            "system",
            "mag_type",
            "mag_zp_offset",
            "flux",
            "flux_err",
        ],
    )


def metadata_table(rows):
    return Table(
        [
            [row["catalog"] for row in rows],
            [row["vizier_id"] for row in rows],
            [row["n_rows"] for row in rows],
            [row["selected_separation_arcsec"] for row in rows],
        ],
        names=["catalog", "vizier_id", "n_rows", "selected_separation_arcsec"],
    )


def write_photometry(rows, output):
    table = photometry_table(rows)
    ascii.write(table, output, overwrite=True)
    return table


def write_metadata(rows, output):
    table = metadata_table(rows)
    ascii.write(table, output, overwrite=True)
    return table


def download_photometry(
    config_path,
    output,
    ra=None,
    dec=None,
    coord=None,
    gaia_id=None,
    radius_arcsec=3.0,
    default_mag_error=0.03,
    timeout=60,
    metadata_output=None,
):
    if config_path is None:
        config_path = default_catalog_config()
    catalogs = load_catalog_config(config_path)
    skycoord = parse_coordinate(ra=ra, dec=dec, coord=coord) if gaia_id is None else None
    rows, metadata, resolved = query_vizier_photometry(
        catalogs,
        coord=skycoord,
        gaia_id=gaia_id,
        radius_arcsec=radius_arcsec,
        default_mag_error=default_mag_error,
        timeout=timeout,
    )
    if not rows:
        raise ValueError("No usable photometry was found in the configured catalogs.")

    output = Path(output)
    table = write_photometry(rows, output)
    if metadata_output is not None:
        write_metadata(metadata, metadata_output)
    return table, metadata, resolved
