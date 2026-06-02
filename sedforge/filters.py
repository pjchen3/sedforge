import glob
import fnmatch
import os

import numpy as np

from astropy.io import ascii
from astropy.table import Table
basedir = os.path.dirname(__file__)
response_dir = os.path.join(basedir, 'transmission_curves')
_vega_cache = None
_filter_info_cache = None


def _vega_file():
    filename = os.path.join(basedir, 'vega.dat')
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            "vega.dat is not available. It is needed to calculate "
            "Vega-weighted effective wavelengths."
        )
    return filename


def _filter_info_file():
    filename = os.path.join(basedir, 'filter_info.dat')
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            "filter_info.dat is not available. It is needed for plotting "
            "filter effective wavelengths and bandwidths."
        )
    return filename


def _load_vega():
    global _vega_cache
    if _vega_cache is None:
        table = np.loadtxt(_vega_file(), comments='#')
        if table.ndim != 2 or table.shape[1] < 2:
            raise ValueError('vega.dat must have at least wavelength and flux columns.')
        wave, flux = table[:, 0], table[:, 1]
        if not np.all(np.diff(wave) > 0):
            raise ValueError('vega.dat wavelengths must be strictly increasing.')
        _vega_cache = (wave, flux)
    return _vega_cache


def _load_filter_info():
    global _filter_info_cache
    if _filter_info_cache is None:
        data = ascii.read(_filter_info_file(), comment=r"\s*#")
        required = {'photband', 'eff_wave', 'bandwidth'}
        missing = required - set(data.colnames)
        if missing:
            raise ValueError(
                'filter_info.dat is missing required columns: {}'.format(
                    ', '.join(sorted(missing))
                )
            )
        _filter_info_cache = data
    return _filter_info_cache


def _response_files():
    return sorted(
        filename for filename in glob.glob(os.path.join(response_dir, '*'))
        if os.path.isfile(filename) and not os.path.basename(filename).startswith('.')
    )


def _response_name(filename):
    name = os.path.basename(filename)
    root, ext = os.path.splitext(name)
    return root if ext.lower() == '.dat' else name


def _name_keys(name):
    root, ext = os.path.splitext(name)
    candidates = {name, root, name.replace('.', '_'), root.replace('.', '_')}
    candidates |= {candidate + '.dat' for candidate in list(candidates)}
    if name.upper().startswith('WISE.'):
        band = name.split('.', 1)[1]
        candidates |= {f'WISE_RSR_{band}', f'WISE_RSR_{band}.dat'}
    return {candidate.lower() for candidate in candidates}


def _canonical_photband(photband):
    try:
        return _response_name(_resolve_response_file(photband))
    except FileNotFoundError:
        return str(photband)


def _filter_info_row(photband):
    try:
        data = _load_filter_info()
    except FileNotFoundError:
        return None

    wanted = _name_keys(str(photband)) | _name_keys(_canonical_photband(photband))
    for row in data:
        if wanted & _name_keys(str(row['photband'])):
            return row
    return None


def _filter_info_value(photband, column):
    row = _filter_info_row(photband)
    if row is None:
        return None
    return float(row[column])


def response_type(photband):
    """
    Return the response convention used by the local passband file.

    ``photon`` means the tabulated response is per incident photon and the
    integration weight is lambda. ``energy`` means the tabulated response is
    already per unit incident energy and no extra lambda factor is applied.
    """
    row = _filter_info_row(photband)
    if row is not None and 'response_type' in row.colnames:
        value = str(row['response_type']).strip().lower()
    else:
        canonical = _canonical_photband(photband)
        energy_prefixes = ('SPITZER_IRAC_', 'WISE_RSR_')
        value = 'energy' if canonical.startswith(energy_prefixes) else 'photon'

    if value not in {'photon', 'energy'}:
        raise ValueError(
            "Unknown response_type '{}' for photband {}. Expected 'photon' "
            "or 'energy'.".format(value, photband)
        )
    return value


def integration_weight(photband, wave):
    """
    Return the wavelength weight matching a local passband response type.
    """
    wave = np.asarray(wave, dtype=float)
    if response_type(photband) == 'photon':
        return wave
    return np.ones_like(wave)


def _resolve_response_file(photband):
    wanted = _name_keys(str(photband))
    for filename in _response_files():
        basename = os.path.basename(filename)
        stem = _response_name(filename)
        keys = _name_keys(basename) | _name_keys(stem)
        if wanted & keys:
            return filename
    raise FileNotFoundError(
        'Can not find transmission curve for photband {} in {}'.format(
            photband, response_dir
        )
    )


def _throughput_weighted_wave(photband):
    wave, trans = get_response(photband)
    weight = trans * integration_weight(photband, wave)
    norm = np.trapz(weight, x=wave)
    if norm <= 0:
        raise ValueError(f'Response curve for {photband} has zero throughput.')
    return np.trapz(wave * weight, x=wave) / norm


def _vega_effective_wave(photband):
    """
    Vega-weighted effective wavelength using the local response convention.

    For photon response curves this uses an extra lambda factor. For energy
    response curves it does not.
    """
    wave, trans = get_response(photband)
    vega_wave, vega_flux = _load_vega()
    vega = np.interp(wave, vega_wave, vega_flux, left=np.nan, right=np.nan)
    valid = np.isfinite(vega) & np.isfinite(trans) & (trans > 0)
    if not np.any(valid):
        return _throughput_weighted_wave(photband)

    wave = wave[valid]
    trans = trans[valid]
    vega = vega[valid]
    weight = vega * trans * integration_weight(photband, wave)
    norm = np.trapz(weight, x=wave)
    if norm <= 0 or not np.isfinite(norm):
        return _throughput_weighted_wave(photband)
    return np.trapz(wave * weight, x=wave) / norm


def _response_bandwidth(photband):
    """
    Calculate the rectangular throughput width of a passband in Angstrom.
    """
    wave, trans = get_response(photband)
    max_trans = np.nanmax(trans)
    if max_trans <= 0 or not np.isfinite(max_trans):
        raise ValueError(f'Response curve for {photband} has zero throughput.')
    return np.trapz(trans, x=wave) / max_trans


def bandwidth(photband):
    """
    Return the rectangular throughput width of a passband in Angstrom.

    The definition is int(T(lambda) dlambda) / max(T), independent of source
    spectrum. This is only a filter-width descriptor; synthetic_flux still uses
    the full response curve.
    """
    value = _filter_info_value(photband, 'bandwidth')
    if value is not None:
        return value
    return _response_bandwidth(photband)


def eff_wave(photband):
    """
    Returns the effective wavelength of the pass band in angstrom

    @param photband: name of the photometric passband
    @type photband: string
    @return: effective wavelength
    @rtype: float
    """

    value = _filter_info_value(photband, 'eff_wave')
    if value is not None:
        return value
    return _vega_effective_wave(photband)


def get_info(photbands=None):
    """
    Return filter metadata for the requested passbands.

    Columns are photband, eff_wave, bandwidth, and response_type. Values are
    read from filter_info.dat when available. Missing bands fall back to direct
    calculation from the local response curves.
    """
    if photbands is None:
        try:
            return _load_filter_info().copy()
        except FileNotFoundError:
            photbands = list_response()
    elif isinstance(photbands, str):
        photbands = [photbands]

    return Table(
        [
            np.array(photbands, dtype=str),
            np.array([eff_wave(photband) for photband in photbands], dtype=float),
            np.array([bandwidth(photband) for photband in photbands], dtype=float),
            np.array([response_type(photband) for photband in photbands], dtype=str),
        ],
        names=['photband', 'eff_wave', 'bandwidth', 'response_type'],
    )


def list_response(name='*', wave_range=(-np.inf, +np.inf)):
    """
    List available response curves.

    Specify a glob string C{name} and/or a wavelength range to make a selection
    of all available curves. If nothing is supplied, all curves will be returned.

    :param name: list all curves containing this string
    :type name: str
    :param wave_range: list all curves within this wavelength range (A)
    :type wave_range: (float, float)
    :return: list of curve files
    :rtype: list of str
    """
    name = '*' if name is None else str(name)
    pattern = name.lower()
    if '*' not in pattern and '?' not in pattern:
        pattern = '*' + pattern + '*'

    responses = []
    for curve_file in _response_files():
        response = _response_name(curve_file)
        if not fnmatch.fnmatch(response.lower(), pattern):
            continue
        try:
            wave, _ = get_response(response)
        except Exception:
            continue
        if np.nanmax(wave) < wave_range[0] or np.nanmin(wave) > wave_range[1]:
            continue
        responses.append(response)

    return responses


def get_response(photband):
    """
    Returns the response/transmission curve of the provided photometric pass band.
    returns wave, transmission
    Wave in AA
    transmission is unitless or more specifically unit independent.
    """

    transmission_file = _resolve_response_file(photband)
    table = np.loadtxt(transmission_file, comments='#')
    if table.ndim != 2 or table.shape[1] < 2:
        raise ValueError(
            'Transmission curve must have at least two numeric columns: {}'.format(
                transmission_file
            )
        )

    return table[:, 0], table[:, 1]


def synthetic_flux(wave, flux, photbands):
    """
    Extract flux measurements from a synthetic SED

    """
    energys = np.zeros(len(photbands))

    for i, photband in enumerate(photbands):

        waver, transr = get_response(photband)
        # -- make wavelength range a bit bigger, otherwise F25 from IRAS has only one Kurucz model point in its
        # wavelength range... this is a bit 'ad hoc' but seems to work.
        region = ((waver[0] - 0.4 * waver[0]) <= wave) & (wave <= (2 * waver[-1]))

        # todo: check if the model needs to be reinterpolated in log scale for some filters
        # # -- if we're working in infrared (>4e4A) and the model is not of high enough resolution (100000 points over
        # # wavelength range), interpolate the model in logscale on to a denser grid (in logscale!)
        # filter_info = filters.get_info()
        # if filter_info['eff_wave'][i] >= 4e4 and 1e5 > sum(region) > 1:
        #     print('%10s: Interpolating model to integrate over response curve' % (photband))
        #     wave_ = np.logspace(np.log10(wave[region][0]), np.log10(wave[region][-1]), 1e5)
        #     flux_ = 10 ** np.interp(np.log10(wave_), np.log10(wave[region]), np.log10(flux[region]), )
        # else:
        #     wave_ = wave[region]
        #     flux_ = flux[region]

        wave_ = wave[region]
        flux_ = flux[region]

        if not len(wave_):
            energys[i] = np.nan
            continue

        # -- perhaps the entire response curve falls in between model points (happends with narrowband UV filters), or
        # there's very few model points covering it
        if (np.searchsorted(wave_, waver[-1]) - np.searchsorted(wave_, waver[0])) < 5:
            wave__ = np.sort(np.hstack([wave_, waver]))
            flux_ = np.interp(wave__, wave_, flux_)
            wave_ = wave__

        # -- interpolate response curve onto model grid
        transr = np.interp(wave_, waver, transr, left=0, right=0)
        weight = integration_weight(photband, wave_)

        # -- WE WORK IN FLAMBDA
        energys[i] = (
            np.trapz(flux_ * transr * weight, x=wave_)
            / np.trapz(transr * weight, x=wave_)
        )

    # -- that's it!
    return energys
