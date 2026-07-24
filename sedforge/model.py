import glob
import hashlib
import json
import re
import os
import shutil
import tempfile
import yaml
from itertools import product
import numpy as np

from astropy.io import fits

from sedforge._compat import trapezoid
from sedforge import interpol, spectral_cache

PC_TO_RSOL = 44365810.04823812
DEFAULT_HDF5_PRELOAD_MAX_GB = 2.0
RUNTIME_CACHE_FORMAT_VERSION = 1
FITS_RUNTIME_CACHE_FORMAT_VERSION = 1
_GRID_CACHE = {}
_SHARED_HDF5_CACHES = []
_SHARED_FITS_GRIDS = []
_GRID_FILE_CAPABILITY_CACHE = {}


def _models_directory_from_env():
    return os.environ.get('SEDFORGE_MODELS')


__defaults__ = dict(grid='ck_all',
                    directory=_models_directory_from_env(), )
defaults = __defaults__.copy()


def _load_grid_description(directory):
    if not directory:
        return None, {}

    path = os.path.join(directory, 'grid_description.yaml')
    try:
        with open(path) as ifile:
            return path, yaml.safe_load(ifile) or {}
    except (OSError, yaml.YAMLError):
        return path, {}


# load a list of all available integrated grids
grid_description_file, grid_description = _load_grid_description(defaults.get('directory'))


def clear_grid_cache():
    """Clear prepared integrated-grid objects cached in this Python process."""
    _GRID_CACHE.clear()
    _SHARED_HDF5_CACHES.clear()
    _SHARED_FITS_GRIDS.clear()
    _GRID_FILE_CAPABILITY_CACHE.clear()


def _freeze_for_cache(value):
    if isinstance(value, np.ndarray):
        return tuple(_freeze_for_cache(item) for item in value.tolist())
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze_for_cache(val))
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_for_cache(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def check_grids(print_bands=False):
    """
    Check which atmospheric model grids are installed and can be used by sedforge.

    :param print_bands: If True, also print the photometric pass bands that are available in the integrated grid.
    :type print_bands: bool
    """
    print("Checking which atmosphere models are available...")

    if defaults['directory'] is not None:
        print(f"Checking for models in {defaults['directory']}")
        print(f"Using grid description file: {grid_description_file}")
    else:
        print("SEDFORGE_MODELS environmental variable not set. CAN NOT find models!")
        print("Please point the SEDFORGE_MODELS variable to the directory where you stored models. On bash use:")
        print("\texport SEDFORGE_MODELS='<path to extracted atmosphere models>'")
        return

    if len(grid_description.keys()) == 0:
        print("grid_description.yaml file not found or no models included in the description file.")
        print("Please add a grid_description.yaml file in the SEDFORGE_MODELS directory.")
        print(f"SEDFORGE_MODELS directory is: {defaults.get('directory', '')}")
        files = glob.glob(os.path.join(defaults.get('directory', ''), '*'))
        print(f"Content of SEDFORGE_MODELS is: {files}")
        return

    for grid in grid_description.keys():
        print(grid)
        gridpath = get_grid_file(integrated=False, grid=grid)
        if raw_spectrum_available(grid):
            if spectral_cache.has_cache(
                    grid, grid_description=grid_description,
                    directory=defaults['directory']):
                print("\t raw cache: available")
            else:
                print("\t raw: available")
        else:
            print("\t raw: NOT FOUND")

        gridpath = get_grid_file(integrated=True, grid=grid)
        if os.path.isfile(gridpath):
            print("\t integrated: available")
        else:
            print("\t integrated: NOT FOUND")

        if print_bands:
            if _grid_integrated_format(grid) == 'hdf5':
                import h5py
                with h5py.File(gridpath, 'r') as h5:
                    bands = [
                        _normalise_photband_name(band)
                        for band in h5['axes/photband'][:]
                    ]
            else:
                hdu = fits.open(gridpath)
                bands = set([b.split('.')[0] for b in hdu[1].data.dtype.names if '.' in b])
            for b in bands:
                print("\t - " + b)

        if 'info' in grid_description[grid]:
            print('\t info: ' + grid_description[grid]['info'])


def _rv_label(rv):
    return f"{float(rv):0.2f}"


def _law_label(law, case1=1):
    if str(law).lower() == 'wc2019' and int(case1) != 1:
        return f"{law}_case{int(case1)}"
    return law


def get_grid_file(integrated=False, **kwargs):
    grid = kwargs.get('grid', defaults['grid'])

    if os.path.isfile(grid):
        return grid

    if grid in grid_description:
        desc = grid_description[grid]
        if integrated:
            if _integrated_format(desc) == 'hdf5':
                filename = desc.get('integrated_path', desc['filename'])
                directory = kwargs.get('directory', defaults['directory'])
                if os.path.isabs(filename):
                    return filename
                if filename.endswith(('.h5', '.hdf5')):
                    return os.path.join(directory, filename)
                return os.path.join(directory, filename + '.h5')
            filename = desc['filename']
        else:
            filename = desc.get('raw_filename', desc['filename'])
    else:
        raise ValueError(f'Grid name ({grid}) not recognized!')

    if integrated:
        law = kwargs.get('reddening_law', kwargs.get('law', 'WC2019'))
        rv = kwargs.get('reddening_Rv', kwargs.get('Rv', 3.1))
        case1 = kwargs.get('reddening_case1', kwargs.get('case1', 1))
        filename = 'i' + filename + '_law' + _law_label(law, case1) + '_Rv' + _rv_label(rv)
        subdir = desc.get('integrated_subdir', '')
        if subdir:
            filename = os.path.join(subdir, filename)

    directory = kwargs.get('directory', defaults['directory'])

    return os.path.join(directory, filename + '.fits')


def _integrated_format(desc):
    return str(desc.get('integrated_format', desc.get('format', 'fits'))).lower()


def _grid_integrated_format(gridname):
    if isinstance(gridname, str) and gridname in grid_description:
        return _integrated_format(grid_description[gridname])
    if isinstance(gridname, dict):
        return _integrated_format(gridname)
    if isinstance(gridname, str) and gridname.endswith(('.h5', '.hdf5')):
        return 'hdf5'
    return 'fits'


def uses_hdf5_integrated_grid(grid):
    """Return whether one or more integrated grids use the HDF5 backend.

    ``grid`` may be a configured grid name, a prepared HDF5 grid object, or a
    list of either.  This is intentionally a public capability check so
    callers can choose numerical methods compatible with the grid backend.
    """
    if isinstance(grid, HDF5IntegratedGrid):
        return True
    if _is_prepared_grid(grid):
        return False
    if isinstance(grid, (str, dict)):
        return _grid_integrated_format(grid) == 'hdf5'
    try:
        return any(uses_hdf5_integrated_grid(item) for item in grid)
    except TypeError:
        return False


def get_spectral_cache_file(grid=None, required=False, **kwargs):
    grid = grid or kwargs.get('grid', defaults['grid'])
    path = spectral_cache.cache_file(
        grid,
        grid_description=grid_description,
        directory=kwargs.get('directory', defaults['directory']),
    )
    if required and (path is None or not os.path.isfile(path)):
        raise FileNotFoundError(f"No spectral cache configured for {grid}")
    return path


def raw_spectrum_available(grid=None, **kwargs):
    grid = grid or kwargs.get('grid', defaults['grid'])
    if spectral_cache.has_cache(
            grid,
            grid_description=grid_description,
            directory=kwargs.get('directory', defaults['directory'])):
        return True
    try:
        return os.path.isfile(get_grid_file(integrated=False, grid=grid,
                                            directory=kwargs.get('directory',
                                                                 defaults['directory'])))
    except Exception:
        return False


def get_grid_ranges(**kwargs):
    grid = kwargs.get('grid', defaults['grid'])

    if str(grid).startswith(('ckm', 'ckp')) or grid == 'ck_all':
        teff = (3500, 10000)
        logg = (4.32, 4.32)
        rad = (0.05, 2.5)

    elif grid == 'ck03_cepheid_rv' or str(grid).startswith('ck03_rv'):
        if str(grid).startswith('ck03_rv'):
            teff = (3500, 50000)
            logg = (0.0, 5.0)
            rad = (0.05, 500.0)
        else:
            teff = (4000, 8000)
            logg = (0.0, 5.0)
            rad = (1.0, 500.0)

    elif str(grid).startswith('tlusty'):
        teff = (15000, 55000)
        logg = (1.75, 4.75)
        rad = (0.1, 25.0)

    elif str(grid).startswith('tmap'):
        teff = (50000, 190000)
        logg = (5.0, 9.0)
        rad = (0.01, 0.5)

    elif grid == 'koester2':
        teff = (5000, 80000)
        logg = (6.5, 9.5)
        rad = (0.001, 0.1)

    elif grid == 'blackbody':
        teff = (20000, 80000)
        logg = (5.0, 5.0)
        rad = (0.01, 2.5)

    elif str(grid).startswith('newera'):
        teff = (2300, 12000)
        logg = (0.0, 6.0)
        rad = (0.01, 100.0)

    else:
        raise ValueError('Grid name ({}) not recognized!'.format(grid))

    ranges = {'teff': teff, 'logg': logg, 'rad': rad}
    if grid == 'ck_all':
        ranges['feh'] = (-2.5, 0.5)
    elif grid == 'ck03_cepheid_rv' or str(grid).startswith('ck03_rv'):
        ranges['feh'] = (-2.5 if str(grid).startswith('ck03_rv') else -2.0, 0.5)
        ranges['rv'] = (2.0, 5.0)
    elif grid == 'tlusty_all':
        ranges['feh'] = (-1.0, 0.3010299956639812)
    elif grid == 'newera_alpha0_rv':
        ranges['feh'] = (-2.5, 0.5)
        ranges['rv'] = (2.0, 5.0)
    elif str(grid).startswith('newera'):
        ranges['feh'] = (-2.0, 0.5)
    return ranges


def get_grid_dimensions(**kwargs):
    """
    Retrieve possible effective temperatures and gravities from a grid.

    E.g. kurucz, sdB, fastwind...

    :rtype: (ndarray,ndarray)
    :return: effective temperatures, gravities
    """
    gridfile = get_grid_file(**kwargs)
    ff = fits.open(gridfile)
    teffs = []
    loggs = []
    for hdu in ff[1:]:
        teffs.append(float(hdu.header['TEFF']))
        loggs.append(float(hdu.header['LOGG']))
    ff.close()

    # # -- maybe the fits extensions are not in right order...
    # matrix = np.vstack([np.array(teffs), np.array(loggs)]).T
    # matrix = numpy_ext.sort_order(matrix, order=[0, 1])
    # teffs, loggs = matrix.T

    return teffs, loggs


_FIELD_ALIASES = {
    'teff': ('teff', 'TEFF'),
    'logg': ('logg', 'LOGG'),
    'av': ('av', 'AV', 'A_V'),
    'ebv': ('ebv', 'EBV', 'E_BV'),
    'rv': ('rv', 'RV', 'R_V'),
    'feh': ('feh', 'FEH', 'FeH', 'M_H', 'MH', 'z', 'Z', 'metallicity'),
}


def _field_name(data, name):
    names = data.dtype.names or ()
    aliases = _FIELD_ALIASES.get(name, (name,))
    by_lower = {n.lower(): n for n in names}
    for alias in aliases:
        if alias in names:
            return alias
        if alias.lower() in by_lower:
            return by_lower[alias.lower()]
    return None


def _grid_file_capabilities(path):
    """Read integrated-grid axes once and reuse them across setup validation."""
    path = os.path.abspath(os.path.expanduser(str(path)))
    stat = os.stat(path)
    key = (path, int(stat.st_size), int(stat.st_mtime_ns))
    cached = _GRID_FILE_CAPABILITY_CACHE.get(key)
    if cached is not None:
        return cached

    axes = set()
    feh_varied = False
    if path.endswith(('.h5', '.hdf5')):
        import h5py
        with h5py.File(path, 'r') as h5:
            axes.update(str(name).lower() for name in h5.get('axes', {}).keys())
            if 'feh' in axes:
                values = np.asarray(h5['axes/feh'][:], dtype=float)
                finite = values[np.isfinite(values)]
                if len(finite):
                    feh_varied = bool(np.any(finite != finite[0]))
    else:
        with fits.open(path, memmap=True) as ff:
            data = ff[1].data
            for name in _FIELD_ALIASES:
                if _field_name(data, name) is not None:
                    axes.add(name)
            if 'feh' in axes:
                field = _field_name(data, 'feh')
                values = np.asarray(data.field(field), dtype=float)
                finite = values[np.isfinite(values)]
                if len(finite):
                    first = finite[0]
                    feh_varied = bool(np.any(finite != first))

    result = {
        'axes': frozenset(axes),
        'feh_varied': bool(feh_varied),
    }
    for stale_key in list(_GRID_FILE_CAPABILITY_CACHE):
        if stale_key[0] == path:
            _GRID_FILE_CAPABILITY_CACHE.pop(stale_key, None)
    _GRID_FILE_CAPABILITY_CACHE[key] = result
    return result


def _infer_grid_metadata(gridname):
    """Return fixed metadata values encoded in a grid description/name."""
    metadata = {}
    desc = grid_description.get(gridname, {}) if isinstance(gridname, str) else {}
    for key in ('feh', 'z', 'metallicity'):
        if key in desc:
            metadata['feh'] = float(desc[key])
    for key in ('he_mass', 'hemass', 'he'):
        if key in desc:
            metadata['he_mass'] = float(desc[key])

    if isinstance(gridname, str):
        match = re.fullmatch(r'ck([mp])(\d{2})', gridname.lower())
        if match:
            sign = -1 if match.group(1) == 'm' else 1
            metadata['feh'] = sign * float(match.group(2)) / 10.0
    return metadata


def _normalise_grid_members(gridname):
    """Accept a grid name, a member dict, or a metallicity stack."""
    if isinstance(gridname, dict) and 'members' in gridname:
        members = gridname['members']
    else:
        members = [gridname]

    out = []
    for member in members:
        if isinstance(member, str):
            item = {'grid': member}
        else:
            item = dict(member)
            if 'grid' not in item and 'name' in item:
                item['grid'] = item.pop('name')
        item.update({k: v for k, v in _infer_grid_metadata(item['grid']).items()
                     if k not in item})
        out.append(item)
    return out


def _grid_supports_feh(gridname):
    """Return True when a grid has a metallicity axis or fixed Fe/H metadata."""
    if isinstance(gridname, dict):
        if any(key in gridname for key in ('feh', 'z', 'metallicity')):
            return True
        if 'members' in gridname:
            return any(_grid_supports_feh(member) for member in gridname['members'])

    if isinstance(gridname, str):
        if gridname.lower() in {
            'ck_all',
            'ck03_rv',
            'ck03_cepheid_rv',
            'tlusty_all',
            'newera_alpha0',
            'newera_alpha0_rv',
        }:
            return True
        if re.fullmatch(r'ck[mp]\d{2}', gridname.lower()):
            return True
        desc = grid_description.get(gridname, {})
        if desc.get('supports_feh') or desc.get('has_feh_axis'):
            return True
        for key in ('axes', 'variables', 'parameters'):
            if 'feh' in [str(item).lower() for item in desc.get(key, [])]:
                return True
        if any(key in desc for key in ('feh', 'z', 'metallicity')):
            return True
        if 'members' in desc:
            return any(_grid_supports_feh(member) for member in desc['members'])
        if os.path.isfile(gridname):
            try:
                return 'feh' in _grid_file_capabilities(gridname)['axes']
            except Exception:
                return False

    if isinstance(gridname, dict) and 'grid' in gridname:
        return _grid_supports_feh(gridname['grid'])

    return False


def grid_has_nonrectangular_coverage(grid):
    """Return whether a grid has missing atmosphere combinations.

    NewEra is distributed on a non-rectangular Teff/logg/[Fe/H] domain.  Its
    prepared FITS representation records missing combinations as non-finite
    pixels, while its HDF5 representation stores only existing spectra.
    """
    if isinstance(grid, HDF5IntegratedGrid):
        return True
    if _is_prepared_grid(grid):
        metadata = grid[3] if len(grid) > 3 and isinstance(grid[3], dict) else {}
        if 'non_rectangular' in metadata:
            return bool(metadata['non_rectangular'])
        pixelgrid = grid[1]
        if not hasattr(pixelgrid, 'shape') or len(pixelgrid.shape) < 2:
            return False
        return not bool(np.all(np.isfinite(pixelgrid[..., 0])))
    if isinstance(grid, dict):
        if 'members' in grid:
            return any(
                grid_has_nonrectangular_coverage(member)
                for member in grid['members']
            )
        if 'grid' in grid:
            return grid_has_nonrectangular_coverage(grid['grid'])
        return bool(grid.get('non_rectangular', False))
    if isinstance(grid, str):
        desc = grid_description.get(grid, {})
        return (
            _grid_integrated_format(grid) == 'hdf5'
            or bool(desc.get('non_rectangular', False))
            or grid.lower().startswith('newera')
        )
    if hasattr(grid, '__iter__'):
        return any(grid_has_nonrectangular_coverage(item) for item in grid)
    return False


def grid_has_axis(gridname, axis):
    """Return True when a grid description or integrated file exposes an axis."""
    axis = str(axis).lower()

    if isinstance(gridname, dict):
        for key in ('axes', 'variables', 'parameters'):
            if axis in [str(item).lower() for item in gridname.get(key, [])]:
                return True
        if 'grid' in gridname:
            return grid_has_axis(gridname['grid'], axis)
        if 'members' in gridname:
            return any(grid_has_axis(member, axis) for member in gridname['members'])
        return False

    if isinstance(gridname, str):
        desc = grid_description.get(gridname, {})
        if axis == 'feh' and _grid_supports_feh(gridname):
            return True
        if axis == 'rv' and gridname in {
            'ck03_rv', 'ck03_cepheid_rv', 'newera_alpha0_rv',
        }:
            return True
        if axis == 'rv' and desc.get('supports_rv'):
            return True
        for key in ('axes', 'variables', 'parameters'):
            if axis in [str(item).lower() for item in desc.get(key, [])]:
                return True
        if os.path.isfile(gridname):
            try:
                return axis in _grid_file_capabilities(gridname)['axes']
            except Exception:
                return False

    return False


def grid_requires_feh_value(gridname):
    """Return True when a grid needs an explicit fitted or fixed Fe/H value."""
    known_metallicity_axes = {
        'ck_all',
        'ck03_rv',
        'tlusty_all',
        'newera_alpha0',
        'newera_alpha0_rv',
    }

    if isinstance(gridname, dict):
        if 'members' in gridname:
            return True
        if any(key in gridname for key in ('feh', 'z', 'metallicity')):
            return False
        if 'grid' in gridname:
            return grid_requires_feh_value(gridname['grid'])
        return False

    if isinstance(gridname, str):
        if gridname.lower() in known_metallicity_axes:
            return True
        desc = grid_description.get(gridname, {})
        if 'members' in desc:
            return True
        if desc.get('supports_feh') or desc.get('has_feh_axis'):
            return True
        for key in ('axes', 'variables', 'parameters'):
            if 'feh' in [str(item).lower() for item in desc.get(key, [])]:
                return True
        if os.path.isfile(gridname):
            try:
                return _grid_file_capabilities(gridname)['feh_varied']
            except Exception:
                return False

    return False


def _snap_range(values, requested):
    """Expand a requested range to real grid points, or return None on no overlap."""
    low_req, high_req = requested
    values = np.unique(values[np.isfinite(values)])
    if len(values) == 0:
        return None
    if np.isfinite(low_req) and high_req < values[0]:
        return None
    if np.isfinite(high_req) and low_req > values[-1]:
        return None

    if np.isfinite(low_req):
        low_candidates = values[values <= low_req]
        low = low_candidates[-1] if len(low_candidates) else values[0]
    else:
        low = values[0]

    if np.isfinite(high_req):
        high_candidates = values[values >= high_req]
        high = high_candidates[0] if len(high_candidates) else values[-1]
    else:
        high = values[-1]
    return low, high


def _range_for(name, pnames, limits, component_suffix=''):
    candidates = []
    if component_suffix and name not in ('av', 'ebv', 'rv'):
        candidates.append(name + component_suffix)
    candidates.append(name)
    for candidate in candidates:
        if candidate in pnames:
            return tuple(limits[pnames.index(candidate)])
    return (-np.inf, np.inf)


def _default_grid_variables(pnames, component_suffix='', gridname=None):
    extinction_axis = 'ebv' if 'ebv' in pnames and 'av' not in pnames else 'av'
    variables = ['teff', 'logg', extinction_axis]
    feh_names = ['feh']
    if component_suffix:
        feh_names.insert(0, 'feh' + component_suffix)
    if any(name in pnames for name in feh_names) and _grid_supports_feh(gridname):
        variables.append('feh')
    if grid_has_axis(gridname, 'rv'):
        variables.append('rv')
    return variables


def _variables_for_component(grid_variables, gridname, index, pnames, component_suffix):
    if grid_variables is None:
        return _default_grid_variables(pnames, component_suffix, gridname)
    if isinstance(grid_variables, dict):
        for key in (index, str(index), gridname):
            if key in grid_variables:
                return list(grid_variables[key])
        return _default_grid_variables(pnames, component_suffix, gridname)
    return list(grid_variables)


def _normalise_photband_name(name):
    return str(name.decode() if isinstance(name, bytes) else name)


def _axis_indices_for_range(axis, requested):
    snapped = _snap_range(axis, requested)
    if snapped is None:
        return None
    low, high = snapped
    axis = np.asarray(axis, dtype=float)
    atol = max(1e-10, 1e-8 * max(1.0, abs(axis[0]), abs(axis[-1])))
    return np.flatnonzero((axis >= low - atol) & (axis <= high + atol))


def _axis_bounds(axis, value):
    stored_axis = np.asarray(axis)
    storage_epsilon = (
        np.finfo(stored_axis.dtype).eps
        if np.issubdtype(stored_axis.dtype, np.floating)
        else np.finfo(float).eps
    )
    axis = np.asarray(stored_axis, dtype=float)
    value = float(value)
    scale = max(1.0, abs(axis[0]), abs(axis[-1]))
    # HDF5 axes may have been read into float64 after decimal values were
    # quantized as float32 (for example 6.2 -> 6.199999809...).  A 1e-7
    # relative endpoint tolerance accepts that representation error while
    # remaining far below the spacing of supported model axes.
    atol = max(1e-10, 1e-7 * scale, 4.0 * storage_epsilon * scale)
    if value < axis[0] - atol or value > axis[-1] + atol:
        raise ValueError(
            "Value {} is outside grid axis [{}, {}].".format(
                value, axis[0], axis[-1]
            )
        )
    if value <= axis[0]:
        return [(0, 1.0)]
    if value >= axis[-1]:
        return [(len(axis) - 1, 1.0)]

    upper = int(np.searchsorted(axis, value))
    lower = upper - 1
    if np.isclose(value, axis[lower], rtol=0, atol=atol):
        return [(lower, 1.0)]
    if np.isclose(value, axis[upper], rtol=0, atol=atol):
        return [(upper, 1.0)]

    frac = (value - axis[lower]) / (axis[upper] - axis[lower])
    return [(lower, 1.0 - frac), (upper, frac)]


class HDF5IntegratedGrid:
    """Lazy integrated-grid reader for grids with an explicit Rv axis."""

    spec_axis_order = ('feh', 'teff', 'logg')

    def __init__(self, path, photbands, variables=None, ranges=None,
                 preload=True, preload_max_gb=DEFAULT_HDF5_PRELOAD_MAX_GB,
                 allow_walker_cache=True, runtime_cache_dir=None):
        self.path = str(path)
        self.photbands = [_normalise_photband_name(name) for name in photbands]
        self.variables = np.array(variables or ['teff', 'logg', 'av', 'feh', 'rv'])
        self.ranges = ranges or {}
        self.preload = bool(preload)
        self.preload_max_gb = float(preload_max_gb)
        self.allow_walker_cache = bool(allow_walker_cache)
        self.runtime_cache_dir = (
            os.path.abspath(os.path.expanduser(str(runtime_cache_dir)))
            if runtime_cache_dir else None
        )
        self._handle = None
        self._active_cache = None
        self._active_caches = []
        self._runtime_cache_hit = False
        self._runtime_cache_path = None
        self._runtime_cache_invalid_path = None
        self._cache_statistics = {
            'cached_points': 0,
            'invalid_cached_points': 0,
            'fallback_points': 0,
        }

        import h5py

        with h5py.File(self.path, 'r') as h5:
            self.axes = {
                name: np.asarray(h5[f'axes/{name}'][:], dtype=float)
                for name in h5['axes']
                if name != 'photband'
            }
            self.available_photbands = [
                _normalise_photband_name(name)
                for name in h5['axes/photband'][:]
            ]
            self.spectra = {
                name: np.asarray(h5[f'spectra/{name}'][:], dtype=float)
                for name in h5['spectra']
            }

        missing = [band for band in self.photbands
                   if band not in self.available_photbands]
        if missing:
            raise ValueError(
                "Photometric passband(s) missing from HDF5 grid {}: {}".format(
                    self.path, ', '.join(missing)
                )
            )
        self.photband_indices = np.array(
            [self.available_photbands.index(band) for band in self.photbands],
            dtype=int,
        )

        for name in self.variables:
            name = str(name)
            if name in self.axes:
                snapped = _snap_range(
                    self.axes[name],
                    self.ranges.get(name, (-np.inf, np.inf)),
                )
            elif name in self.spectra:
                snapped = _snap_range(
                    self.spectra[name],
                    self.ranges.get(name, (-np.inf, np.inf)),
                )
            else:
                raise ValueError(
                    f"HDF5 grid {self.path} does not provide axis '{name}'."
                )
            if snapped is None:
                raise ValueError(
                    f"No grid points left for {self.path} after applying "
                    f"limits on '{name}'."
                )

        self.spec_axes = [
            name for name in self.spec_axis_order if name in self.spectra
        ]
        self.spec_axis_values = {
            name: np.unique(self.spectra[name][np.isfinite(self.spectra[name])])
            for name in self.spec_axes
        }
        shape = tuple(len(self.spec_axis_values[name]) for name in self.spec_axes)
        self.spec_index = -np.ones(shape, dtype=int)
        lookup = {
            name: {float(value): i for i, value in enumerate(values)}
            for name, values in self.spec_axis_values.items()
        }
        nspec = len(next(iter(self.spectra.values())))
        for ispec in range(nspec):
            index = tuple(
                lookup[name][float(self.spectra[name][ispec])]
                for name in self.spec_axes
            )
            self.spec_index[index] = ispec

        self._attach_compatible_shared_cache()
        if self.preload and not self._active_caches:
            self._preload_active_subgrid()

    def _cache_covers_active_ranges(self, cache):
        """Return whether a shared union cache covers this grid's limits."""
        for name in self.spec_axes:
            needed = _axis_indices_for_range(
                self.spec_axis_values[name],
                self.ranges.get(name, (-np.inf, np.inf)),
            )
            if needed is None or not np.all(np.isin(
                    self.spec_axis_values[name][needed], cache['spec_values'][name])):
                return False
        for name in ('rv', 'av'):
            if name not in self.axes:
                continue
            needed = _axis_indices_for_range(
                self.axes[name],
                self.ranges.get(name, (-np.inf, np.inf)),
            )
            if needed is None or not np.all(np.isin(
                    self.axes[name][needed], cache['axis_values'][name])):
                return False
        return True

    def _attach_compatible_shared_cache(self):
        if not self.allow_walker_cache:
            return False
        path = os.path.realpath(self.path)
        for entry in reversed(_SHARED_HDF5_CACHES):
            if entry['path'] != path:
                continue
            if any(band not in entry['photbands'] for band in self.photbands):
                continue
            cache = entry['cache']
            if not self._cache_covers_active_ranges(cache):
                continue
            attached = dict(cache)
            attached['band_indices'] = np.asarray(
                [entry['photbands'].index(band) for band in self.photbands],
                dtype=int,
            )
            attached['band_indices'].flags.writeable = False
            self._active_caches = [attached]
            self._active_cache = attached
            return True
        return False

    def register_active_cache_as_shared(self):
        """Expose a full union cache to compatible grids created after fork."""
        full_caches = [
            cache for cache in self._active_caches if cache['is_full_active_grid']
        ]
        if not full_caches:
            return False
        cache = full_caches[0]
        entry = {
            'path': os.path.realpath(self.path),
            'photbands': list(self.photbands),
            'cache': cache,
        }
        if not any(existing['cache'] is cache for existing in _SHARED_HDF5_CACHES):
            _SHARED_HDF5_CACHES.append(entry)
        return True

    @property
    def h5(self):
        if self._handle is None:
            import h5py
            self._handle = h5py.File(self.path, 'r')
        return self._handle

    def close(self):
        """Close a lazily opened HDF5 handle before process-level parallelism."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    @staticmethod
    def _sample_axis_indices(values, indices, count, logarithmic=False):
        """Select representative grid-axis indices, retaining both endpoints."""
        indices = np.asarray(indices, dtype=int)
        if len(indices) <= int(count):
            return indices

        selected_values = np.asarray(values, dtype=float)[indices]
        if logarithmic and np.all(selected_values >= 0):
            coordinates = np.log1p(selected_values)
        else:
            coordinates = selected_values
        targets = np.linspace(coordinates[0], coordinates[-1], int(count))
        positions = np.searchsorted(coordinates, targets)
        positions = np.clip(positions, 0, len(indices) - 1)
        previous = np.maximum(positions - 1, 0)
        use_previous = np.abs(coordinates[previous] - targets) < np.abs(
            coordinates[positions] - targets
        )
        positions[use_previous] = previous[use_previous]
        return np.unique(indices[positions])

    def _seed_spectrum_indices(self, maximum):
        """Return valid atmosphere rows inside the active parameter ranges."""
        nspec = len(self.spectra['teff'])
        keep = np.ones(nspec, dtype=bool)
        for name in self.spec_axes:
            values = np.asarray(self.spectra[name], dtype=float)
            snapped = _snap_range(
                values,
                self.ranges.get(name, (-np.inf, np.inf)),
            )
            if snapped is None:
                return np.array([], dtype=int)
            low, high = snapped
            keep &= np.isfinite(values) & (values >= low) & (values <= high)

        indices = np.flatnonzero(keep)
        if len(indices) <= int(maximum):
            return indices

        # HDF5 spectra are ordered by their atmosphere axes. Even spacing in
        # row index preserves a deterministic broad coverage at large scale.
        positions = np.linspace(0, len(indices) - 1, int(maximum), dtype=int)
        return np.unique(indices[positions])

    def _seed_axis_indices(self, name):
        if name not in self.axes:
            return np.array([0], dtype=int)
        indices = _axis_indices_for_range(
            self.axes[name],
            self.ranges.get(name, (-np.inf, np.inf)),
        )
        return np.array([], dtype=int) if indices is None else indices

    @staticmethod
    def _profile_scale_and_chi2(flux, obs, weights):
        """Profile the positive flux normalization for a set of model vectors."""
        flux = np.asarray(flux, dtype=float)
        valid = np.all(np.isfinite(flux) & (flux > 0), axis=1)
        scale = np.full(len(flux), np.nan, dtype=float)
        chi2 = np.full(len(flux), np.inf, dtype=float)
        if not np.any(valid):
            return scale, chi2

        values = flux[valid]
        numerator = np.sum(values * (obs * weights), axis=1)
        denominator = np.sum(values ** 2 * weights, axis=1)
        usable = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
        profile_scale = np.full(len(values), np.nan, dtype=float)
        profile_scale[usable] = numerator[usable] / denominator[usable]
        usable &= profile_scale > 0
        residual = values[usable] * profile_scale[usable, None] - obs
        profile_chi2 = np.sum(residual ** 2 * weights, axis=1)

        valid_indices = np.flatnonzero(valid)
        usable_indices = valid_indices[usable]
        scale[usable_indices] = profile_scale[usable]
        chi2[usable_indices] = profile_chi2
        return scale, chi2

    def _profile_cache_mapping(self, spectrum_indices, rv_indices, av_indices):
        """Map HDF5 node indices into an existing read-only flux cache."""
        spectrum_indices = np.asarray(spectrum_indices, dtype=int)
        rv_indices = np.asarray(rv_indices, dtype=int)
        av_indices = np.asarray(av_indices, dtype=int)
        for cache in self._active_caches:
            spectrum_rows = cache['spectrum_row_lookup'][spectrum_indices]
            if np.any(spectrum_rows < 0):
                continue

            local_axes = []
            covered = True
            for name, global_indices in (('rv', rv_indices), ('av', av_indices)):
                lookup = cache['axis_row_lookup'].get(name)
                if lookup is None:
                    local = np.zeros(len(global_indices), dtype=int)
                else:
                    local = lookup[global_indices]
                if np.any(local < 0):
                    covered = False
                    break
                local_axes.append(local)
            if covered:
                return cache, spectrum_rows, local_axes[0], local_axes[1]
        return None

    def _append_profile_candidates(self, candidates, spectrum_indices,
                                   rv_indices, av_indices, obs, weights,
                                   keep_count, chunk_size=2048,
                                   unique_spectra=False):
        """Scan selected HDF5 nodes and retain only the best profile matches."""
        spectrum_indices = np.unique(np.asarray(spectrum_indices, dtype=int))
        if len(spectrum_indices) == 0:
            return candidates

        cache_mapping = self._profile_cache_mapping(
            spectrum_indices, rv_indices, av_indices,
        )
        dset = None if cache_mapping is not None else self.h5['flux']
        cache = cache_spectrum_rows = cache_rv = cache_av = None
        if cache_mapping is not None:
            cache, cache_spectrum_rows, cache_rv, cache_av = cache_mapping
            spectrum_row_lookup = {
                int(ispec): int(cache_row)
                for ispec, cache_row in zip(spectrum_indices, cache_spectrum_rows)
            }
            rv_row_lookup = {
                int(global_index): int(local_index)
                for global_index, local_index in zip(rv_indices, cache_rv)
            }
            av_row_lookup = {
                int(global_index): int(local_index)
                for global_index, local_index in zip(av_indices, cache_av)
            }
        for rv_index in np.asarray(rv_indices, dtype=int):
            for av_index in np.asarray(av_indices, dtype=int):
                for start in range(0, len(spectrum_indices), int(chunk_size)):
                    chunk = spectrum_indices[start:start + int(chunk_size)]
                    if cache is None:
                        rows = np.asarray(
                            dset[chunk, int(rv_index), int(av_index), :],
                            dtype=float,
                        )[:, self.photband_indices]
                    else:
                        cached_rows = np.array(
                            [spectrum_row_lookup[int(ispec)] for ispec in chunk],
                            dtype=int,
                        )
                        log_rows = cache['log_flux'][
                            cached_rows,
                            rv_row_lookup[int(rv_index)],
                            av_row_lookup[int(av_index)],
                            :,
                        ]
                        log_rows = log_rows[:, cache['band_indices']]
                        rows = 10.0 ** np.asarray(log_rows, dtype=float)
                    scales, chi2 = self._profile_scale_and_chi2(rows, obs, weights)
                    count = min(int(keep_count), len(chunk))
                    if count == 0:
                        continue
                    selected = np.argpartition(chi2, count - 1)[:count]
                    for local_index in selected:
                        if not np.isfinite(chi2[local_index]):
                            continue
                        ispec = int(chunk[local_index])
                        candidate = {
                            'spec_index': ispec,
                            'rv_index': int(rv_index),
                            'av_index': int(av_index),
                            'scale': float(scales[local_index]),
                            'profile_chi2': float(chi2[local_index]),
                        }
                        for name in self.spec_axes:
                            candidate[name] = float(self.spectra[name][ispec])
                        if 'rv' in self.axes:
                            candidate['rv'] = float(self.axes['rv'][rv_index])
                        if 'av' in self.axes:
                            candidate['av'] = float(self.axes['av'][av_index])
                        candidates.append(candidate)

                candidates.sort(key=lambda item: item['profile_chi2'])
                if unique_spectra:
                    retained = []
                    seen = set()
                    for candidate in candidates:
                        ispec = int(candidate['spec_index'])
                        if ispec in seen:
                            continue
                        seen.add(ispec)
                        retained.append(candidate)
                        if len(retained) >= int(keep_count):
                            break
                    candidates[:] = retained
                else:
                    del candidates[int(keep_count):]
        return candidates

    def _append_refined_profile_candidates(self, candidates, spectrum_indices,
                                           rv_indices, av_indices, obs, weights,
                                           keep_count, chunk_size=8):
        """Scan small full Rv-Av cubes with batched HDF5 reads.

        The refinement uses few atmosphere spectra but every active extinction
        node. Reading a whole small cube avoids hundreds of thousands of tiny
        HDF5 slice operations while evaluating exactly the same discrete nodes.
        """
        spectrum_indices = np.unique(np.asarray(spectrum_indices, dtype=int))
        rv_indices = np.asarray(rv_indices, dtype=int)
        av_indices = np.asarray(av_indices, dtype=int)
        if len(spectrum_indices) == 0 or len(rv_indices) == 0 or len(av_indices) == 0:
            return candidates

        cache_mapping = self._profile_cache_mapping(
            spectrum_indices, rv_indices, av_indices,
        )
        dset = None if cache_mapping is not None else self.h5['flux']
        cache = cache_spectrum_rows = cache_rv = cache_av = None
        if cache_mapping is not None:
            cache, cache_spectrum_rows, cache_rv, cache_av = cache_mapping
        for start in range(0, len(spectrum_indices), int(chunk_size)):
            chunk = spectrum_indices[start:start + int(chunk_size)]
            if cache is None:
                cube = np.asarray(dset[chunk, :, :, :], dtype=float)
                cube = cube[:, rv_indices, :, :]
                cube = cube[:, :, av_indices, :]
                rows = cube[..., self.photband_indices].reshape(-1, len(self.photbands))
            else:
                cached_rows = cache_spectrum_rows[start:start + len(chunk)]
                log_cube = cache['log_flux'][cached_rows]
                log_cube = log_cube[:, cache_rv, :, :]
                log_cube = log_cube[:, :, cache_av, :]
                log_cube = log_cube[..., cache['band_indices']]
                rows = (10.0 ** np.asarray(log_cube, dtype=float)).reshape(
                    -1, len(self.photbands)
                )
            scales, chi2 = self._profile_scale_and_chi2(rows, obs, weights)
            count = min(int(keep_count), len(chi2))
            if count == 0:
                continue

            selected = np.argpartition(chi2, count - 1)[:count]
            for flat_index in selected:
                if not np.isfinite(chi2[flat_index]):
                    continue
                local_spec, local_rv, local_av = np.unravel_index(
                    int(flat_index),
                    (len(chunk), len(rv_indices), len(av_indices)),
                )
                ispec = int(chunk[local_spec])
                rv_index = int(rv_indices[local_rv])
                av_index = int(av_indices[local_av])
                candidate = {
                    'spec_index': ispec,
                    'rv_index': rv_index,
                    'av_index': av_index,
                    'scale': float(scales[flat_index]),
                    'profile_chi2': float(chi2[flat_index]),
                }
                for name in self.spec_axes:
                    candidate[name] = float(self.spectra[name][ispec])
                if 'rv' in self.axes:
                    candidate['rv'] = float(self.axes['rv'][rv_index])
                if 'av' in self.axes:
                    candidate['av'] = float(self.axes['av'][av_index])
                candidates.append(candidate)

            candidates.sort(key=lambda item: item['profile_chi2'])
            del candidates[int(keep_count):]
        return candidates

    def profile_seed_candidates(self, obs, obs_err, maximum_spectra=12000,
                                coarse_rv_points=7, coarse_av_points=11,
                                coarse_keep_count=256, refine_spectra=64,
                                result_count=256):
        """Return high-likelihood discrete seeds without gradient optimization.

        The first pass profiles the flux normalization over all active
        atmosphere nodes and a sparse extinction grid. The second pass scans
        the complete Av/Rv axes around the most promising atmosphere spectra.
        These candidates are for MCMC initialization only; posterior sampling
        continues to use the full interpolated model likelihood.
        """
        obs = np.asarray(obs, dtype=float)
        obs_err = np.asarray(obs_err, dtype=float)
        valid = np.isfinite(obs) & np.isfinite(obs_err) & (obs > 0) & (obs_err > 0)
        if not np.all(valid):
            obs, obs_err = obs[valid], obs_err[valid]
        if len(obs) != len(self.photbands):
            raise ValueError("Grid seed search requires finite positive observations in every fitted band.")
        weights = 1.0 / obs_err ** 2

        spectrum_indices = self._seed_spectrum_indices(maximum_spectra)
        rv_indices = self._seed_axis_indices('rv')
        av_indices = self._seed_axis_indices('av')
        if len(spectrum_indices) == 0 or len(rv_indices) == 0 or len(av_indices) == 0:
            return []

        coarse_rv = self._sample_axis_indices(
            self.axes.get('rv', np.array([0.0])),
            rv_indices,
            coarse_rv_points,
        )
        coarse_av = self._sample_axis_indices(
            self.axes.get('av', np.array([0.0])),
            av_indices,
            coarse_av_points,
            logarithmic=True,
        )
        candidates = self._append_profile_candidates(
            [],
            spectrum_indices,
            coarse_rv,
            coarse_av,
            obs,
            weights,
            keep_count=coarse_keep_count,
            unique_spectra=True,
        )
        if not candidates:
            return []

        refined_indices = np.array(
            list(dict.fromkeys(candidate['spec_index'] for candidate in candidates))[:int(refine_spectra)],
            dtype=int,
        )
        candidates = self._append_refined_profile_candidates(
            candidates,
            refined_indices,
            rv_indices,
            av_indices,
            obs,
            weights,
            keep_count=result_count,
        )
        candidates.sort(key=lambda item: item['profile_chi2'])
        return candidates[:int(result_count)]

    def profile_continuous_seed_candidate(self, obs, obs_err, maxiter=80,
                                          popsize=12, seed=20260711):
        """Find a continuous profile-likelihood seed with global DE search.

        This is a rescue path for cases where the fast discrete atmosphere
        scan has an implausibly poor fit.  Radius/distance normalization is
        still profiled analytically, so the optimizer only explores physical
        integrated-grid coordinates.  The method is derivative-free and is
        intended to run against an in-memory full active-subgrid cache.
        """
        from scipy.optimize import differential_evolution

        obs = np.asarray(obs, dtype=float)
        obs_err = np.asarray(obs_err, dtype=float)
        weights = 1.0 / obs_err ** 2
        search_names = [
            str(name) for name in self.variables
            if str(name) in self.spec_axes or str(name) in self.axes
        ]
        all_bounds = {}
        for name in search_names:
            axis = self.spec_axis_values[name] if name in self.spec_axes else self.axes[name]
            allowed = self.ranges.get(name, (-np.inf, np.inf))
            low = max(float(axis[0]), float(allowed[0]))
            high = min(float(axis[-1]), float(allowed[1]))
            if low > high:
                return None
            all_bounds[name] = (low, high)

        variable_names = [
            name for name in search_names
            if all_bounds[name][1] > all_bounds[name][0]
        ]
        fixed_values = {
            name: all_bounds[name][0]
            for name in search_names if name not in variable_names
        }

        def objective(points):
            points = np.asarray(points, dtype=float)
            if points.ndim == 1:
                points = points[:, None]
            values = {
                name: points[index]
                for index, name in enumerate(variable_names)
            }
            npoint = points.shape[1]
            values.update({
                name: np.full(npoint, value, dtype=float)
                for name, value in fixed_values.items()
            })
            flux, _labs = self.evaluate(**values)
            flux = np.asarray(flux, dtype=float)
            if flux.ndim == 1:
                flux = flux[:, None]
            _scales, chi2 = self._profile_scale_and_chi2(
                flux.T, obs, weights,
            )
            return chi2

        if variable_names:
            result = differential_evolution(
                objective,
                [all_bounds[name] for name in variable_names],
                maxiter=int(maxiter),
                popsize=int(popsize),
                seed=int(seed),
                polish=False,
                updating='deferred',
                vectorized=True,
            )
            if not np.isfinite(result.fun):
                return None
            values = dict(fixed_values)
            values.update({
                name: float(value)
                for name, value in zip(variable_names, result.x)
            })
        else:
            values = dict(fixed_values)

        flux, _labs = self.evaluate(**values)
        scales, chi2 = self._profile_scale_and_chi2(
            np.atleast_2d(np.asarray(flux, dtype=float)), obs, weights,
        )
        if len(chi2) == 0 or not np.isfinite(chi2[0]):
            return None
        candidate = dict(values)
        candidate['scale'] = float(scales[0])
        candidate['profile_chi2'] = float(chi2[0])
        return candidate

    def _prepare_inputs(self, values_by_name):
        arrays = {}
        npoint = 1
        scalar_input = True
        for name in self.variables:
            value = values_by_name.get(str(name))
            if value is None:
                raise ValueError(f"Missing model parameter '{name}' for grid interpolation.")
            raw = np.asarray(value, dtype=float)
            if raw.ndim > 0:
                scalar_input = False
            arr = np.atleast_1d(raw)
            if arr.ndim > 1:
                arr = arr.reshape(-1)
            npoint = max(npoint, len(arr))
            arrays[str(name)] = arr

        for name, arr in list(arrays.items()):
            if len(arr) == 1 and npoint > 1:
                arrays[name] = np.full(npoint, float(arr[0]))
            elif len(arr) != npoint:
                raise ValueError(
                    f"Parameter '{name}' has {len(arr)} values; expected 1 or {npoint}."
                )
        return arrays, npoint, scalar_input

    def _corner_bounds(self, arrays, ipoint):
        bounds = {}
        for name in self.spec_axes:
            bounds[name] = _axis_bounds(self.spec_axis_values[name], arrays[name][ipoint])
        for name in ('rv', 'av'):
            if name in self.axes:
                bounds[name] = _axis_bounds(self.axes[name], arrays[name][ipoint])
        return bounds

    def _preload_axes(self, ranges=None):
        ranges = self.ranges if ranges is None else ranges
        spec_indices = {}
        spec_values = {}
        for name in self.spec_axes:
            values = self.spec_axis_values[name]
            indices = _axis_indices_for_range(
                values,
                ranges.get(name, (-np.inf, np.inf)),
            )
            if indices is None or len(indices) == 0:
                return None
            spec_indices[name] = indices
            spec_values[name] = values[indices]

        axis_indices = {}
        axis_values = {}
        for name in ('rv', 'av'):
            if name not in self.axes:
                continue
            indices = _axis_indices_for_range(
                self.axes[name],
                ranges.get(name, (-np.inf, np.inf)),
            )
            if indices is None or len(indices) == 0:
                return None
            axis_indices[name] = indices
            axis_values[name] = self.axes[name][indices]

        return spec_indices, spec_values, axis_indices, axis_values

    def _subgrid_estimate_gb(self, ranges):
        layout = self._subgrid_layout(ranges)
        if layout is None:
            return None
        _axes, spec_shape, spec_records = layout
        _spec_indices, _spec_values, _axis_indices, axis_values = _axes
        rv_len = len(axis_values.get('rv', np.array([0.0])))
        av_len = len(axis_values.get('av', np.array([0.0])))
        nrow = len(spec_records)
        flux_bytes = nrow * rv_len * av_len * len(self.photbands) * np.dtype('f4').itemsize
        lookup_bytes = int(np.prod(spec_shape, dtype=np.int64)) * np.dtype('i4').itemsize
        lookup_bytes += len(self.spectra['teff']) * np.dtype('i4').itemsize
        lookup_bytes += sum(len(self.axes[name]) * np.dtype('i4').itemsize
                            for name in ('rv', 'av') if name in self.axes)
        labs_bytes = nrow * np.dtype('f4').itemsize
        return (flux_bytes + lookup_bytes + labs_bytes) / 1024.0 ** 3

    def _subgrid_layout(self, ranges):
        """Describe only valid atmosphere rows inside a requested subgrid.

        Atmosphere grids are frequently non-rectangular.  Keeping one flux
        cube row per real spectrum avoids allocating the much larger Cartesian
        product of all Teff, logg, and metallicity axis values.
        """
        axes = self._preload_axes(ranges=ranges)
        if axes is None:
            return None
        spec_indices, spec_values, _axis_indices, _axis_values = axes
        spec_shape = tuple(len(spec_values[name]) for name in self.spec_axes)
        spec_local_lookup = {
            name: {
                int(global_index): local_index
                for local_index, global_index in enumerate(spec_indices[name])
            }
            for name in self.spec_axes
        }
        spec_records = []
        for global_spec_index in product(*[spec_indices[name] for name in self.spec_axes]):
            spec_index = tuple(int(index) for index in global_spec_index)
            ispec = int(self.spec_index[spec_index])
            if ispec < 0:
                continue
            local_spec_index = tuple(
                spec_local_lookup[name][int(index)]
                for name, index in zip(self.spec_axes, spec_index)
            )
            spec_records.append((ispec, local_spec_index))
        spec_records.sort(key=lambda item: item[0])
        return axes, spec_shape, spec_records

    @staticmethod
    def _cache_range_key(ranges):
        if ranges is None:
            return ('full_active_grid',)
        return tuple(
            (str(name), float(bounds[0]), float(bounds[1]))
            for name, bounds in sorted(ranges.items())
        )

    def _runtime_cache_identity(self):
        if self.runtime_cache_dir is None:
            return None, None
        source = os.stat(self.path)
        payload = {
            'format_version': RUNTIME_CACHE_FORMAT_VERSION,
            'source_path': os.path.realpath(self.path),
            'source_size': int(source.st_size),
            'source_mtime_ns': int(source.st_mtime_ns),
            'photbands': list(self.photbands),
            'variables': [str(name) for name in self.variables],
            'ranges': repr(_freeze_for_cache(self.ranges)),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
        digest = hashlib.sha256(encoded).hexdigest()[:24]
        return payload, os.path.join(self.runtime_cache_dir, 'hdf5_' + digest)

    @staticmethod
    def _runtime_array_path(directory, name):
        return os.path.join(directory, name + '.npy')

    def _load_runtime_cache(self):
        identity, directory = self._runtime_cache_identity()
        if directory is None or not os.path.isdir(directory):
            return None
        try:
            with open(os.path.join(directory, 'metadata.json')) as handle:
                metadata = json.load(handle)
            if metadata.get('identity') != identity:
                self._runtime_cache_invalid_path = directory
                return None

            def load(name):
                return np.load(
                    self._runtime_array_path(directory, name),
                    mmap_mode='r',
                    allow_pickle=False,
                )
            spec_names = metadata['spec_names']
            axis_names = metadata['axis_names']
            cache = {
                'spec_values': {name: load('spec_values_' + name) for name in spec_names},
                'axis_values': {name: load('axis_values_' + name) for name in axis_names},
                'log_flux': load('log_flux'),
                'log_labs': load('log_labs'),
                'spec_row_index': load('spec_row_index'),
                'spectrum_row_lookup': load('spectrum_row_lookup'),
                'axis_row_lookup': {
                    name: load('axis_row_lookup_' + name) for name in axis_names
                },
                'band_indices': load('band_indices'),
                'estimate_gb': float(metadata['estimate_gb']),
                'is_full_active_grid': True,
                'range_key': self._cache_range_key(None),
                'persistent': True,
            }
            expected_shape = tuple(metadata['log_flux_shape'])
            if cache['log_flux'].shape != expected_shape:
                self._runtime_cache_invalid_path = directory
                return None
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            self._runtime_cache_invalid_path = directory
            return None

        self._runtime_cache_hit = True
        self._runtime_cache_path = directory
        return cache

    def _write_runtime_cache(self, cache):
        identity, directory = self._runtime_cache_identity()
        if directory is None:
            return False
        replace_invalid = (
            os.path.isdir(directory)
            and self._runtime_cache_invalid_path == directory
        )
        if os.path.isdir(directory) and not replace_invalid:
            return False
        os.makedirs(self.runtime_cache_dir, exist_ok=True)
        temporary = tempfile.mkdtemp(prefix='.sedforge-cache-', dir=self.runtime_cache_dir)
        try:
            def save(name, values):
                np.save(
                    self._runtime_array_path(temporary, name),
                    np.asarray(values),
                    allow_pickle=False,
                )

            save('log_flux', cache['log_flux'])
            save('log_labs', cache['log_labs'])
            save('spec_row_index', cache['spec_row_index'])
            save('spectrum_row_lookup', cache['spectrum_row_lookup'])
            save('band_indices', cache['band_indices'])
            for name, values in cache['spec_values'].items():
                save('spec_values_' + name, values)
            for name, values in cache['axis_values'].items():
                save('axis_values_' + name, values)
                save('axis_row_lookup_' + name, cache['axis_row_lookup'][name])

            metadata = {
                'identity': identity,
                'spec_names': list(cache['spec_values']),
                'axis_names': list(cache['axis_values']),
                'estimate_gb': float(cache['estimate_gb']),
                'log_flux_shape': list(cache['log_flux'].shape),
            }
            with open(os.path.join(temporary, 'metadata.json'), 'w') as handle:
                json.dump(metadata, handle, sort_keys=True)
            backup = None
            if replace_invalid and os.path.isdir(directory):
                backup = directory + '.invalid-{}'.format(os.getpid())
                os.rename(directory, backup)
            try:
                os.rename(temporary, directory)
            except FileExistsError:
                if backup is not None and not os.path.exists(directory):
                    os.rename(backup, directory)
                return False
            if backup is not None and os.path.isdir(backup):
                shutil.rmtree(backup)
            self._runtime_cache_path = directory
            self._runtime_cache_invalid_path = None
            return True
        finally:
            if os.path.isdir(temporary):
                shutil.rmtree(temporary)

    def _preload_active_subgrid(self, ranges=None, append=False, max_gb=None):
        if ranges is None and not append:
            persistent_cache = self._load_runtime_cache()
            if persistent_cache is not None:
                self._active_caches = [persistent_cache]
                self._active_cache = persistent_cache
                print(
                    "Memory-mapped persistent HDF5 cache for {}: shape {}, {:.3f} GB.".format(
                        os.path.basename(self.path),
                        persistent_cache['log_flux'].shape,
                        persistent_cache['estimate_gb'],
                    )
                )
                return
        layout = self._subgrid_layout(ranges)
        if layout is None:
            return
        axes, spec_shape, spec_records = layout
        spec_indices, spec_values, axis_indices, axis_values = axes
        rv_len = len(axis_values.get('rv', np.array([0.0])))
        av_len = len(axis_values.get('av', np.array([0.0])))
        nband = len(self.photbands)
        nrow = len(spec_records)
        flux_bytes = nrow * rv_len * av_len * nband * np.dtype('f4').itemsize
        lookup_bytes = int(np.prod(spec_shape, dtype=np.int64)) * np.dtype('i4').itemsize
        lookup_bytes += len(self.spectra['teff']) * np.dtype('i4').itemsize
        lookup_bytes += sum(len(self.axes[name]) * np.dtype('i4').itemsize
                            for name in ('rv', 'av') if name in self.axes)
        labs_bytes = nrow * np.dtype('f4').itemsize
        estimate_gb = (flux_bytes + lookup_bytes + labs_bytes) / 1024.0 ** 3
        limit_gb = self.preload_max_gb if max_gb is None else float(max_gb)
        if estimate_gb > limit_gb:
            print(
                "HDF5 grid preload skipped for {}: active subgrid is {:.2f} GB "
                "(limit {:.2f} GB).".format(
                    os.path.basename(self.path),
                    estimate_gb,
                    limit_gb,
                )
            )
            return

        log_flux = np.full((nrow, rv_len, av_len, nband), np.nan, dtype=np.float32)
        log_labs = np.full(nrow, np.nan, dtype=np.float32)
        spec_row_index = np.full(spec_shape, -1, dtype=np.int32)
        spectrum_row_lookup = np.full(len(self.spectra['teff']), -1, dtype=np.int32)
        axis_row_lookup = {}
        for name in ('rv', 'av'):
            if name not in self.axes:
                continue
            lookup = np.full(len(self.axes[name]), -1, dtype=np.int32)
            lookup[axis_indices[name]] = np.arange(len(axis_indices[name]), dtype=np.int32)
            lookup.flags.writeable = False
            axis_row_lookup[name] = lookup
        rv_index = axis_indices.get('rv', np.array([0], dtype=int))
        av_index = axis_indices.get('av', np.array([0], dtype=int))
        rv_slice = slice(int(rv_index[0]), int(rv_index[-1]) + 1)
        av_slice = slice(int(av_index[0]), int(av_index[-1]) + 1)
        band_order = np.argsort(self.photband_indices)
        sorted_bands = self.photband_indices[band_order]
        restore_band_order = np.argsort(band_order)
        indexed_records = []
        for cache_row, (ispec, local_spec_index) in enumerate(spec_records):
            spec_row_index[local_spec_index] = cache_row
            spectrum_row_lookup[ispec] = cache_row
            labs_value = float(self.spectra['Labs'][ispec])
            if labs_value > 0 and np.isfinite(labs_value):
                log_labs[cache_row] = np.log10(labs_value)
            indexed_records.append((ispec, cache_row))

        spec_blocks = []
        for ispec, cache_row in indexed_records:
            if not spec_blocks or ispec != spec_blocks[-1]['stop']:
                spec_blocks.append({
                    'start': ispec,
                    'stop': ispec + 1,
                    'rows': [cache_row],
                })
            else:
                spec_blocks[-1]['stop'] = ispec + 1
                spec_blocks[-1]['rows'].append(cache_row)

        import h5py

        with h5py.File(self.path, 'r') as h5:
            dset = h5['flux']
            for block in spec_blocks:
                rows = np.asarray(
                    dset[
                        slice(block['start'], block['stop']),
                        rv_slice,
                        av_slice,
                        sorted_bands,
                    ],
                    dtype=np.float32,
                )
                rows = rows[:, :, :, restore_band_order]
                with np.errstate(invalid='ignore', divide='ignore'):
                    rows = np.log10(rows)
                rows[~np.isfinite(rows)] = np.nan
                for irow, cache_row in enumerate(block['rows']):
                    log_flux[cache_row] = rows[irow]

        log_flux.flags.writeable = False
        log_labs.flags.writeable = False
        spec_row_index.flags.writeable = False
        spectrum_row_lookup.flags.writeable = False
        band_indices = np.arange(nband, dtype=int)
        band_indices.flags.writeable = False
        cache = {
            'spec_values': spec_values,
            'axis_values': axis_values,
            'log_flux': log_flux,
            'log_labs': log_labs,
            'spec_row_index': spec_row_index,
            'spectrum_row_lookup': spectrum_row_lookup,
            'axis_row_lookup': axis_row_lookup,
            'band_indices': band_indices,
            'estimate_gb': estimate_gb,
            'is_full_active_grid': ranges is None,
            'range_key': self._cache_range_key(ranges),
            'persistent': False,
        }
        if append:
            self._active_caches.append(cache)
        else:
            self._active_caches = [cache]
        self._active_cache = self._active_caches[0]
        if ranges is None and not append:
            self._write_runtime_cache(cache)
        print(
            "Preloaded HDF5 active subgrid for {}: shape {}, {:.3f} GB.".format(
                os.path.basename(self.path),
                log_flux.shape,
                estimate_gb,
            )
        )

    @staticmethod
    def _neighborhood_range(axis, values, allowed_range, padding):
        """Return a grid-node window around finite parameter values."""
        axis = np.asarray(axis, dtype=float)
        indices = _axis_indices_for_range(axis, allowed_range)
        if indices is None or len(indices) == 0:
            return None
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return None

        low_value = max(float(np.min(values)), float(axis[indices[0]]))
        high_value = min(float(np.max(values)), float(axis[indices[-1]]))
        low_index = int(np.searchsorted(axis, low_value, side='right') - 1)
        high_index = int(np.searchsorted(axis, high_value, side='left'))
        low_index = max(int(indices[0]), low_index - int(padding))
        high_index = min(int(indices[-1]), high_index + int(padding))
        if low_index > high_index:
            return None
        return float(axis[low_index]), float(axis[high_index])

    def _neighborhood_ranges(self, values_by_name, padding):
        ranges = {}
        for name in self.spec_axes:
            if name not in values_by_name:
                return None
            window = self._neighborhood_range(
                self.spec_axis_values[name],
                values_by_name[name],
                self.ranges.get(name, (-np.inf, np.inf)),
                padding,
            )
            if window is None:
                return None
            ranges[name] = window
        for name in ('rv', 'av'):
            if name not in self.axes:
                continue
            if name not in values_by_name:
                return None
            window = self._neighborhood_range(
                self.axes[name],
                values_by_name[name],
                self.ranges.get(name, (-np.inf, np.inf)),
                padding,
            )
            if window is None:
                return None
            ranges[name] = window
        return ranges

    def preload_neighborhoods(self, neighborhoods, padding=1,
                              max_total_gb=None):
        """Preload one small cache per posterior mode without restricting it.

        The caches are read-only accelerators. ``evaluate`` falls back to the
        original HDF5 interpolation when a proposed point is outside every
        cache or lies at a non-rectangular missing atmosphere node.
        """
        if not self.allow_walker_cache:
            return False
        if any(cache['is_full_active_grid'] for cache in self._active_caches):
            return True

        created = 0
        current_gb = sum(cache['estimate_gb'] for cache in self._active_caches)
        existing_keys = {cache['range_key'] for cache in self._active_caches}
        for values_by_name in neighborhoods:
            ranges = self._neighborhood_ranges(values_by_name, padding)
            if ranges is None:
                continue
            range_key = self._cache_range_key(ranges)
            if range_key in existing_keys:
                continue
            estimate_gb = self._subgrid_estimate_gb(ranges)
            if estimate_gb is None:
                continue
            if max_total_gb is not None and current_gb + estimate_gb > float(max_total_gb):
                continue
            before = len(self._active_caches)
            self._preload_active_subgrid(ranges=ranges, append=True)
            if len(self._active_caches) > before:
                created += 1
                current_gb += estimate_gb
                existing_keys.add(range_key)
        return created > 0

    def preload_mode_envelope(self, neighborhoods, padding=1, max_gb=2.0):
        """Preload one bounded envelope covering all retained seed modes."""
        if not self.allow_walker_cache or max_gb is None or float(max_gb) <= 0:
            return False
        if any(cache['is_full_active_grid'] for cache in self._active_caches):
            return True

        ranges_list = [
            self._neighborhood_ranges(values_by_name, padding)
            for values_by_name in neighborhoods
        ]
        ranges_list = [ranges for ranges in ranges_list if ranges is not None]
        if not ranges_list:
            return False
        names = set().union(*(ranges.keys() for ranges in ranges_list))
        envelope = {
            name: (
                min(ranges[name][0] for ranges in ranges_list),
                max(ranges[name][1] for ranges in ranges_list),
            )
            for name in names
        }
        estimate_gb = self._subgrid_estimate_gb(envelope)
        if estimate_gb is None or estimate_gb > float(max_gb):
            return False
        self._preload_active_subgrid(
            ranges=envelope,
            append=False,
            max_gb=max_gb,
        )
        return bool(self._active_caches)

    def preload_full_active_subgrid(self, max_gb=2.0):
        """Promote to a full active-subgrid cache only within an explicit cap."""
        if not self.allow_walker_cache or max_gb is None or float(max_gb) <= 0:
            return False
        if any(cache['is_full_active_grid'] for cache in self._active_caches):
            return True
        estimate_gb = self._subgrid_estimate_gb(None)
        if estimate_gb is None or estimate_gb > float(max_gb):
            return False
        self._preload_active_subgrid(ranges=None, append=False, max_gb=max_gb)
        return bool(self._active_caches)

    def preload_neighborhood(self, values_by_name, padding=1):
        """Compatibility wrapper for a one-mode walker cache."""
        return self.preload_neighborhoods([values_by_name], padding=padding)

    def cache_diagnostics(self):
        """Return read-cache metadata for single-process performance checks."""
        diagnostics = dict(self._cache_statistics)
        diagnostics['active'] = bool(self._active_caches)
        diagnostics['cache_count'] = len(self._active_caches)
        diagnostics['estimate_gb'] = float(sum(
            cache['estimate_gb'] for cache in self._active_caches
        ))
        diagnostics['shapes'] = [
            list(cache['log_flux'].shape) for cache in self._active_caches
        ]
        diagnostics['runtime_cache_hit'] = bool(self._runtime_cache_hit)
        diagnostics['runtime_cache_path'] = self._runtime_cache_path
        return diagnostics

    def reset_cache_statistics(self):
        """Reset per-fit counters while retaining reusable read-only caches."""
        for name in self._cache_statistics:
            self._cache_statistics[name] = 0

    @staticmethod
    def _cache_covers(cache, arrays):
        axes = {}
        axes.update(cache['spec_values'])
        axes.update(cache['axis_values'])
        for name, axis in axes.items():
            values = np.asarray(arrays[name], dtype=float)
            atol = max(1e-10, 1e-8 * max(1.0, abs(axis[0]), abs(axis[-1])))
            if np.any(values < axis[0] - atol) or np.any(values > axis[-1] + atol):
                return False
        return True

    def _corner_bounds_cached(self, cache, arrays, ipoint):
        bounds = {}
        for name in self.spec_axes:
            bounds[name] = _axis_bounds(cache['spec_values'][name], arrays[name][ipoint])
        for name in ('rv', 'av'):
            if name in cache['axis_values']:
                bounds[name] = _axis_bounds(cache['axis_values'][name], arrays[name][ipoint])
        return bounds

    def _evaluate_cached(self, cache, arrays, npoint, scalar_input):
        nband = len(self.photbands)
        flux = np.empty((nband, npoint), dtype=float)
        labs = np.empty(npoint, dtype=float)
        log_flux_grid = cache['log_flux']
        log_labs_grid = cache['log_labs']
        spec_row_index = cache['spec_row_index']

        for ipoint in range(npoint):
            bounds = self._corner_bounds_cached(cache, arrays, ipoint)
            spec_items = [bounds[name] for name in self.spec_axes]
            rv_items = bounds.get('rv', [(0, 1.0)])
            av_items = bounds.get('av', [(0, 1.0)])

            log_flux = np.zeros(nband, dtype=float)
            log_labs = 0.0
            total_weight = 0.0

            for spec_corner in product(*spec_items):
                spec_indices = tuple(item[0] for item in spec_corner)
                spec_weight = float(np.prod([item[1] for item in spec_corner]))
                if spec_weight == 0.0:
                    continue
                cache_row = int(spec_row_index[spec_indices])
                if cache_row < 0:
                    continue
                labs_value = float(log_labs_grid[cache_row])
                if not np.isfinite(labs_value):
                    continue

                for rv_index, rv_weight in rv_items:
                    for av_index, av_weight in av_items:
                        weight = spec_weight * rv_weight * av_weight
                        if weight == 0.0:
                            continue
                        row = np.asarray(
                            log_flux_grid[cache_row, rv_index, av_index, :],
                            dtype=float,
                        )
                        row = row[cache['band_indices']]
                        if not np.all(np.isfinite(row)):
                            continue
                        log_flux += weight * row
                        log_labs += weight * labs_value
                        total_weight += weight

            if total_weight <= 0:
                flux[:, ipoint] = np.nan
                labs[ipoint] = np.nan
            else:
                log_flux /= total_weight
                log_labs /= total_weight
                flux[:, ipoint] = 10.0 ** log_flux
                labs[ipoint] = 10.0 ** log_labs

        if scalar_input:
            return flux[:, 0], labs[:1]
        return flux, labs

    def _evaluate_uncached(self, arrays, npoint, scalar_input):

        nband = len(self.photbands)
        flux = np.empty((nband, npoint), dtype=float)
        labs = np.empty(npoint, dtype=float)
        dset = self.h5['flux']

        for ipoint in range(npoint):
            bounds = self._corner_bounds(arrays, ipoint)
            spec_items = [bounds[name] for name in self.spec_axes]
            rv_items = bounds.get('rv', [(0, 1.0)])
            av_items = bounds.get('av', [(0, 1.0)])

            log_flux = np.zeros(nband, dtype=float)
            log_labs = 0.0
            total_weight = 0.0

            for spec_corner in product(*spec_items):
                spec_indices = tuple(item[0] for item in spec_corner)
                spec_weight = float(np.prod([item[1] for item in spec_corner]))
                ispec = int(self.spec_index[spec_indices])
                if ispec < 0 or spec_weight == 0.0:
                    continue

                labs_value = float(self.spectra['Labs'][ispec])
                if labs_value <= 0 or not np.isfinite(labs_value):
                    continue

                for rv_index, rv_weight in rv_items:
                    for av_index, av_weight in av_items:
                        weight = spec_weight * rv_weight * av_weight
                        if weight == 0.0:
                            continue
                        row = np.asarray(dset[ispec, rv_index, av_index, :], dtype=float)
                        row = row[self.photband_indices]
                        if np.any(row <= 0) or not np.all(np.isfinite(row)):
                            continue
                        log_flux += weight * np.log10(row)
                        log_labs += weight * np.log10(labs_value)
                        total_weight += weight

            if total_weight <= 0:
                flux[:, ipoint] = np.nan
                labs[ipoint] = np.nan
            else:
                log_flux /= total_weight
                log_labs /= total_weight
                flux[:, ipoint] = 10.0 ** log_flux
                labs[ipoint] = 10.0 ** log_labs

        if scalar_input:
            return flux[:, 0], labs[:1]
        return flux, labs

    def evaluate(self, **values_by_name):
        arrays, npoint, scalar_input = self._prepare_inputs(values_by_name)
        for cache in self._active_caches:
            if not self._cache_covers(cache, arrays):
                continue
            flux, labs = self._evaluate_cached(cache, arrays, npoint, scalar_input)
            if np.all(np.isfinite(flux)) and np.all(np.isfinite(labs)):
                self._cache_statistics['cached_points'] += int(npoint)
            else:
                # A covering cache contains every valid atmosphere corner in
                # its axis range.  A non-finite result therefore represents a
                # genuinely unsupported point in a non-rectangular grid; the
                # raw HDF5 interpolation would return the same result.
                self._cache_statistics['invalid_cached_points'] += int(npoint)
            return flux, labs
        if self._active_caches:
            self._cache_statistics['fallback_points'] += int(npoint)
        return self._evaluate_uncached(arrays, npoint, scalar_input)


def _fits_source_identity(gridname, reddening_law, reddening_rv,
                          reddening_case1):
    files = []
    for member in _normalise_grid_members(gridname):
        path = get_grid_file(
            integrated=True,
            grid=member['grid'],
            reddening_law=reddening_law,
            reddening_Rv=reddening_rv,
            reddening_case1=reddening_case1,
        )
        stat = os.stat(path)
        files.append((
            os.path.realpath(path),
            int(stat.st_size),
            int(stat.st_mtime_ns),
        ))
    return tuple(files)


def _fits_shared_signature(gridname, variables, reddening_law,
                           reddening_rv, reddening_case1):
    return (
        _fits_source_identity(
            gridname, reddening_law, reddening_rv, reddening_case1,
        ),
        tuple(str(name) for name in variables),
        str(reddening_law),
        float(reddening_rv),
        int(reddening_case1),
    )


def register_shared_fits_grid(gridname, variables, photbands, axis_values,
                              pixelgrid, grid_names, reddening_law='WC2019',
                              reddening_rv=3.1, reddening_case1=1):
    """Register one parent-built FITS union grid for forked source workers."""
    entry = {
        'signature': _fits_shared_signature(
            gridname, variables, reddening_law, reddening_rv, reddening_case1,
        ),
        'photbands': [_normalise_photband_name(name) for name in photbands],
        'axis_values': axis_values,
        'pixelgrid': pixelgrid,
        'grid_names': np.asarray(grid_names),
        'gridname': gridname,
        'non_rectangular': grid_has_nonrectangular_coverage(gridname),
    }
    _SHARED_FITS_GRIDS.append(entry)
    return True


def _shared_fits_grid(gridname, variables, ranges, photbands,
                      reddening_law='WC2019', reddening_rv=3.1,
                      reddening_case1=1):
    signature = _fits_shared_signature(
        gridname, variables, reddening_law, reddening_rv, reddening_case1,
    )
    requested_bands = [_normalise_photband_name(name) for name in photbands]
    for entry in reversed(_SHARED_FITS_GRIDS):
        if entry['signature'] != signature:
            continue
        if any(name not in entry['photbands'] for name in requested_bands):
            continue
        covered = True
        for name, axis in zip(entry['grid_names'], entry['axis_values']):
            low, high = ranges.get(str(name), (-np.inf, np.inf))
            tolerance = max(1e-10, 1e-8 * max(1.0, abs(axis[0]), abs(axis[-1])))
            if (np.isfinite(low) and low < axis[0] - tolerance) or \
                    (np.isfinite(high) and high > axis[-1] + tolerance):
                covered = False
                break
        if not covered:
            continue
        output_indices = np.asarray(
            [entry['photbands'].index(name) for name in requested_bands]
            + [len(entry['photbands'])],
            dtype=int,
        )
        output_indices.flags.writeable = False
        metadata = {
            'output_indices': output_indices,
            'shared_union': True,
            'source_grid': entry['gridname'],
            'non_rectangular': entry['non_rectangular'],
        }
        return [
            entry['axis_values'],
            entry['pixelgrid'],
            entry['grid_names'],
            metadata,
        ]
    return None


def load_grids(gridnames, pnames, limits, photbands,
               grid_variables=None, reddening_law='WC2019',
               reddening_Rv=3.1, reddening_case1=1,
               use_cache=True, hdf5_preload=True,
               hdf5_preload_max_gb=DEFAULT_HDF5_PRELOAD_MAX_GB,
               hdf5_walker_cache=True, hdf5_runtime_cache_dir=None,
               runtime_cache_dir=None):
    """
    prepares the integrated photometry grid by loading the grid and cutting it to the size
    given in limits.
    """
    if runtime_cache_dir is None:
        runtime_cache_dir = hdf5_runtime_cache_dir
    cache_key = None
    if use_cache:
        cache_key = _freeze_for_cache((
            gridnames,
            pnames,
            np.asarray(limits, dtype=float),
            [_normalise_photband_name(name) for name in photbands],
            grid_variables,
            reddening_law,
            float(reddening_Rv),
            int(reddening_case1),
            bool(hdf5_preload),
            float(hdf5_preload_max_gb),
            bool(hdf5_walker_cache),
            runtime_cache_dir,
        ))
        if cache_key in _GRID_CACHE:
            return _GRID_CACHE[cache_key]

    pnames = list(pnames)
    grids = []
    for i, name in enumerate(gridnames):
        ind = '' if i == 0 else str(i + 1)
        variables = _variables_for_component(grid_variables, name, i, pnames, ind)
        ranges = {var: _range_for(var, pnames, limits, ind) for var in variables}

        if _grid_integrated_format(name) == 'hdf5':
            grids.append(prepare_hdf5_grid(
                photbands,
                name,
                variables=variables,
                ranges=ranges,
                preload=hdf5_preload,
                preload_max_gb=hdf5_preload_max_gb,
                allow_walker_cache=hdf5_walker_cache,
                runtime_cache_dir=runtime_cache_dir,
            ))
        else:
            shared = _shared_fits_grid(
                name,
                variables,
                ranges,
                photbands,
                reddening_law=reddening_law,
                reddening_rv=reddening_Rv,
                reddening_case1=reddening_case1,
            )
            if shared is not None:
                grids.append(shared)
                continue
            axis_values, grid_pars, pixelgrid, grid_names = prepare_grid(
                photbands, name,
                variables=variables,
                ranges=ranges,
                reddening_law=reddening_law,
                reddening_Rv=reddening_Rv,
                reddening_case1=reddening_case1,
                runtime_cache_dir=runtime_cache_dir,
            )

            grids.append([
                axis_values,
                pixelgrid,
                grid_names,
                {
                    'source_grid': name,
                    'non_rectangular': grid_has_nonrectangular_coverage(name),
                },
            ])

    if cache_key is not None:
        _GRID_CACHE[cache_key] = grids
    return grids


def prepare_hdf5_grid(photbands, gridname, variables=None, ranges=None,
                      preload=True,
                      preload_max_gb=DEFAULT_HDF5_PRELOAD_MAX_GB,
                      allow_walker_cache=True, runtime_cache_dir=None):
    desc = grid_description.get(gridname, {}) if isinstance(gridname, str) else {}
    axes = [str(axis).lower() for axis in desc.get('axes', [])]
    if variables is None:
        variables = axes or ['teff', 'logg', 'av', 'feh', 'rv']
    variables = list(dict.fromkeys([str(variable).lower() for variable in variables]))
    for required in axes:
        if required not in variables:
            variables.append(required)
    ranges = {} if ranges is None else dict(ranges)
    for variable in variables:
        ranges.setdefault(variable, (-np.inf, np.inf))
    return HDF5IntegratedGrid(
        get_grid_file(integrated=True, grid=gridname),
        photbands,
        variables=variables,
        ranges=ranges,
        preload=preload,
        preload_max_gb=preload_max_gb,
        allow_walker_cache=allow_walker_cache,
        runtime_cache_dir=runtime_cache_dir,
    )


def _fits_runtime_cache_identity(runtime_cache_dir, gridname, photbands,
                                 variables, ranges, reddening_law,
                                 reddening_rv, reddening_case1):
    if not runtime_cache_dir:
        return None, None
    sources = [
        {'path': path, 'size': size, 'mtime_ns': mtime}
        for path, size, mtime in _fits_source_identity(
            gridname, reddening_law, reddening_rv, reddening_case1,
        )
    ]
    payload = {
        'format_version': FITS_RUNTIME_CACHE_FORMAT_VERSION,
        'sources': sources,
        'gridname': repr(_freeze_for_cache(gridname)),
        'photbands': [_normalise_photband_name(name) for name in photbands],
        'variables': [str(name) for name in variables],
        'ranges': repr(_freeze_for_cache(ranges)),
        'reddening_law': str(reddening_law),
        'reddening_rv': float(reddening_rv),
        'reddening_case1': int(reddening_case1),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    digest = hashlib.sha256(encoded).hexdigest()[:24]
    root = os.path.abspath(os.path.expanduser(str(runtime_cache_dir)))
    return payload, os.path.join(root, 'fits_' + digest)


def _load_fits_runtime_grid(identity, directory):
    if directory is None or not os.path.isdir(directory):
        return None
    try:
        with open(os.path.join(directory, 'metadata.json')) as handle:
            metadata = json.load(handle)
        if metadata.get('identity') != identity:
            return None
        axis_values = [
            np.load(os.path.join(directory, 'axis_{}.npy'.format(index)),
                    mmap_mode='r', allow_pickle=False)
            for index in range(int(metadata['naxes']))
        ]
        pixelgrid = np.load(
            os.path.join(directory, 'pixelgrid.npy'),
            mmap_mode='r',
            allow_pickle=False,
        )
        grid_names = np.load(
            os.path.join(directory, 'grid_names.npy'),
            mmap_mode='r',
            allow_pickle=False,
        )
        if pixelgrid.shape != tuple(metadata['pixelgrid_shape']):
            return None
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    return axis_values, None, pixelgrid, grid_names


def _write_fits_runtime_grid(identity, directory, axis_values, pixelgrid,
                             grid_names):
    if directory is None or os.path.isdir(directory):
        return False
    root = os.path.dirname(directory)
    os.makedirs(root, exist_ok=True)
    temporary = tempfile.mkdtemp(prefix='.sedforge-fits-cache-', dir=root)
    try:
        np.save(os.path.join(temporary, 'pixelgrid.npy'), np.asarray(pixelgrid),
                allow_pickle=False)
        np.save(os.path.join(temporary, 'grid_names.npy'), np.asarray(grid_names),
                allow_pickle=False)
        for index, axis in enumerate(axis_values):
            np.save(os.path.join(temporary, 'axis_{}.npy'.format(index)),
                    np.asarray(axis), allow_pickle=False)
        metadata = {
            'identity': identity,
            'naxes': len(axis_values),
            'pixelgrid_shape': list(pixelgrid.shape),
        }
        with open(os.path.join(temporary, 'metadata.json'), 'w') as handle:
            json.dump(metadata, handle, sort_keys=True)
        try:
            os.rename(temporary, directory)
        except FileExistsError:
            return False
        return True
    finally:
        if os.path.isdir(temporary):
            shutil.rmtree(temporary)


def prepare_grid(photbands, gridname,
                 teffrange=(-np.inf, np.inf), loggrange=(-np.inf, np.inf),
                 avrange=(-np.inf, np.inf), ebvrange=(-np.inf, np.inf),
                 variables=None, ranges=None,
                 reddening_law='WC2019', reddening_Rv=3.1,
                 reddening_case1=1,
                 runtime_cache_dir=None,
                 **kwargs):
    fluxes = []
    grid_pars = []
    if variables is None:
        variables = ['teff', 'logg', 'av']
    grid_names = np.array(variables)
    if ranges is None:
        ranges = {
            'teff': teffrange,
            'logg': loggrange,
            'av': avrange,
            'ebv': ebvrange,
        }

    runtime_identity, runtime_directory = _fits_runtime_cache_identity(
        runtime_cache_dir,
        gridname,
        photbands,
        variables,
        ranges,
        reddening_law,
        reddening_Rv,
        reddening_case1,
    )
    runtime_grid = _load_fits_runtime_grid(runtime_identity, runtime_directory)
    if runtime_grid is not None:
        print(
            "Memory-mapped persistent FITS grid cache for {}: shape {}, {:.3f} GB.".format(
                gridname,
                runtime_grid[2].shape,
                runtime_grid[2].nbytes / 1024.0 ** 3,
            )
        )
        return runtime_grid

    for member in _normalise_grid_members(gridname):
        gridfilename = get_grid_file(integrated=True, grid=member['grid'],
                                     reddening_law=reddening_law,
                                     reddening_Rv=reddening_Rv,
                                     reddening_case1=reddening_case1)

        with fits.open(gridfilename) as ff:
            # -- make an alias for further reference
            ext = ff[1]

            # -- the grid is already cut here to limit memory usage
            keep = np.ones(len(ext.data), bool)
            variable_values = []
            for name in variables:
                field = _field_name(ext.data, name)
                if field is not None:
                    values = np.asarray(ext.data.field(field), dtype=float)
                elif name == 'av' and _field_name(ext.data, 'ebv') is not None:
                    values = (
                        np.asarray(ext.data.field(_field_name(ext.data, 'ebv')), dtype=float)
                        * float(reddening_Rv)
                    )
                elif name == 'ebv' and _field_name(ext.data, 'av') is not None:
                    values = (
                        np.asarray(ext.data.field(_field_name(ext.data, 'av')), dtype=float)
                        / float(reddening_Rv)
                    )
                elif name in member:
                    values = np.full(len(ext.data), float(member[name]))
                else:
                    raise ValueError(
                        f"Grid {member['grid']} does not provide axis '{name}'. "
                        "Add it as a FITS column, in grid_description.yaml, or "
                        "as member metadata in the setup file."
                    )

                snapped = _snap_range(values, ranges.get(name, (-np.inf, np.inf)))
                if snapped is None:
                    keep = np.zeros(len(ext.data), bool)
                    low, high = np.nan, np.nan
                else:
                    low, high = snapped

                # -- we need to be careful for rounding errors
                in_range = (low <= values) & (values <= high)
                on_edge = np.allclose(values, low) | np.allclose(values, high)

                keep = keep & (in_range | on_edge)
                variable_values.append(values)

            if not np.any(keep):
                continue

            grid_pars.append(np.vstack([values[keep] for values in variable_values]))
            fluxes.append(_get_flux_from_table(ext, photbands,
                                               include_Labs=True)[keep])

    if not fluxes:
        raise ValueError(f"No grid points left for {gridname} after applying limits.")

    grid_pars = np.hstack(grid_pars)
    flux = np.vstack(fluxes)
    with np.errstate(divide='ignore', invalid='ignore'):
        flux = np.log10(flux)

    # -- create the pixeltype grid
    axis_values, pixelgrid = interpol.create_pixeltypegrid(grid_pars, flux.T)
    _write_fits_runtime_grid(
        runtime_identity,
        runtime_directory,
        axis_values,
        pixelgrid,
        grid_names,
    )
    return axis_values, grid_pars.T, pixelgrid, grid_names


def get_itable_single(teff=None, logg=None, g=None, av=None, ebv=None, feh=None, rv=None,
                      distance=None, dist=None, **kwargs):
    reddening_rv = float(kwargs.get('reddening_Rv', kwargs.get('Rv', 3.1)))
    if av is not None and ebv is not None:
        raise ValueError("Use either av or ebv for grid interpolation, not both.")
    if av is None and ebv is None:
        av = 0.0
        ebv = 0.0
    elif av is None:
        av = np.asarray(ebv, dtype=float) * reddening_rv
    elif ebv is None:
        ebv = np.asarray(av, dtype=float) / reddening_rv

    # -- check if logg or g is given
    if logg is None and g is not None:
        logg = np.log10(g)

    # -- get the grid from the keyword, or prepare it if a grid name is given
    integrated_hdf5 = isinstance(kwargs['grid'], HDF5IntegratedGrid)
    if isinstance(kwargs['grid'], HDF5IntegratedGrid):
        flux, Labs = kwargs['grid'].evaluate(
            teff=teff,
            logg=logg,
            av=av,
            feh=feh,
            rv=rv,
        )
        scalar_input = flux.ndim == 1
    elif isinstance(kwargs['grid'], str):
        if _grid_integrated_format(kwargs['grid']) == 'hdf5':
            integrated_hdf5 = True
            hdf5_grid = prepare_hdf5_grid(
                kwargs['photbands'],
                kwargs['grid'],
            )
            flux, Labs = hdf5_grid.evaluate(
                teff=teff,
                logg=logg,
                av=av,
                feh=feh,
                rv=rv,
            )
            scalar_input = flux.ndim == 1
        else:
            extinction_axis = 'ebv' if 'ebv' in (kwargs.get('grid_variables') or []) else 'av'
            variables = ['teff', 'logg', extinction_axis]
            extinction_value = ebv if extinction_axis == 'ebv' else av
            ranges = {
                'teff': (np.min(teff), np.max(teff)),
                'logg': (np.min(logg), np.max(logg)),
                extinction_axis: (np.min(extinction_value), np.max(extinction_value)),
            }
            if feh is not None and grid_has_axis(kwargs['grid'], 'feh'):
                variables = ['teff', 'logg', 'feh', extinction_axis]
                ranges['feh'] = (np.min(feh), np.max(feh))
            axis_values, grid_pars, pixelgrid, grid_names = prepare_grid(kwargs['photbands'], kwargs['grid'],
                                                                         variables=variables,
                                                                         ranges=ranges)
    else:
        axis_values, pixelgrid = kwargs['grid'][:2]
        if len(kwargs['grid']) > 2:
            grid_names = kwargs['grid'][2]
        else:
            grid_names = np.array(['teff', 'logg', 'av'][:len(axis_values)])

    values_by_name = {
        'teff': teff,
        'logg': logg,
        'av': av,
        'ebv': ebv,
        'feh': feh,
        'rv': rv,
    }
    if not integrated_hdf5:
        p_values = []
        scalar_input = True
        for name in grid_names:
            value = values_by_name.get(name)
            if value is None:
                raise ValueError(f"Missing model parameter '{name}' for grid interpolation.")
            arr = np.asarray(value)
            if arr.ndim > 0:
                scalar_input = False
            p_values.append(np.atleast_1d(arr.astype(float)).reshape(-1))

        # Vectorized likelihood calls combine sampled walker arrays with
        # scalar fixed atmosphere parameters. Broadcast those fixed values to
        # the common walker count before stacking interpolation coordinates.
        # Non-scalar inputs must already agree in length so shape mistakes are
        # reported explicitly instead of being silently tiled.
        target_size = max(value.size for value in p_values)
        incompatible = [value.size for value in p_values if value.size not in (1, target_size)]
        if incompatible:
            raise ValueError(
                "Model parameters must be scalars or arrays with a common length; "
                "received lengths {}.".format([value.size for value in p_values])
            )
        if target_size > 1:
            p_values = [
                np.full(target_size, float(value[0]), dtype=float)
                if value.size == 1 else value
                for value in p_values
            ]

        p = np.vstack(p_values)

        values = interpol.interpolate(p, axis_values, pixelgrid)

        if len(kwargs['grid']) > 3 and isinstance(kwargs['grid'][3], dict):
            output_indices = kwargs['grid'][3].get('output_indices')
            if output_indices is not None:
                values = values[np.asarray(output_indices, dtype=int)]

        # -- switch logarithm to normal
        values = 10 ** values
        flux, Labs = values[:-1], values[-1]

    # -- Take radius into account when provided
    if 'rad' in kwargs:
        rad = np.array(kwargs['rad'])
        flux, Labs = flux * rad ** 2, Labs * rad ** 2

    if distance is None:
        distance = dist
    if distance is not None:
        d = np.array(distance) * PC_TO_RSOL
        flux = flux / d ** 2

    if 'd' in kwargs:
        d = np.array(kwargs['d'])
        flux, Labs = flux / d ** 2, Labs * d ** 2

    if scalar_input:
        flux = flux.flatten()
        Labs = Labs.flatten()

    return flux, Labs


def _is_prepared_grid(grid):
    if isinstance(grid, HDF5IntegratedGrid):
        return True
    return (
        isinstance(grid, (list, tuple)) and len(grid) in (2, 3, 4)
        and hasattr(grid[1], 'shape')
    )


def _ordered_components(components):
    def key(component):
        if component == '':
            return 0
        if str(component).isdigit():
            return int(component) - 1
        return 1000

    return sorted(components, key=key)


def _normalise_component_grids(grid, components):
    ordered_components = _ordered_components(components)
    if not ordered_components:
        ordered_components = ['']
    if isinstance(grid, str):
        grids = [grid]
    elif _is_prepared_grid(grid):
        grids = [grid]
    elif hasattr(grid, '__iter__') and len(grid) > 0 and isinstance(grid[0], str):
        grids = list(grid)
    elif hasattr(grid, '__iter__') and len(grid) > 0 and _is_prepared_grid(grid[0]):
        grids = list(grid)
    else:
        grids = [grid]

    if len(grids) == 1 and len(ordered_components) > 1:
        grids = grids * len(ordered_components)
    elif len(grids) != len(ordered_components):
        raise ValueError(
            "Received {} model component(s) but {} grid(s). Provide one grid "
            "per component.".format(len(ordered_components), len(grids))
        )

    return ordered_components, grids


def get_itable(grid=[], **kwargs):
    values, parameters, components = {}, set(), set()
    for key in list(kwargs.keys()):
        if re.search(r"^(teff|logg|g|av|ebv|feh|rv|rad|distance|dist)\d*$", key):
            par, comp = re.findall(r"^(teff|logg|g|av|ebv|feh|rv|rad|distance|dist)(\d*)$", key)[0]
            values[key] = kwargs.pop(key)
            parameters.add(par)
            components.add(comp)

    ordered_components, grids = _normalise_component_grids(grid, components)

    # -- If there is only one component, we can directly return the result
    if len(components) == 1:
        kwargs.update(values)
        fluxes, Labs = get_itable_single(grid=grids[0], **kwargs)
        return fluxes, {'L': Labs}

    fluxes, Labs = [], {}
    for i, (comp, grid) in enumerate(zip(ordered_components, grids)):
        kwargs_ = kwargs.copy()
        for par in parameters:
            kwargs_[par] = values[par + comp] if par + comp in values else values[par]

        f, L = get_itable_single(grid=grid, **kwargs_)

        fluxes.append(f)
        Labs['L' + comp] = np.sum(L, axis=0)

    fluxes = np.sum(fluxes, axis=0)
    return fluxes, Labs


def get_table_single(teff=None, logg=None, g=None, av=None, ebv=None, feh=None, he_mass=None,
                     rv=None, distance=None, dist=None, **kwargs):
    """
   No interpolating, just returns the closest gridpoint
   """

    if logg is None and g is not None:
        logg = np.log10(g)

    # -- get the grid
    gridname = kwargs['grid']
    metadata = _infer_grid_metadata(gridname)
    if feh is None and 'feh' in metadata:
        feh = metadata['feh']
    if he_mass is None and 'he_mass' in metadata:
        he_mass = metadata['he_mass']

    cache_path = get_spectral_cache_file(grid=gridname)
    if cache_path is not None and os.path.isfile(cache_path):
        return spectral_cache.read_spectrum(
            cache_path,
            teff=teff,
            logg=logg,
            feh=feh,
            he_mass=he_mass,
            rad=kwargs.get('rad'),
            distance=distance,
            dist=dist,
            d=kwargs.get('d'),
        )

    gridfilename = get_grid_file(integrated=False, grid=gridname)

    hdu = fits.open(gridfilename)

    teffs, loggs = np.zeros(len(hdu) - 1), np.zeros(len(hdu) - 1)
    for i in range(1, len(hdu)):
        teffs[i - 1] = hdu[i].header['TEFF']
        loggs[i - 1] = hdu[i].header['LOGG']

    dteff = abs(teffs - teff) / teff
    dlogg = abs(loggs - logg) / logg

    s = np.where(np.sqrt(dteff ** 2 + dlogg ** 2) == np.min(np.sqrt(dteff ** 2 + dlogg ** 2)))

    model = hdu[s[0][0] + 1].data
    wave, flux = model['wavelength'], model['flux']

    # -- Take radius into account when provided
    if 'rad' in kwargs:
        rad = np.array(kwargs['rad'])
        flux = flux * rad ** 2

    if distance is None:
        distance = dist
    if distance is not None:
        d = np.array(distance) * PC_TO_RSOL
        flux = flux / d ** 2

    if 'd' in kwargs:
        d = np.array(kwargs['d'])
        flux = flux / d ** 2

    return wave, flux


def get_table(grid=[], **kwargs):
    """
   Returns the closest model atmosphere available in the grid. No interpolation is done!
   """
    values, parameters, components = {}, set(), set()
    for key in list(kwargs.keys()):
        if re.search(r"^(teff|logg|g|av|ebv|feh|rv|he_mass|rad|distance|dist)\d*$", key):
            par, comp = re.findall(r"^(teff|logg|g|av|ebv|feh|rv|he_mass|rad|distance|dist)(\d*)$", key)[0]
            values[key] = kwargs.pop(key)
            parameters.add(par)
            components.add(comp)

    ordered_components, grids = _normalise_component_grids(grid, components)

    # -- If there is only one component, we can directly return the result
    if len(components) == 1:
        kwargs.update(values)
        wave, flux = get_table_single(grid=grids[0], **kwargs)
        return wave, flux

    waves, fluxes = [], []
    for i, (comp, grid) in enumerate(zip(ordered_components, grids)):
        kwargs_ = kwargs.copy()
        for par in parameters:
            kwargs_[par] = values[par + comp] if par + comp in values else values[par]

        w, f = get_table_single(grid=grid, **kwargs_)

        waves.append(w)
        fluxes.append(f)

    # -- interpolate and combine the models
    wave = waves[0]
    flux = np.zeros_like(wave)
    for w, f in zip(waves, fluxes):
        flux += np.interp(wave, w, f)

    return wave, flux


def _get_flux_from_table(fits_ext, photbands, index=None, include_Labs=True):
    """
   Retrieve fluxes from an integrated SED table.

   @param fits_ext: fits extension containing integrated flux
   @type fits_ext: FITS extension
   @param photbands: list of photometric passbands
   @type photbands: list of str
   @param index: slice or index of rows to retrieve
   @type index: slice or integer
   @return: fluxes or flux ratios
   #@rtype: list
   """
    if index is None:
        index = slice(None)  # -- full range
    fluxes = []
    for photband in photbands:
        try:
            fluxes.append(fits_ext.data.field(photband)[index])
        except KeyError:
            print('Passband %s missing from table' % (photband))
            fluxes.append(np.nan * np.ones(len(fits_ext.data)))
    # -- possibly include absolute luminosity
    if include_Labs:
        fluxes.append(fits_ext.data.field("Labs")[index])
    fluxes = np.array(fluxes).T
    if index is not None:
        fluxes = fluxes
    return fluxes


def luminosity(wave, flux, radius=1.):
    """
    Calculate the bolometric luminosity of a model SED.

    Flux should be in cgs per unit wavelength (same unit as wave).
    The latter is integrated out, so it is of no importance. After integration,
    flux, should have units erg/s/cm2.

    Returned luminosity is in solar units.

    If you give radius=1 and want to correct afterwards, multiply the obtained
    Labs with radius**2.

    :param wave: model wavelengths
    :type wave: ndarray
    :param flux: model fluxes (Flam)
    :type flux: ndarray
    :param radius: stellar radius in solar units
    :type radius: float
    :return: total bolometric luminosity
    :rtype: float
    """
    Lsol_cgs = 3.846e33
    Rsol_cgs = 6.95508e10
    Lint = trapezoid(flux, x=wave)
    Labs = Lint * 4 * np.pi / Lsol_cgs * (radius * Rsol_cgs) ** 2
    return Labs
