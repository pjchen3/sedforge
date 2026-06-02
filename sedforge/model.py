import glob
import re
import os
import yaml
from itertools import product
import numpy as np

from astropy.io import fits

from sedforge._compat import trapezoid
from sedforge import interpol, spectral_cache

PC_TO_RSOL = 44365810.04823812


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

    elif grid == 'ck03_cepheid_rv':
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
    elif grid == 'ck03_cepheid_rv':
        ranges['feh'] = (-2.0, 0.5)
        ranges['rv'] = (2.0, 5.0)
    elif grid == 'tlusty_all':
        ranges['feh'] = (-1.0, 0.3010299956639812)
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
                with fits.open(gridname, memmap=True) as ff:
                    return _field_name(ff[1].data, 'feh') is not None
            except Exception:
                return False

    if isinstance(gridname, dict) and 'grid' in gridname:
        return _grid_supports_feh(gridname['grid'])

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
        if axis == 'rv' and desc.get('supports_rv'):
            return True
        for key in ('axes', 'variables', 'parameters'):
            if axis in [str(item).lower() for item in desc.get(key, [])]:
                return True
        if os.path.isfile(gridname):
            if gridname.endswith(('.h5', '.hdf5')):
                try:
                    import h5py
                    with h5py.File(gridname, 'r') as h5:
                        return axis in h5.get('axes', {})
                except Exception:
                    return False
            try:
                with fits.open(gridname, memmap=True) as ff:
                    return _field_name(ff[1].data, axis) is not None
            except Exception:
                return False

    return False


def grid_requires_feh_value(gridname):
    """Return True when a grid needs an explicit fitted or fixed Fe/H value."""
    known_metallicity_axes = {'ck_all', 'tlusty_all', 'newera_alpha0'}

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
                with fits.open(gridname, memmap=True) as ff:
                    field = _field_name(ff[1].data, 'feh')
                    if field is None:
                        return False
                    values = np.asarray(ff[1].data.field(field), dtype=float)
                    values = values[np.isfinite(values)]
                    return len(np.unique(values)) > 1
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


def _axis_bounds(axis, value):
    axis = np.asarray(axis, dtype=float)
    value = float(value)
    atol = max(1e-10, 1e-8 * max(1.0, abs(axis[0]), abs(axis[-1])))
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

    def __init__(self, path, photbands, variables=None, ranges=None):
        self.path = str(path)
        self.photbands = [_normalise_photband_name(name) for name in photbands]
        self.variables = np.array(variables or ['teff', 'logg', 'av', 'feh', 'rv'])
        self.ranges = ranges or {}
        self._handle = None

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

    @property
    def h5(self):
        if self._handle is None:
            import h5py
            self._handle = h5py.File(self.path, 'r')
        return self._handle

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

    def evaluate(self, **values_by_name):
        arrays, npoint, scalar_input = self._prepare_inputs(values_by_name)
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


def load_grids(gridnames, pnames, limits, photbands,
               grid_variables=None, reddening_law='WC2019',
               reddening_Rv=3.1, reddening_case1=1):
    """
    prepares the integrated photometry grid by loading the grid and cutting it to the size
    given in limits.
    """
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
            ))
        else:
            axis_values, grid_pars, pixelgrid, grid_names = prepare_grid(
                photbands, name,
                variables=variables,
                ranges=ranges,
                reddening_law=reddening_law,
                reddening_Rv=reddening_Rv,
                reddening_case1=reddening_case1,
            )

            grids.append([axis_values, pixelgrid, grid_names])

    return grids


def prepare_hdf5_grid(photbands, gridname, variables=None, ranges=None):
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
    )


def prepare_grid(photbands, gridname,
                 teffrange=(-np.inf, np.inf), loggrange=(-np.inf, np.inf),
                 avrange=(-np.inf, np.inf), ebvrange=(-np.inf, np.inf),
                 variables=None, ranges=None,
                 reddening_law='WC2019', reddening_Rv=3.1,
                 reddening_case1=1,
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
    flux = np.log10(flux)

    # -- create the pixeltype grid
    axis_values, pixelgrid = interpol.create_pixeltypegrid(grid_pars, flux.T)
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
            p_values.append(np.atleast_1d(arr.astype(float)))

        p = np.vstack(p_values)

        values = interpol.interpolate(p, axis_values, pixelgrid)

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
        isinstance(grid, (list, tuple)) and len(grid) in (2, 3)
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
