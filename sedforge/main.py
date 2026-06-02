import os
import re

import yaml
import argparse
import numpy as np
import pylab as pl
import corner

from astropy.io import ascii

from numpy.lib.recfunctions import repack_fields

from sedforge import mcmc, model, plotting, fileio, catalog_photometry
from sedforge.default_setup import default_binary, default_single

MAG_PHOTOMETRY_COLUMNS = ('photband', 'mag', 'mag_err')
FLUX_PHOTOMETRY_COLUMNS = ('photband', 'flux', 'flux_err')
PHOTOMETRY_COLUMNS = MAG_PHOTOMETRY_COLUMNS
MAG_TYPE_COLUMNS = ('mag_type', 'magnitude_type')
MAG_ZP_OFFSET_COLUMNS = ('mag_zp_offset', 'zp_offset', 'mag_offset')
LEGACY_PHOTOMETRY_KEYS = ('photband_index', 'obs_index', 'err_index')


def _normalise_fixed_parameters(setup):
    fixed = setup.get('fixed', {})
    if fixed is None:
        return {}
    if type(fixed) is not dict:
        raise ValueError("Fixed parameters must be provided as a dictionary, e.g. fixed: {feh: 0.0}.")

    values = {}
    for par, value in fixed.items():
        if hasattr(value, '__iter__') and not isinstance(value, str):
            raise ValueError(
                "Fixed parameter '{}' must be a single numeric value, not {}.".format(
                    par, value
                )
            )
        try:
            values[str(par)] = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                "Fixed parameter '{}' must be numeric; received {}.".format(par, value)
            )

    if 'distance' in values and 'dist' in values:
        raise ValueError("Use only one distance alias in fixed parameters: 'distance' or 'dist'.")

    return values


def _reject_retired_setup_keys(setup):
    if 'constraints' in setup:
        raise ValueError("The setup key 'constraints' is retired. Use 'priors' for Gaussian priors.")
    if 'derived_limits' in setup:
        raise ValueError("The setup key 'derived_limits' is retired. Derived quantities are outputs only.")


def _reject_reddening_rv_for_rv_grids(setup):
    grids = setup.get('grids', [])
    if isinstance(grids, str):
        grids = [grids]
    if not any(model.grid_has_axis(grid, 'rv') for grid in grids):
        return
    if 'reddening_Rv' in setup or 'Rv' in setup:
        raise ValueError(
            "Grids with an explicit 'rv' axis fit/use Rv as a model parameter. "
            "Remove setup key 'reddening_Rv'/'Rv' and put 'rv' in pnames/limits "
            "or fixed."
        )


def _normalised_parameter_name(name):
    text = str(name).strip().lower()
    for char in ('_', '-', '(', ')', ' '):
        text = text.replace(char, '')
    return text


def _is_legacy_ebv_parameter(name):
    text = _normalised_parameter_name(name)
    return text == 'ebv' or (text.startswith('ebv') and text[3:].isdigit())


def _collect_legacy_ebv_names(values, location, places, keys=False):
    if values is None:
        return
    if isinstance(values, dict):
        candidates = values.keys() if keys else values.values()
    elif isinstance(values, str):
        candidates = [values]
    else:
        try:
            candidates = list(values)
        except TypeError:
            return

    for value in candidates:
        if isinstance(value, dict) or (
                not isinstance(value, str)
                and hasattr(value, '__iter__')
        ):
            _collect_legacy_ebv_names(value, location, places, keys=keys)
        elif _is_legacy_ebv_parameter(value):
            places.append(location)


def _reject_legacy_ebv_setup_parameter(setup):
    places = []
    _collect_legacy_ebv_names(setup.get('pnames', []), 'pnames', places)
    _collect_legacy_ebv_names(setup.get('fixed', {}), 'fixed', places, keys=True)
    _collect_legacy_ebv_names(setup.get('priors', {}), 'priors', places, keys=True)
    _collect_legacy_ebv_names(setup.get('grid_variables', None),
                              'grid_variables', places)

    for key, value in setup.items():
        if not str(key).startswith('plot') or not isinstance(value, dict):
            continue
        _collect_legacy_ebv_names(value.get('parameters', None),
                                  f'{key}.parameters', places)
        _collect_legacy_ebv_names(value.get('units', None),
                                  f'{key}.units', places, keys=True)

    if places:
        raise ValueError(
            "YAML/setup files no longer accept legacy E(B-V) parameter 'ebv'. "
            "Use 'av' for A(V) in magnitudes instead. Found legacy parameter in: "
            "{}.".format(', '.join(dict.fromkeys(places)))
        )


def _split_fixed_and_varied_parameters(pnames, limits, fixed_from_setup):
    pnames = list(pnames)
    limits = np.asarray(limits, dtype=float)
    fixed_variables = dict(fixed_from_setup)
    varied_pnames, varied_limits = [], []

    for par, limit in zip(pnames, limits):
        low, high = limit

        if par in fixed_variables:
            raise ValueError(
                "Parameter '{}' is both fitted in pnames/limits and fixed in the fixed section. "
                "Choose one: put it in pnames with a non-zero range, or put one value in fixed.".format(par)
            )

        if not low < high:
            raise ValueError(
                "Parameter '{}' has limits [{}, {}]. Fixed parameters must be listed in fixed, "
                "not as identical or reversed limits.".format(par, low, high)
            )

        varied_pnames.append(par)
        varied_limits.append([float(low), float(high)])

    return varied_pnames, np.asarray(varied_limits, dtype=float), fixed_variables


def _parameters_for_grid(varied_pnames, varied_limits, fixed_variables):
    grid_pnames = list(varied_pnames)
    grid_limits = list(np.asarray(varied_limits, dtype=float))
    for par, value in fixed_variables.items():
        if par not in grid_pnames:
            grid_pnames.append(par)
            grid_limits.append([value, value])
    return grid_pnames, np.asarray(grid_limits, dtype=float)


def _component_suffixes_from_parameters(pnames, fixed_variables):
    suffixes = {''}
    names = list(pnames) + list(fixed_variables)
    for name in names:
        match = re.fullmatch(r'(teff|logg|g|rad)(\d*)', str(name))
        if match is not None:
            suffixes.add(match.group(2))
        match = re.fullmatch(r'feh(\d+)', str(name))
        if match is not None:
            suffixes.add(match.group(1))

    def key(suffix):
        return 1 if suffix == '' else int(suffix)

    suffixes = sorted(suffixes, key=key)
    expected = [''] + [str(i) for i in range(2, len(suffixes) + 1)]
    if suffixes != expected:
        raise ValueError(
            "Model component suffixes must be contiguous: primary uses no suffix, "
            "then 2, 3, ... Found suffixes {}.".format(
                [suffix or 'primary' for suffix in suffixes]
            )
        )
    return suffixes


def _validate_grid_component_count(gridnames, pnames, fixed_variables):
    if isinstance(gridnames, str):
        gridnames = [gridnames]
    suffixes = _component_suffixes_from_parameters(pnames, fixed_variables)
    if len(gridnames) != len(suffixes):
        raise ValueError(
            "The setup defines {} model component(s) ({}) but provides {} grid(s). "
            "Provide one grid per component, e.g. grids: [primary_grid, secondary_grid, "
            "tertiary_grid] for a three-component fit.".format(
                len(suffixes),
                ', '.join(suffix or 'primary' for suffix in suffixes),
                len(gridnames),
            )
        )


def _component_grid_variables(grid_variables, gridname, index):
    if grid_variables is None:
        return None
    if isinstance(grid_variables, dict):
        keys = [index, str(index)]
        if isinstance(gridname, str):
            keys.append(gridname)
        for key in keys:
            if key in grid_variables:
                return list(grid_variables[key])
        return None
    return list(grid_variables)


def _validate_grid_parameter_requirements(gridnames, pnames, fixed_variables,
                                          grid_variables=None):
    if isinstance(gridnames, str):
        gridnames = [gridnames]
    available = set(pnames) | set(fixed_variables)

    for index, gridname in enumerate(gridnames):
        if not model.grid_requires_feh_value(gridname):
            continue

        suffix = '' if index == 0 else str(index + 1)
        component_feh = 'feh' + suffix if suffix else 'feh'
        allowed = {'feh', component_feh}
        if available & allowed:
            variables = _component_grid_variables(grid_variables, gridname, index)
            if variables is not None and 'feh' not in [str(v).lower() for v in variables]:
                raise ValueError(
                    "Grid '{}' requires [Fe/H], but grid_variables for this "
                    "component does not include 'feh'.".format(gridname)
                )
            continue

        raise ValueError(
            "Grid '{}' requires an explicit [Fe/H] value. Add '{}' to pnames/limits "
            "to fit it, or set fixed: {{{}: value}}. For additional components you "
            "may also use shared fixed/pname 'feh'.".format(
                gridname,
                component_feh,
                component_feh,
            )
        )


def _component_parameter(name, suffix):
    if name in ('av', 'ebv', 'rv', 'distance', 'dist'):
        return name
    if suffix:
        return name + suffix
    return name


def _available_parameter(allowed, pnames, fixed_variables):
    available = set(pnames) | set(fixed_variables)
    return available & set(allowed)


def _component_required_parameters(gridname, index, pnames, fixed_variables,
                                   grid_variables=None):
    suffix = '' if index == 0 else str(index + 1)
    available = list(dict.fromkeys(list(pnames) + list(fixed_variables)))
    variables = _component_grid_variables(grid_variables, gridname, index)
    if variables is None:
        variables = model._variables_for_component(
            None,
            gridname,
            index,
            available,
            suffix,
        )
        if model.grid_requires_feh_value(gridname) and 'feh' not in [
                str(variable).lower() for variable in variables]:
            variables.append('feh')
        if model.grid_has_axis(gridname, 'rv') and 'rv' not in [
                str(variable).lower() for variable in variables]:
            variables.append('rv')

    required = []
    for variable in variables:
        variable = str(variable).lower()
        if variable == 'feh':
            component_feh = 'feh' + suffix if suffix else 'feh'
            required.append((component_feh, {'feh', component_feh}))
        else:
            required.append((_component_parameter(variable, suffix),
                             {_component_parameter(variable, suffix)}))

    rad_name = _component_parameter('rad', suffix)
    required.append((rad_name, {rad_name}))
    return required


def _validate_required_fit_parameters(gridnames, pnames, fixed_variables,
                                      grid_variables=None):
    if isinstance(gridnames, str):
        gridnames = [gridnames]

    all_required = []
    allowed_sampled = {'distance', 'dist'}
    for index, gridname in enumerate(gridnames):
        component_required = _component_required_parameters(
            gridname,
            index,
            pnames,
            fixed_variables,
            grid_variables=grid_variables,
        )
        all_required.extend(component_required)
        for _label, allowed in component_required:
            allowed_sampled.update(allowed)

    if not _available_parameter({'distance', 'dist'}, pnames, fixed_variables):
        all_required.append(('distance', {'distance', 'dist'}))

    missing = [
        label for label, allowed in all_required
        if not _available_parameter(allowed, pnames, fixed_variables)
    ]
    if missing:
        raise ValueError(
            "Missing required model parameter(s): {}. Every model parameter must "
            "be either fitted in pnames/limits or supplied as one numeric value "
            "in fixed.".format(', '.join(missing))
        )

    if 'distance' in pnames and 'dist' in pnames:
        raise ValueError("Use only one distance alias in pnames: 'distance' or 'dist'.")
    if ('distance' in pnames and 'dist' in fixed_variables) or (
            'dist' in pnames and 'distance' in fixed_variables):
        raise ValueError("Use only one distance alias across pnames and fixed.")

    extras = [par for par in pnames if par not in allowed_sampled]
    if extras:
        raise ValueError(
            "Parameter(s) in pnames are not used by the selected model grid(s): {}. "
            "Remove them or add the corresponding model support first.".format(
                ', '.join(extras)
            )
        )

    fixed_extras = [par for par in fixed_variables if par not in allowed_sampled]
    if fixed_extras:
        raise ValueError(
            "Parameter(s) in fixed are not used by the selected model grid(s): {}. "
            "Remove them from fixed.".format(', '.join(fixed_extras))
        )


def _normalise_priors_for_fit(setup, pnames, fixed_variables):
    priors = dict(setup.get('priors', {}))
    for con, val in list(priors.items()):
        if not hasattr(val, '__len__') or len(val) not in (2, 3):
            raise ValueError(
                "Prior '{}' must have the form [value, error] or [value, -error, +error].".format(con)
            )
        if len(val) == 2:
            priors[con] = [val[0], val[1], val[1]]
        if priors[con][1] <= 0 or priors[con][2] <= 0:
            raise ValueError("Prior '{}' must have positive uncertainty values.".format(con))

    sampled = set(pnames)
    fixed = set(fixed_variables)
    for con in priors:
        if con in fixed:
            raise ValueError(
                "Prior '{}' targets a fixed parameter. Remove it from priors "
                "or fit it by adding it to pnames/limits.".format(con)
            )
        if con not in sampled:
            raise ValueError(
                "Prior '{}' is not a fitted parameter. Priors may only target sampled "
                "parameters in pnames. Derived quantities such as L, mass, and q are "
                "outputs/checks, not sampled parameters.".format(con)
            )

    return priors


def _photband_system_name(photband):
    photband = str(photband)
    if '.' in photband:
        return photband.split('.', 1)[0]
    if '_' in photband:
        return photband.rsplit('_', 1)[0]
    return photband


def _jitter_parameter_name(group):
    safe = re.sub(r'[^0-9A-Za-z]+', '_', str(group)).strip('_')
    return 'jitter_' + safe


def _normalise_limits(value, name):
    if value is None:
        value = [0.0, 0.2]
    if not hasattr(value, '__len__') or len(value) != 2:
        raise ValueError("{} must have the form [low, high].".format(name))
    low, high = float(value[0]), float(value[1])
    if low < 0 or not low < high:
        raise ValueError("{} must be non-negative and increasing.".format(name))
    return [low, high]


def _normalise_prior_value(name, value):
    if value is None:
        return None
    if not hasattr(value, '__len__') or len(value) not in (2, 3):
        raise ValueError(
            "Prior '{}' must have the form [value, error] or [value, -error, +error].".format(name)
        )
    if len(value) == 2:
        value = [value[0], value[1], value[1]]
    value = [float(value[0]), float(value[1]), float(value[2])]
    if value[1] <= 0 or value[2] <= 0:
        raise ValueError("Prior '{}' must have positive uncertainty values.".format(name))
    return value


def _normalise_bool_switch(name, value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        aliases = {
            'true': True,
            'yes': True,
            'on': True,
            '1': True,
            'false': False,
            'no': False,
            'off': False,
            '0': False,
        }
        if text in aliases:
            return aliases[text]
    raise ValueError("{} must be a boolean switch: true or false.".format(name))


def _error_model_type(value):
    text = str(value).strip().lower()
    aliases = {
        'off': 'none',
        'false': 'none',
        'none': 'none',
        'fixed_fraction': 'fixed_fraction_by_group',
        'fixed_fraction_by_group': 'fixed_fraction_by_group',
        'fitted_fraction': 'fitted_fraction_by_group',
        'fitted_fraction_by_group': 'fitted_fraction_by_group',
    }
    if text not in aliases:
        raise ValueError(
            "Unknown error_model type '{}'. Use 'none', "
            "'fixed_fraction_by_group', or 'fitted_fraction_by_group'.".format(value)
        )
    return aliases[text]


def _group_config(groups, group):
    if not isinstance(groups, dict):
        raise ValueError("error_model.groups must be a dictionary.")
    for key in (group, str(group).upper(), str(group).lower()):
        if key in groups:
            cfg = groups[key]
            if isinstance(cfg, dict):
                return cfg
            return {'fraction': cfg}
    return {}


def _normalise_error_model(setup, photbands):
    spec = setup.get('error_model', None)
    jitter_switch = setup.get('jitter', setup.get('use_jitter', None))
    if 'jitter' in setup and 'use_jitter' in setup:
        raise ValueError("Use only one jitter switch: 'jitter' or 'use_jitter'.")

    if jitter_switch is not None:
        jitter_enabled = _normalise_bool_switch('jitter', jitter_switch)
        if not jitter_enabled:
            if spec not in (None, False):
                spec_type = spec if isinstance(spec, str) else spec.get('type', None)
                if spec_type is None or _error_model_type(spec_type) != 'none':
                    raise ValueError(
                        "jitter: false conflicts with a non-empty error_model. "
                        "Remove error_model or set jitter: true."
                    )
            return {'type': 'none'}, [], np.empty((0, 2), dtype=float), {}
        if spec is None:
            spec = {
                'type': 'fitted_fraction_by_group',
                'default_limits': [0.0, 0.2],
                'default_prior': [0.03, 0.03],
            }

    if spec is None or spec is False:
        return {'type': 'none'}, [], np.empty((0, 2), dtype=float), {}

    if isinstance(spec, str):
        spec = {'type': spec}
    if not isinstance(spec, dict):
        raise ValueError("error_model must be a dictionary or a type string.")

    model_type = _error_model_type(spec.get('type', 'fitted_fraction_by_group'))
    if model_type == 'none':
        return {'type': 'none'}, [], np.empty((0, 2), dtype=float), {}

    group_by = str(spec.get('group_by', 'system')).lower()
    if group_by != 'system':
        raise ValueError("Only error_model.group_by: system is currently supported.")

    photbands = np.asarray(photbands, dtype=str)
    groups = list(dict.fromkeys(_photband_system_name(band) for band in photbands))
    group_specs = spec.get('groups', {}) or {}
    default_limits = _normalise_limits(spec.get('default_limits', [0.0, 0.2]),
                                       'error_model.default_limits')
    default_prior = _normalise_prior_value(
        'error_model.default_prior',
        spec.get('default_prior', None),
    )
    default_fraction = spec.get('default_fraction', spec.get('default', None))

    jitter_pnames, jitter_limits, jitter_priors = [], [], {}
    error_model = {
        'type': model_type,
        'band_groups': {},
        'group_parameters': {},
        'fixed_fractions': {},
        'include_log_norm': model_type == 'fitted_fraction_by_group',
    }

    for group in groups:
        cfg = _group_config(group_specs, group)
        bands = [band for band in photbands if _photband_system_name(band) == group]
        error_model['band_groups'][group] = bands

        if model_type == 'fitted_fraction_by_group':
            par = str(cfg.get('parameter', _jitter_parameter_name(group)))
            if par in jitter_pnames:
                raise ValueError("Duplicate fitted jitter parameter '{}'.".format(par))
            limits = _normalise_limits(cfg.get('limits', default_limits),
                                       "limits for '{}'".format(par))
            prior = _normalise_prior_value(
                par,
                cfg.get('prior', default_prior),
            )
            jitter_pnames.append(par)
            jitter_limits.append(limits)
            error_model['group_parameters'][group] = par
            if prior is not None:
                jitter_priors[par] = prior
        else:
            fraction = cfg.get('fraction', cfg.get('value', default_fraction))
            if fraction is None:
                raise ValueError(
                    "Fixed group-level jitter requires error_model.default_fraction "
                    "or a per-group fraction."
                )
            fraction = float(fraction)
            if fraction < 0:
                raise ValueError("Fixed jitter fraction for '{}' must be non-negative.".format(group))
            error_model['fixed_fractions'][group] = fraction

    return error_model, jitter_pnames, np.asarray(jitter_limits, dtype=float), jitter_priors


def _matches_photband_selector(photband, selector):
    band = str(photband).replace('.', '_').upper()
    wanted = str(selector).replace('.', '_').upper()
    return band == wanted or band.startswith(wanted + '_')


def select_photometry(photbands, obs, obs_err, remove_nan=True, include=None, exclude=None,
                      verbose=True):
    """
    Function to select the wanted photometry.

    - removes photometry with nan values in value or error
    - selects photometry based on include and exclude arrays
    """

    # -- remove photometry with nan values in measurement or error.
    if remove_nan:
        nani = ~np.isfinite(obs) | ~np.isfinite(obs_err)
        if any(nani) and verbose:
            print("Warning: there are NaN values in the following photometric bands:")
            for p in photbands[nani]:
                print("\t {}".format(p))
            print("They were removed")
        obs, obs_err, photbands = obs[~nani], obs_err[~nani], photbands[~nani]

    # -- only include bands that are requested based on the include keyword, or exclude all non wanted photometry
    #    based on the exclude keyword.
    if include is not None:
        incband = []
        for i, photband in enumerate(photbands):
            if any(_matches_photband_selector(photband, inc) for inc in include):
                incband.append(i)
        incband = (np.array(incband),)
        photbands, obs, obs_err = photbands[incband], obs[incband], obs_err[incband]

    elif exclude is not None:
        incband = []
        for i, photband in enumerate(photbands):
            if not any(_matches_photband_selector(photband, exc) for exc in exclude):
                incband.append(i)
        incband = (np.array(incband),)
        photbands, obs, obs_err = photbands[incband], obs[incband], obs_err[incband]

    return photbands, obs, obs_err


def _optional_magnitude_types(data, photbands):
    for name in MAG_TYPE_COLUMNS:
        if name in data.colnames:
            return np.array(data[name], dtype=str)
    return np.array([
        catalog_photometry.default_magnitude_type(photband)
        for photband in photbands
    ], dtype=str)


def _optional_mag_zp_offsets(data, photbands):
    for name in MAG_ZP_OFFSET_COLUMNS:
        if name in data.colnames:
            return np.array(data[name], dtype=float)
    return np.array([
        catalog_photometry.default_mag_zp_offset(photband)
        for photband in photbands
    ], dtype=float)


def _read_magnitude_photometry(data):
    photbands = np.array(data['photband'], dtype=str)
    mags = np.array(data['mag'], dtype=float)
    mag_errs = np.array(data['mag_err'], dtype=float)
    if 'system' in data.colnames:
        systems = np.array(data['system'], dtype=str)
    else:
        systems = np.array([
            catalog_photometry.default_magnitude_system(photband)
            for photband in photbands
        ], dtype=str)
    mag_types = _optional_magnitude_types(data, photbands)
    mag_offsets = _optional_mag_zp_offsets(data, photbands)

    if np.any(~np.isfinite(mags)) or np.any(~np.isfinite(mag_errs)):
        raise ValueError('Photometry mag and mag_err values must be finite.')
    if np.any(mag_errs <= 0):
        raise ValueError('Photometry mag_err values must be positive.')
    if np.any(~np.isfinite(mag_offsets)):
        raise ValueError('Optional magnitude zero-point offsets must be finite.')

    obs = np.empty(len(photbands), dtype=float)
    obs_err = np.empty(len(photbands), dtype=float)
    for i, (photband, mag, mag_err, system, mag_type, offset) in enumerate(
            zip(photbands, mags, mag_errs, systems, mag_types, mag_offsets)):
        obs[i], obs_err[i] = catalog_photometry.mag_to_flux(
            photband,
            mag,
            mag_err,
            system=system,
            mag_type=mag_type,
            mag_zp_offset=offset,
        )

    if np.any(~np.isfinite(obs)) or np.any(~np.isfinite(obs_err)):
        raise ValueError('Converted photometry flux and flux_err values must be finite.')
    if np.any(obs <= 0):
        raise ValueError('Converted photometry flux values must be positive.')
    if np.any(obs_err <= 0):
        raise ValueError('Converted photometry flux_err values must be positive.')

    return photbands, obs, obs_err


def _read_flux_photometry(data):
    photbands = np.array(data['photband'], dtype=str)
    obs = np.array(data['flux'], dtype=float)
    obs_err = np.array(data['flux_err'], dtype=float)

    if np.any(~np.isfinite(obs)) or np.any(~np.isfinite(obs_err)):
        raise ValueError('Photometry flux and flux_err values must be finite.')
    if np.any(obs <= 0):
        raise ValueError('Photometry flux values must be positive.')
    if np.any(obs_err <= 0):
        raise ValueError('Photometry flux_err values must be positive.')

    return photbands, obs, obs_err


def read_photometry(filename):
    """
    Read photometry from a whitespace/comment compatible table.

    The standard format is:
        photband mag mag_err system
    Magnitudes are converted to band-averaged Flambda using the same response
    curves as the integrated model grids. ``system`` may be omitted only for
    filter families with an unambiguous default AB/Vega convention.
    ``mag_type`` is optional; SDSS defaults to asinh magnitudes and other
    filters default to Pogson magnitudes.

    Advanced flux input is still accepted:
        photband flux flux_err
    with fluxes in band-averaged erg/s/cm2/Angstrom.
    """
    data = ascii.read(filename)
    has_mag = all(column in data.colnames for column in MAG_PHOTOMETRY_COLUMNS)
    has_flux = all(column in data.colnames for column in FLUX_PHOTOMETRY_COLUMNS)

    if has_mag:
        return _read_magnitude_photometry(data)

    if has_flux:
        return _read_flux_photometry(data)

    raise ValueError(
        "Photometry files must contain magnitude columns "
        "'photband mag mag_err [system]' or advanced flux columns "
        "'photband flux flux_err'. Missing required input columns."
    )


def get_observations(setup):
    photbands, obs, obs_err = read_photometry(setup['photometryfile'])

    # -- select the requested photometry
    photbands, obs, obs_err = select_photometry(photbands, obs, obs_err, remove_nan=True,
                                                include=setup.get('photband_include', None),
                                                exclude=setup.get('photband_exclude', None))

    return photbands, obs, obs_err


def validate_setup(setup):

    # check the photometry file
    assert 'photometryfile' in setup, "You need to provide a photometry file in the setup using the photometryfile" \
                                      " argument. This can be a n absolute or relative path."
    assert os.path.isfile(setup['photometryfile']), f"The photometry file: {setup['photometryfile']} does not exist."

    legacy_keys = [key for key in LEGACY_PHOTOMETRY_KEYS if key in setup]
    if legacy_keys:
        raise ValueError(
            "Legacy photometry column-index setup keys are no longer supported: "
            "{}. Use a photometry file with columns 'photband mag mag_err system' "
            "and remove those keys from the setup.".format(', '.join(legacy_keys))
        )

    # check if the number of parameters and limits match up.
    assert len(setup['pnames']) == len(setup['limits']), \
        f"The number of parameters fitted has to match the provided limits. Received: \n{len(setup['pnames'])} " \
        f"pnames: {setup['pnames']} \n and \n{len(setup['limits'])} limits: {setup['limits']}"

    _reject_retired_setup_keys(setup)
    _reject_legacy_ebv_setup_parameter(setup)
    _reject_reddening_rv_for_rv_grids(setup)

    _normalise_fixed_parameters(setup)
    if 'error_model' in setup and setup['error_model'] is not None:
        if not isinstance(setup['error_model'], (dict, str)):
            raise ValueError("error_model must be a dictionary or a type string.")
    for switch_name in ('jitter', 'use_jitter'):
        if switch_name in setup:
            _normalise_bool_switch(switch_name, setup[switch_name])
    if 'jitter' in setup and 'use_jitter' in setup:
        raise ValueError("Use only one jitter switch: 'jitter' or 'use_jitter'.")

    # check priors
    priors = setup.get('priors', {})
    assert type(priors) is dict, "Priors need to be provided as a dictionary in the setup."
    for i, (prior, val) in enumerate(list(priors.items())):
        assert len(val) == 2 or len(val) == 3, f"Prior {i}: {prior} = {val} can not be parsed. Priors need " \
                                f"to have the form of: parameter: [val, error] or parameter: [val, -error, +error]"
        assert val[1] > 0 and (len(val) == 2 or val[2] > 0), \
            f"Prior {i}: {prior} = {val} must have positive uncertainty values."

    return True


def fit_sed(setup, photbands, obs, obs_err):

    _reject_retired_setup_keys(setup)
    _reject_legacy_ebv_setup_parameter(setup)
    _reject_reddening_rv_for_rv_grids(setup)

    # -- pars limits
    raw_pnames = list(setup['pnames'])
    raw_limits = np.array(setup['limits'], dtype=float)
    fixed_from_setup = _normalise_fixed_parameters(setup)
    pnames, limits, fixed_variables = _split_fixed_and_varied_parameters(
        raw_pnames,
        raw_limits,
        fixed_from_setup,
    )

    _validate_grid_component_count(setup['grids'], pnames, fixed_variables)
    _validate_grid_parameter_requirements(
        setup['grids'],
        pnames,
        fixed_variables,
        setup.get('grid_variables', None),
    )
    _validate_required_fit_parameters(
        setup['grids'],
        pnames,
        fixed_variables,
        setup.get('grid_variables', None),
    )

    # -- Gaussian priors on sampled parameters
    priors = _normalise_priors_for_fit(
        setup,
        pnames,
        fixed_variables,
    )
    error_model, jitter_pnames, jitter_limits, jitter_priors = _normalise_error_model(
        setup,
        photbands,
    )
    mcmc_pnames = list(pnames) + list(jitter_pnames)
    if len(jitter_limits) > 0:
        mcmc_limits = np.vstack([limits, jitter_limits])
    else:
        mcmc_limits = limits
    priors.update(jitter_priors)

    print("Applied priors: ")
    for prior, val in list(priors.items()):
        print("\t {} = {} - {} + {}".format(prior, val[0], val[1], val[2]))
    if error_model.get('type') == 'fitted_fraction_by_group':
        print("Fitted group-level fractional jitter: ")
        for group, par in error_model['group_parameters'].items():
            print("\t {} -> {}".format(group, par))
    elif error_model.get('type') == 'fixed_fraction_by_group':
        print("Fixed group-level fractional jitter: ")
        for group, value in error_model['fixed_fractions'].items():
            print("\t {} = {}".format(group, value))

    # -- pars grid
    gridnames = setup['grids']
    grid_pnames, grid_limits = _parameters_for_grid(pnames, limits, fixed_variables)
    grids = model.load_grids(gridnames, grid_pnames, grid_limits, photbands,
                             grid_variables=setup.get('grid_variables', None),
                             reddening_law=setup.get('reddening_law', 'WC2019'),
                             reddening_Rv=setup.get('reddening_Rv',
                                                    setup.get('Rv', 3.1)),
                             reddening_case1=setup.get('reddening_case1',
                                                       setup.get('case1', 1)))

    # -- pars mcmc setup
    nwalkers = setup.get('nwalkers', 24)
    nsteps = setup.get('nsteps', 4000)
    nrelax = setup.get('nrelax', 500)
    a = setup.get('a', 10)

    # -- MCMC
    results, samples = mcmc.MCMC(obs, obs_err, photbands,
                                 mcmc_pnames, mcmc_limits, grids,
                                 fixed_variables=fixed_variables,
                                 priors=priors,
                                 error_model=error_model,
                                 nwalkers=nwalkers, nsteps=nsteps, nrelax=nrelax,
                                 a=a)

    # -- add fixed variables to results dictionary
    for par, val in list(fixed_variables.items()):
        results[par] = [val, val, 0, 0]

    percentiles = setup.get('percentiles', [16, 50, 84])
    pc = np.percentile(samples.view(np.float64).reshape(samples.shape + (-1,)), percentiles, axis=0)
    pars = {}
    for p, v, e1, e2 in zip(samples.dtype.names, pc[1], pc[1] - pc[0], pc[2] - pc[1]):
        results[p] = [results[p], v, e1, e2]
        pars[p] = v

    return results, samples, priors, gridnames


def write_results(setup, results, samples, obs, obs_err, photbands):

    outpars, outvals = [], []
    suffixes = _component_suffixes_from_parameters(results.keys(), {})
    preferred = []
    for suffix in suffixes:
        for par in ['teff', 'logg', 'g', 'feh', 'rad']:
            name = par + suffix
            if name in results:
                preferred.append(name)
    preferred += ['distance', 'dist', 'd', 'L', 'av', 'rv', 'chi2']
    for suffix in [s for s in suffixes if s]:
        for par in ['L', 'mass', 'q', 'lr', 'rr']:
            name = par + ('' if suffix == '2' and par in ('q', 'lr', 'rr') else suffix)
            if name in results:
                preferred.append(name)
    if 'mass' in results:
        preferred.append('mass')
    result_names = [p for p in preferred if p in results]
    result_names += [p for p in results if p not in result_names]

    for par in result_names:
        if not hasattr(results[par], '__iter__') or len(results[par]) < 4:
            continue
        outpars.append(par)
        outpars.append(par + '_err_minus')
        outpars.append(par + '_err_plus')
        outvals.append(results[par][1])
        outvals.append(results[par][2])
        outvals.append(results[par][3])

    resultfile = setup.get('resultfile', None)
    if resultfile is not None:
        import pandas as pd
        data = pd.DataFrame(data=[outvals], columns=outpars)
        data.to_csv(resultfile, index=False)

    datafile = setup.get('datafile', None)
    if datafile is not None:
        fileio.write2fits(samples, datafile, setup=setup)

def plot_results(setup, results, samples, priors, gridnames, obs, obs_err, photbands):
    _reject_legacy_ebv_setup_parameter(setup)
    plot_error_model, _, _, _ = _normalise_error_model(setup, photbands)

    # check for 10 possible plots. Should be enough for now.
    for i in range(10):

        pindex = 'plot' + str(i)
        if pindex not in setup:
            continue
        plot_dpi = setup[pindex].get('dpi', setup.get('plot_dpi', 200))

        if setup[pindex]['type'] == 'sed_fit':

            res = setup[pindex].get('result', 'best')

            pl.figure(i)
            pl.clf()
            pl.subplots_adjust(wspace=0.25)
            figsize = setup[pindex].get('figsize', (9, 7))
            if figsize is not None:
                figsize = tuple(figsize)
            plotting.plot_fit(obs, obs_err, photbands, pars=results, grids=setup['grids'],
                              gridnames=gridnames, result=res,
                              reddening_law=setup.get('reddening_law', 'WC2019'),
                              reddening_Rv=setup.get('reddening_Rv', setup.get('Rv', 3.1)),
                              reddening_case1=setup.get('reddening_case1',
                                                        setup.get('case1', 1)),
                              observations_path=setup[pindex].get('observations_path',
                                                                  'observations.txt'),
                              residual_ylim=setup[pindex].get('residual_ylim', None),
                              xlim=setup[pindex].get('xlim', None),
                              figsize=figsize,
                              error_model=plot_error_model)

            if setup[pindex].get('path', None) is not None:
                pl.savefig(setup[pindex].get('path', 'sed_fit.png'), dpi=plot_dpi)

        if setup[pindex]['type'] == 'priors':

            pl.figure(i, figsize=(2 * max(1, len(priors)), 6))
            pl.clf()
            pl.subplots_adjust(wspace=0.40, left=0.07, right=0.98)

            if len(priors) > 0:
                plotting.plot_priors(priors, samples, results)

            if setup[pindex].get('path', None) is not None:
                pl.savefig(setup[pindex].get('path', 'priors.png'), dpi=plot_dpi)

        if setup[pindex]['type'] == 'distribution':

            pars1 = []
            for p in setup[pindex].get('parameters', ['teff', 'rad', 'L', 'distance']):
                if p in samples.dtype.names:
                    pars1.append(p)

            data = repack_fields(samples[pars1])
            data = plotting.sample_for_corner(
                data,
                max_samples=setup[pindex].get('max_samples', 100000),
                random_seed=setup[pindex].get('random_seed', 12345),
            )

            corner_data = data.view(np.float64).reshape(data.shape + (-1,))
            corner_labels = plotting.corner_labels(
                data.dtype.names,
                labels=setup[pindex].get('labels', None),
                units=setup[pindex].get('units', None),
                include_units=setup[pindex].get('label_units', True),
            )
            titles = setup[pindex].get(
                'titles',
                plotting.corner_labels(data.dtype.names, include_units=False),
            )
            best_array = (
                [results[p][0] for p in data.dtype.names]
                if setup[pindex].get('show_best', True)
                else None
            )

            corner_kwargs = {
                'labels': corner_labels,
                'quantiles': setup[pindex].get('quantiles', [0.16, 0.5, 0.84]),
                'titles': titles,
                'show_titles': True,
                'truths': best_array,
                'truth_color': setup[pindex].get('truth_color', 'tab:red'),
                'title_kwargs': setup[pindex].get('title_kwargs', {"fontsize": 20}),
                'label_kwargs': setup[pindex].get('label_kwargs', {"fontsize": 20}),
                'smooth': setup[pindex].get('smooth', 0.5),
                'labelpad': setup[pindex].get('labelpad', 0.12),
                'max_n_ticks': setup[pindex].get('max_n_ticks', 4),
                'use_math_text': setup[pindex].get('use_math_text', True),
            }
            if 'levels' in setup[pindex]:
                corner_kwargs['levels'] = setup[pindex]['levels']

            npars = len(data.dtype.names)
            if 'figsize' in setup[pindex]:
                figsize = tuple(setup[pindex]['figsize'])
            else:
                side = max(8.0, 3.2 * npars)
                figsize = (side, side)
            corner_kwargs['fig'] = pl.figure(i, figsize=figsize)
            fig = corner.corner(corner_data, **corner_kwargs)
            fig.subplots_adjust(
                left=setup[pindex].get('subplots_left', 0.10),
                bottom=setup[pindex].get('subplots_bottom', 0.10),
                right=setup[pindex].get('subplots_right', 0.98),
                top=setup[pindex].get('subplots_top', 0.98),
                wspace=setup[pindex].get('subplots_wspace', 0.08),
                hspace=setup[pindex].get('subplots_hspace', 0.08),
            )
            for ax in fig.get_axes():
                ax.tick_params(
                    axis="both",
                    labelsize=setup[pindex].get('tick_labelsize', 18),
                    direction=setup[pindex].get('tick_direction', 'out'),
                    length=setup[pindex].get('tick_length', 4),
                    width=setup[pindex].get('tick_width', 1.2),
                    pad=setup[pindex].get('tick_pad', 7),
                )
                ax.xaxis.labelpad = setup[pindex].get('axis_labelpad', 18)
                ax.yaxis.labelpad = setup[pindex].get('axis_labelpad', 18)
                for spine in ax.spines.values():
                    spine.set_linewidth(setup[pindex].get('spine_width', 1.4))
                    spine.set_edgecolor(setup[pindex].get('spine_color', 'black'))

            if setup[pindex].get('path', None) is not None:
                pl.savefig(setup[pindex].get('path', 'distribution.png'), dpi=plot_dpi)


# ====================================================================================================================
# Command line stuff below.

def create_setup(args):
    object_name = args. object_name
    grid  = args.grid

    filename = "{}_setup_{}.yaml".format(object_name, grid)

    # excluded photometry
    photband_exclude = "['GALEX', 'SkyMapper', 'SDSS', 'WISE_RSR_W3', 'WISE_RSR_W4']"

    # parameter ranges
    if grid != 'binary':
        ranges = model.get_grid_ranges(grid=grid)
        ranges['distance'] = (1, 10000)
        ranges['av'] = (0, 0.30)
        fixed = {}
        fitted = []
        parameter_limits = ""
        for par in ['teff', 'logg', 'rad', 'distance', 'av']:
            low, high = ranges[par]
            if low < high:
                fitted.append(par)
                parameter_limits += "\n- [{}, {}]  # {}".format(low, high, par)
            else:
                fixed[par] = low
        if 'feh' in ranges:
            fixed['feh'] = 0.0
        parameter_names = '[' + ', '.join(fitted) + ']'
        if fixed:
            fixed_parameters = ''.join("\n  {}: {}".format(par, val)
                                       for par, val in fixed.items())
        else:
            fixed_parameters = " {}"
    else:
        fixed_parameters = None
        parameter_names = None
        parameter_limits = ""

    priors = "{}"

    # grids
    if grid == 'binary':
        model_grids = "- ck_all\n- tmap_he000"
    else:
        model_grids = "- {}".format(grid)

    out = default_single if grid != 'binary' else default_binary
    out = out.replace('<objectname>', object_name)
    out = out.replace('<photfilename>', object_name + '.phot')
    out = out.replace('<photband_exclude>', photband_exclude)
    if parameter_names is not None:
        out = out.replace('<parameter_names>', parameter_names)
    out = out.replace('<parameter_limits>', parameter_limits)
    out = out.replace('<priors>', priors)
    out = out.replace('<model_grids>', model_grids)
    out = out.replace('<postfix>', grid)
    if fixed_parameters is not None:
        out = out.replace('fixed: {}', 'fixed:{}'.format(fixed_parameters))

    ofile = open(filename, 'w')
    ofile.write(out)
    ofile.close()

    print(f"To start the fit run:\n\tsedforge fit {filename}")


def perform_fit(args):
    setup_file = args.setup_file
    noplot = args.noplot

    # -- load the setup file
    ifile = open(setup_file)
    setup = yaml.safe_load(ifile)
    ifile.close()

    # -- check if the provided setup is valid and provide some useful feedback to the user if not.
    validate_setup(setup)

    # -- obtain the observations
    photbands, obs, obs_err = get_observations(setup)

    # -- perform the SED fit
    results, samples, priors, gridnames = fit_sed(setup, photbands, obs, obs_err)

    # -- write the results
    write_results(setup, results, samples, obs, obs_err, photbands)

    # -- create plots
    plot_results(setup, results, samples, priors, gridnames, obs, obs_err, photbands)

    print("================================================================================")
    print("")
    print("Resulting parameter values and errors:")
    print("   Par             Best        Pc       emin       emax")
    for p in samples.dtype.names:
        print("   {:10s} = {}   {}   -{}   +{}".format(p, *plotting.format_parameter(p, results[p])))

    if not noplot:
        pl.show()


def check_grids(args):
    """
    Run the check model grids function and report the results to the user.
    """
    print_bands = args.bands

    model.check_grids(print_bands=print_bands)


def create_photometry(args):
    """
    Download VizieR photometry and write the magnitude table used by the fitter.
    """
    if args.gaia_id is not None and (args.ra is not None or args.dec is not None):
        raise ValueError("Use either --gaia-id or --ra/--dec, not both.")

    config_path = args.catalog_config or catalog_photometry.default_catalog_config()
    table, _metadata, coord = catalog_photometry.download_photometry(
        config_path=config_path,
        output=args.output,
        ra=args.ra,
        dec=args.dec,
        coord=args.coord,
        gaia_id=args.gaia_id,
        radius_arcsec=args.radius,
        default_mag_error=args.default_mag_error,
        timeout=args.timeout,
        metadata_output=args.metadata_output,
    )
    print("Resolved coordinate: ra={:.8f} deg dec={:.8f} deg".format(
        coord.ra.deg,
        coord.dec.deg,
    ))
    print("Catalog config: {}".format(config_path))
    print("Wrote {} photometric bands to {}".format(len(table), args.output))
    if args.metadata_output is not None:
        print("Wrote catalog query summary to {}".format(args.metadata_output))


def main():
    parser = argparse.ArgumentParser(description="Fit flux-calibrated photometric SEDs")

    subparsers = parser.add_subparsers(dest='action')

    # --setup--
    setup_parser = subparsers.add_parser('setup', help='Create yaml setup files for the SED fit')

    setup_parser.add_argument('object_name', default=None,
                             help='Target identifier used to name the setup file')
    setup_parser.add_argument('-grid', default='ck_all',
                             help='The model grid to use (for example ck_all, tlusty_all, koester2, '
                                  'tmap_he000, or binary). Parameter ranges are set automatically '
                                  'based on the grid name.')
    setup_parser.set_defaults(func=create_setup)

    # --fit--
    fit_parser = subparsers.add_parser('fit', help='Fit an SED based on the obtained photometry and setup file')

    fit_parser.add_argument('setup_file', default=None,
                            help='Name of the setup yaml file with all information necessary for the fit')
    fit_parser.add_argument('--noplot', dest='noplot', action='store_true',
                            help="Don't show any plots, only save to disk.")
    fit_parser.set_defaults(func=perform_fit)

    # --check grids--
    grid_parser = subparsers.add_parser('checkgrids', help='Check which model atmosphere grids are installed')

    grid_parser.add_argument('--bands', dest='bands', action='store_true',
                             help="List the photometric bands included in the integrated grids.")
    grid_parser.set_defaults(func=check_grids)

    # --photometry--
    phot_parser = subparsers.add_parser(
        'photometry',
        help='Download VizieR photometry and write a magnitude-format photometry file',
    )
    target = phot_parser.add_mutually_exclusive_group()
    target.add_argument('--gaia-id', dest='gaia_id',
                        help='Gaia DR3 source_id. The YAML config must include a source_id catalog '
                             'with coordinates for resolution.')
    target.add_argument('--coord', dest='coord',
                        help='SkyCoord-readable coordinate string, for example "10.1 -20.2".')
    phot_parser.add_argument('--ra', dest='ra', type=float,
                             help='Right ascension in degrees. Use with --dec.')
    phot_parser.add_argument('--dec', dest='dec', type=float,
                             help='Declination in degrees. Use with --ra.')
    phot_parser.add_argument('--catalog-config',
                             help='YAML file defining which VizieR catalogs and columns to use. '
                                  'Default: bundled Gaia/2MASS/WISE/PS1/SDSS/'
                                  'GLIMPSE/SkyMapper/GALEX config.')
    phot_parser.add_argument('-o', '--output', required=True,
                             help='Output photometry file in photband mag mag_err system format '
                                  'with optional mag_type/mag_zp_offset columns.')
    phot_parser.add_argument('--metadata-output',
                             help='Optional table summarizing catalog hits and selected separations.')
    phot_parser.add_argument('--radius', type=float, default=3.0,
                             help='Cone-search radius in arcsec. Default: 3.')
    phot_parser.add_argument('--default-mag-error', type=float, default=0.03,
                             help='Magnitude error used when a catalog lacks an error column. Default: 0.03.')
    phot_parser.add_argument('--timeout', type=float, default=60,
                             help='Timeout per VizieR request in seconds. Default: 60.')
    phot_parser.set_defaults(func=create_photometry)

    args = parser.parse_args()
    if not hasattr(args, 'func'):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
