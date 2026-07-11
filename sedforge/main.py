import os
import re
import csv
import copy
import time
import traceback
import warnings

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
_MAP_INITIALIZATION_METHODS = ('map', 'best', 'optimize', 'optimise')


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


def _initialization_method(setup):
    """Return the requested walker initializer, preserving the legacy alias."""
    return str(setup.get('init_method', setup.get('initialization', 'auto'))).lower()


def _reject_map_initialization_for_hdf5_grids(setup):
    """L-BFGS-B MAP initialization is unsafe on piecewise HDF5 grids."""
    if _initialization_method(setup) not in _MAP_INITIALIZATION_METHODS:
        return

    gridnames = setup.get('grids', [])
    if model.uses_hdf5_integrated_grid(gridnames):
        raise ValueError(
            "MAP initialization uses L-BFGS-B and is disabled for HDF5 integrated "
            "grids because their likelihood can be piecewise and non-rectangular. "
            "Use init_method: auto or init_method: grid for the grid-aware "
            "initializer."
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
    _reject_map_initialization_for_hdf5_grids(setup)

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


def _prepare_fit_parameters(setup):
    """Validate physical parameters before any model-grid I/O."""
    _reject_retired_setup_keys(setup)
    _reject_legacy_ebv_setup_parameter(setup)
    _reject_reddening_rv_for_rv_grids(setup)
    _reject_map_initialization_for_hdf5_grids(setup)

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

    return pnames, limits, fixed_variables


def _load_fit_grids(setup, photbands, pnames, limits, fixed_variables):
    """Construct grids after setup-level parameter and prior validation."""
    gridnames = setup['grids']
    grid_pnames, grid_limits = _parameters_for_grid(pnames, limits, fixed_variables)
    grids = model.load_grids(gridnames, grid_pnames, grid_limits, photbands,
                             grid_variables=setup.get('grid_variables', None),
                             reddening_law=setup.get('reddening_law', 'WC2019'),
                             reddening_Rv=setup.get('reddening_Rv',
                                                    setup.get('Rv', 3.1)),
                             reddening_case1=setup.get('reddening_case1',
                                                       setup.get('case1', 1)),
                             use_cache=setup.get('grid_cache', True),
                             hdf5_preload=setup.get('hdf5_preload', False),
                             hdf5_preload_max_gb=setup.get(
                                 'hdf5_preload_max_gb',
                                 model.DEFAULT_HDF5_PRELOAD_MAX_GB,
                             ),
                             hdf5_walker_cache=setup.get('hdf5_walker_cache', True),
                             hdf5_runtime_cache_dir=setup.get(
                                 'hdf5_runtime_cache_dir',
                                 os.environ.get('SEDFORGE_RUNTIME_CACHE'),
                             ),
                             runtime_cache_dir=setup.get(
                                 'runtime_grid_cache_dir',
                                 setup.get(
                                     'hdf5_runtime_cache_dir',
                                     os.environ.get('SEDFORGE_RUNTIME_CACHE'),
                                 ),
                             ))
    return gridnames, grids


def fit_sed(setup, photbands, obs, obs_err):

    pnames, limits, fixed_variables = _prepare_fit_parameters(setup)

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

    gridnames, grids = _load_fit_grids(
        setup, photbands, pnames, limits, fixed_variables,
    )

    # -- pars mcmc setup
    nwalkers = setup.get('nwalkers', 24)
    nsteps = setup.get('nsteps', 4000)
    nrelax = setup.get('nrelax', 500)
    nworkers = setup.get('nworkers', setup.get('threads', 1))
    a = setup.get('a', 10)
    init_method = _initialization_method(setup)
    init_ntries = setup.get('init_ntries', 8)
    init_spread = setup.get('init_spread', 1e-3)
    init_grid_max_spectra = setup.get('init_grid_max_spectra', 12000)
    init_grid_rv_points = setup.get('init_grid_rv_points', 7)
    init_grid_av_points = setup.get('init_grid_av_points', 11)
    init_grid_top_candidates = setup.get('init_grid_top_candidates', 256)
    init_grid_max_modes = setup.get('init_grid_max_modes', 6)
    init_grid_min_separation = setup.get('init_grid_min_separation', 0.05)
    init_max_delta_logprob = setup.get('init_max_delta_logprob', 25.0)
    init_grid_rescue = setup.get('init_grid_rescue', True)
    init_grid_rescue_chi2_threshold = setup.get(
        'init_grid_rescue_chi2_threshold', None)
    init_grid_rescue_cache_max_gb = setup.get(
        'init_grid_rescue_cache_max_gb', 2.0)
    init_grid_rescue_maxiter = setup.get('init_grid_rescue_maxiter', 80)
    init_grid_rescue_popsize = setup.get('init_grid_rescue_popsize', 12)
    hdf5_walker_cache_padding = setup.get('hdf5_walker_cache_padding', 4)
    hdf5_walker_cache_max_gb = setup.get('hdf5_walker_cache_max_gb', 0.25)
    hdf5_walker_cache_refresh = setup.get('hdf5_walker_cache_refresh', 0)
    hdf5_walker_cache_max_modes = setup.get('hdf5_walker_cache_max_modes', 6)
    hdf5_walker_cache_envelope_max_gb = setup.get(
        'hdf5_walker_cache_envelope_max_gb', 2.0)
    hdf5_auto_full_cache_max_gb = setup.get(
        'hdf5_auto_full_cache_max_gb', 2.0)
    convergence_rhat_threshold = setup.get('convergence_rhat_threshold', 1.05)
    convergence_min_acceptance = setup.get('convergence_min_acceptance', 0.01)
    convergence_min_bulk_ess = setup.get('convergence_min_bulk_ess', 100.0)
    convergence_min_tail_ess = setup.get('convergence_min_tail_ess', 100.0)
    convergence_action = setup.get('convergence_action', 'warn')
    autostop = setup.get('autostop', False)
    autostop_check_interval = setup.get('autostop_check_interval', 200)
    autostop_tau_factor = setup.get('autostop_tau_factor', 50.0)
    autostop_tolerance = setup.get('autostop_tolerance', 0.01)

    # -- MCMC
    results, samples = mcmc.MCMC(obs, obs_err, photbands,
                                 mcmc_pnames, mcmc_limits, grids,
                                 fixed_variables=fixed_variables,
                                 priors=priors,
                                 error_model=error_model,
                                 nwalkers=nwalkers, nsteps=nsteps, nrelax=nrelax,
                                 a=a, nworkers=nworkers,
                                 init_method=init_method,
                                 init_ntries=init_ntries,
                                 init_spread=init_spread,
                                 init_grid_max_spectra=init_grid_max_spectra,
                                 init_grid_rv_points=init_grid_rv_points,
                                 init_grid_av_points=init_grid_av_points,
                                 init_grid_top_candidates=init_grid_top_candidates,
                                 init_grid_max_modes=init_grid_max_modes,
                                 init_grid_min_separation=init_grid_min_separation,
                                 init_max_delta_logprob=init_max_delta_logprob,
                                 init_grid_rescue=init_grid_rescue,
                                 init_grid_rescue_chi2_threshold=init_grid_rescue_chi2_threshold,
                                 init_grid_rescue_cache_max_gb=init_grid_rescue_cache_max_gb,
                                 init_grid_rescue_maxiter=init_grid_rescue_maxiter,
                                 init_grid_rescue_popsize=init_grid_rescue_popsize,
                                 hdf5_walker_cache_padding=hdf5_walker_cache_padding,
                                 hdf5_walker_cache_max_gb=hdf5_walker_cache_max_gb,
                                 hdf5_walker_cache_refresh=hdf5_walker_cache_refresh,
                                 hdf5_walker_cache_max_modes=hdf5_walker_cache_max_modes,
                                 hdf5_walker_cache_envelope_max_gb=hdf5_walker_cache_envelope_max_gb,
                                 hdf5_auto_full_cache_max_gb=hdf5_auto_full_cache_max_gb,
                                 convergence_rhat_threshold=convergence_rhat_threshold,
                                 convergence_min_acceptance=convergence_min_acceptance,
                                 convergence_min_bulk_ess=convergence_min_bulk_ess,
                                 convergence_min_tail_ess=convergence_min_tail_ess,
                                 convergence_action=convergence_action,
                                 autostop=autostop,
                                 autostop_check_interval=autostop_check_interval,
                                 autostop_tau_factor=autostop_tau_factor,
                                 autostop_tolerance=autostop_tolerance,
                                 progress=setup.get('progress', True),
                                 vectorized_likelihood=setup.get(
                                     'vectorized_likelihood', True,
                                 ))

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
        if par.startswith('_') or isinstance(results[par], dict):
            continue
        if not hasattr(results[par], '__iter__') or len(results[par]) < 4:
            continue
        outpars.append(par)
        outpars.append(par + '_err_minus')
        outpars.append(par + '_err_plus')
        outvals.append(results[par][1])
        outvals.append(results[par][2])
        outvals.append(results[par][3])

    diagnostics = results.get('_mcmc_diagnostics', {})
    if isinstance(diagnostics, dict):
        for name in ('status', 'post_burn_steps', 'nwalkers', 'max_split_rhat',
                     'mean_acceptance_fraction', 'min_acceptance_fraction',
                     'min_bulk_ess', 'min_tail_ess',
                     'initialization_seconds', 'grid_initialization_seconds',
                     'hdf5_cache_preload_seconds',
                     'hdf5_walker_cache_preloaded', 'vectorized_likelihood'):
            if name in diagnostics:
                outpars.append('mcmc_' + name)
                outvals.append(diagnostics[name])
        hdf5_cache = diagnostics.get('hdf5_cache', {})
        if isinstance(hdf5_cache, dict):
            for name in ('cached_points', 'invalid_cached_points',
                         'fallback_points', 'estimate_gb', 'runtime_cache_hit'):
                if name in hdf5_cache:
                    outpars.append('mcmc_hdf5_cache_' + name)
                    outvals.append(hdf5_cache[name])

    resultfile = setup.get('resultfile', None)
    if resultfile is not None:
        import pandas as pd
        data = pd.DataFrame(data=[outvals], columns=outpars)
        data.to_csv(resultfile, index=False)

    diagnosticsfile = setup.get('diagnosticsfile', None)
    if diagnosticsfile is not None and isinstance(diagnostics, dict):
        with open(diagnosticsfile, 'w') as handle:
            yaml.safe_dump(diagnostics, handle, sort_keys=False)

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


def run_fit(setup, noplot=True, make_plots=True):
    validate_setup(setup)
    photbands, obs, obs_err = get_observations(setup)
    results, samples, priors, gridnames = fit_sed(setup, photbands, obs, obs_err)
    write_results(setup, results, samples, obs, obs_err, photbands)
    if make_plots:
        plot_results(setup, results, samples, priors, gridnames, obs, obs_err, photbands)
        if not noplot:
            pl.show()
        else:
            pl.close('all')
    return results, samples, priors, gridnames


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

    results, samples, _priors, _gridnames = run_fit(
        setup,
        noplot=noplot,
        make_plots=True,
    )

    print("================================================================================")
    print("")
    print("Resulting parameter values and errors:")
    print("   Par             Best        Pc       emin       emax")
    for p in samples.dtype.names:
        print("   {:10s} = {}   {}   -{}   +{}".format(p, *plotting.format_parameter(p, results[p])))


def check_grids(args):
    """
    Run the check model grids function and report the results to the user.
    """
    print_bands = args.bands

    model.check_grids(print_bands=print_bands)


def _parse_batch_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text == '':
            return None
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError:
            return text
    return value


def _set_nested_value(mapping, dotted_key, value):
    keys = str(dotted_key).split('.')
    target = mapping
    for key in keys[:-1]:
        if key not in target or target[key] is None:
            target[key] = {}
        if not isinstance(target[key], dict):
            raise ValueError(
                "Cannot set '{}': '{}' is already a non-dictionary value.".format(
                    dotted_key,
                    key,
                )
            )
        target = target[key]
    target[keys[-1]] = value


def _read_batch_manifest(path):
    with open(path, newline='') as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        return [dict(row) for row in reader]


def _resolve_setup_paths(setup, base_dir):
    for key in ('photometryfile',):
        value = setup.get(key)
        if value is not None and not os.path.isabs(str(value)):
            setup[key] = os.path.abspath(os.path.join(base_dir, str(value)))


def _setup_from_batch_row(row, template, manifest_dir, index):
    setup = copy.deepcopy(template)
    for key, raw_value in row.items():
        if key in ('setup_file', 'output_dir'):
            continue
        value = _parse_batch_value(raw_value)
        if value is None:
            continue
        if key == 'source_id':
            setup.setdefault('object_name', str(value))
            continue
        if '.' in key:
            _set_nested_value(setup, key, value)
        else:
            setup[key] = value

    if 'photometryfile' in setup and not os.path.isabs(str(setup['photometryfile'])):
        setup['photometryfile'] = os.path.abspath(
            os.path.join(manifest_dir, str(setup['photometryfile']))
        )

    output_dir = _parse_batch_value(row.get('output_dir'))
    if output_dir is not None:
        output_dir = str(output_dir)
        if not os.path.isabs(output_dir):
            output_dir = os.path.abspath(os.path.join(manifest_dir, output_dir))
        os.makedirs(output_dir, exist_ok=True)
        name = str(setup.get('object_name', row.get('source_id') or f'source_{index:05d}'))
        safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', name).strip('_') or f'source_{index:05d}'
        setup.setdefault('resultfile', os.path.join(output_dir, f'{safe_name}_results.csv'))
        setup.setdefault('datafile', os.path.join(output_dir, f'{safe_name}_samples.fits'))

    return setup


_BATCH_CONTEXT = {}


def _batch_worker_init(template, manifest_dir, make_plots, noplot,
                       fit_workers_per_source, resolved_setups=None):
    for var in (
            'OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS',
            'NUMEXPR_NUM_THREADS',
    ):
        os.environ.setdefault(var, '1')
    _BATCH_CONTEXT.clear()
    _BATCH_CONTEXT.update({
        'template': template,
        'manifest_dir': manifest_dir,
        'make_plots': make_plots,
        'noplot': noplot,
        'fit_workers_per_source': fit_workers_per_source,
        'resolved_setups': {} if resolved_setups is None else resolved_setups,
    })


def _summary_from_results(row, index, setup, results):
    summary = {
        'index': index,
        'source_id': row.get('source_id', setup.get('object_name', index)),
        'photometryfile': setup.get('photometryfile', ''),
        'resultfile': setup.get('resultfile', ''),
        'datafile': setup.get('datafile', ''),
        'status': 'ok',
        'message': '',
    }
    for par, values in results.items():
        if str(par).startswith('_') or isinstance(values, dict):
            continue
        if not hasattr(values, '__iter__') or len(values) < 4:
            continue
        summary[f'{par}_best'] = values[0]
        summary[par] = values[1]
        summary[f'{par}_err_minus'] = values[2]
        summary[f'{par}_err_plus'] = values[3]

    diagnostics = results.get('_mcmc_diagnostics', {})
    if isinstance(diagnostics, dict):
        for name in (
                'status', 'passed', 'max_split_rhat',
                'mean_acceptance_fraction', 'min_acceptance_fraction',
                'min_bulk_ess', 'min_tail_ess',
                'initialization_seconds', 'grid_initialization_seconds',
                'hdf5_cache_preload_seconds',
                'hdf5_cache_preloaded_before_grid_initialization',
                'hdf5_cache_reused_at_fit_start', 'vectorized_likelihood'):
            if name in diagnostics:
                summary[f'mcmc_{name}'] = diagnostics[name]
        cache = diagnostics.get('hdf5_cache', {})
        if isinstance(cache, dict):
            for name in (
                    'cached_points', 'invalid_cached_points',
                    'fallback_points', 'estimate_gb', 'runtime_cache_hit'):
                if name in cache:
                    summary[f'mcmc_hdf5_cache_{name}'] = cache[name]
    return summary


def _batch_task_setup(index, row):
    """Resolve one manifest row into the same setup used by a source worker."""
    resolved = _BATCH_CONTEXT.setdefault('resolved_setups', {})
    if index in resolved:
        return resolved[index]

    setup_file = _parse_batch_value(row.get('setup_file'))
    if setup_file is not None:
        setup_file = str(setup_file)
        if not os.path.isabs(setup_file):
            setup_file = os.path.abspath(
                os.path.join(_BATCH_CONTEXT['manifest_dir'], setup_file)
            )
        with open(setup_file) as handle:
            setup = yaml.safe_load(handle)
        _resolve_setup_paths(setup, os.path.dirname(setup_file))
        resolved[index] = setup
        return setup

    if _BATCH_CONTEXT['template'] is None:
        raise ValueError(
            "Batch rows without a setup_file require --setup-template."
        )
    setup = _setup_from_batch_row(
        row,
        _BATCH_CONTEXT['template'],
        _BATCH_CONTEXT['manifest_dir'],
        index,
    )
    resolved[index] = setup
    return setup


def _prewarm_batch_shared_grid_cache(tasks, max_gb=4.0, runtime_cache_dir=None):
    """Load one union-of-limits integrated grid before source workers fork."""
    common_signature = None
    common_gridname = None
    common_photbands = None
    common_variables = None
    common_format = None
    common_reddening = None
    union_ranges = {}

    for index, row in tasks:
        setup = _batch_task_setup(index, row)
        if not setup.get('grid_cache', True):
            print("Shared batch grid cache skipped because grid caching is disabled.")
            return False
        validate_setup(setup)
        photbands, _obs, _obs_err = get_observations(setup)
        pnames, limits, fixed_variables = _prepare_fit_parameters(setup)
        raw_gridnames = setup['grids']
        gridnames = [raw_gridnames] if isinstance(raw_gridnames, str) else list(raw_gridnames)
        if len(gridnames) != 1:
            print("Shared batch grid cache is available for one component only.")
            return False
        integrated_format = model._grid_integrated_format(gridnames[0])
        if integrated_format == 'hdf5' and not setup.get('hdf5_walker_cache', True):
            print("Shared HDF5 batch cache skipped because hdf5_walker_cache is disabled.")
            return False

        grid_pnames, grid_limits = _parameters_for_grid(
            pnames, limits, fixed_variables,
        )
        variables = model._variables_for_component(
            setup.get('grid_variables', None),
            gridnames[0],
            0,
            grid_pnames,
            '',
        )
        ranges = {
            variable: model._range_for(variable, grid_pnames, grid_limits, '')
            for variable in variables
        }
        signature = (
            str(gridnames[0]),
            tuple(str(name) for name in variables),
            integrated_format,
            str(setup.get('reddening_law', 'WC2019')),
            float(setup.get('reddening_Rv', setup.get('Rv', 3.1))),
            int(setup.get('reddening_case1', setup.get('case1', 1))),
        )
        if common_signature is None:
            common_signature = signature
            common_gridname = gridnames[0]
            common_photbands = []
            common_variables = list(variables)
            common_format = integrated_format
            common_reddening = signature[3:]
        elif signature != common_signature:
            print(
                "Shared batch grid cache skipped because rows use different "
                "grids or model variables."
            )
            return False

        for photband in photbands:
            if photband not in common_photbands:
                common_photbands.append(str(photband))

        for name, bounds in ranges.items():
            low, high = map(float, bounds)
            if name not in union_ranges:
                union_ranges[name] = [low, high]
            else:
                union_ranges[name][0] = min(union_ranges[name][0], low)
                union_ranges[name][1] = max(union_ranges[name][1], high)

    if common_signature is None:
        return False
    cache_limit = 4.0 if max_gb is None else float(max_gb)
    if common_format == 'hdf5':
        shared_grid = model.prepare_hdf5_grid(
            common_photbands,
            common_gridname,
            variables=common_variables,
            ranges=union_ranges,
            preload=False,
            allow_walker_cache=True,
            runtime_cache_dir=runtime_cache_dir,
        )
        loaded = shared_grid.preload_full_active_subgrid(max_gb=cache_limit)
        if loaded:
            shared_grid.register_active_cache_as_shared()
        shared_grid.close()
        estimate_gb = shared_grid.cache_diagnostics()['estimate_gb'] if loaded else 0.0
    else:
        reddening_law, reddening_rv, reddening_case1 = common_reddening
        axis_values, _grid_pars, pixelgrid, grid_names = model.prepare_grid(
            common_photbands,
            common_gridname,
            variables=common_variables,
            ranges=union_ranges,
            reddening_law=reddening_law,
            reddening_Rv=reddening_rv,
            reddening_case1=reddening_case1,
            runtime_cache_dir=runtime_cache_dir,
        )
        estimate_gb = pixelgrid.nbytes / 1024.0 ** 3
        loaded = estimate_gb <= cache_limit
        if loaded:
            model.register_shared_fits_grid(
                common_gridname,
                common_variables,
                common_photbands,
                axis_values,
                pixelgrid,
                grid_names,
                reddening_law=reddening_law,
                reddening_rv=reddening_rv,
                reddening_case1=reddening_case1,
            )
        else:
            print(
                "Shared FITS union grid is {:.3f} GB, above the {:.3f} GB cap.".format(
                    estimate_gb, cache_limit,
                )
            )
    if loaded:
        print(
            "Prewarmed {:.3f} GB shared batch union grid cache for {} sources; "
            "workers inherit it through fork.".format(
                estimate_gb, len(tasks)
            )
        )
    return loaded


def _batch_worker_fit(task):
    index, row = task
    started = time.perf_counter()
    try:
        setup = _batch_task_setup(index, row)

        setup['nworkers'] = int(_BATCH_CONTEXT['fit_workers_per_source'])
        setup['progress'] = False
        results, _samples, _priors, _gridnames = run_fit(
            setup,
            noplot=_BATCH_CONTEXT['noplot'],
            make_plots=_BATCH_CONTEXT['make_plots'],
        )
        summary = _summary_from_results(row, index, setup, results)
        summary['elapsed_seconds'] = time.perf_counter() - started
        return summary
    except Exception as exc:
        return {
            'index': index,
            'source_id': row.get('source_id', index),
            'photometryfile': row.get('photometryfile', ''),
            'resultfile': '',
            'datafile': '',
            'status': 'failed',
            'message': '{}\n{}'.format(exc, traceback.format_exc()),
            'elapsed_seconds': time.perf_counter() - started,
        }


def _write_batch_summary(rows, path):
    if path is None:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_batch(args):
    batch_started = time.perf_counter()
    manifest = os.path.abspath(args.manifest)
    manifest_dir = os.path.dirname(manifest)
    rows = _read_batch_manifest(manifest)
    if len(rows) == 0:
        raise ValueError("Batch manifest is empty.")

    template = None
    if args.setup_template is not None:
        with open(args.setup_template) as handle:
            template = yaml.safe_load(handle)

    summary_path = args.summary
    if summary_path is not None and not os.path.isabs(summary_path):
        summary_path = os.path.abspath(os.path.join(manifest_dir, summary_path))

    make_plots = bool(args.plots)
    noplot = True
    workers = int(args.workers)
    fit_workers_per_source = int(args.fit_workers_per_source)
    max_tasks_per_worker = getattr(args, 'max_tasks_per_worker', None)
    if max_tasks_per_worker is not None:
        max_tasks_per_worker = int(max_tasks_per_worker)
        if max_tasks_per_worker < 1:
            raise ValueError("--max-tasks-per-worker must be at least 1.")
    tasks = list(enumerate(rows, start=1))

    _batch_worker_init(
        template,
        manifest_dir,
        make_plots,
        noplot,
        fit_workers_per_source,
    )

    shared_grid_cache = bool(getattr(args, 'shared_grid_cache', True))
    shared_grid_cache_max_gb = getattr(args, 'shared_grid_cache_max_gb', None)
    runtime_grid_cache_dir = getattr(args, 'runtime_grid_cache_dir', None)
    shared_grid_cache_preloaded = False
    shared_grid_cache_preload_seconds = 0.0
    if workers > 1 and shared_grid_cache:
        shared_grid_cache_started = time.perf_counter()
        try:
            shared_grid_cache_preloaded = _prewarm_batch_shared_grid_cache(
                tasks,
                max_gb=shared_grid_cache_max_gb,
                runtime_cache_dir=runtime_grid_cache_dir,
            )
        except Exception as exc:
            warnings.warn(
                "Shared batch grid-cache prewarm failed ({}); continuing with "
                "per-worker loading.".format(exc),
                RuntimeWarning,
                stacklevel=2,
            )
        shared_grid_cache_preload_seconds = (
            time.perf_counter() - shared_grid_cache_started
        )
        print(
            "Shared batch cache preparation completed in {:.2f} s (loaded={}).".format(
                shared_grid_cache_preload_seconds,
                shared_grid_cache_preloaded,
            )
        )

    summaries = []
    if workers <= 1:
        for task in tasks:
            summary = _batch_worker_fit(task)
            summaries.append(summary)
            print("[{}/{}] {} {}".format(
                len(summaries),
                len(tasks),
                summary['status'],
                summary.get('source_id', summary['index']),
            ))
    else:
        from multiprocessing import get_context

        with get_context('fork').Pool(
                processes=workers,
                initializer=_batch_worker_init,
                initargs=(template, manifest_dir, make_plots, noplot,
                          fit_workers_per_source,
                          _BATCH_CONTEXT.get('resolved_setups', {})),
                maxtasksperchild=max_tasks_per_worker) as pool:
            for summary in pool.imap_unordered(_batch_worker_fit, tasks):
                summaries.append(summary)
                print("[{}/{}] {} {}".format(
                    len(summaries),
                    len(tasks),
                    summary['status'],
                    summary.get('source_id', summary['index']),
                ), flush=True)

    batch_elapsed_seconds = time.perf_counter() - batch_started
    for summary in summaries:
        summary['batch_shared_grid_cache_preloaded'] = shared_grid_cache_preloaded
        summary['batch_shared_grid_cache_preload_seconds'] = (
            shared_grid_cache_preload_seconds
        )
        summary['batch_elapsed_seconds'] = batch_elapsed_seconds
    summaries.sort(key=lambda item: int(item['index']))
    _write_batch_summary(summaries, summary_path)
    nfail = sum(row['status'] != 'ok' for row in summaries)
    print(
        "Batch complete in {:.2f} s: {} ok, {} failed.".format(
            batch_elapsed_seconds,
            len(summaries) - nfail,
            nfail,
        )
    )
    if summary_path is not None:
        print("Wrote batch summary to {}".format(summary_path))


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

    # --batch--
    batch_parser = subparsers.add_parser(
        'batch',
        help='Fit many SEDs with source-level parallelism',
    )
    batch_parser.add_argument(
        'manifest',
        help='CSV/TSV table. Use a setup_file column, or provide --setup-template '
             'and one row per source with photometryfile plus optional setup overrides.',
    )
    batch_parser.add_argument(
        '--setup-template',
        help='YAML setup used as the base for rows that do not provide setup_file.',
    )
    batch_parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of sources to fit in parallel. Default: 1.',
    )
    batch_parser.add_argument(
        '--fit-workers-per-source',
        type=int,
        default=1,
        help='Inner MCMC workers per source. Default: 1; keep this at 1 for large batches.',
    )
    batch_parser.add_argument(
        '--max-tasks-per-worker',
        type=int,
        default=None,
        help='Recycle a source worker after this many fits. Use 1 for very large '
             'HDF5 active subgrids so memory is returned between sources.',
    )
    shared_cache_group = batch_parser.add_mutually_exclusive_group()
    shared_cache_group.add_argument(
        '--shared-grid-cache',
        dest='shared_grid_cache',
        action='store_true',
        help='Preload one read-only HDF5 active grid before forking source workers '
             '(default). Matching source setups share its physical memory.',
    )
    shared_cache_group.add_argument(
        '--no-shared-grid-cache',
        dest='shared_grid_cache',
        action='store_false',
        help='Disable parent-process integrated-grid cache prewarming.',
    )
    batch_parser.set_defaults(shared_grid_cache=True)
    batch_parser.add_argument(
        '--shared-grid-cache-max-gb',
        type=float,
        default=4.0,
        help='Parent union-cache cap in GB. Default: 4.0. This cache is loaded '
             'once and its physical pages are shared by forked source workers.',
    )
    batch_parser.add_argument(
        '--runtime-grid-cache-dir',
        default=os.environ.get('SEDFORGE_RUNTIME_CACHE'),
        help='Optional directory for persistent memory-mapped integrated-grid runtime '
             'caches. The SEDFORGE_RUNTIME_CACHE environment variable provides '
             'the default.',
    )
    batch_parser.add_argument(
        '--plots',
        action='store_true',
        help='Also generate plots for each source. Disabled by default for speed.',
    )
    batch_parser.add_argument(
        '--summary',
        default='batch_summary.csv',
        help='Output CSV summary path. Default: batch_summary.csv beside the manifest.',
    )
    batch_parser.set_defaults(func=run_batch)

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
