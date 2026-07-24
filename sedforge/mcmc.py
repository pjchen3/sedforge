import warnings
import time
from itertools import product

import numpy as np
from multiprocessing import get_context

import emcee

from . import statfunc, model

_WORKER_STATE = None
_MAP_INITIALIZATION_METHODS = ('map', 'best', 'optimize', 'optimise')
_GRID_INITIALIZATION_METHODS = ('grid', 'grid_search', 'hdf5_grid')
_CONVERGENCE_ACTIONS = ('warn', 'error', 'ignore')


def _init_worker_state(obs, obs_err, limits, kwargs):
    global _WORKER_STATE
    _WORKER_STATE = {
        'obs': obs,
        'obs_err': obs_err,
        'limits': limits,
        'kwargs': kwargs,
    }


def _lnprob_from_worker_state(theta):
    state = _WORKER_STATE
    return lnprob(
        theta,
        state['obs'],
        state['obs_err'],
        state['limits'],
        **state['kwargs'],
    )

def lnlike(pars, derived_properties, y, yerr, **kwargs):
    """
    log likelihood function

    Calculates the chi2 of the model defined by theta compared to observed
    fluxes.
    """
    model_func = kwargs.pop('model_func', model.get_itable)
    stat_func = kwargs.pop('stat_func', statfunc.stat_chi2)


    #-- calculate synthetic fluxes; kwargs contains info about which grid to use
    kwargs.update(pars)
    y_syn, extra_drv = model_func(**kwargs)
    if 'L' in extra_drv:
        for luminosity_name in [name for name in extra_drv if name.startswith('L') and name[1:].isdigit()]:
            suffix = luminosity_name[1:]
            ratio_suffix = '' if suffix == '2' else suffix
            extra_drv['lr' + ratio_suffix] = extra_drv['L'] / extra_drv[luminosity_name]
    derived_properties.update(extra_drv)


    stat_value = stat_func(y,
                           yerr,
                           y_syn,
                           pars,
                           photbands=kwargs.get('photbands', None),
                           error_model=kwargs.get('error_model', None))

    if isinstance(stat_value, tuple):
        chi2, deviance = stat_value
    else:
        chi2 = stat_value
        deviance = stat_value

    #-- add distance to extra derived parameter (which already contains luminosities)
    #   Distance is a real sampled or fixed parameter in parsec.
    if 'distance' in pars:
        extra_drv['d'] = pars['distance']
    elif 'dist' in pars:
        extra_drv['d'] = pars['dist']
    else:
        extra_drv['d'] = 0
    extra_drv['chi2'] = chi2

    return -deviance/2, extra_drv


def _normalise_priors(priors):
    priors = dict(priors or {})
    for par, val in list(priors.items()):
        if not hasattr(val, '__len__') or len(val) not in (2, 3):
            raise ValueError(
                "Prior '{}' must have the form [value, error] or [value, -error, +error].".format(par)
            )
        if len(val) == 2:
            priors[par] = (val[0], val[1], val[1])
        else:
            priors[par] = tuple(val)
        if priors[par][1] <= 0 or priors[par][2] <= 0:
            raise ValueError("Prior '{}' must have positive uncertainty values.".format(par))
    return priors


def lnprior(theta, derived_properties, limits, **kwargs):
    """
    Uniform parameter limits plus optional Gaussian priors on sampled
    parameters.

    If all parameters are within the provided limits, the returned log
    probability is the Gaussian-prior contribution. Otherwise it is -inf.

    :param theta: list of model parameters
    :type theta: list
    :param limits: limits on the model parameters
    :type limits: list of tuples

    :return: logarithm of the probability of the parameters (theta) given the
             model limits
    :rtype: float
    """

    pnames = kwargs.get('pnames', ())
    priors = _normalise_priors(kwargs.get('priors', {}))

    #-- check if all parameters are within their limits
    if any(theta < limits[:,0]) or any(theta > limits[:,1]):
        return -np.inf

    pars = dict(zip(pnames, theta))
    log_prior = 0.0
    for par, (center, err_minus, err_plus) in priors.items():
        if par not in pars:
            raise ValueError("Prior '{}' is not a sampled parameter.".format(par))
        sigma = err_minus if pars[par] < center else err_plus
        log_prior += -0.5 * ((pars[par] - center) / sigma) ** 2

    return log_prior

def lnprob(theta, y, yerr, limits, **kwargs):
    """
    full log probability function combining the prior and the likelihood

    will return -inf if any of :py:func:`lnprior` or :py:func:`lnlikelyhood` is
    infite, otherwise it will return the sum of both functions.

    :param theta: list of model parameters (normaly mass, fe/h and age)
    :type theta: list
    :param y: 1D array of observables
    :type y: array
    :param yerr: 1D array containing errors on every observable
    :type yerr: array
    :param limits: limits on the model parameters
    :type limits: list of tuples

    :return: the sum of the log prior and log likelihood
    :rtype: float
    """

    #-- create keyword parameters from theta
    pars = {}
    for name, value in zip(kwargs['pnames'], theta):
        pars[name]=value

    #-- add extra variables which are not fitted to pars.
    pars.update(kwargs.pop('fixed_variables', {}))

    #-- get derived properties
    prop_func = kwargs.pop('prop_func', statfunc.get_derived_properties)
    syn_drv = prop_func(**pars)

    #-- calculate prior probability
    lp = lnprior(theta, syn_drv, limits, **kwargs)
    if not np.isfinite(lp):
        return -np.inf, syn_drv

    #-- calculate likelihood
    ll, extra_drv = lnlike(pars, syn_drv, y, yerr, **kwargs)
    syn_drv.update(extra_drv)
    if not np.isfinite(ll):
        return -np.inf, syn_drv

    return lp + ll, syn_drv


def _vectorized_log_prior(theta, limits, pnames, priors):
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    limits = np.asarray(limits, dtype=float)
    values = np.zeros(len(theta), dtype=float)
    inside = np.all(
        (theta >= limits[:, 0][None, :])
        & (theta <= limits[:, 1][None, :]),
        axis=1,
    )
    values[~inside] = -np.inf
    normalised = _normalise_priors(priors)
    lookup = {str(name): index for index, name in enumerate(pnames)}
    for name, (center, err_minus, err_plus) in normalised.items():
        if name not in lookup:
            raise ValueError("Prior '{}' is not a sampled parameter.".format(name))
        parameter = theta[:, lookup[name]]
        sigma = np.where(parameter < center, err_minus, err_plus)
        values[inside] += -0.5 * ((parameter[inside] - center) / sigma[inside]) ** 2
    return values


def _slice_derived_properties(properties, index):
    result = {}
    for name, value in properties.items():
        array = np.asarray(value)
        if array.ndim == 0:
            result[name] = float(array)
        else:
            result[name] = float(array[index])
    return result


def lnprob_vectorized(theta, y, yerr, limits, **kwargs):
    """Evaluate one ensemble proposal in a single model interpolation call."""
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    pnames = list(kwargs['pnames'])
    fixed_variables = dict(kwargs.get('fixed_variables', {}))
    prop_func = kwargs.get('prop_func', statfunc.get_derived_properties)
    priors = kwargs.get('priors', {})

    pars_all = {
        name: theta[:, index]
        for index, name in enumerate(pnames)
    }
    pars_all.update(fixed_variables)
    base_derived = prop_func(**pars_all)
    log_prior = _vectorized_log_prior(theta, limits, pnames, priors)
    finite_indices = np.flatnonzero(np.isfinite(log_prior))
    log_probability = np.full(len(theta), -np.inf, dtype=float)
    finite_derived = None

    if len(finite_indices):
        pars = {
            name: values[finite_indices]
            for name, values in pars_all.items()
            if np.asarray(values).ndim > 0
        }
        pars.update({
            name: value
            for name, value in fixed_variables.items()
        })
        if len(finite_indices) == len(theta):
            finite_derived = base_derived
        else:
            finite_derived = prop_func(**pars)
        likelihood_kwargs = dict(kwargs)
        likelihood_kwargs.pop('fixed_variables', None)
        likelihood_kwargs.pop('prop_func', None)
        likelihood, extra = lnlike(
            pars,
            finite_derived,
            y,
            yerr,
            **likelihood_kwargs,
        )
        finite_derived.update(extra)
        likelihood = np.atleast_1d(np.asarray(likelihood, dtype=float))
        usable = np.isfinite(likelihood)
        log_probability[finite_indices[usable]] = (
            log_prior[finite_indices[usable]] + likelihood[usable]
        )

    results = []
    finite_lookup = {
        int(global_index): local_index
        for local_index, global_index in enumerate(finite_indices)
    }
    for index, value in enumerate(log_probability):
        if index in finite_lookup and finite_derived is not None:
            blob = _slice_derived_properties(finite_derived, finite_lookup[index])
        else:
            blob = _slice_derived_properties(base_derived, index)
        results.append((float(value), blob))
    return results


def _random_positions(limits, nwalkers):
    pos = [np.random.uniform(lim[0], lim[1], nwalkers) for lim in limits]
    return np.array(pos).T


def _negative_lnprob(theta, obs, obs_err, limits, kwargs):
    value, _derived = lnprob(theta, obs, obs_err, limits, **kwargs)
    if not np.isfinite(value):
        return np.inf
    return -float(value)


def _initial_center(pnames, limits, priors):
    priors = _normalise_priors(priors)
    center = []
    for name, (low, high) in zip(pnames, limits):
        if name in priors:
            value = priors[name][0]
        else:
            value = 0.5 * (low + high)
        center.append(float(np.clip(value, low, high)))
    return np.asarray(center, dtype=float)


def _map_positions(obs, obs_err, limits, kwargs, nwalkers,
                   ntries=8, spread=1e-3):
    from scipy.optimize import minimize

    limits = np.asarray(limits, dtype=float)
    pnames = kwargs.get('pnames', ())
    center = _initial_center(pnames, limits, kwargs.get('priors', {}))
    bounds = [tuple(lim) for lim in limits]
    rng = np.random.default_rng()
    starts = [center]
    width = limits[:, 1] - limits[:, 0]
    for _ in range(max(0, int(ntries) - 1)):
        starts.append(rng.uniform(limits[:, 0], limits[:, 1]))

    best_x = None
    best_fun = np.inf
    for start in starts:
        result = minimize(
            _negative_lnprob,
            start,
            args=(obs, obs_err, limits, kwargs),
            method='L-BFGS-B',
            bounds=bounds,
        )
        value = result.fun if np.isfinite(result.fun) else np.inf
        if value < best_fun:
            best_fun = value
            best_x = np.asarray(result.x, dtype=float)

    if best_x is None or not np.all(np.isfinite(best_x)):
        raise ValueError("Could not find a finite MAP starting point.")

    jitter = rng.normal(0.0, np.maximum(width * float(spread), 1e-10),
                        size=(nwalkers, len(bounds)))
    pos = best_x + jitter
    pos = np.clip(pos, limits[:, 0], limits[:, 1])
    return pos


def _limit_lookup(pnames, limits):
    return {
        name: tuple(np.asarray(limit, dtype=float))
        for name, limit in zip(pnames, np.asarray(limits, dtype=float))
    }


def _initial_parameter_value(name, bounds, priors):
    if name in priors:
        return float(np.clip(priors[name][0], bounds[0], bounds[1]))
    return 0.5 * (bounds[0] + bounds[1])


def _candidate_theta(candidate, pnames, limits, fixed_variables, priors):
    """Convert a profiled HDF5 node candidate to one physical MCMC position."""
    pnames = list(pnames)
    bounds = _limit_lookup(pnames, limits)
    values = {}

    for name in pnames:
        if name in candidate:
            values[name] = float(candidate[name])
        else:
            values[name] = _initial_parameter_value(name, bounds[name], priors)

    distance_name = 'distance' if 'distance' in bounds or 'distance' in fixed_variables else 'dist'
    if distance_name not in bounds and distance_name not in fixed_variables:
        raise ValueError("Grid initialization requires a fitted or fixed distance parameter.")
    if 'rad' not in bounds and 'rad' not in fixed_variables:
        raise ValueError("Grid initialization requires a fitted or fixed radius parameter.")

    scale = float(candidate['scale'])
    if not np.isfinite(scale) or scale <= 0:
        return None
    scale_root = np.sqrt(scale)
    conversion = model.PC_TO_RSOL * scale_root

    radius_fixed = fixed_variables.get('rad')
    distance_fixed = fixed_variables.get(distance_name)
    if radius_fixed is not None and distance_fixed is not None:
        values.pop('rad', None)
        values.pop(distance_name, None)
    elif distance_fixed is not None:
        radius = float(distance_fixed) * conversion
        if 'rad' in bounds and not bounds['rad'][0] <= radius <= bounds['rad'][1]:
            return None
        values['rad'] = radius
    elif radius_fixed is not None:
        distance = float(radius_fixed) / conversion
        if distance_name in bounds and not bounds[distance_name][0] <= distance <= bounds[distance_name][1]:
            return None
        values[distance_name] = distance
    else:
        radius_bounds = bounds['rad']
        distance_bounds = bounds[distance_name]
        allowed_distance = (
            max(distance_bounds[0], radius_bounds[0] / conversion),
            min(distance_bounds[1], radius_bounds[1] / conversion),
        )
        if allowed_distance[0] > allowed_distance[1]:
            return None

        if distance_name in priors:
            preferred_distance = priors[distance_name][0]
        elif 'rad' in priors:
            preferred_distance = priors['rad'][0] / conversion
        else:
            preferred_distance = np.sqrt(allowed_distance[0] * allowed_distance[1])
        distance = float(np.clip(preferred_distance, *allowed_distance))
        values[distance_name] = distance
        values['rad'] = distance * conversion

    theta = np.asarray([values[name] for name in pnames], dtype=float)
    if not np.all(np.isfinite(theta)):
        return None
    if np.any(theta < np.asarray(limits, dtype=float)[:, 0]) or \
            np.any(theta > np.asarray(limits, dtype=float)[:, 1]):
        return None
    return theta


def _distinct_grid_seeds(scored, limits, maximum_modes, min_separation,
                         max_delta_logprob):
    """Keep distinct high-posterior seed basins without assuming one mode."""
    if not scored:
        return []
    best_logprob = scored[0][0]
    width = np.maximum(np.asarray(limits, dtype=float)[:, 1] -
                       np.asarray(limits, dtype=float)[:, 0], 1e-12)
    seeds = []
    for logprob, theta in scored:
        if logprob < best_logprob - float(max_delta_logprob):
            break
        if all(np.linalg.norm((theta - other) / width) >= float(min_separation)
               for other in seeds):
            seeds.append(theta)
        if len(seeds) >= int(maximum_modes):
            break
    return seeds or [scored[0][1]]


def _sample_seed_axis_indices(values, count, logarithmic=False):
    indices = np.arange(len(values), dtype=int)
    if len(indices) <= int(count):
        return indices
    selected_values = np.asarray(values, dtype=float)
    coordinates = (
        np.log1p(selected_values)
        if logarithmic and np.all(selected_values >= 0)
        else selected_values
    )
    targets = np.linspace(coordinates[0], coordinates[-1], int(count))
    positions = np.searchsorted(coordinates, targets)
    positions = np.clip(positions, 0, len(indices) - 1)
    previous = np.maximum(positions - 1, 0)
    use_previous = np.abs(coordinates[previous] - targets) < np.abs(
        coordinates[positions] - targets
    )
    positions[use_previous] = previous[use_previous]
    return np.unique(indices[positions])


def _profile_scale_and_chi2(flux, obs, weights):
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


def _prepared_fits_seed_layout(grid, nobs):
    axis_values = [np.asarray(axis) for axis in grid[0]]
    pixelgrid = grid[1]
    names = [str(name).lower() for name in np.asarray(grid[2])]
    metadata = grid[3] if len(grid) > 3 and isinstance(grid[3], dict) else {}
    output_indices = metadata.get('output_indices')
    if output_indices is None:
        output_indices = np.arange(int(nobs), dtype=int)
    else:
        output_indices = np.asarray(output_indices, dtype=int)[:int(nobs)]
    if len(output_indices) != int(nobs) or np.any(output_indices >= pixelgrid.shape[-1]):
        raise ValueError(
            "Prepared FITS grid does not contain every fitted photometric band."
        )

    scan_positions = [
        index for index, name in enumerate(names)
        if name in {'av', 'ebv', 'rv'}
    ]
    base_positions = [
        index for index in range(len(names))
        if index not in scan_positions
    ]
    if not base_positions:
        raise ValueError(
            "Grid-aware FITS initialization requires at least one atmosphere axis."
        )
    return (
        axis_values,
        pixelgrid,
        names,
        output_indices,
        base_positions,
        scan_positions,
    )


def _prepared_fits_valid_base_nodes(layout):
    axis_values, pixelgrid, _names, output_indices, base_positions, scan_positions = layout
    reference = []
    for position in range(len(axis_values)):
        reference.append(slice(None) if position in base_positions else 0)
    finite = np.isfinite(
        np.asarray(pixelgrid[tuple(reference) + (int(output_indices[0]),)])
    )
    return np.flatnonzero(finite.reshape(-1))


def _prepared_fits_node_rows(layout, base_indices, scan_indices):
    axis_values, pixelgrid, _names, output_indices, base_positions, scan_positions = layout
    base_shape = tuple(len(axis_values[position]) for position in base_positions)
    base_coordinates = np.unravel_index(
        np.asarray(base_indices, dtype=int),
        base_shape,
    )
    base_lookup = {
        position: np.asarray(coordinates, dtype=int)
        for position, coordinates in zip(base_positions, base_coordinates)
    }
    scan_lookup = {
        position: int(index)
        for position, index in zip(scan_positions, scan_indices)
    }
    count = len(np.asarray(base_indices))
    indices = []
    for position in range(len(axis_values)):
        if position in base_lookup:
            indices.append(base_lookup[position])
        else:
            indices.append(np.full(count, scan_lookup[position], dtype=int))
    log_rows = np.asarray(pixelgrid[tuple(indices)], dtype=float)
    if log_rows.ndim == 1:
        log_rows = log_rows[None, :]
    return 10.0 ** log_rows[:, output_indices]


def _prepared_fits_candidate(layout, base_index, scan_indices, scale, chi2):
    axis_values, _pixelgrid, names, _outputs, base_positions, scan_positions = layout
    base_shape = tuple(len(axis_values[position]) for position in base_positions)
    base_coordinates = np.unravel_index(int(base_index), base_shape)
    index_lookup = {
        position: int(index)
        for position, index in zip(base_positions, base_coordinates)
    }
    index_lookup.update({
        position: int(index)
        for position, index in zip(scan_positions, scan_indices)
    })
    candidate = {
        'spec_index': int(base_index),
        'scale': float(scale),
        'profile_chi2': float(chi2),
    }
    for position, name in enumerate(names):
        candidate[name] = float(axis_values[position][index_lookup[position]])
    return candidate


def _append_prepared_fits_candidates(candidates, layout, base_indices,
                                     scan_combinations, obs, weights,
                                     keep_count, unique_spectra=False,
                                     chunk_size=2048):
    base_indices = np.unique(np.asarray(base_indices, dtype=int))
    for scan_indices in scan_combinations:
        for start in range(0, len(base_indices), int(chunk_size)):
            chunk = base_indices[start:start + int(chunk_size)]
            rows = _prepared_fits_node_rows(layout, chunk, scan_indices)
            scales, chi2 = _profile_scale_and_chi2(rows, obs, weights)
            count = min(int(keep_count), len(chunk))
            if count == 0:
                continue
            selected = np.argpartition(chi2, count - 1)[:count]
            for local_index in selected:
                if not np.isfinite(chi2[local_index]):
                    continue
                candidates.append(_prepared_fits_candidate(
                    layout,
                    chunk[local_index],
                    scan_indices,
                    scales[local_index],
                    chi2[local_index],
                ))

        candidates.sort(key=lambda item: item['profile_chi2'])
        if unique_spectra:
            retained = []
            seen = set()
            for candidate in candidates:
                key = int(candidate['spec_index'])
                if key in seen:
                    continue
                seen.add(key)
                retained.append(candidate)
                if len(retained) >= int(keep_count):
                    break
            candidates[:] = retained
        else:
            del candidates[int(keep_count):]
    return candidates


def _prepared_fits_profile_seed_candidates(
        grid, obs, obs_err, maximum_spectra=12000,
        coarse_rv_points=7, coarse_av_points=11,
        coarse_keep_count=256, refine_spectra=64, result_count=256):
    """Profile normalization on sparse FITS nodes without using gradients."""
    obs = np.asarray(obs, dtype=float)
    obs_err = np.asarray(obs_err, dtype=float)
    valid = np.isfinite(obs) & np.isfinite(obs_err) & (obs > 0) & (obs_err > 0)
    if not np.all(valid):
        raise ValueError(
            "Grid seed search requires finite positive observations and errors."
        )
    weights = 1.0 / obs_err ** 2
    layout = _prepared_fits_seed_layout(grid, len(obs))
    axis_values, _pixels, names, _outputs, _base_positions, scan_positions = layout

    base_indices = _prepared_fits_valid_base_nodes(layout)
    if len(base_indices) > int(maximum_spectra):
        selected = np.linspace(
            0, len(base_indices) - 1, int(maximum_spectra), dtype=int,
        )
        base_indices = np.unique(base_indices[selected])
    if len(base_indices) == 0:
        return []

    coarse_axes = []
    full_axes = []
    for position in scan_positions:
        name = names[position]
        values = axis_values[position]
        full_axes.append(np.arange(len(values), dtype=int))
        coarse_axes.append(_sample_seed_axis_indices(
            values,
            coarse_rv_points if name == 'rv' else coarse_av_points,
            logarithmic=name in {'av', 'ebv'},
        ))
    coarse_combinations = list(product(*coarse_axes)) if coarse_axes else [()]
    candidates = _append_prepared_fits_candidates(
        [],
        layout,
        base_indices,
        coarse_combinations,
        obs,
        weights,
        keep_count=coarse_keep_count,
        unique_spectra=True,
    )
    if not candidates:
        return []

    refined = np.asarray(
        list(dict.fromkeys(
            int(candidate['spec_index']) for candidate in candidates
        ))[:int(refine_spectra)],
        dtype=int,
    )
    full_combinations = product(*full_axes) if full_axes else [()]
    candidates = _append_prepared_fits_candidates(
        candidates,
        layout,
        refined,
        full_combinations,
        obs,
        weights,
        keep_count=result_count,
    )
    candidates.sort(key=lambda item: item['profile_chi2'])
    return candidates[:int(result_count)]


def _grid_list(grids):
    """Return component grids without splitting a prepared FITS tuple."""
    if model._is_prepared_grid(grids):
        return [grids]
    if isinstance(grids, str):
        return [grids]
    return list(grids) if hasattr(grids, '__iter__') else [grids]


def _grid_positions(obs, obs_err, limits, kwargs, nwalkers,
                    maximum_spectra=12000, coarse_rv_points=7,
                    coarse_av_points=11, top_candidates=256,
                    maximum_modes=6, min_separation=0.05,
                    max_delta_logprob=25.0, spread=1e-3,
                    rescue=True, rescue_chi2_threshold=None,
                    rescue_cache_max_gb=2.0, rescue_maxiter=80,
                    rescue_popsize=12,
                    return_seeds=False):
    """Initialize one prepared grid from discrete high-likelihood nodes."""
    grids = _grid_list(kwargs['grid'])
    if len(grids) != 1 or not model._is_prepared_grid(grids[0]):
        raise ValueError(
            "Grid initialization currently supports one prepared stellar component. "
            "Use an explicit scientifically justified initialization for multi-component fits."
        )

    grid = grids[0]
    hdf5_grid = isinstance(grid, model.HDF5IntegratedGrid)
    if hdf5_grid:
        try:
            candidates = grid.profile_seed_candidates(
                obs,
                obs_err,
                maximum_spectra=maximum_spectra,
                coarse_rv_points=coarse_rv_points,
                coarse_av_points=coarse_av_points,
                result_count=top_candidates,
            )
        finally:
            # Do not carry a parent HDF5 handle into forked likelihood workers.
            grid.close()
    else:
        candidates = _prepared_fits_profile_seed_candidates(
            grid,
            obs,
            obs_err,
            maximum_spectra=maximum_spectra,
            coarse_rv_points=coarse_rv_points,
            coarse_av_points=coarse_av_points,
            result_count=top_candidates,
        )
    if not candidates:
        raise ValueError("Grid search found no finite photometric seed candidates.")

    normalised_priors = _normalise_priors(kwargs.get('priors', {}))

    def score_candidates(items):
        scored_items = []
        for candidate in items:
            theta = _candidate_theta(
                candidate,
                kwargs['pnames'],
                limits,
                kwargs.get('fixed_variables', {}),
                normalised_priors,
            )
            if theta is None:
                continue
            logprob, derived = lnprob(theta, obs, obs_err, limits, **kwargs)
            if np.isfinite(logprob):
                scored_items.append((
                    float(logprob),
                    theta,
                    float(derived.get('chi2', np.inf)),
                ))
        scored_items.sort(key=lambda item: item[0], reverse=True)
        return scored_items

    scored = score_candidates(candidates)
    if not scored:
        raise ValueError(
            "Grid search found candidates, but none satisfy the physical "
            "radius/distance limits and priors."
        )

    threshold = rescue_chi2_threshold
    if threshold is None:
        threshold = len(obs) + 6.0 * np.sqrt(2.0 * len(obs))
    if bool(rescue) and hdf5_grid and scored[0][2] > float(threshold):
        initial_chi2 = scored[0][2]
        rescue_candidates = []
        if grid.preload_full_active_subgrid(max_gb=rescue_cache_max_gb):
            candidate = grid.profile_continuous_seed_candidate(
                obs,
                obs_err,
                maxiter=rescue_maxiter,
                popsize=rescue_popsize,
            )
            if candidate is not None:
                rescue_candidates.append(candidate)
        else:
            try:
                rescue_candidates = grid.profile_seed_candidates(
                    obs,
                    obs_err,
                    maximum_spectra=maximum_spectra,
                    coarse_rv_points=max(25, int(coarse_rv_points) * 3),
                    coarse_av_points=max(31, int(coarse_av_points) * 3),
                    coarse_keep_count=max(512, int(top_candidates)),
                    refine_spectra=max(256, int(top_candidates)),
                    result_count=max(512, int(top_candidates) * 2),
                )
            finally:
                grid.close()

        rescued = score_candidates(rescue_candidates)
        if rescued:
            scored.extend(rescued)
            scored.sort(key=lambda item: item[0], reverse=True)
            print(
                "Grid initialization rescue improved best chi2 from {:.3f} to {:.3f}.".format(
                    initial_chi2, scored[0][2],
                )
            )

    scored_pairs = [(logprob, theta) for logprob, theta, _chi2 in scored]

    seeds = _distinct_grid_seeds(
        scored_pairs,
        limits,
        maximum_modes=maximum_modes,
        min_separation=min_separation,
        max_delta_logprob=max_delta_logprob,
    )
    best_logprob = scored_pairs[0][0]
    limits = np.asarray(limits, dtype=float)
    width = limits[:, 1] - limits[:, 0]
    rng = np.random.default_rng()
    pnames = [str(name) for name in kwargs['pnames']]
    parameter_index = {name: index for index, name in enumerate(pnames)}
    distance_name = 'distance' if 'distance' in parameter_index else 'dist'

    def local_seed_sigma(seed):
        local_scale = np.maximum(np.abs(seed), 0.01 * width)
        sigma = np.minimum(width * float(spread), local_scale * float(spread))
        for name, (_center, err_minus, err_plus) in normalised_priors.items():
            if name in parameter_index:
                sigma[parameter_index[name]] = min(
                    sigma[parameter_index[name]],
                    0.25 * 0.5 * (err_minus + err_plus),
                )
        return np.maximum(sigma, 1e-10)

    def preserve_flux_scale(seed, trial, shrink):
        if 'rad' not in parameter_index or distance_name not in parameter_index:
            return trial
        radius_index = parameter_index['rad']
        distance_index = parameter_index[distance_name]
        ratio = seed[radius_index] / seed[distance_index]
        if 'rad' in normalised_priors and distance_name not in normalised_priors:
            trial[distance_index] = trial[radius_index] / ratio
        else:
            trial[radius_index] = trial[distance_index] * ratio
        # A tiny orthogonal displacement avoids placing the ensemble in an
        # exactly lower-dimensional affine subspace.
        trial[radius_index] *= 1.0 + rng.normal(
            0.0, max(1e-8, min(1e-4, float(spread) * 0.01)) * shrink,
        )
        return trial

    positions = []
    for index in range(int(nwalkers)):
        seed = seeds[index % len(seeds)]
        accepted = None
        sigma = local_seed_sigma(seed)
        for attempt in range(48):
            shrink = 0.25 ** (attempt // 12)
            trial = seed + rng.normal(
                0.0,
                sigma * shrink,
            )
            trial = preserve_flux_scale(seed, trial, shrink)
            trial = np.clip(trial, limits[:, 0], limits[:, 1])
            logprob, _derived = lnprob(trial, obs, obs_err, limits, **kwargs)
            if np.isfinite(logprob) and logprob >= best_logprob - float(max_delta_logprob):
                accepted = trial
                break
        if accepted is None:
            raise ValueError(
                "Could not draw a valid walker near a high-posterior grid seed."
            )
        positions.append(accepted)

    print(
        "Initialized {} walkers from {} integrated-grid seed basin(s); "
        "best seed log-probability {:.3f}.".format(
            nwalkers,
            len(seeds),
            best_logprob,
        )
    )
    positions = np.asarray(positions, dtype=float)
    if return_seeds:
        return positions, np.asarray(seeds, dtype=float)
    return positions


def _split_rhat(chain):
    """Return classical split-R-hat values for a (step, walker, parameter) chain."""
    chain = np.asarray(chain, dtype=float)
    if chain.ndim != 3:
        raise ValueError("MCMC chain must have shape (step, walker, parameter).")
    nsteps, nwalkers, ndim = chain.shape
    half = nsteps // 2
    if half < 2 or nwalkers < 2:
        return np.full(ndim, np.nan, dtype=float)

    split = np.concatenate((chain[:half], chain[-half:]), axis=1)
    means = np.mean(split, axis=0)
    variances = np.var(split, axis=0, ddof=1)
    within = np.mean(variances, axis=0)
    between = half * np.var(means, axis=0, ddof=1)
    variance_estimate = ((half - 1.0) / half) * within + between / half

    rhat = np.full(ndim, np.inf, dtype=float)
    nonzero = within > 0
    rhat[nonzero] = np.sqrt(variance_estimate[nonzero] / within[nonzero])
    stationary = ~nonzero & (between == 0)
    rhat[stationary] = 1.0
    return rhat


def _split_chains(chain):
    chain = np.asarray(chain, dtype=float)
    if chain.ndim != 3:
        raise ValueError("MCMC chain must have shape (step, walker, parameter).")
    half = chain.shape[0] // 2
    if half < 2 or chain.shape[1] < 2:
        return None
    return np.concatenate((chain[:half], chain[-half:]), axis=1)


def _rank_normalize(values):
    from scipy.stats import norm, rankdata

    values = np.asarray(values, dtype=float)
    result = np.empty_like(values, dtype=float)
    for parameter in range(values.shape[-1]):
        flat = values[..., parameter].reshape(-1)
        ranks = rankdata(flat, method='average')
        probability = (ranks - 3.0 / 8.0) / (len(flat) + 1.0 / 4.0)
        result[..., parameter] = norm.ppf(probability).reshape(
            values.shape[:-1]
        )
    return result


def _rhat_from_split(split):
    nsteps = split.shape[0]
    means = np.mean(split, axis=0)
    variances = np.var(split, axis=0, ddof=1)
    within = np.mean(variances, axis=0)
    between = nsteps * np.var(means, axis=0, ddof=1)
    variance_estimate = ((nsteps - 1.0) / nsteps) * within + between / nsteps
    result = np.full(split.shape[-1], np.inf, dtype=float)
    nonzero = within > 0
    result[nonzero] = np.sqrt(variance_estimate[nonzero] / within[nonzero])
    result[~nonzero & (between == 0)] = 1.0
    return result


def _rank_normalized_split_rhat(chain):
    """Return Vehtari-style rank-normalized and folded split R-hat."""
    split = _split_chains(chain)
    if split is None:
        ndim = np.asarray(chain).shape[-1]
        empty = np.full(ndim, np.nan, dtype=float)
        return empty, empty, empty
    rank_rhat = _rhat_from_split(_rank_normalize(split))
    median = np.median(split, axis=(0, 1))
    folded = np.abs(split - median[None, None, :])
    folded_rhat = _rhat_from_split(_rank_normalize(folded))
    return np.maximum(rank_rhat, folded_rhat), rank_rhat, folded_rhat


def _autocovariance_fft(values):
    values = np.asarray(values, dtype=float)
    nsteps = values.shape[0]
    centered = values - np.mean(values, axis=0, keepdims=True)
    size = 1 << int(np.ceil(np.log2(max(2, 2 * nsteps))))
    transformed = np.fft.rfft(centered, n=size, axis=0)
    return np.fft.irfft(transformed * np.conjugate(transformed), n=size, axis=0)[:nsteps] / nsteps


def _effective_sample_size_from_split(split):
    """Estimate ESS using Geyer's initial positive monotone sequence."""
    split = np.asarray(split, dtype=float)
    nsteps, nchains, ndim = split.shape
    if nsteps < 3 or nchains < 2:
        return np.full(ndim, np.nan, dtype=float)
    autocov = _autocovariance_fft(split)
    within = np.mean(autocov[0], axis=0)
    chain_means = np.mean(split, axis=0)
    between = nsteps * np.var(chain_means, axis=0, ddof=1)
    variance = ((nsteps - 1.0) / nsteps) * within + between / nsteps
    ess = np.empty(ndim, dtype=float)
    total = float(nsteps * nchains)

    for parameter in range(ndim):
        if not np.isfinite(variance[parameter]) or variance[parameter] <= 0:
            ess[parameter] = total
            continue
        rho = np.ones(nsteps, dtype=float)
        rho[1:] = 1.0 - (
            within[parameter] - np.mean(autocov[1:, :, parameter], axis=1)
        ) / variance[parameter]
        pair_sums = []
        for lag in range(0, nsteps - 1, 2):
            pair = rho[lag] + rho[lag + 1]
            if pair < 0:
                break
            if pair_sums:
                pair = min(pair, pair_sums[-1])
            pair_sums.append(pair)
        tau = -1.0 + 2.0 * np.sum(pair_sums)
        ess[parameter] = np.clip(total / max(tau, 1.0), 1.0, total)
    return ess


def _bulk_tail_ess(chain):
    split = _split_chains(chain)
    if split is None:
        ndim = np.asarray(chain).shape[-1]
        empty = np.full(ndim, np.nan, dtype=float)
        return empty, empty
    bulk = _effective_sample_size_from_split(_rank_normalize(split))
    lower = np.quantile(split, 0.05, axis=(0, 1))
    upper = np.quantile(split, 0.95, axis=(0, 1))
    lower_indicator = (split <= lower[None, None, :]).astype(float)
    upper_indicator = (split >= upper[None, None, :]).astype(float)
    tail = np.minimum(
        _effective_sample_size_from_split(lower_indicator),
        _effective_sample_size_from_split(upper_indicator),
    )
    return bulk, tail


def _identifiability_diagnostics(chain, pnames, limits):
    flat = np.asarray(chain, dtype=float).reshape(-1, len(pnames))
    limits = np.asarray(limits, dtype=float)
    diagnostics = {}
    for index, name in enumerate(pnames):
        low, high = limits[index]
        width = high - low
        q05, q95 = np.quantile(flat[:, index], [0.05, 0.95])
        scale = max(width, np.finfo(float).eps)
        edge_width = 0.02 * scale
        diagnostics[str(name)] = {
            'posterior_90_width': float(q95 - q05),
            'posterior_90_to_limit_width': float((q95 - q05) / scale),
            'lower_edge_fraction': float(np.mean(flat[:, index] <= low + edge_width)),
            'upper_edge_fraction': float(np.mean(flat[:, index] >= high - edge_width)),
        }
    return diagnostics


def chain_diagnostics(chain, log_prob, pnames, acceptance_fraction,
                      rhat_threshold=1.05, min_acceptance=0.01,
                      min_bulk_ess=100.0, min_tail_ess=100.0,
                      limits=None):
    """Summarize chain mixing without altering posterior samples.

    The returned status is intentionally conservative: it requires finite
    post-burn-in samples, split-R-hat below the requested threshold for every
    sampled parameter, and no essentially immobile walker.
    """
    chain = np.asarray(chain, dtype=float)
    log_prob = np.asarray(log_prob, dtype=float)
    acceptance_fraction = np.asarray(acceptance_fraction, dtype=float)
    rhat, rank_rhat, folded_rhat = _rank_normalized_split_rhat(chain)
    bulk_ess, tail_ess = _bulk_tail_ess(chain)
    rhat_by_parameter = {
        str(name): float(value) for name, value in zip(pnames, rhat)
    }
    rank_rhat_by_parameter = {
        str(name): float(value) for name, value in zip(pnames, rank_rhat)
    }
    folded_rhat_by_parameter = {
        str(name): float(value) for name, value in zip(pnames, folded_rhat)
    }
    bulk_ess_by_parameter = {
        str(name): float(value) for name, value in zip(pnames, bulk_ess)
    }
    tail_ess_by_parameter = {
        str(name): float(value) for name, value in zip(pnames, tail_ess)
    }
    finite_rhat = rhat[np.isfinite(rhat)]
    max_rhat = float(np.max(finite_rhat)) if len(finite_rhat) else np.nan
    finite_fraction = float(np.mean(np.isfinite(log_prob))) if log_prob.size else 0.0
    min_accept = float(np.min(acceptance_fraction)) if acceptance_fraction.size else np.nan
    mean_accept = float(np.mean(acceptance_fraction)) if acceptance_fraction.size else np.nan
    finite_bulk = bulk_ess[np.isfinite(bulk_ess)]
    finite_tail = tail_ess[np.isfinite(tail_ess)]
    minimum_bulk = float(np.min(finite_bulk)) if len(finite_bulk) else np.nan
    minimum_tail = float(np.min(finite_tail)) if len(finite_tail) else np.nan

    enough_samples = chain.ndim == 3 and chain.shape[0] >= 4 and chain.shape[1] >= 2
    passed = (
        enough_samples
        and finite_fraction == 1.0
        and np.all(np.isfinite(rhat))
        and max_rhat <= float(rhat_threshold)
        and np.isfinite(min_accept)
        and min_accept >= float(min_acceptance)
        and np.isfinite(minimum_bulk)
        and minimum_bulk >= float(min_bulk_ess)
        and np.isfinite(minimum_tail)
        and minimum_tail >= float(min_tail_ess)
    )
    if passed:
        status = 'passed'
    elif not enough_samples:
        status = 'insufficient_post_burn_samples'
    elif finite_fraction < 1.0:
        status = 'nonfinite_post_burn_samples'
    elif not np.all(np.isfinite(rhat)):
        status = 'rhat_unavailable'
    elif np.isfinite(min_accept) and min_accept < float(min_acceptance):
        status = 'low_walker_acceptance'
    elif (not np.isfinite(minimum_bulk) or not np.isfinite(minimum_tail)):
        status = 'ess_unavailable'
    elif max_rhat > float(rhat_threshold) and (
            minimum_bulk < float(min_bulk_ess)
            or minimum_tail < float(min_tail_ess)):
        status = 'rhat_and_ess'
    elif max_rhat > float(rhat_threshold):
        status = 'rhat_above_threshold'
    elif minimum_bulk < float(min_bulk_ess) or minimum_tail < float(min_tail_ess):
        status = 'low_effective_sample_size'
    else:
        status = 'rhat_above_threshold'

    diagnostics = {
        'status': status,
        'passed': bool(passed),
        'post_burn_steps': int(chain.shape[0]) if chain.ndim == 3 else 0,
        'nwalkers': int(chain.shape[1]) if chain.ndim == 3 else 0,
        'finite_logprob_fraction': finite_fraction,
        'mean_acceptance_fraction': mean_accept,
        'min_acceptance_fraction': min_accept,
        'rhat_threshold': float(rhat_threshold),
        'max_split_rhat': max_rhat,
        'split_rhat': rhat_by_parameter,
        'rank_normalized_rhat': rank_rhat_by_parameter,
        'folded_rank_normalized_rhat': folded_rhat_by_parameter,
        'bulk_ess': bulk_ess_by_parameter,
        'tail_ess': tail_ess_by_parameter,
        'min_bulk_ess': minimum_bulk,
        'min_tail_ess': minimum_tail,
        'min_bulk_ess_threshold': float(min_bulk_ess),
        'min_tail_ess_threshold': float(min_tail_ess),
    }
    diagnostics['sampling_quality'] = {
        'status': diagnostics['status'],
        'passed': diagnostics['passed'],
        'max_rank_normalized_rhat': diagnostics['max_split_rhat'],
        'min_bulk_ess': diagnostics['min_bulk_ess'],
        'min_tail_ess': diagnostics['min_tail_ess'],
        'mean_acceptance_fraction': diagnostics['mean_acceptance_fraction'],
        'min_acceptance_fraction': diagnostics['min_acceptance_fraction'],
    }
    diagnostics['identifiability'] = (
        _identifiability_diagnostics(chain, pnames, limits)
        if limits is not None else {}
    )
    return diagnostics


def _apply_convergence_policy(diagnostics, action):
    action = str(action).lower()
    if action not in _CONVERGENCE_ACTIONS:
        raise ValueError(
            "convergence_action must be one of: {}.".format(', '.join(_CONVERGENCE_ACTIONS))
        )
    if diagnostics['passed'] or action == 'ignore':
        return

    message = (
        "MCMC convergence check failed: status={status}, max split-R-hat={rhat:.3f} "
        "(threshold {threshold:.3f}), minimum bulk/tail ESS={bulk:.1f}/{tail:.1f}, "
        "minimum walker acceptance={acceptance:.3f}."
    ).format(
        status=diagnostics['status'],
        rhat=diagnostics['max_split_rhat'],
        threshold=diagnostics['rhat_threshold'],
        bulk=diagnostics['min_bulk_ess'],
        tail=diagnostics['min_tail_ess'],
        acceptance=diagnostics['min_acceptance_fraction'],
    )
    if action == 'error':
        raise RuntimeError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _preload_hdf5_walker_neighborhood(grids, pnames, fixed_variables, positions,
                                      padding, seed_positions=None,
                                      max_cache_gb=None,
                                      mode_envelope_max_gb=2.0,
                                      full_active_max_gb=2.0):
    """Try to cache the local HDF5 region occupied by initialized walkers."""
    grid_list = _grid_list(grids)
    hdf5_grids = [
        grid for grid in grid_list if isinstance(grid, model.HDF5IntegratedGrid)
    ]
    if len(grid_list) != 1 or len(hdf5_grids) != 1:
        return False

    seed_positions = positions if seed_positions is None else seed_positions
    neighborhoods = []
    for theta in np.atleast_2d(np.asarray(seed_positions, dtype=float)):
        values = dict(fixed_variables)
        for index, name in enumerate(pnames):
            values[str(name)] = np.asarray([theta[index]], dtype=float)
        neighborhoods.append(values)
    try:
        if seed_positions is not None and hdf5_grids[0].preload_full_active_subgrid(
                max_gb=full_active_max_gb,
        ):
            return True
        if seed_positions is not None and hdf5_grids[0].preload_mode_envelope(
                neighborhoods,
                padding=padding,
                max_gb=mode_envelope_max_gb,
        ):
            return True
        return hdf5_grids[0].preload_neighborhoods(
            neighborhoods,
            padding=padding,
            max_total_gb=max_cache_gb,
        )
    except (OSError, ValueError) as exc:
        print("HDF5 walker-neighborhood cache skipped: {}".format(exc))
        return False


def _preload_hdf5_full_active(grids, max_gb):
    """Preload one single-component HDF5 grid before seed generation."""
    grid_list = _grid_list(grids)
    if len(grid_list) != 1 or not isinstance(grid_list[0], model.HDF5IntegratedGrid):
        return False
    try:
        return grid_list[0].preload_full_active_subgrid(max_gb=max_gb)
    except (OSError, ValueError) as exc:
        print("Early HDF5 active-grid cache skipped: {}".format(exc))
        return False


def _representative_cache_positions(positions, log_prob, limits, maximum=4,
                                    min_separation=0.05):
    """Select separated high-posterior walkers for cache refreshes."""
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    log_prob = np.asarray(log_prob, dtype=float)
    if len(positions) == 0:
        return positions
    order = np.argsort(np.where(np.isfinite(log_prob), log_prob, -np.inf))[::-1]
    width = np.maximum(
        np.asarray(limits, dtype=float)[:, 1] - np.asarray(limits, dtype=float)[:, 0],
        1e-12,
    )
    selected = []
    for index in order:
        theta = positions[index]
        if not np.all(np.isfinite(theta)):
            continue
        if all(np.linalg.norm((theta - other) / width) >= float(min_separation)
               for other in selected):
            selected.append(theta)
        if len(selected) >= int(maximum):
            break
    return np.asarray(selected or [positions[order[0]]], dtype=float)


def _hdf5_cache_diagnostics(grids, nworkers):
    """Expose parent-process cache metadata without changing the likelihood."""
    grid_list = _grid_list(grids)
    hdf5_grids = [
        grid for grid in grid_list if isinstance(grid, model.HDF5IntegratedGrid)
    ]
    if len(grid_list) != 1 or len(hdf5_grids) != 1:
        return {}
    diagnostics = hdf5_grids[0].cache_diagnostics()
    diagnostics['statistics_scope'] = 'single_process' if int(nworkers) == 1 else 'parent_only'
    return diagnostics


def _reset_hdf5_cache_statistics(grids):
    """Start per-source cache accounting without dropping reusable arrays."""
    grid_list = _grid_list(grids)
    reused = False
    for grid in grid_list:
        if not isinstance(grid, model.HDF5IntegratedGrid):
            continue
        reused = reused or bool(grid.cache_diagnostics().get('active'))
        grid.reset_cache_statistics()
    return reused


def _run_sampler(sampler, pos, total_steps, nrelax, autostop=False,
                 autostop_check_interval=200, autostop_tau_factor=50.0,
                 autostop_tolerance=0.01, progress=True,
                 cache_refresher=None, cache_refresh_interval=25):
    if not autostop:
        if cache_refresher is None:
            sampler.run_mcmc(pos, total_steps, progress=progress)
            return
        refresh_interval = max(1, int(cache_refresh_interval))
        for step, state in enumerate(
                sampler.sample(pos, iterations=total_steps, progress=progress), start=1):
            if step % refresh_interval == 0:
                cache_refresher(state)
        return

    check_interval = max(1, int(autostop_check_interval))
    tau_factor = float(autostop_tau_factor)
    tolerance = float(autostop_tolerance)
    old_tau = np.inf
    state = pos
    nrun = 0
    while nrun < total_steps:
        chunk = min(check_interval, total_steps - nrun)
        state = sampler.run_mcmc(state, chunk, progress=progress)
        nrun += chunk
        if cache_refresher is not None:
            cache_refresher(state)
        if nrun <= nrelax + check_interval:
            continue
        try:
            tau = sampler.get_autocorr_time(tol=0)
        except Exception:
            continue
        post_burn_steps = max(1, nrun - nrelax)
        long_enough = np.all(tau * tau_factor < post_burn_steps)
        stable = np.all(np.abs(old_tau - tau) / tau < tolerance)
        if long_enough and stable:
            print(
                "Stopping MCMC early after {} steps; autocorrelation time is stable.".format(
                    nrun
                )
            )
            return
        old_tau = tau


def MCMC(obs, obs_err, photbands,
         pnames, limits, grids,
         fixed_variables=None, priors=None,
         error_model=None,
         nwalkers=24, nsteps=4000, nrelax=500, a=2, pos=None,
         nworkers=1, init_method='auto', init_ntries=8,
         init_spread=1e-3, autostop=False,
         autostop_check_interval=200, autostop_tau_factor=50.0,
         autostop_tolerance=0.01,
         init_grid_max_spectra=12000, init_grid_rv_points=7,
         init_grid_av_points=11, init_grid_top_candidates=256,
         init_grid_max_modes=6, init_grid_min_separation=0.05,
         init_max_delta_logprob=25.0,
         init_grid_rescue=True, init_grid_rescue_chi2_threshold=None,
         init_grid_rescue_cache_max_gb=2.0,
         init_grid_rescue_maxiter=80, init_grid_rescue_popsize=12,
         hdf5_walker_cache_padding=4,
         hdf5_walker_cache_max_gb=0.25,
         hdf5_walker_cache_refresh=0,
         hdf5_walker_cache_max_modes=6,
         hdf5_walker_cache_envelope_max_gb=2.0,
         hdf5_auto_full_cache_max_gb=2.0,
         convergence_rhat_threshold=1.05,
         convergence_min_acceptance=0.01,
         convergence_min_bulk_ess=100.0,
         convergence_min_tail_ess=100.0,
         convergence_action='warn', progress=True,
         vectorized_likelihood=True):
    fixed_variables = {} if fixed_variables is None else fixed_variables
    priors = {} if priors is None else priors
    nworkers = int(nworkers or 1)

    grid_list = _grid_list(grids)
    prepared_single_grid = (
        len(grid_list) == 1 and model._is_prepared_grid(grid_list[0])
    )
    nonrectangular_grid = model.grid_has_nonrectangular_coverage(grids)
    init_method = str(init_method).lower()
    if init_method == 'auto':
        init_method = (
            'grid'
            if prepared_single_grid and nonrectangular_grid
            else 'random'
        )

    if init_method in _MAP_INITIALIZATION_METHODS and \
            nonrectangular_grid:
        raise ValueError(
            "MAP initialization uses L-BFGS-B and is disabled for HDF5, sparse, "
            "or non-rectangular integrated grids because their likelihood is piecewise. "
            "Use init_method='auto' or init_method='grid'."
        )

    hdf5_cache_reused = _reset_hdf5_cache_statistics(grids)

    #-- setup the sampler
    ndim = len(pnames)
    kwargs = {'pnames':pnames,
              'grid':grids,
              'fixed_variables':fixed_variables,
              'priors':priors,
              'photbands':photbands,
              'error_model':error_model,
              'prop_func': statfunc.get_derived_properties}

    # Load the YAML-limit cache before the discrete seed scan.  The scan and
    # subsequent likelihood evaluations can then share one HDF5 read.
    initialization_start = time.perf_counter()
    early_cache_preload_start = time.perf_counter()
    early_cache_preloaded = False
    if pos is None and init_method in _GRID_INITIALIZATION_METHODS:
        early_cache_preloaded = _preload_hdf5_full_active(
            grids,
            max_gb=hdf5_auto_full_cache_max_gb,
        )
    early_cache_preload_seconds = time.perf_counter() - early_cache_preload_start

    #-- initialize the walkers if no starting positions are given
    grid_initialization_start = time.perf_counter()
    seed_positions = None
    if pos is None:
        if init_method in _MAP_INITIALIZATION_METHODS:
            try:
                pos = _map_positions(
                    obs,
                    obs_err,
                    limits,
                    kwargs,
                    nwalkers,
                    ntries=init_ntries,
                    spread=init_spread,
                )
                print("Initialized walkers around a MAP estimate.")
            except Exception as exc:
                print("MAP initialization failed ({}); falling back to random walkers.".format(exc))
                pos = _random_positions(limits, nwalkers)
        elif init_method in _GRID_INITIALIZATION_METHODS:
            pos, seed_positions = _grid_positions(
                obs,
                obs_err,
                limits,
                kwargs,
                nwalkers,
                maximum_spectra=init_grid_max_spectra,
                coarse_rv_points=init_grid_rv_points,
                coarse_av_points=init_grid_av_points,
                top_candidates=init_grid_top_candidates,
                maximum_modes=init_grid_max_modes,
                min_separation=init_grid_min_separation,
                max_delta_logprob=init_max_delta_logprob,
                spread=init_spread,
                rescue=init_grid_rescue,
                rescue_chi2_threshold=init_grid_rescue_chi2_threshold,
                rescue_cache_max_gb=init_grid_rescue_cache_max_gb,
                rescue_maxiter=init_grid_rescue_maxiter,
                rescue_popsize=init_grid_rescue_popsize,
                return_seeds=True,
            )
        else:
            pos = _random_positions(limits, nwalkers)
    else:
        nwalkers = pos.shape[0]

    initial_log_probability = np.asarray([
        lnprob(theta, obs, obs_err, limits, **kwargs)[0]
        for theta in np.asarray(pos, dtype=float)
    ])
    if not np.all(np.isfinite(initial_log_probability)):
        invalid = int(np.sum(~np.isfinite(initial_log_probability)))
        raise ValueError(
            "Walker initialization produced {} non-finite posterior position(s). "
            "Refusing to start emcee; use the grid-aware initializer or provide "
            "validated starting positions.".format(invalid)
        )

    grid_initialization_seconds = time.perf_counter() - grid_initialization_start
    cache_preload_start = time.perf_counter()
    walker_cache_preloaded = _preload_hdf5_walker_neighborhood(
        grids,
        pnames,
        fixed_variables,
        pos,
        padding=hdf5_walker_cache_padding,
        seed_positions=seed_positions,
        max_cache_gb=hdf5_walker_cache_max_gb,
        mode_envelope_max_gb=hdf5_walker_cache_envelope_max_gb,
        full_active_max_gb=hdf5_auto_full_cache_max_gb,
    )
    cache_preload_seconds = (
        early_cache_preload_seconds + time.perf_counter() - cache_preload_start
    )
    initialization_seconds = time.perf_counter() - initialization_start
    print(
        "Walker initialization completed in {:.2f} s{}.".format(
            initialization_seconds,
            "; preloaded local HDF5 cache" if walker_cache_preloaded else "",
        )
    )

    # Derived-field names depend on the component setup, so emcee keeps one
    # dictionary blob per sample. Final conversion below is column-oriented.
    pool_context = None
    log_prob_func = lnprob
    log_prob_args = (obs, obs_err, limits)
    log_prob_kwargs = kwargs
    use_vectorized_likelihood = (
        bool(vectorized_likelihood)
        and nworkers == 1
        and len(grid_list) == 1
        and model._is_prepared_grid(grid_list[0])
    )
    if use_vectorized_likelihood:
        log_prob_func = lnprob_vectorized
    elif nworkers > 1:
        pool_context = get_context("fork").Pool(
            processes=nworkers,
            initializer=_init_worker_state,
            initargs=(obs, obs_err, limits, kwargs),
        )
        log_prob_func = _lnprob_from_worker_state
        log_prob_args = ()
        log_prob_kwargs = {}

    try:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_prob_func,
            a=a,
            args=log_prob_args,
            kwargs=log_prob_kwargs,
            blobs_dtype=[('blob','O')],
            pool=pool_context,
            vectorize=use_vectorized_likelihood,
        )

        cache_refresher = None
        if nworkers == 1 and walker_cache_preloaded and int(hdf5_walker_cache_refresh) > 0:
            def cache_refresher(state):
                representative_positions = _representative_cache_positions(
                    state.coords,
                    state.log_prob,
                    limits,
                    maximum=hdf5_walker_cache_max_modes,
                )
                return _preload_hdf5_walker_neighborhood(
                    grids,
                    pnames,
                    fixed_variables,
                    representative_positions,
                    padding=hdf5_walker_cache_padding,
                    max_cache_gb=hdf5_walker_cache_max_gb,
                    mode_envelope_max_gb=0.0,
                    full_active_max_gb=0.0,
                )

        #================
        # MCMC part

        #-- run the sampler, both burn in and actual run in one
        _run_sampler(
            sampler,
            pos,
            nsteps + nrelax,
            nrelax,
            autostop=autostop,
            autostop_check_interval=autostop_check_interval,
            autostop_tau_factor=autostop_tau_factor,
            autostop_tolerance=autostop_tolerance,
            cache_refresher=cache_refresher,
            cache_refresh_interval=hdf5_walker_cache_refresh,
            progress=bool(progress),
        )
    except Exception:
        if pool_context is not None:
            pool_context.terminate()
            pool_context.join()
        raise
    else:
        if pool_context is not None:
            pool_context.close()
            pool_context.join()

    #-- combine the results from the individual walkers discarding the burn in steps
    chain = sampler.get_chain(discard=nrelax, thin=1, flat=False)
    log_prob_chain = sampler.get_log_prob(discard=nrelax, thin=1, flat=False)
    diagnostics = chain_diagnostics(
        chain,
        log_prob_chain,
        pnames,
        sampler.acceptance_fraction,
        rhat_threshold=convergence_rhat_threshold,
        min_acceptance=convergence_min_acceptance,
        min_bulk_ess=convergence_min_bulk_ess,
        min_tail_ess=convergence_min_tail_ess,
        limits=limits,
    )
    diagnostics['initialization_seconds'] = float(initialization_seconds)
    diagnostics['grid_initialization_seconds'] = float(grid_initialization_seconds)
    diagnostics['hdf5_cache_preload_seconds'] = float(cache_preload_seconds)
    diagnostics['hdf5_cache_preloaded_before_grid_initialization'] = bool(
        early_cache_preloaded
    )
    diagnostics['hdf5_cache_reused_at_fit_start'] = bool(hdf5_cache_reused)
    diagnostics['initial_log_probability_min'] = float(
        np.min(initial_log_probability)
    )
    diagnostics['initial_log_probability_max'] = float(
        np.max(initial_log_probability)
    )
    diagnostics['vectorized_likelihood'] = bool(use_vectorized_likelihood)
    diagnostics['hdf5_walker_cache_preloaded'] = bool(walker_cache_preloaded)
    diagnostics['hdf5_cache'] = _hdf5_cache_diagnostics(grids, nworkers)
    _apply_convergence_policy(diagnostics, convergence_action)

    samples = sampler.get_chain(discard=nrelax, thin=1, flat=True)
    blobs = sampler.get_blobs(discard=nrelax, thin=1, flat=True)
    probabilities = sampler.get_log_prob(discard=nrelax, thin=1, flat=True)

    #-- clear the samples to save memory
    sampler.reset()

    #-- remove all steps that are not accepted (lnprob == -inf)
    accept = np.isfinite(probabilities)
    samples = samples[accept]
    blobs = blobs[accept]
    probabilities = probabilities[accept]

    if len(samples) == 0:
        raise ValueError('No models were accepted, all probabilities were -inf')

    #-- convert to recarrays
    sample_dtype = [(name, 'f8') for name in pnames]
    structured_samples = np.empty(len(samples), dtype=sample_dtype)
    for index, name in enumerate(pnames):
        structured_samples[name] = samples[:, index]
    samples = structured_samples

    blob_values = blobs['blob'] if blobs.dtype.names else np.asarray(
        [blob[0] for blob in blobs], dtype=object
    )
    names = list(blob_values[0].keys())
    blob_dtype = [(name, 'f8') for name in names]
    structured_blobs = np.empty(len(blob_values), dtype=blob_dtype)
    for name in names:
        structured_blobs[name] = np.fromiter(
            (blob[name] for blob in blob_values),
            dtype=float,
            count=len(blob_values),
        )
    blobs = structured_blobs

    #-- remove all steps where model creation failed (d == 0)
    accept = blobs['d'] > 0
    samples = samples[accept]
    blobs = blobs[accept]
    probabilities = probabilities[accept]

    #-- merge all results in 1 recarray and select best model
    overlap = set(samples.dtype.names) & set(blobs.dtype.names)
    if overlap:
        raise ValueError(
            "Sampled and derived output fields overlap: {}".format(
                ', '.join(sorted(overlap))
            )
        )
    data = np.empty(
        len(samples),
        dtype=samples.dtype.descr + blobs.dtype.descr,
    ).view(np.recarray)
    for name in samples.dtype.names:
        data[name] = samples[name]
    for name in blobs.dtype.names:
        data[name] = blobs[name]
    best = np.where(probabilities == np.max(probabilities))

    results = {}
    for n, v in zip(data.dtype.names, data[best][0]):
        results[n] = v
    results['_mcmc_diagnostics'] = diagnostics


    return results, data
   
   

   
   
