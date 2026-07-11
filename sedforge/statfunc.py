# -*- coding: utf-8 -*-
import numpy as np

import warnings


def get_derived_properties(**pars):
    """
    Function that will derive several properties based on the chosen models
   
    Currently the following properties are calculated:
    - mass, mass2, mass3, ...
    - q, q3, ...
    - lr, lr3, ...
    - rr, rr3, ...
   
    returns dictionary of all properties that could be calculated.
    """

    derived_properties = {}

    GG = 6.67384e-08
    Rsol = 69550800000.0
    Msol = 1.988547e+33

    suffixes = _component_suffixes(pars)

    # -- derive masses
    for suffix in suffixes:
        rad_name = 'rad' + suffix
        mass_name = 'mass' + suffix
        logg_name = 'logg' + suffix
        g_name = 'g' + suffix
        if rad_name in pars and logg_name in pars:
            mass = 10 ** pars[logg_name] * (pars[rad_name] * Rsol) ** 2 / GG
            derived_properties[mass_name] = mass / Msol
        elif rad_name in pars and g_name in pars:
            mass = pars[g_name] * (pars[rad_name] * Rsol) ** 2 / GG
            derived_properties[mass_name] = mass / Msol

    primary_mass = derived_properties.get('mass')
    primary_rad = pars.get('rad')
    primary_luminosity_proxy = None
    if 'rad' in pars and 'teff' in pars:
        primary_luminosity_proxy = pars['rad'] ** 2 * pars['teff'] ** 4

    for suffix in [s for s in suffixes if s]:
        ratio_suffix = '' if suffix == '2' else suffix
        mass_name = 'mass' + suffix
        rad_name = 'rad' + suffix
        teff_name = 'teff' + suffix

        if primary_mass is not None and mass_name in derived_properties:
            derived_properties['q' + ratio_suffix] = primary_mass / derived_properties[mass_name]

        if primary_luminosity_proxy is not None and rad_name in pars and teff_name in pars:
            luminosity_proxy = pars[rad_name] ** 2 * pars[teff_name] ** 4
            derived_properties['lr' + ratio_suffix] = primary_luminosity_proxy / luminosity_proxy

        if primary_rad is not None and rad_name in pars:
            derived_properties['rr' + ratio_suffix] = primary_rad / pars[rad_name]

    # -- add empty values for luminosity and distance to prevent problems with
    #   failed models
    derived_properties.update({'d': 0, 'L': 0})
    for suffix in [s for s in suffixes if s]:
        derived_properties['L' + suffix] = 0
    if hasattr(pars['teff'], '__iter__'):
        derived_properties['d'] = np.zeros_like(pars['teff'])
        derived_properties['L'] = np.zeros_like(pars['teff'])
        for suffix in [s for s in suffixes if s]:
            derived_properties['L' + suffix] = np.zeros_like(pars['teff'])

    return derived_properties


def _component_suffixes(pars):
    suffixes = {''}
    for name in pars:
        match = None
        for prefix in ('teff', 'logg', 'g', 'rad'):
            if str(name) == prefix:
                match = ''
                break
            if str(name).startswith(prefix) and str(name)[len(prefix):].isdigit():
                match = str(name)[len(prefix):]
                break
        if match is not None:
            suffixes.add(match)

    def key(suffix):
        return 1 if suffix == '' else int(suffix)

    return sorted(suffixes, key=key)


def _error_model_variance(meas, e_meas, pars=None, photbands=None,
                          error_model=None):
    meas = np.asarray(meas, dtype=float)
    base_variance = np.asarray(e_meas, dtype=float) ** 2
    if not error_model or error_model.get('type', 'none') == 'none':
        return base_variance

    if photbands is None:
        raise ValueError("A group-level error model requires photbands.")

    pars = {} if pars is None else pars
    photbands = np.asarray(photbands, dtype=str)
    band_groups = error_model.get('band_groups', {})
    group_parameters = error_model.get('group_parameters', {})
    fixed_fractions = error_model.get('fixed_fractions', {})

    fractions = {}
    npoint = 1
    for group in band_groups:
        if group in group_parameters:
            parameter = group_parameters[group]
            if parameter not in pars:
                raise ValueError(
                    "Missing fitted jitter parameter '{}' for error model group '{}'.".format(
                        parameter,
                        group,
                    )
                )
            fraction = np.asarray(pars[parameter], dtype=float)
        else:
            fraction = np.asarray(fixed_fractions[group], dtype=float)
        fraction = np.atleast_1d(fraction).reshape(-1)
        npoint = max(npoint, len(fraction))
        fractions[group] = fraction

    if npoint == 1:
        variance = base_variance.copy()
    else:
        variance = np.repeat(base_variance[:, None], npoint, axis=1)

    for group, band_indices in band_groups.items():
        frac = fractions[group]
        if len(frac) == 1 and npoint > 1:
            frac = np.full(npoint, float(frac[0]))
        if len(frac) != npoint:
            raise ValueError(
                "Jitter group '{}' has {} values; expected 1 or {}.".format(
                    group, len(frac), npoint,
                )
            )
        if np.any(frac < 0):
            return np.full_like(variance, np.nan)
        band_indices = np.asarray(band_indices)
        if band_indices.dtype.kind in 'iu':
            indices = band_indices.astype(int)
        else:
            indices = np.where(np.isin(photbands, band_indices.astype(str)))[0]
        if len(indices) == 0:
            continue
        if npoint == 1:
            variance[indices] += (float(frac[0]) * meas[indices]) ** 2
        else:
            variance[indices] += (meas[indices, None] * frac[None, :]) ** 2
    return variance


def effective_errors(meas, e_meas, pars=None, photbands=None, error_model=None):
    """
    Return the total flux errors used by the likelihood.

    For the group-level jitter model this is
    ``sqrt(flux_err**2 + (fraction_group * flux)**2)``.
    """
    return np.sqrt(
        _error_model_variance(
            meas,
            e_meas,
            pars=pars,
            photbands=photbands,
            error_model=error_model,
        )
    )


def stat_chi2(meas, e_meas, syn, pars=None, **kwargs):
    """
    Calculate chi-square for absolute flux measurements.

    @param meas: array of measurements
    @type meas: 1D array
    @param e_meas: array containing measurements errors
    @type e_meas: 1D array
    @param syn: synthetic fluxes
    @type syn: 1D array
    @param full_output: set to True if you want individual chisq
    @type full_output: boolean
    @return: chi-square
    @rtype: float
    """

    # check for NaNs in the observations
    nani = np.isnan(meas) | np.isnan(e_meas)
    if np.any(nani):
        warnings.warn('{} of the observed fluxes are NaN values! (NaN values are ignored)'.format(np.sum(nani)))
        photbands = kwargs.get('photbands', None)
        if photbands is not None:
            photbands = np.asarray(photbands, dtype=str)[~nani]
        meas, e_meas, syn = meas[~nani], e_meas[~nani], syn[~nani]
    else:
        photbands = kwargs.get('photbands', None)

    variance = _error_model_variance(
        meas,
        e_meas,
        pars=pars,
        photbands=photbands,
        error_model=kwargs.get('error_model', None),
    )

    # The model is already physically normalised by radius and distance.
    if np.asarray(syn).ndim > 1:
        residual = syn - meas[:, None]
        likelihood_variance = (
            variance[:, None] if np.asarray(variance).ndim == 1 else variance
        )
    else:
        residual = syn - meas
        likelihood_variance = variance
    chisq = residual ** 2 / likelihood_variance
    chi2 = chisq.sum(axis=0)

    error_model = kwargs.get('error_model', None) or {}
    if error_model.get('include_log_norm', False):
        deviance = chi2 + np.log(2 * np.pi * variance).sum(axis=0)
        return chi2, deviance

    return chi2
