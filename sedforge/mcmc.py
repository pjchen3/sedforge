import numpy as np
from numpy.lib.recfunctions import merge_arrays

import emcee

from . import statfunc, model

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


def MCMC(obs, obs_err, photbands,
         pnames, limits, grids,
         fixed_variables=None, priors=None,
         error_model=None,
         nwalkers=24, nsteps=4000, nrelax=500, a=10, pos=None):
    fixed_variables = {} if fixed_variables is None else fixed_variables
    priors = {} if priors is None else priors


    #-- initialize the walkers randomly is no starting positions are given
    if pos is None:
        pos = [ np.random.uniform(lim[0], lim[1], nwalkers) for lim in limits]
        pos = np.array(pos).T
    else:
        nwalkers = pos.shape[0]

    #-- setup the sampler
    ndim = len(pnames)
    kwargs = {'pnames':pnames,
              'grid':grids,
              'fixed_variables':fixed_variables,
              'priors':priors,
              'photbands':photbands,
              'error_model':error_model,
              'prop_func': statfunc.get_derived_properties}

    # TODO: storing the blobs as dictionary with dtype object and then later converting to recarray is inefficient.
    # This needs to be addressed: provide the correct dtypes here and let emcee directly store them in recarray
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, a=a,
                                    args=(obs, obs_err, limits), kwargs=kwargs, blobs_dtype=[('blob','O')])

    #================
    # MCMC part

    #-- run the sampler, both burn in and actual run in one
    sampler.run_mcmc(pos, nsteps+nrelax, progress=True)

    #-- combine the results from the individual walkers discarding the burn in steps
    samples = sampler.get_chain(discard=nrelax, thin=1, flat=True)
    blobs = sampler.get_blobs(discard=nrelax, thin=1, flat=True)
    probabilities = sampler.get_log_prob(discard=nrelax, thin=1, flat=True)

    #-- clear the samples to save memory
    sampler.reset()

    #-- remove all steps that are not accepted (lnprob == -inf)
    accept = np.where(np.isfinite(probabilities))
    samples = samples[accept]
    blobs = blobs[accept]
    probabilities = probabilities[accept]

    if len(samples) == 0:
        raise ValueError('No models were accepted, all probabilities were -inf')

    #-- convert to recarrays
    dtypes = [(n, 'f8') for n in pnames]
    samples = np.array([tuple(s) for s in samples], dtype=dtypes)

    names = list(blobs[0][0].keys())
    pars = []
    for b in blobs:
        pars.append(tuple([b[0][n] for n in names]))
    dtypes = [(n, 'f8') for n in names]
    blobs = np.array(pars, dtype=dtypes)

    #-- remove all steps where model creation failed (d == 0)
    accept = np.where(blobs['d'] > 0)
    samples = samples[accept]
    blobs = blobs[accept]
    probabilities = probabilities[accept]

    #-- merge all results in 1 recarray and select best model
    data = merge_arrays((samples, blobs), asrecarray=True, flatten=True)
    best = np.where(probabilities == np.max(probabilities))

    results = {}
    for n, v in zip(data.dtype.names, data[best][0]):
        results[n] = v


    return results, data
   
   

   
   
