import re

import numpy as np
import pylab as pl

import matplotlib.patches as patches
import matplotlib.ticker as ticker

from scipy.stats import gaussian_kde

from . import model, filters, reddening, statfunc

from astropy.io import ascii


_MODEL_PARAMETERS = ('teff', 'logg', 'g', 'rad', 'distance', 'dist', 'av', 'rv', 'ebv', 'feh', 'he_mass')
_COMPONENT_STYLES = [
   {'color': 'b', 'lw': 1.0, 'linestyle': (0, (5, 5)), 'zorder': 2},
   {'color': 'r', 'lw': 1.0, 'linestyle': (0, (5, 5)), 'zorder': 2},
   {'color': 'tab:green', 'lw': 1.0, 'linestyle': (0, (5, 5)), 'zorder': 2},
   {'color': 'tab:orange', 'lw': 1.0, 'linestyle': (0, (5, 5)), 'zorder': 2},
   {'color': 'tab:purple', 'lw': 1.0, 'linestyle': (0, (5, 5)), 'zorder': 2},
]
_CORNER_LABELS = {
   'teff': (r'$T_{\rm eff}$', 'K'),
   'logg': (r'$\log g$', 'dex'),
   'g': (r'$g$', r'cm s$^{-2}$'),
   'feh': (r'$\mathrm{[Fe/H]}$', 'dex'),
   'rad': (r'$R$', r'$R_\odot$'),
   'distance': (r'$d$', 'pc'),
   'dist': (r'$d$', 'pc'),
   'd': (r'$d$', 'pc'),
   'av': (r'$A_V$', 'mag'),
   'rv': (r'$R_V$', None),
   'ebv': (r'$E(B-V)$', 'mag'),
   'L': (r'$L$', r'$L_\odot$'),
   'mass': (r'$M$', r'$M_\odot$'),
   'q': (r'$q$', None),
   'lr': (r'$L_1/L_2$', None),
   'rr': (r'$R_1/R_2$', None),
   'chi2': (r'$\chi^2$', None),
}


def format_parameter(name, value):

   temp = "{:7.3f}"
   if str(name).startswith('jitter_'):
      temp = "{:7.4f}"
      if hasattr(value, '__iter__'):
         return [temp.format(v) for v in value]
      return temp.format(value)
   if 'teff' in name or name in ('d', 'distance', 'dist'):
      temp = "{:7.0f}"

   elif  'logg' in name or 'rad' in name or 'mass' in name or name=='L' or name == 'L2':
      temp = "{:7.2f}"

   elif 'av' in name or 'rv' in name or 'ebv' in name or 'q' in name:
      temp = "{:7.3f}"

   if hasattr(value, '__iter__'):
      return [temp.format(v) for v in value]
   else:
      return temp.format(value)


def _split_parameter_component(name):
   if name == 'chi2':
      return name, ''
   match = re.fullmatch(r'([A-Za-z_]+)(\d*)', str(name))
   if match is None:
      return str(name), ''
   return match.groups()


def _add_component_suffix(label, component):
   if not component:
      return label
   if label.startswith('$') and label.endswith('$'):
      return '${{{}}}_{{{}}}$'.format(label[1:-1], component)
   return '{}{}'.format(label, component)


def corner_label(parameter, include_unit=True):
   """
   Return a default corner-plot label with physical units when known.
   """
   base, component = _split_parameter_component(parameter)
   if str(parameter).startswith('jitter_'):
      group = str(parameter)[len('jitter_'):].replace('_', r'\_')
      label, unit = r'$f_{\rm ' + group + r'}$', None
   else:
      label, unit = _CORNER_LABELS.get(base, (str(parameter), None))
   label = _add_component_suffix(label, component)
   if unit is None or not include_unit:
      return label
   return '{} ({})'.format(label, unit)


def _label_has_unit(label, unit):
   if unit is None:
      return True
   return '({})'.format(unit) in str(label)


def corner_labels(parameters, labels=None, units=None, include_units=True):
   """
   Return corner labels, using explicit labels when provided.

   ``units`` can be a mapping such as ``{'teff': 'kK'}`` to override or add
   units without replacing the parameter label itself.
   """
   if labels is not None:
      out = list(labels)
   else:
      default_units = include_units and units is None
      out = [
         corner_label(parameter, include_unit=default_units)
         for parameter in parameters
      ]

   if units is not None and include_units:
      for i, parameter in enumerate(parameters):
         unit = units.get(parameter)
         if unit is not None and not _label_has_unit(out[i], unit):
            out[i] = '{} ({})'.format(out[i], unit)
   return out


def _is_iterable_result(value):
   return hasattr(value, '__iter__') and not isinstance(value, (str, bytes))


def _result_parameters(results, result):
   """Split a results dictionary into scalar plotting values and model values."""
   resi = 0 if result == 'best' else 1
   pars = {}
   ipars = {}
   for key, value in results.items():
      if str(key).startswith('_') or isinstance(value, dict):
         continue
      if _is_iterable_result(value):
         value = value[resi]
      pars[key] = value
      ipars[key] = [value]

   # ``d`` is an output alias for distance; model calls use distance/dist.
   pars.pop('d', None)
   ipars.pop('d', None)
   return pars, ipars


def _photband_system(photband):
   photband = str(photband)
   if '.' in photband:
      return photband.split('.', 1)[0]
   if '_' in photband:
      return photband.rsplit('_', 1)[0]
   return photband


def _photband_label(photband):
   photband = str(photband)
   if '.' in photband:
      return photband.split('.')[-1]
   if '_' in photband:
      return photband.rsplit('_', 1)[-1]
   return photband


def _unique_in_order(values):
   return list(dict.fromkeys(values))


def _as_grid_list(gridnames):
   if isinstance(gridnames, str):
      return [gridnames]
   return list(gridnames)


def _raw_grid_available(gridname):
   return model.raw_spectrum_available(gridname)


def _component_model_parameters(pars, component=''):
   component_pars = {}
   for par in _MODEL_PARAMETERS:
      key = par + component
      if key in pars:
         component_pars[par] = pars[key]
      elif par in pars:
         component_pars[par] = pars[par]
   return component_pars


def _component_suffixes_from_parameters(pars):
   suffixes = {''}
   for name in pars:
      match = re.fullmatch(r'(teff|logg|g|rad)(\d*)', str(name))
      if match is not None:
         suffixes.add(match.group(2))

   def key(suffix):
      return 1 if suffix == '' else int(suffix)

   return sorted(suffixes, key=key)


def _write_observations(path, waves, bandwidths, fluxes, errors, photbands,
                        raw_errors=None):
   if path is None:
      return
   if raw_errors is None:
      raw_errors = errors
   ascii.write(
      [waves, bandwidths, fluxes, errors, raw_errors, errors, photbands],
      path,
      names=[
         'wave',
         'bandwidth',
         'flux',
         'error',
         'raw_error',
         'total_error',
         'photband',
      ],
      overwrite=True,
   )


def _positive_finite(values):
   values = np.asarray(values, dtype=float)
   return values[np.isfinite(values) & (values > 0)]


def _lambda_flux(wave, flux):
   return np.asarray(wave, dtype=float) * np.asarray(flux, dtype=float)


def _style_sed_axis(ax):
   for spine in ax.spines.values():
      spine.set_linewidth(1.4)
      spine.set_edgecolor('black')
   ax.minorticks_on()
   ax.tick_params(axis='both', which='both', top=True, right=True,
                  labelsize=18, direction='in')
   ax.tick_params(axis='both', which='major', length=8, width=1.2)
   ax.tick_params(axis='both', which='minor', length=4, width=1.0)


def _auto_residual_ylim(residual, residual_err=None, minimum=0.05, pad=1.25):
   residual = np.asarray(residual, dtype=float)
   finite = np.isfinite(residual)
   if not np.any(finite):
      return (-minimum, minimum)

   values = [residual[finite]]
   if residual_err is not None:
      residual_err = np.asarray(residual_err, dtype=float)
      finite_err = finite & np.isfinite(residual_err)
      if np.any(finite_err):
         values.extend([
            residual[finite_err] - residual_err[finite_err],
            residual[finite_err] + residual_err[finite_err],
         ])

   values = np.concatenate(values)
   span = max(minimum, pad * np.nanmax(np.abs(values)))
   return (-span, span)


def _auto_sed_xlim(waves, bandwidths, pad_fraction=0.06, minimum_pad=0.04):
   waves = np.asarray(waves, dtype=float)
   bandwidths = np.asarray(bandwidths, dtype=float)
   finite = np.isfinite(waves) & (waves > 0)
   if not np.any(finite):
      return (1.0, 10.0)

   half_widths = np.where(
      np.isfinite(bandwidths) & (bandwidths > 0),
      0.5 * bandwidths,
      0.0,
   )
   left_edge = waves[finite] - half_widths[finite]
   right_edge = waves[finite] + half_widths[finite]
   positive_left = left_edge[left_edge > 0]
   xmin = np.min(positive_left) if len(positive_left) > 0 else 0.85 * np.min(waves[finite])
   xmax = np.max(right_edge)

   if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin <= 0 or xmax <= xmin:
      center = np.nanmedian(waves[finite])
      return (0.85 * center, 1.15 * center)

   log_min = np.log10(xmin)
   log_max = np.log10(xmax)
   log_pad = max(minimum_pad, pad_fraction * (log_max - log_min))
   return (10 ** (log_min - log_pad), 10 ** (log_max + log_pad))


def sample_for_corner(data, max_samples=None, random_seed=12345):
   """
   Return a deterministic subset for expensive corner plots.

   ``corner`` can dominate runtime for long MCMC chains.  The fit results still
   use the full chain; this only thins the plotted points.
   """
   if max_samples is None or max_samples <= 0 or len(data) <= max_samples:
      return data
   rng = np.random.default_rng(random_seed)
   indices = np.sort(rng.choice(len(data), size=int(max_samples), replace=False))
   return data[indices]


def plot_distribution(data, parameters=None, percentiles=[16, 50, 84]):
   """
   Plots a histogram of the requested parameters, or all parameters if none are given.
   """

   if parameters is None:
      parameters = data.dtype.names

   x = len(parameters)

   for i, par in enumerate(parameters):
      d = data[par]

      pc  = np.percentile(d, percentiles, axis=0)

      pl.subplot(1,x,i+1)

      pl.hist(d, 50, density=True)

      for l in pc:
         pl.axvline(x=l, ls='--', color='k')

         emin, emax = pc[1]-pc[0], pc[2]-pc[1]

         if np.min([emin, emax]) > 10:
            temp = "{}={:0.0f} -{:0.0f} +{:0.0f}"
         elif np.min([emin, emax]) > 1:
            temp = "{}={:0.1f} -{:0.1f} +{:0.1f}"
         else:
            temp = "{}={:0.2f} -{:0.2f} +{:0.2f}"

      pl.title(temp.format(par, pc[1], emin, emax))

def plot_distribution_2d(data, xpar, ypar, percentiles=[16, 50, 84]):

   x, y = data[xpar], data[ypar]

   xpc  = np.percentile(x, percentiles, axis=0)
   ypc = np.percentile(y, percentiles, axis=0)

   pl.hist2d(x, y, bins=50, density=True)

   for i in xpc:
      pl.axvline(x=i, ls='--', color='w')
   for i in ypc:
      pl.axhline(y=i, ls='--', color='w')

   pl.xlabel(xpar)
   pl.ylabel(ypar)


def plot_distribution_density(data, xpar, ypar, percentiles=[16, 50, 84]):

   x, y = data[xpar][::10], data[ypar][::10]

   # Keep the signature aligned with plot_distribution; density plots do not
   # currently draw percentile guide lines.
   _ = percentiles

   # Calculate the point density
   xy = np.vstack([x,y])
   z = gaussian_kde(xy)(xy)

   # Sort the points by density, so that the densest points are plotted last
   idx = z.argsort()
   x, y, z = x[idx], y[idx], z[idx]

   pl.scatter(x, y, c=z, s=50, edgecolors='none')


def plot_priors(priors, samples, results):
   priors = priors.copy()

   pars = sorted(priors.keys())

   for i, par in enumerate(pars):

      ax = pl.subplot(1, len(pars), i+1)

      pc = np.percentile(samples[par], [0.2, 16, 50, 84, 99.8])

      #-- plot 1 sigma range as box
      ax.add_patch(
         patches.Rectangle(
            (0.5, pc[1]),
            1.0,
            pc[3]-pc[1],
            fill=False 
         )
      )

      #-- plot best fit and 50 percentile fit
      pl.plot([0.5,1.5], [results[par][0], results[par][0]], '--r', lw=1.5)
      pl.plot([0.5,1.5], [results[par][1], results[par][1]], '-b', lw=1.5)

      #-- plot 3 sigma range as wiskers
      pl.plot([1.0, 1.0], [pc[0], pc[1]], '-k', lw=1.5, zorder=0)
      pl.plot([1.0, 1.0], [pc[3], pc[4]], '-k', lw=1.5, zorder=0)

      #pl.boxplot(samples[par], usermedians=usermedians)

      if par in priors:
         pl.errorbar([1], priors[par][0],
                     yerr=[[priors[par][1]],[priors[par][2]]],
                     color='r', marker='x', mew=2, lw=2)

      ax.axes.get_xaxis().set_visible(False)

      if 'teff' in par:
         ticks = ticker.FuncFormatter(lambda x, pos: '{:0.1f}'.format(x/1000.))
         ax.yaxis.set_major_formatter(ticks)

      pl.title(par)

 
def plot_fit(obs, obs_err, photbands, pars=None, grids=None, gridnames=None,
             result='best', plot_components=True,
             reddening_law='WC2019',
             reddening_Rv=3.1, reddening_case1=1,
             observations_path='observations.txt',
             residual_ylim=None, xlim=None, figsize=(9, 7),
             error_model=None):

   if pars is None:
      pars = {}
   if grids is None:
      grids = []
   if gridnames is None:
      gridnames = []

   results = pars.copy()
   pars = {}
   ipars = {}
   obs = np.asarray(obs, dtype=float)
   obs_err = np.asarray(obs_err, dtype=float)
   photbands = np.asarray(photbands)

   if len(photbands) == 0:
      raise ValueError('plot_fit needs at least one absolute-flux photometric band.')

   filter_info = filters.get_info(photbands)
   waves = np.asarray(filter_info['eff_wave'], dtype=float)
   bandwidths = np.asarray(filter_info['bandwidth'], dtype=float)

   finite_widths = np.where(np.isfinite(bandwidths) & (bandwidths > 0), bandwidths, 0.0)
   abs_xlim = tuple(xlim) if xlim is not None else _auto_sed_xlim(waves, bandwidths)

   has_model = len(results) > 0
   syn = None
   model_curves = []
   if has_model:
      pars, ipars = _result_parameters(results, result)
      syn, Labs = model.get_itable(grid=grids, photbands=photbands, **ipars)
      syn = syn[:,0]

   total_obs_err = statfunc.effective_errors(
      obs,
      obs_err,
      pars=pars,
      photbands=photbands,
      error_model=error_model,
   )

   if has_model:
      #-- if possible, plot a non integrated model.
      #   Integrated-only stacks such as ck_all do not have a raw spectrum file.
      gridname_list = _as_grid_list(gridnames)
      raw_grid_available = (
         len(gridname_list) > 0
         and all(_raw_grid_available(gridname) for gridname in gridname_list)
      )

      if raw_grid_available:
         #-- synthetic model
         reddening_kwargs = {'av': pars['av']} if 'av' in pars else {'ebv': pars.get('ebv', 0.0)}
         curve_rv = pars.get('rv', reddening_Rv)
         wave, flux = model.get_table(grid=gridnames, **pars)
         flux = reddening.redden(flux, wave=wave, rtype='flux',
                                 law=reddening_law, Rv=curve_rv,
                                 case1=reddening_case1, **reddening_kwargs)
         total_wave, total_flux = wave, flux

         #-- plot components
         component_suffixes = _component_suffixes_from_parameters(pars)
         if plot_components and len(component_suffixes) > 1:
            component_grids = gridname_list
            if len(component_grids) == 1:
               component_grids = component_grids * len(component_suffixes)
            elif len(component_grids) != len(component_suffixes):
               raise ValueError(
                  'plot_fit received {} component(s) but {} grid name(s).'.format(
                     len(component_suffixes),
                     len(component_grids),
                  )
               )

            for icomp, (suffix, component_grid) in enumerate(zip(component_suffixes, component_grids)):
               wave, flux = model.get_table(
                  grid=component_grid,
                  **_component_model_parameters(pars, suffix),
               )
               flux = reddening.redden(flux, wave=wave, rtype='flux',
                                       law=reddening_law, Rv=curve_rv,
                                       case1=reddening_case1, **reddening_kwargs)
               style = _COMPONENT_STYLES[icomp % len(_COMPONENT_STYLES)].copy()
               model_curves.append(('component' + (suffix or '1'), wave, flux, style))

         model_curves.append(('total', total_wave, total_flux, {
            'color': 'k',
            'lw': 1.2,
            'zorder': 3,
         }))

   _write_observations(
      observations_path,
      waves,
      bandwidths,
      obs,
      total_obs_err,
      [str(p) for p in photbands],
      raw_errors=obs_err,
   )

   fig = pl.gcf()
   if figsize is not None:
      fig.set_size_inches(*figsize, forward=True)
   grid = fig.add_gridspec(2, 1, height_ratios=[3.0, 0.5], hspace=0.0)
   ax1 = fig.add_subplot(grid[0, 0])
   ax2 = fig.add_subplot(grid[1, 0], sharex=ax1)

   for _, wave, flux, style in model_curves:
      ax1.plot(wave, _lambda_flux(wave, flux), **style)

   valid_obs = (
      np.isfinite(waves) & (waves > 0)
      & np.isfinite(obs) & (obs > 0)
      & np.isfinite(total_obs_err) & (total_obs_err >= 0)
   )
   if np.any(valid_obs):
      ax1.errorbar(
         waves[valid_obs],
         _lambda_flux(waves[valid_obs], obs[valid_obs]),
         xerr=0.5 * finite_widths[valid_obs],
         yerr=_lambda_flux(waves[valid_obs], total_obs_err[valid_obs]),
         fmt=',',
         ecolor='tab:blue',
         capsize=0.0,
         c='aqua',
         elinewidth=1.0,
         zorder=1.0,
      )
      ax1.scatter(
         waves[valid_obs],
         _lambda_flux(waves[valid_obs], obs[valid_obs]),
         edgecolors='black',
         marker='D',
         c='aqua',
         s=35,
         zorder=5,
         alpha=1.0,
         lw=0.6,
      )

   valid_syn = np.zeros(len(obs), dtype=bool)
   if has_model:
      syn = np.asarray(syn, dtype=float)
      valid_syn = valid_obs & np.isfinite(syn) & (syn > 0)
      if np.any(valid_syn):
         ax1.scatter(
            waves[valid_syn],
            _lambda_flux(waves[valid_syn], syn[valid_syn]),
            marker='D',
            edgecolors='xkcd:magenta',
            s=45,
            facecolors='none',
            zorder=4,
            lw=2.0,
         )

   ax1.set_xlim(abs_xlim)
   y_reference = []
   if np.any(valid_obs):
      y_reference.append(_lambda_flux(waves[valid_obs], obs[valid_obs]))
   if np.any(valid_syn):
      y_reference.append(_lambda_flux(waves[valid_syn], syn[valid_syn]))
   if y_reference:
      positive_y = _positive_finite(np.concatenate(y_reference))
      if len(positive_y) > 0:
         ax1.set_ylim(0.75 * np.min(positive_y), 1.35 * np.max(positive_y))

   ax1.set_xscale('log', nonpositive='clip')
   ax1.set_yscale('log', nonpositive='clip')
   ax1.set_ylabel(r'$\lambda F_\lambda$ (erg cm$^{-2}$ s$^{-1}$)', fontsize=20)

   ax2.axhline(y=0, lw=1.5, ls='--', c='k', alpha=0.8, zorder=0.0)
   residual = np.array([])
   residual_err = np.array([])
   if has_model and np.any(valid_syn):
      residual = 2.5 * np.log10(obs[valid_syn] / syn[valid_syn])
      residual_err = 2.5 / np.log(10.0) * total_obs_err[valid_syn] / obs[valid_syn]
      ax2.errorbar(
         waves[valid_syn],
         residual,
         xerr=0.5 * finite_widths[valid_syn],
         yerr=residual_err,
         fmt=',',
         ecolor='tab:blue',
         capsize=0.0,
         c='aqua',
         elinewidth=1.0,
         zorder=1.0,
      )
      ax2.scatter(
         waves[valid_syn],
         residual,
         edgecolors='black',
         marker='D',
         c='aqua',
         s=30,
         zorder=5,
         alpha=1.0,
         lw=0.6,
      )

   ax2.set_xlim(abs_xlim)
   ax2.set_xscale('log', nonpositive='clip')
   if residual_ylim is None:
      ax2.set_ylim(_auto_residual_ylim(residual, residual_err))
   else:
      ax2.set_ylim(residual_ylim)
   ax2.set_ylabel(r'O-C (mag)', fontsize=20)
   ax2.set_xlabel(r'$\lambda\,(\mathrm{\AA})$', fontsize=20)

   for ax in [ax1, ax2]:
      _style_sed_axis(ax)
   pl.setp(ax1.get_xticklabels(), visible=False)
   fig.tight_layout()
   fig.subplots_adjust(left=0.16, bottom=0.13, right=0.98, top=0.97, hspace=0.0)
