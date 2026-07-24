
default_single = """
# SED fitting setup file
# Photometry format: photband mag mag_err system
# Optional columns: mag_type mag_zp_offset
# Magnitudes are converted internally to band-averaged Flambda using the
# same response curves as the integrated grids. Extra flux columns are checks.

# ============================================================================
# Target And Photometry
# ============================================================================
objectname: <objectname>
photometryfile: <photfilename>
photband_exclude: <photband_exclude>

# ============================================================================
# Model Grids And Extinction
# ============================================================================
grids: 
<model_grids>
reddening_law: WC2019
reddening_Rv: 3.1
reddening_case1: 1

# ============================================================================
# Fitted Parameters
# ============================================================================
pnames: <parameter_names>
limits: <parameter_limits>

# ============================================================================
# Fixed Parameters
# ============================================================================
# Parameters listed here are kept constant and are not sampled by MCMC.
# For a metallicity grid, use e.g. fixed: {feh: 0.0} when [Fe/H] is not fitted.
fixed: {}

# ============================================================================
# Priors
# ============================================================================
# Gaussian priors on sampled parameters only: [value, sigma] or [value, -sigma, +sigma]
priors: <priors>

# ============================================================================
# Error Model
# ============================================================================
# Optional group-level fractional jitter. Off by default. Set jitter: true to
# fit one extra fractional error term per filter system, e.g. GAIA3E, 2MASS,
# WISE_RSR. Use error_model only if you need custom limits/priors.
jitter: false
# error_model:
#   type: fitted_fraction_by_group
#   default_limits: [0.0, 0.2]
#   default_prior: [0.03, 0.03]

# ============================================================================
# MCMC Sampler
# ============================================================================
nwalkers: 24     # total number of walkers
nsteps: 4000     # steps taken by each walker (not including burn-in)
nrelax: 500      # burn-in steps taken by each walker
a: 2             # emcee stretch-move scale
init_method: auto # grid-aware for one non-rectangular grid; random otherwise
init_grid_rescue: true # globally re-search only when the fast grid seed has implausibly high chi2
init_grid_rescue_cache_max_gb: 2.0 # full YAML-limit cache allowed for rescue initialization
hdf5_preload: false # full active-subgrid cache; useful only for a single long fit
hdf5_walker_cache: true # sparse valid-spectrum cache with exact HDF5 fallback
hdf5_walker_cache_padding: 4 # local cache only; out-of-cache proposals use HDF5
hdf5_walker_cache_max_gb: 0.25 # fallback local-cache budget for one source
hdf5_walker_cache_envelope_max_gb: 2.0 # initial multi-mode cache envelope
hdf5_auto_full_cache_max_gb: 2.0 # complete YAML-limit cache budget per source
hdf5_walker_cache_refresh: 0 # set positive to enable adaptive cache updates
hdf5_walker_cache_max_modes: 6 # high-posterior representatives per refresh
vectorized_likelihood: true # batch all walkers through one integrated-grid interpolation call
# runtime_grid_cache_dir: /path/to/cache # optional persistent read-only NPY cache
convergence_rhat_threshold: 1.05
convergence_min_acceptance: 0.01
convergence_min_bulk_ess: 100
convergence_min_tail_ess: 100
convergence_action: warn # use error to reject failed chains in production batches
percentiles: [0.2, 50, 99.8] # 16 - 84 corresponds to 1 sigma

# ============================================================================
# Outputs
# ============================================================================
resultfile: <objectname>_results_<postfix>.csv   # filepath to write results
plot1:
 type: sed_fit
 result: pc
 path: <objectname>_sed_<postfix>.png
plot2:
 type: distribution
 show_best: true
 path: <objectname>_distribution_<postfix>.png
 parameters: ['teff', 'rad', 'distance', 'av']
"""

default_binary = """
# SED fitting setup file
# Photometry format: photband mag mag_err system
# Optional columns: mag_type mag_zp_offset
# Magnitudes are converted internally to band-averaged Flambda using the
# same response curves as the integrated grids. Extra flux columns are checks.

# ============================================================================
# Target And Photometry
# ============================================================================
objectname: <objectname>
photometryfile: <photfilename>
photband_exclude: ['GALEX', 'SDSS', 'WISE_RSR_W3', 'WISE_RSR_W4']

# ============================================================================
# Model Grids And Extinction
# ============================================================================
grids: 
- ck_all
- tmap_he000
reddening_law: WC2019
reddening_Rv: 3.1
reddening_case1: 1

# ============================================================================
# Fitted Parameters
# ============================================================================
pnames: [teff, rad, teff2, rad2, distance, av]
limits:
- [3500, 10000]   # teff
- [0.01, 2.5]     # rad
- [20000, 80000]  # teff2
- [0.01, 0.5]     # rad2
- [100, 10000]    # distance
- [0, 0.30]       # av

# ============================================================================
# Fixed Parameters
# ============================================================================
# Shared parameters apply to both components unless a component-specific value
# such as feh2 is also provided.
fixed:
  feh: 0.0
  logg: 4.31
  logg2: 5.8

# ============================================================================
# Priors
# ============================================================================
# Gaussian priors on sampled parameters only: [value, sigma] or [value, -sigma, +sigma]
priors: <priors>

# ============================================================================
# Error Model
# ============================================================================
# Optional group-level fractional jitter. Off by default. Set jitter: true to
# fit one extra fractional error term per filter system, e.g. GAIA3E, 2MASS,
# WISE_RSR. Use error_model only if you need custom limits/priors.
jitter: false
# error_model:
#   type: fitted_fraction_by_group
#   default_limits: [0.0, 0.2]
#   default_prior: [0.03, 0.03]

# ============================================================================
# MCMC Sampler
# ============================================================================
nwalkers: 24     # total number of walkers
nsteps: 4000     # steps taken by each walker (not including burn-in)
nrelax: 500      # burn-in steps taken by each walker
a: 2             # emcee stretch-move scale
init_method: auto # grid-aware for one non-rectangular grid; random otherwise
init_grid_rescue: true # globally re-search only when the fast grid seed has implausibly high chi2
init_grid_rescue_cache_max_gb: 2.0 # full YAML-limit cache allowed for rescue initialization
hdf5_preload: false # full active-subgrid cache; useful only for a single long fit
hdf5_walker_cache: true # sparse valid-spectrum cache with exact HDF5 fallback
hdf5_walker_cache_padding: 4 # local cache only; out-of-cache proposals use HDF5
hdf5_walker_cache_max_gb: 0.25 # fallback local-cache budget for one source
hdf5_walker_cache_envelope_max_gb: 2.0 # initial multi-mode cache envelope
hdf5_auto_full_cache_max_gb: 2.0 # complete YAML-limit cache budget per source
hdf5_walker_cache_refresh: 0 # set positive to enable adaptive cache updates
hdf5_walker_cache_max_modes: 6 # high-posterior representatives per refresh
vectorized_likelihood: true # batch all walkers through one integrated-grid interpolation call
# runtime_grid_cache_dir: /path/to/cache # optional persistent read-only NPY cache
convergence_rhat_threshold: 1.05
convergence_min_acceptance: 0.01
convergence_min_bulk_ess: 100
convergence_min_tail_ess: 100
convergence_action: warn # use error to reject failed chains in production batches
percentiles: [16, 50, 84] # 16 - 84 corresponds to 1 sigma

# ============================================================================
# Outputs
# ============================================================================
resultfile: <objectname>_results_<postfix>.csv   # filepath to write results
plot1:
 type: sed_fit
 path: <objectname>_sed_<postfix>.png
plot2:
 type: distribution
 path: <objectname>_distribution_<postfix>.png
 parameters: ['teff', 'rad', 'teff2', 'rad2', 'distance', 'av']
"""
