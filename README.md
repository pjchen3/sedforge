# sedforge

<p align="right">
  <a href="README.md"><img src="https://img.shields.io/badge/English-README-2563eb" alt="English README"></a>
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-README-c2410c" alt="简体中文 README"></a>
</p>

sedforge is a Python package for fitting stellar photometric spectral energy
distributions (SEDs). It fits absolute fluxes with MCMC and supports single,
binary, and triple unresolved stellar components.

This repository is a research fork of
[Speedyfit](https://github.com/vosjo/speedyfit). The fitting workflow, input
files, grid handling, extinction axes, plotting, and catalog photometry helpers
have been adapted substantially for the local magnitude-first workflow
described below.

This fork has been adapted for a magnitude-first workflow:

- photometry is read from a simple `photband mag mag_err system` table and is
  converted internally to band-averaged `Flambda`;
- `distance` is a physical fit or fixed parameter in parsec;
- `[Fe/H]` can be fitted as a real grid axis when the integrated model grid
  provides it;
- any model parameter can be fixed through a readable `fixed:` setup section;
- integrated grids are built by applying extinction wavelength-by-wavelength
  before filter convolution;
- bundled filter response curves are taken from the
  [SVO Filter Profile Service](https://svo2.cab.inta-csic.es/theory/fps/),
  with photon/energy response conventions recorded in `filter_info.dat`;
- the default extinction law is `WC2019` with `case1=1`.

> [!IMPORTANT]
> **Do not convert broad-band magnitudes to monochromatic flux densities at a
> fixed effective wavelength.** The source-dependent effective wavelength of a
> wide passband shifts with the SED and extinction. sedforge therefore
> recommends native catalog magnitudes (`mag`, `mag_err`, and `system`): it
> integrates the AB/Vega zero point and every model SED through the same full
> filter response, then fits response-weighted band-averaged `Flambda`. In the
> standard integrated-grid likelihood, `eff_wave` is plotting/filter metadata,
> not the wavelength at which the model is evaluated. Direct flux input is
> valid only when it is already a response-weighted band-averaged `Flambda` in
> the convention documented below.

The project is based on the original GPLv3 Speedyfit package. If this fork is
published or redistributed, keep the GPLv3 license and the original attribution.

## Installation

sedforge requires Python 3.9 or newer. Clone the repository, enter the
repository root, and install the package:

```bash
git clone https://github.com/pjchen3/sedforge.git
cd sedforge
python -m pip install .
```

For development, install the package in editable mode with test/build tools:

```bash
python -m pip install -e ".[dev]"
```

Optional extras keep non-core dependencies out of the default install. Run
these commands from the same repository root only when you need the matching
feature:

```bash
python -m pip install ".[photometry]"  # VizieR catalog downloads with astroquery
python -m pip install ".[svo]"         # SVO filter update helper
python -m pip install ".[hdf5]"        # HDF5 model grids
```

The source code does not include the large model grids. Prepared model grid
archives are available from Zenodo:
[doi:10.5281/zenodo.20520723](https://doi.org/10.5281/zenodo.20520723).
For the `ck_all` Quick Start below, download at least the integrated-grid
archive. Download the spectral-cache archive too if you want SED plots with
continuous model spectra. The current v2026.06.03 data release includes the
legacy `ck03_cepheid_rv` HDF5 grid. sedforge 0.3.0 also supports the full
`ck03_rv` and `newera_alpha0_rv` grids, but those two files are not yet part of
that Zenodo record; generate them locally with the bundled scripts or obtain
them from a future data release. Any HDF5 grid requires the `hdf5` extra.

Unpack the archives in the same parent directory so that they merge into one
`sed_models/` directory. The model grid directory is selected with
`SEDFORGE_MODELS`:

```bash
export SEDFORGE_MODELS=/path/to/sed_models
```

That directory should contain `grid_description.yaml` plus the model files it
references. A tidy local layout is:

```text
sed_models/
  grid_description.yaml
  raw/              # original model spectra
  integrated/       # passband-integrated fitting grids
  ck03_rv/          # optional HDF5 grid with an explicit Rv axis
  newera_alpha0_rv/ # optional HDF5 grid with an explicit Rv axis
  spectral_cache/   # continuous spectra used only for plotting
```

## Quick Start

This example uses target name `my_target` and model grid `ck_all`.

Create a magnitude photometry file named `my_target.phot` in your working
directory. It must contain these columns:

```text
photband  mag    mag_err  system
GAIA3E_G  12.30  0.01     vega
PS1_g     18.42  0.02     ab
2MASS_Ks  10.10  0.02     vega
```

Magnitudes are converted to the same band-averaged `Flambda` definition used by
the integrated grids. The `photband` names must match bundled response curves
such as `GAIA3E_G`, `2MASS_Ks`, `WISE_RSR_W1`, `HST_WFC3_F814W`, or another
band in `sedforge/transmission_curves`.

Create a starter setup file for target `my_target` and grid `ck_all`:

```bash
sedforge setup my_target -grid ck_all
```

This command writes `my_target_setup_ck_all.yaml`. By default, that setup file
expects the photometry file `my_target.phot` beside the setup file unless you
edit the generated YAML.

Run the fit with the generated setup file:

```bash
sedforge fit my_target_setup_ck_all.yaml --noplot
```

The `--noplot` option skips SED and corner plots, so the Quick Start can run
with only the integrated-grid archive. If you also unpacked the spectral-cache
archive, omit `--noplot` to create the plots. Outputs include a CSV result
summary, accepted MCMC samples in FITS format, and, when plotting is enabled,
an SED plot and a corner plot.

Example output from a synthetic `ck_all` recovery is shown below. The first
figure shows the fitted SED, and the second shows the corresponding MCMC
posterior corner plot.

<p align="center">
  <img src="docs/assets/example_synthetic_ck_all_sedfit.png" alt="Example synthetic ck_all SED fit" width="720">
</p>

<p align="center">
  <img src="docs/assets/example_synthetic_ck_all_mcmc_posterior.png" alt="Example synthetic ck_all MCMC posterior corner plot" width="720">
</p>

## Photometry Input

Photometry files must contain these columns:

```text
photband  mag    mag_err  system
GAIA3E_G  12.30  0.01     vega
PS1_g     18.42  0.02     ab
2MASS_Ks  10.10  0.02     vega
```

The fitter converts these magnitudes to band-averaged `Flambda` using the
same SVO response curves as the model grids. Files created by
`sedforge photometry` include internally converted flux columns for checking:

```text
photband  mag    mag_err  system  mag_type  mag_zp_offset  flux       flux_err
GAIA3E_G  12.30  0.01     vega    pogson    0.00           1.23e-13  1.13e-15
```

If both magnitude and flux columns are present, the fitter uses the magnitude
columns and recomputes the flux. Advanced input with `photband flux flux_err`
is still accepted only when the fluxes are already band-averaged
`erg/s/cm2/Angstrom`.

The default magnitude systems for common bundled filters are:

- Vega: `GAIA3E`, `2MASS`, `WISE_RSR`, `SPITZER_IRAC`, `WFCAM`
- AB: `GALEX`, `PS1`, `SDSS`, `SkyMapper`, `ZTF`

HST filters do not have a default because the same passband can be reported as
VegaMag, ABMag, or STMag. Provide `system: vega` or `system: ab` explicitly for
HST photometry. STMag input is not currently supported.

The bundled filter set currently covers the instrument families and wavelength
ranges shown below. Use the exact `photband` names from
`sedforge/transmission_curves`; the figure summarizes filter coverage by
instrument.

![sedforge supported photometric filters](docs/assets/sedforge_supported_filters_by_instrument.png)

AB and Vega zero points are computed by integrating the AB reference spectrum
or `vega.dat` through the same local response curve used for model convolution.
The optional `mag_type` column can be `pogson` or `asinh`; if omitted, SDSS
filters default to `asinh` and all other filters default to `pogson`.

SDSS catalog magnitudes are luptitudes/asinh magnitudes, so sedforge does not
convert them with the high-S/N Pogson approximation. For `SDSS_u/g/r/i/z` it
uses the SDSS softening parameters
`1.4, 0.9, 1.2, 1.8, 7.4 x 10^-10` and inverts

```text
m = -2.5/ln(10) * [asinh((f/f0)/(2b)) + ln(b)] .
```

The bundled SDSS DR12 downloader also applies the commonly used AB offsets
`SDSS_u: -0.04` and `SDSS_z: +0.02` through `mag_zp_offset`; the other bundled
catalogs use zero additional fixed offset. If your SDSS values have already
been converted to ordinary AB/Pogson magnitudes, set `mag_type` to `pogson` and
`mag_zp_offset` to `0.0` explicitly in the photometry table. If another catalog
requires a fixed correction, either apply it to the input magnitude or add a
`mag_zp_offset` column.
Legacy column-index setup keys are not used by this fork.

Use `photband_include` or `photband_exclude` in the setup file to select bands.
Selectors can match a family prefix, for example `GAIA3E` matches
`GAIA3E_G`, `GAIA3E_BP`, and `GAIA3E_RP`.

### Downloading Photometry From VizieR

The package can create this magnitude table from coordinates or a Gaia DR3
source id.
By default it uses the bundled VizieR catalog list:

- Gaia DR3: `I/355/gaiadr3`
- 2MASS: `II/246/out`
- AllWISE: `II/328/allwise` (`W1`, `W2`)
- Pan-STARRS1: `II/349/ps1`
- SDSS DR12: `V/147/sdss12`
- GLIMPSE: `II/293/glimpse` (`IRAC 3.6`, `IRAC 4.5`)
- SkyMapper DR2: `II/379`
- GALEX AIS: `II/312/ais`

Only catalogs listed in the YAML config are queried. This keeps the query
policy explicit and easy to audit.

Example:

```bash
sedforge photometry \
  --gaia-id 1234567890123456789 \
  --output my_target.phot \
  --metadata-output my_target_catalogs.dat
```

Coordinate input is also supported:

```bash
sedforge photometry \
  --ra 10.6847083 --dec 41.26875 \
  --output my_target.phot
```

Pass `--catalog-config my_catalogs.yaml` to override the bundled catalog set.
Catalog configs contain the VizieR id and column-to-filter mapping:

```yaml
catalogs:
  - name: GaiaDR3
    vizier_id: I/355/gaiadr3
    source_id_column: Source
    ra_column: RA_ICRS
    dec_column: DE_ICRS
    bands:
      - photband: GAIA3E_G
        mag: Gmag
        mag_err: e_Gmag
        system: vega
      - photband: GAIA3E_BP
        mag: BPmag
        mag_err: e_BPmag
        system: vega
```

For magnitude columns, `system` can be `vega` or `ab`. The conversion uses the
local filter response curves and produces band-averaged `erg/s/cm2/Angstrom`,
matching the fitter. Direct flux columns are only for advanced custom catalog
configs; if the same `photband` is found in more than one catalog, the first
catalog in the YAML file wins.
The bundled catalog config applies basic quality cuts for 2MASS, AllWISE,
Pan-STARRS1, SDSS, SkyMapper, and GALEX.

## Model Grids

Integrated model grids are FITS tables. The required columns are model axes
such as `teff`, `logg`, `av`, optionally `feh`, plus one flux column per
filter. A `Labs` column stores bolometric luminosity information used by the
fit output.

The current sedforge model configuration supports these spectral model
families. The large grid files are kept outside the Git repository and should
be downloaded from a data release or generated locally.

- Castelli & Kurucz 2003 (`ckm25`, `ckm20`, `ckm15`, `ckm10`, `ckm05`,
  `ckp00`, `ckp02`, `ckp05`, and the combined `ck_all` stack), including the
  HDF5 `ck03_rv` grid with explicit `Rv` and `Av` axes.
- TLUSTY/SYNSPEC hot-star grids (`tlusty00`, `tlusty01`, `tlusty02`,
  `tlusty05`, `tlusty10`, `tlusty20`, and `tlusty_all`), with
  `feh = log10(Z/Zsun)` for the non-zero metallicity stack.
- PHOENIX NewEra V3 LowRes alpha=0 spectra (`newera_alpha0`), represented as
  a grid with a real `[Fe/H]` axis; the HDF5 `newera_alpha0_rv` grid also has
  explicit `Rv` and `Av` axes.
- Koester DA white-dwarf spectra (`koester2`).
- TMAP H+He spectra (`tmap_he000` through `tmap_he100`) with fixed helium mass
  fraction.
- A disc-integrated blackbody grid (`blackbody`) for simple continuum
  components.

The current prepared model-grid coverage in effective temperature and surface
gravity is summarized below. The plotted regions mark valid, nonzero spectra;
zero-flux placeholder spectra in the source FITS files are excluded. Exact
axes, metallicity coverage, and file paths are defined by
`grid_description.yaml` and the released grid files.

![sedforge model grid coverage](docs/assets/model_teff_logg_shaded_grid.png)

sedforge is not limited to these model families. Users can convolve a new
atmosphere or spectrum library with the same filter response curves and
extinction law, then add the resulting grid to `grid_description.yaml`. A
custom integrated grid should use the same contract as the built-in grids:
model-axis columns such as `teff`, `logg`, `av`, and optional physical axes;
one band-averaged `Flambda` column per filter; a `Labs` column when luminosity
output is needed; and, for plotting, an optional `spectral_cache` FITS file
with wavelength in Angstrom and flux in `erg/s/cm2/Angstrom`. If a new grid
name is not covered by the built-in setup defaults, provide suitable parameter
ranges in the setup file or extend the package defaults before running fits.

The model directory is described by `grid_description.yaml`. A fixed-metallicity
grid can be described like this:

```yaml
ckp00:
  filename: ck03_p00
  raw_filename: raw/ck/ck03_p00
  feh: 0.0
  spectral_cache: spectral_cache/ck_all_plot_spectra.fits
  info: Castelli & Kurucz 2003, [Fe/H] = 0.0
```

A metallicity stack can be described as multiple members:

```yaml
ck_all:
  filename: ck_all
  integrated_subdir: integrated
  spectral_cache: spectral_cache/ck_all_plot_spectra.fits
  info: Combined Castelli & Kurucz metallicity stack
  members:
    - grid: ckm05
      feh: -0.5
    - grid: ckp00
      feh: 0.0
    - grid: ckp05
      feh: 0.5
```

A grid with an actual `[Fe/H]` FITS column can advertise that axis directly:

```yaml
newera_alpha0:
  filename: newera_alpha0
  integrated_subdir: integrated
  spectral_cache: spectral_cache/newera_alpha0_plot_spectra.fits
  supports_feh: true
  info: PHOENIX NewEra alpha=0 integrated grid
```

Large model grids should not be committed to GitHub. The prepared sedforge
model-grid release is archived on Zenodo:
[doi:10.5281/zenodo.20520723](https://doi.org/10.5281/zenodo.20520723).
For new or modified grids, keep the files in a local or server-side
`sed_models` directory, document how they were generated, and publish large
artifacts through a data repository rather than Git history.

### Raw Spectrum Cache For Plotting

Fitting uses integrated grids because they are compact and fast. Continuous
spectra for figures should be stored separately in the same cache format for
all model families:

```yaml
newera_alpha0:
  filename: newera_alpha0
  integrated_subdir: integrated
  spectral_cache: spectral_cache/newera_alpha0_plot_spectra.fits
  supports_feh: true
  info: PHOENIX NewEra alpha=0 integrated grid
```

The spectral cache is a FITS file with:

- `PARAMS`: one row per model spectrum, with columns such as `teff`, `logg`,
  `feh`, and `he_mass` when relevant;
- `WAVE`: the common wavelength grid in Angstrom;
- `FLUX`: a two-dimensional `(n_spectra, n_wave)` array in
  `erg/s/cm2/Angstrom`.

The fitter opens these files with FITS memory mapping and reads only the
nearest spectrum needed for plotting. This keeps CK, TLUSTY, Koester, TMAP,
blackbody, and NewEra spectra on the same code path without loading a whole
spectral library into memory.

For the local model set used during development, the plotting caches are built
with CK, blackbody, and Koester kept on their native sampling, while TLUSTY,
TMAP, and NewEra are resampled to a `2 Angstrom` grid for plotting.
Fixed-metallicity CK grids share `ck_all_plot_spectra.fits`, non-zero TLUSTY
metallicity grids share `tlusty_all_plot_spectra.fits`, and all fixed-helium
TMAP grids share `tmap_plot_spectra.fits`.

## Extinction And Filter Convolution

Integrated grids are generated by applying the extinction law to each model
spectrum at each wavelength and then integrating through each filter response.
This avoids treating extinction as a single effective-wavelength correction.

The default law is WC2019, following Wang & Chen (2019), *The Optical to
Mid-infrared Extinction Law Based on the APOGEE, Gaia DR2, Pan-STARRS1, SDSS,
APASS, 2MASS, and WISE Surveys*, ApJ, 877, 116,
doi:[10.3847/1538-4357/ab1c61](https://doi.org/10.3847/1538-4357/ab1c61).

The default setup is:

```yaml
reddening_law: WC2019
reddening_Rv: 3.1
reddening_case1: 1
```

For ordinary FITS integrated grids, `reddening_Rv` is a grid-selection
constant, not a sampled parameter. Do not put `rv` in `pnames` for grids such
as `ck_all`, `newera_alpha0`, `tlusty_all`, `koester2`, or `blackbody`.

HDF5 grids such as `ck03_rv` and `newera_alpha0_rv` have an explicit `rv`
axis. In that case remove `reddening_Rv`/`Rv` from the setup and provide `rv`
either as a fitted parameter:

```yaml
grids: [ck03_rv]
pnames: [teff, logg, feh, rad, distance, av, rv]
limits:
  - [3500, 50000]     # teff, K
  - [0.0, 5.0]        # logg, dex
  - [-2.5, 0.5]       # [Fe/H], dex
  - [0.05, 500.0]     # radius, Rsun
  - [100, 100000]     # distance, pc
  - [0.0, 4.0]        # Av, mag
  - [2.0, 5.0]        # Rv
```

or as a fixed value:

```yaml
grids: [ck03_rv]
fixed:
  rv: 3.1
```

For `newera_alpha0_rv`, use the same `rv` pattern but choose NewEra parameter
limits, for example `teff = 2300..12000 K`, `logg = 0..6`, and
`[Fe/H] = -2.5..0.5`.

Only filters fully covered by the model wavelength range should be used for a
grid. Filter metadata such as effective wavelength and bandwidth can be stored
in `filter_info.dat` for plotting.

Bundled response curves are generated from `filter_svo_map.dat` using the
[SVO Filter Profile Service](https://svo2.cab.inta-csic.es/theory/fps/).
`filter_info.dat` records the SVO id and the local
`response_type`: photon responses use an extra wavelength weight in synthetic
photometry, while energy responses use no extra wavelength weight. SVO WISE and
Spitzer/IRAC curves are energy responses.

## Setup Files

A setup YAML file controls one fit. The main sections are:

- target and photometry;
- model grids and extinction;
- fitted parameters and hard limits;
- fixed parameters;
- Gaussian priors on fitted parameters;
- MCMC sampler settings;
- output files and plots.

`pnames` and `limits` define the parameters sampled by MCMC. `fixed` defines
parameters that are held constant and are not sampled. Every model parameter
needed by the selected grid must appear in one of those two places.

### Fixed Parameters

Use `fixed:` whenever a parameter should be supplied to the model but should
not participate in the fit:

```yaml
fixed:
  feh: 0.0
```

This is the preferred way to use a metallicity grid at a fixed `[Fe/H]`. The
fixed value is still passed into the grid interpolation, so a grid such as
`ck_all` or `newera_alpha0` is evaluated at the requested metallicity.

Do not use identical limits to fix a parameter. A parameter in `pnames` must
have a real non-zero fitting range. Put all fixed values in `fixed`.

Fixed parameters are written to the result table with zero uncertainty. They
are not included in the MCMC sample FITS table or corner plot because they have
no posterior width.

Extinction in setup files is always `av`, meaning `A(V)` in magnitudes. The
legacy `ebv` / `E(B-V)` parameter is intentionally rejected in YAML files.

### Priors Versus Fixed

`fixed` is a hard value. The parameter is not sampled.

`priors` are Gaussian priors in the posterior. The parameter is still sampled,
so every prior must refer to a name in `pnames`:

```yaml
priors:
  distance: [1000.0, 50.0]
```

Use `fixed` for known constants and `priors` for external measurements
with uncertainty.

Derived quantities such as `L`, `mass`, and `q` are outputs/checks, not fitted
parameters. They cannot be placed in `priors`.

### Group-Level Jitter

Catalogue errors can be too optimistic, and different surveys often carry
survey-level zero-point, calibration, saturation, or filter-curve systematics.
Jitter is off by default. The simple switch is:

```yaml
jitter: false
```

Set `jitter: true` to fit one fractional extra error by filter system. Filters
are grouped automatically by their name prefix, for example
`GAIA3E_G/BP/RP` share the `GAIA3E` group, `2MASS_J/H/Ks` share `2MASS`, and
`WISE_RSR_W1/W2` share `WISE_RSR`.

The switch uses these defaults:

```yaml
jitter: true
```

```text
type = fitted_fraction_by_group
default_limits = [0.0, 0.2]
default_prior = [0.03, 0.03]
```

For custom limits, priors, or fixed survey-level error floors, use the
explicit `error_model` section instead of the simple switch.

The fitter automatically adds one MCMC parameter per filter system, such as
`jitter_GAIA3E`, `jitter_2MASS`, and `jitter_WISE_RSR`. Do not add these names
manually to `pnames`; they are error-model parameters, not model-grid
parameters. The effective error for band `i` is

```text
sigma_eff,i^2 = sigma_i^2 + (f_group * F_obs,i)^2
```

where `f_group` is the fitted fractional jitter for that filter system. Because
the jitter is a fitted uncertainty, the likelihood includes the Gaussian
normalisation term:

```text
sum_i log(2*pi*sigma_eff,i^2)
```

which prevents the sampler from simply driving all jitter values to the upper
limit.

SED plots use the same `sigma_eff` for the photometric and residual error bars.
The plot-side observations table keeps both `raw_error` and `total_error`;
the legacy `error` column is the plotted total error.

You can override one group:

```yaml
error_model:
  type: fitted_fraction_by_group
  default_limits: [0.0, 0.2]
  groups:
    GAIA3E:
      limits: [0.0, 0.05]
      prior: [0.02, 0.01]
```

For a fixed survey-level error floor instead of a fitted parameter:

```yaml
error_model:
  type: fixed_fraction_by_group
  default_fraction: 0.03
  groups:
    WISE_RSR:
      fraction: 0.05
```

## Single-Star Example

This example fits `teff`, `logg`, `rad`, `distance`, and `av`, while keeping
`[Fe/H]=0.0` fixed:

```yaml
objectname: example_single
photometryfile: example_single.phot
photband_exclude: []

grids:
  - ck_all
reddening_law: WC2019
reddening_Rv: 3.1
reddening_case1: 1

pnames: [teff, logg, rad, distance, av]
limits:
  - [5000, 9000]      # teff, K
  - [3.0, 5.0]        # logg, dex
  - [0.1, 10.0]       # radius, Rsun
  - [100, 5000]       # distance, pc
  - [0.0, 3.1]        # Av, mag

fixed:
  feh: 0.0

priors: {}

nwalkers: 80
nsteps: 1000
nrelax: 300
a: 2
percentiles: [16, 50, 84]

resultfile: example_single_results.csv
datafile: example_single_samples.fits
plot1:
  type: sed_fit
  result: pc
  path: example_single_sed.png
plot2:
  type: distribution
  show_best: true
  path: example_single_corner.png
  parameters: [teff, logg, rad, distance, av]
```

Run it with:

```bash
sedforge fit example_single_setup.yaml --noplot
```

## Multi-Component Examples

A binary setup uses one grid per component. Shared parameters such as
`distance`, `av`, and `feh` can be supplied once. Component-specific secondary
parameters use the suffix `2`.

This example fixes one shared metallicity for both components:

```yaml
objectname: example_binary
photometryfile: example_binary.phot
photband_exclude: []

grids:
  - ck_all
  - ck_all
reddening_law: WC2019
reddening_Rv: 3.1
reddening_case1: 1

pnames: [teff, logg, rad, teff2, logg2, rad2, distance, av]
limits:
  - [8000, 16000]     # teff
  - [3.5, 5.0]        # logg
  - [0.5, 5.0]        # rad
  - [3500, 6500]      # teff2
  - [2.0, 4.5]        # logg2
  - [1.0, 20.0]       # rad2
  - [100, 5000]       # distance
  - [0.0, 1.55]       # Av

fixed:
  feh: 0.0

priors:
  distance: [1000.0, 50.0]

nwalkers: 120
nsteps: 2000
nrelax: 500
a: 2
percentiles: [16, 50, 84]

resultfile: example_binary_results.csv
datafile: example_binary_samples.fits
plot1:
  type: sed_fit
  result: pc
  path: example_binary_sed.png
plot2:
  type: distribution
  show_best: true
  path: example_binary_corner.png
  parameters: [teff, rad, teff2, rad2, distance, av]
```

If the two components should use different metallicities, provide both values:

```yaml
fixed:
  feh: 0.0
  feh2: -0.5
```

Three-component fits follow the same convention: give three grids and add
suffix `3` for the third component's atmospheric and radius parameters. Shared
parameters still appear only once. This example uses a blue CK component, a red
CK component, and a cool blackbody component that mainly contributes in the
mid-infrared:

```yaml
grids:
  - ck_all
  - ck_all
  - blackbody

pnames: [teff, logg, rad, teff2, logg2, rad2, teff3, rad3, distance, av]
limits:
  - [12000, 18000]    # blue component teff
  - [3.5, 5.0]        # logg
  - [1.0, 5.0]        # rad
  - [3500, 5000]      # red component teff2
  - [0.5, 3.0]        # logg2
  - [20.0, 200.0]     # rad2
  - [300, 1500]       # cool blackbody teff3
  - [10.0, 5000.0]    # blackbody scale radius rad3
  - [100, 5000]       # distance
  - [0.0, 1.55]       # Av

fixed:
  feh: 0.0
  feh2: -0.5
  logg3: 5.0
```

Do not use identical limits to fix a parameter; put it in `fixed` instead.
For component-specific metallicities, use `feh2`, `feh3`, etc. The number of
grids must match the number of components.

## Performance, Batch Runs, And Diagnostics

Explicit-`Rv` HDF5 grids are substantially larger than fixed-`Rv` FITS grids.
For ordinary fits, the defaults are designed to keep memory bounded while
preserving the exact likelihood:

```yaml
init_method: auto
hdf5_preload: false
hdf5_walker_cache: true
hdf5_auto_full_cache_max_gb: 2.0
vectorized_likelihood: true
```

For a single HDF5 component, `init_method: auto` selects a grid-aware walker
initializer. It searches real atmosphere/extinction nodes, profiles the
radius-distance normalization, and re-ranks candidates with the complete
posterior. If the fast seed is implausibly poor, `init_grid_rescue: true`
enables a derivative-free global rescue search. MAP initialization
(`init_method: map`) remains available for smooth FITS grids but is rejected
for piecewise, non-rectangular HDF5 grids.

The HDF5 cache contains only real spectra inside the active setup limits.
Proposals outside a local cache automatically fall back to exact HDF5
interpolation, so caching does not restrict or alter the posterior. Set
`hdf5_preload: true` only when non-MCMC code also benefits from eager loading.

For many targets, use source-level parallelism with a CSV manifest:

```bash
sedforge batch sources.csv --setup-template template.yaml --workers 8
```

The manifest may contain a `setup_file` column, or columns that override the
template. Dotted names update nested YAML values:

```text
source_id,photometryfile,output_dir,priors.distance,fixed.feh
src001,phot/src001.phot,runs/src001,"[262.8, 10.0]",0.0
src002,phot/src002.phot,runs/src002,"[120.5, 5.0]",0.0
```

Batch mode defaults to one MCMC worker per source and no plots. Homogeneous
single-component jobs prewarm the union of their grid limits and passbands
before workers fork, allowing compatible workers to share read-only cache
pages. Use `--plots` for selected diagnostic runs, and adjust the shared-cache
limit with `--shared-grid-cache-max-gb` when needed.

Repeated batches can reuse a persistent runtime cache:

```bash
sedforge batch sources.csv --setup-template template.yaml --workers 8 \
  --runtime-grid-cache-dir /path/to/sedforge-runtime-cache
```

The same location can be set with `SEDFORGE_RUNTIME_CACHE` or
`runtime_grid_cache_dir` in a setup. Cache identities include the source grid
metadata, active limits, variables, and passbands; stale or incomplete entries
are ignored automatically.

Post-burn chains are checked with rank-normalized/folded split R-hat, bulk and
tail effective sample sizes, and walker acceptance fractions. Diagnostics are
recorded in the result CSV and can also be written to YAML:

```yaml
convergence_rhat_threshold: 1.05
convergence_min_acceptance: 0.01
convergence_min_bulk_ess: 100
convergence_min_tail_ess: 100
convergence_action: warn
diagnosticsfile: target_mcmc_diagnostics.yaml
```

Use `convergence_action: error` for production batches that should reject a
chain failing the sampling-quality thresholds. Identifiability indicators such
as posterior width and boundary occupancy are reported but do not fail a fit.

## Useful Commands

Create a starter setup:

```bash
sedforge setup my_target -grid ck_all
```

Run a fit:

```bash
sedforge fit my_target_setup_ck_all.yaml --noplot
```

Check installed model grids:

```bash
sedforge checkgrids
sedforge checkgrids --bands
```

Download a magnitude photometry file from configured VizieR catalogs:

```bash
sedforge photometry --ra 10.6847083 --dec 41.26875 \
  --output my_target.phot
```

Run the local synthetic recovery scripts:

```bash
python synthetic_recovery_ck_all.py
python synthetic_recovery_binary_ck_all.py
```

Run tests:

```bash
python -m pytest
```

Build source and wheel distributions:

```bash
python -m build
```

## Outputs

Typical outputs are:

- `resultfile`: one-row CSV with median values and separate 16th/84th
  percentile uncertainties (`*_err_minus`, `*_err_plus`) plus MCMC quality
  fields such as `mcmc_status` and `mcmc_max_split_rhat`;
- `datafile`: accepted MCMC samples as a FITS table;
- `diagnosticsfile` (optional): YAML containing convergence, initialization,
  and grid-cache diagnostics;
- SED plot: observed fluxes and best/percentile model SED;
- corner plot: posterior distributions for sampled parameters.

The corner plot labels include physical units for common parameters:
`teff`, `logg`, `feh`, `rad`, `distance`, and `av`.

## License And Citation

This fork is derived from the original
[Speedyfit](https://github.com/vosjo/speedyfit) package by Joris Vos and keeps
the original GPLv3 license. See `LICENSE` for the full license text.

If you publish results based on this code, cite sedforge using `CITATION.cff`.
Because sedforge is derived from Speedyfit, also cite the Speedyfit repository
and the related Speedyfit science papers:

- [Speedyfit](https://github.com/vosjo/speedyfit), the original software
  repository;
- Vos et al. (2017), *The orbits of subdwarf-B + main-sequence binaries. III.
  The period-eccentricity distribution*, A&A, 605, A109,
  doi:[10.1051/0004-6361/201730958](https://doi.org/10.1051/0004-6361/201730958);
- Vos et al. (2018), *Composite hot subdwarf binaries - I. The
  spectroscopically confirmed sdB sample*, MNRAS, 473, 693-709,
  doi:[10.1093/mnras/stx2198](https://doi.org/10.1093/mnras/stx2198).

Also cite the model atmosphere grids, filter curves/catalogs, and extinction
law used for the integrated grids. For example, a paper should document:

- the prepared sedforge model-grid release if those archives are used:
  [doi:10.5281/zenodo.20520723](https://doi.org/10.5281/zenodo.20520723);
- the model family and grid release, such as Castelli & Kurucz, PHOENIX/NewEra,
  TLUSTY, Koester, TMAP, or blackbody grids, and the corresponding original
  papers or model-grid documentation for each spectral model family used;
- the source of filter response curves, such as the
  [SVO Filter Profile Service](https://svo2.cab.inta-csic.es/theory/fps/),
  following the service's acknowledgement and citation guidance;
- the photometry catalogs queried, such as Gaia DR3, 2MASS, AllWISE, PS1, SDSS,
  GLIMPSE, SkyMapper, or GALEX;
- the extinction law and parameters, such as `WC2019`, `Rv`, and `case1`.
  For the bundled WC2019 law, cite Wang & Chen (2019),
  doi:[10.3847/1538-4357/ab1c61](https://doi.org/10.3847/1538-4357/ab1c61).

For issues, reproducible examples, or release questions, open an issue in the
GitHub repository after publishing the project.
