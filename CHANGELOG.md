# Changelog

All notable user-facing changes to sedforge are documented here.

## 0.3.1 - 2026-07-24

- Fixed interpolation near missing combinations in non-rectangular NewEra FITS
  grids by renormalizing the available multilinear corner weights.
- Added grid-aware walker initialization for sparse FITS grids, including
  fixed-`Rv` NewEra models, and rejected finite-difference MAP initialization on
  piecewise model domains.
- Validate that every initial MCMC walker has a finite posterior before
  sampling begins.
- Broadcast fixed scalar atmosphere parameters across vectorized walker
  evaluations.
- Restored the standard emcee stretch-move scale (`a = 2`) as the generated
  setup and API default.

## 0.3.0 - 2026-07-11

- Added grid-aware initialization and bounded sparse caching for explicit-`Rv`
  HDF5 grids.
- Added vectorized ensemble likelihood evaluation for compatible
  single-component fits.
- Added source-level `sedforge batch` execution with shared and persistent grid
  caches.
- Added rank-normalized split-R-hat, bulk/tail ESS, acceptance, and
  identifiability diagnostics with configurable quality policies.
- Added full built-in metadata for `ck03_rv` and `newera_alpha0_rv` grids.
- Added generic CK03 `Rv` grid construction and HDF5 grid subsetting tools.
- Expanded tests and release documentation for the new execution paths.

## 0.2.5 - 2026-06-02

- Prepared the initial public sedforge release with modern package metadata,
  magnitude-based photometry input, model-grid documentation, and tests.
