# Changelog

All notable user-facing changes to sedforge are documented here.

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
