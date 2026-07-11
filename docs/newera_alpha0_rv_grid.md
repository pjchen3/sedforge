# HDF5 Rv grid builds

This note records the server-side HDF5 grids that have explicit `Rv` and `Av`
axes, plus the build command used for the PHOENIX NewEra V3 LowRes alpha=0
grid.

Expected local model-directory layout after downloading or building the HDF5
Rv grids:

```text
sed_models/
  grid_description.yaml
  ck03_rv/
    ck03_rv_grid.h5
  newera_alpha0_rv/
    newera_alpha0_rv_grid.h5
```

Recommended `grid_description.yaml` entries:

```yaml
ck03_rv:
  filename: ck03_rv
  integrated_path: ck03_rv/ck03_rv_grid.h5
  integrated_format: hdf5
  axes:
    - teff
    - logg
    - feh
    - rv
    - av
  supports_feh: true
  supports_rv: true
  info: Castelli & Kurucz 2003 / ATLAS9 integrated grid with explicit Rv and Av axes.

newera_alpha0_rv:
  filename: newera_alpha0_rv
  integrated_path: newera_alpha0_rv/newera_alpha0_rv_grid.h5
  integrated_format: hdf5
  axes:
    - teff
    - logg
    - feh
    - rv
    - av
  supports_feh: true
  supports_rv: true
  info: PHOENIX NewEra V3 LowRes alpha=0 integrated grid with explicit Rv and Av axes; [Fe/H] >= -2.5.
```

Use these grids by setting:

```bash
export SEDFORGE_MODELS=/path/to/sed_models
```

Then select `ck03_rv` or `newera_alpha0_rv` in the setup file. Since `rv` is
an explicit HDF5 axis, remove `reddening_Rv`/`Rv` from the setup and either fit
`rv` in `pnames` or put it under `fixed:`.

Target grid:

- model family: PHOENIX NewEra V3 LowRes, alpha=0
- metallicity: `[Fe/H] >= -2.5`
- extinction: `Av = 0.0..4.0`
- reddening parameter: `Rv = 2.0..5.0`
- stellar parameters: full available `Teff` and `logg` coverage after the
  metallicity cut
- output format: chunked HDF5 virtual grid

The build writes one HDF5 chunk per `Rv` value under
`$SEDFORGE_MODELS/newera_alpha0_rv/chunks/`, then writes the virtual grid:

```text
$SEDFORGE_MODELS/newera_alpha0_rv/newera_alpha0_rv_grid.h5
```

The chunk files make the build resumable. If the command is interrupted, rerun
the same command and existing valid `Rv` chunks will be skipped.

## Environment

Set these paths on the server:

```bash
export NEWERA_DIR=/path/to/NewEra
export SEDFORGE_MODELS=/path/to/sed_models
```

`NEWERA_DIR` must contain:

```text
PHOENIX-NewEraV3-LowRes-SPECTRA.tar.gz
```

If the extra `Z+0.5` text file is available, place it in the same directory:

```text
PHOENIX-NewEraV3-add001-LowRes-SPECTRA.Z+0.5.txt
```

## Smoke test

Run a small build before launching the full grid:

```bash
python scripts/build_newera_alpha0_rv_grid.py \
  --newera-dir "$NEWERA_DIR" \
  --model-dir "$SEDFORGE_MODELS" \
  --package-dir . \
  --feh-min -2.5 \
  --av-min 0.0 \
  --av-max 4.0 \
  --rv-min 2.0 \
  --rv-max 5.0 \
  --rv-step 0.50 \
  --max-spectra 16 \
  --smoke \
  --threads 2
```

The smoke run should create two `Rv` chunks and
`newera_alpha0_rv_grid.h5`.

## Production run

For the full grid:

```bash
nohup python scripts/build_newera_alpha0_rv_grid.py \
  --newera-dir "$NEWERA_DIR" \
  --model-dir "$SEDFORGE_MODELS" \
  --package-dir . \
  --grid-name newera_alpha0_rv \
  --feh-min -2.5 \
  --feh-max 0.5 \
  --av-min 0.0 \
  --av-max 4.0 \
  --rv-min 2.0 \
  --rv-max 5.0 \
  --rv-step 0.01 \
  --law WC2019 \
  --case1 1 \
  --threads 30 \
  --chunk-size 96 \
  --update-grid-description \
  > newera_alpha0_rv_build.log 2>&1 &
```

Adjust `--threads` and `--chunk-size` for the server memory budget. The script
prints the number of spectra, filter count, Av/Rv axis lengths, approximate
loaded spectral-array size, per-`Rv` chunk size, and full logical grid size
before computation starts.

## After the build

Check the model registration:

```bash
export SEDFORGE_MODELS=/path/to/sed_models
sedforge checkgrids
```

The generated `grid_description.yaml` entry should include:

```yaml
newera_alpha0_rv:
  integrated_format: hdf5
  axes:
    - teff
    - logg
    - feh
    - rv
    - av
  supports_feh: true
  supports_rv: true
```
