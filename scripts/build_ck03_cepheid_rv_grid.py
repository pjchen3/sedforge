#!/usr/bin/env python3
"""Build a CK03 Cepheid integrated grid with Av and Rv axes."""

import argparse
import os
import sys
import time
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402


CK_MEMBERS = [
    ("ck03_m20.fits", -2.0),
    ("ck03_m15.fits", -1.5),
    ("ck03_m10.fits", -1.0),
    ("ck03_m05.fits", -0.5),
    ("ck03_p00.fits", 0.0),
    ("ck03_p02.fits", 0.2),
    ("ck03_p05.fits", 0.5),
]

DEFAULT_RESPONSES = [
    "GAIA3E",
    "SDSS",
    "PS1",
    "2MASS",
    "WISE_RSR_W1",
    "WISE_RSR_W2",
]

_WAVE = None
_FLUX = None
_AVS = None
_WEIGHTS = None
_INVALID = None
_PHOTBANDS = None
_OUTDIR = None
_CHUNK_SIZE = None
_LAW = None
_CASE1 = None


def _default_root():
    here = Path(__file__).resolve()
    if here.parents[1].name == "sedforge":
        return here.parents[2]
    return Path.cwd()


def _hdu_parameters(hdu):
    header = hdu.header
    return float(header["TEFF"]), float(header["LOGG"])


def _load_ck_spectra(model_dir, teff_min, teff_max):
    from sedforge import model

    rows = []
    waves = []
    fluxes = []
    for filename, feh in CK_MEMBERS:
        path = model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        with fits.open(path, memmap=True) as hdul:
            for hdu in hdul[1:]:
                teff, logg = _hdu_parameters(hdu)
                if teff < teff_min or teff > teff_max:
                    continue
                wave = np.asarray(hdu.data["wavelength"], dtype=np.float64)
                flux = np.asarray(hdu.data["flux"], dtype=np.float32)
                if waves and not np.array_equal(wave, waves[0]):
                    raise ValueError(f"Non-matching wavelength grid in {path}:{hdu.name}")
                if not waves:
                    waves.append(wave)
                labs = float(model.luminosity(wave, flux))
                rows.append((teff, logg, feh, labs))
                fluxes.append(flux)

    if not rows:
        raise ValueError("No CK03 spectra matched the requested Teff range.")

    rows = np.asarray(rows, dtype=[
        ("teff", "f8"),
        ("logg", "f8"),
        ("feh", "f8"),
        ("Labs", "f8"),
    ])
    fluxes = np.asarray(fluxes, dtype=np.float32)
    order = np.lexsort((rows["logg"], rows["teff"], rows["feh"]))
    return waves[0], rows[order], fluxes[order]


def _chunk_name(rv):
    return f"rv_{rv:05.2f}.h5"


def _worker_init(wave, flux, avs, weights, invalid, photbands, outdir,
                 chunk_size, law, case1):
    global _WAVE, _FLUX, _AVS, _WEIGHTS, _INVALID, _PHOTBANDS
    global _OUTDIR, _CHUNK_SIZE, _LAW, _CASE1
    _WAVE = wave
    _FLUX = flux
    _AVS = avs
    _WEIGHTS = weights
    _INVALID = invalid
    _PHOTBANDS = photbands
    _OUTDIR = outdir
    _CHUNK_SIZE = int(chunk_size)
    _LAW = law
    _CASE1 = int(case1)


def _valid_existing_chunk(path, nspec, nav, nfilter, rv):
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as h5:
            if "flux" not in h5:
                return False
            if h5["flux"].shape != (nspec, nav, nfilter):
                return False
            if not np.isclose(float(h5.attrs["rv"]), float(rv)):
                return False
    except Exception:
        return False
    return True


def _write_rv_chunk(rv):
    from sedforge import reddening

    nspec, nwave = _FLUX.shape
    nav = len(_AVS)
    nfilter = len(_PHOTBANDS)
    path = Path(_OUTDIR) / "chunks" / _chunk_name(float(rv))
    if _valid_existing_chunk(path, nspec, nav, nfilter, rv):
        return str(path), "skipped"

    tmp_path = path.with_suffix(".tmp.h5")
    if tmp_path.exists():
        tmp_path.unlink()

    _, redmag = reddening.get_law(
        _LAW,
        wave=_WAVE,
        norm="Av",
        Rv=float(rv),
        case1=_CASE1,
    )
    red = 10.0 ** (-0.4 * _AVS[:, None] * redmag[None, :])
    red = np.asarray(red, dtype=np.float32)

    with h5py.File(tmp_path, "w") as h5:
        dset = h5.create_dataset(
            "flux",
            shape=(nspec, nav, nfilter),
            dtype="f4",
            chunks=(min(_CHUNK_SIZE, nspec), min(64, nav), nfilter),
            compression="lzf",
            shuffle=True,
        )
        h5.attrs["rv"] = float(rv)
        h5.attrs["law"] = str(_LAW)
        h5.attrs["case1"] = int(_CASE1)
        h5.attrs["flux_layout"] = "spec,av,filter"

        for start in range(0, nspec, _CHUNK_SIZE):
            stop = min(start + _CHUNK_SIZE, nspec)
            block = _FLUX[start:stop, None, :] * red[None, :, :]
            out = np.tensordot(block, _WEIGHTS, axes=([2], [0]))
            out = np.asarray(out, dtype=np.float32)
            if len(_INVALID):
                out[:, :, _INVALID] = np.nan
            dset[start:stop, :, :] = out

    tmp_path.replace(path)
    return str(path), "done"


def _write_virtual_grid(outdir, grid_path, spectra, avs, rvs, photbands,
                        chunk_files, attrs):
    nspec = len(spectra)
    nav = len(avs)
    nrv = len(rvs)
    nfilter = len(photbands)
    if grid_path.exists():
        grid_path.unlink()

    layout = h5py.VirtualLayout(
        shape=(nspec, nrv, nav, nfilter),
        dtype=np.dtype("f4"),
    )
    for irv, chunk in enumerate(chunk_files):
        rel = os.path.relpath(chunk, start=grid_path.parent)
        source = h5py.VirtualSource(rel, "flux", shape=(nspec, nav, nfilter))
        layout[:, irv, :, :] = source

    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(grid_path, "w", libver="latest") as h5:
        h5.create_virtual_dataset("flux", layout, fillvalue=np.nan)
        axes = h5.create_group("axes")
        axes.create_dataset("av", data=np.asarray(avs, dtype=np.float32))
        axes.create_dataset("rv", data=np.asarray(rvs, dtype=np.float32))
        axes.create_dataset("photband", data=np.asarray(photbands, dtype=object),
                            dtype=str_dtype)
        spec = h5.create_group("spectra")
        for name in spectra.dtype.names:
            spec.create_dataset(name, data=np.asarray(spectra[name], dtype=np.float64))
        h5.create_dataset("chunk_files",
                          data=np.asarray([os.path.relpath(p, grid_path.parent)
                                           for p in chunk_files], dtype=object),
                          dtype=str_dtype)
        for key, value in attrs.items():
            h5.attrs[key] = value
        h5.attrs["flux_layout"] = "spec,rv,av,filter"
        h5.attrs["grid"] = "ck03_cepheid_rv"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Build a CK03 Cepheid HDF5 integrated grid with an Rv axis."
    )
    parser.add_argument("--root", type=Path, default=_default_root())
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--package-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=96)
    parser.add_argument("--teff-min", type=float, default=4000.0)
    parser.add_argument("--teff-max", type=float, default=8000.0)
    parser.add_argument("--rv-min", type=float, default=2.0)
    parser.add_argument("--rv-max", type=float, default=5.0)
    parser.add_argument("--rv-step", type=float, default=0.01)
    parser.add_argument("--law", default="WC2019")
    parser.add_argument("--case1", type=int, default=1)
    parser.add_argument(
        "--responses",
        default=",".join(DEFAULT_RESPONSES),
        help="Comma-separated response systems/passbands, or 'all'.",
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Build only two Rv slices for a quick validation run.")
    return parser.parse_args()


def main():
    args = _parse_args()
    root = args.root.resolve()
    model_dir = (args.model_dir or root / "sed_models").resolve()
    package_dir = (args.package_dir or root / "sedforge").resolve()
    output_dir = (args.output_dir or model_dir / "ck03_cepheid_rv").resolve()
    chunks_dir = output_dir / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(package_dir))
    os.environ["SEDFORGE_MODELS"] = str(model_dir)

    from sedforge import integrate_grid

    avs = integrate_grid.default_av_grid()
    rvs = np.round(
        np.arange(args.rv_min, args.rv_max + 0.5 * args.rv_step, args.rv_step),
        6,
    )
    if args.smoke:
        rvs = np.asarray([args.rv_min, args.rv_max], dtype=float)

    t0 = time.time()
    wave, spectra, flux = _load_ck_spectra(model_dir, args.teff_min, args.teff_max)
    responses = None if args.responses.lower() == "all" else [
        item.strip() for item in args.responses.split(",") if item.strip()
    ]
    photbands = integrate_grid.get_responses(responses=responses, wave=wave)
    weights, invalid = integrate_grid._response_weight_matrix(wave, photbands)
    weights = np.asarray(weights, dtype=np.float32)
    invalid = np.asarray(invalid, dtype=np.int64)

    print("CK03 Cepheid Rv grid")
    print(f"  spectra: {len(spectra)}")
    print(f"  teff: {args.teff_min:g}..{args.teff_max:g} K")
    print(f"  feh: {np.min(spectra['feh']):g}..{np.max(spectra['feh']):g}")
    print(f"  logg: {np.min(spectra['logg']):g}..{np.max(spectra['logg']):g}")
    print(f"  Av points: {len(avs)} ({avs[0]:g}..{avs[-1]:g})")
    print(f"  Rv points: {len(rvs)} ({rvs[0]:g}..{rvs[-1]:g})")
    print(f"  filters: {len(photbands)}")
    print(f"  threads: {args.threads}")
    print(f"  output: {output_dir}")
    print("  one numerical thread per worker enforced by environment")

    from multiprocessing import get_context

    ctx = get_context("fork")
    chunk_files = [str(chunks_dir / _chunk_name(float(rv))) for rv in rvs]
    with ctx.Pool(
        processes=args.threads,
        initializer=_worker_init,
        initargs=(
            wave,
            flux,
            np.asarray(avs, dtype=np.float32),
            weights,
            invalid,
            np.asarray(photbands, dtype=object),
            str(output_dir),
            args.chunk_size,
            args.law,
            args.case1,
        ),
    ) as pool:
        completed = 0
        for path, status in pool.imap_unordered(_write_rv_chunk, rvs):
            completed += 1
            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0.0
            remain = (len(rvs) - completed) / rate if rate > 0 else np.nan
            print(
                f"[{completed:4d}/{len(rvs):4d}] {status:7s} "
                f"{Path(path).name}  elapsed={elapsed/60:.1f}m eta={remain/60:.1f}m",
                flush=True,
            )

    grid_path = output_dir / "ck03_cepheid_rv_grid.h5"
    _write_virtual_grid(
        output_dir,
        grid_path,
        spectra,
        avs,
        rvs,
        photbands,
        chunk_files,
        attrs={
            "teff_min": float(args.teff_min),
            "teff_max": float(args.teff_max),
            "feh_min": float(np.min(spectra["feh"])),
            "feh_max": float(np.max(spectra["feh"])),
            "law": str(args.law),
            "case1": int(args.case1),
            "extaxis": "Av",
            "created_by": Path(__file__).name,
        },
    )
    print(f"Wrote virtual grid: {grid_path}")
    print(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
