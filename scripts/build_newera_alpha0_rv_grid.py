#!/usr/bin/env python3
"""Build a PHOENIX NewEra alpha=0 integrated HDF5 grid with Av and Rv axes."""

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

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _path in (_REPO_ROOT, _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from build_newera_alpha0_integrated import (  # noqa: E402
    expected_alpha0_count,
    iter_alpha0_spectra,
    select_responses,
    wavelength_from_header,
)
from sedforge import integrate_grid, model  # noqa: E402


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
    if len(here.parents) > 1 and here.parents[1].name == "sedforge":
        return here.parents[2]
    return Path.cwd()


def _parse_responses(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in {"all", "auto"}:
        return None
    return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]


def _chunk_name(rv):
    return f"rv_{float(rv):05.2f}.h5"


def _rv_axis(rv_min, rv_max, rv_step, smoke=False):
    if rv_step <= 0:
        raise ValueError("--rv-step must be positive.")
    if rv_max < rv_min:
        raise ValueError("--rv-max must be greater than or equal to --rv-min.")
    if smoke:
        return np.unique(np.round(np.asarray([rv_min, rv_max], dtype=float), 8))
    rvs = np.arange(rv_min, rv_max + 0.5 * rv_step, rv_step, dtype=float)
    return np.unique(np.round(rvs, 8))


def _load_newera_spectra(args):
    responses = _parse_responses(args.responses)
    expected = expected_alpha0_count(args.newera_dir)
    if expected is not None:
        print(f"Expected alpha=0 NewEra spectra before filtering: {expected}")

    rows = []
    fluxes = []
    seen = set()
    wave0 = None
    photbands = None
    t0 = time.time()

    for header, mh, flux in iter_alpha0_spectra(args.newera_dir):
        teff = float(header["teff"])
        logg = float(header["logg"])
        mh = float(mh)
        if mh < args.feh_min or mh > args.feh_max:
            continue
        if teff < args.teff_min or teff > args.teff_max:
            continue
        if logg < args.logg_min or logg > args.logg_max:
            continue

        key = (round(teff, 6), round(logg, 6), round(mh, 6))
        if key in seen:
            continue
        seen.add(key)

        wave = wavelength_from_header(header)
        if wave0 is None:
            wave0 = np.asarray(wave, dtype=np.float64)
            photbands = select_responses(
                wave0,
                requested=responses,
                min_coverage=args.min_filter_coverage,
            )
        elif len(wave) != len(wave0) or not np.allclose(wave, wave0, rtol=0, atol=1e-8):
            raise ValueError(
                "NewEra spectra in this build do not share one wavelength grid; "
                "this HDF5 builder expects the LowRes common wavelength grid."
            )

        flux = np.asarray(flux, dtype=np.float32)
        labs = float(model.luminosity(wave0, flux))
        rows.append((teff, logg, mh, labs))
        fluxes.append(flux)

        if len(rows) % args.progress_every == 0:
            elapsed = time.time() - t0
            rate = len(rows) / elapsed if elapsed > 0 else np.nan
            print(f"Loaded {len(rows)} spectra after filtering ({rate:.2f} spectra/s)")

        if args.max_spectra is not None and len(rows) >= args.max_spectra:
            break

    if not rows:
        raise RuntimeError("No NewEra alpha=0 spectra matched the requested limits.")
    if photbands is None or wave0 is None:
        raise RuntimeError("No photometric responses were selected.")

    spectra = np.asarray(
        rows,
        dtype=[
            ("teff", "f8"),
            ("logg", "f8"),
            ("feh", "f8"),
            ("Labs", "f8"),
        ],
    )
    fluxes = np.asarray(fluxes, dtype=np.float32)
    order = np.lexsort((spectra["logg"], spectra["teff"], spectra["feh"]))
    return wave0, spectra[order], fluxes[order], photbands


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

    nspec, _ = _FLUX.shape
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


def _write_virtual_grid(grid_path, spectra, avs, rvs, photbands, chunk_files,
                        grid_name, attrs):
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
        axes.create_dataset(
            "photband",
            data=np.asarray(photbands, dtype=object),
            dtype=str_dtype,
        )
        spec = h5.create_group("spectra")
        for name in spectra.dtype.names:
            spec.create_dataset(name, data=np.asarray(spectra[name], dtype=np.float64))
        h5.create_dataset(
            "chunk_files",
            data=np.asarray(
                [os.path.relpath(p, grid_path.parent) for p in chunk_files],
                dtype=object,
            ),
            dtype=str_dtype,
        )
        for key, value in attrs.items():
            h5.attrs[key] = value
        h5.attrs["flux_layout"] = "spec,rv,av,filter"
        h5.attrs["grid"] = grid_name


def _update_grid_description(model_dir, grid_name, grid_path):
    desc_path = Path(model_dir) / "grid_description.yaml"
    if desc_path.is_file():
        with desc_path.open() as handle:
            desc = yaml.safe_load(handle) or {}
    else:
        desc = {}

    integrated_path = os.path.relpath(grid_path, start=Path(model_dir))
    desc.setdefault(grid_name, {})
    desc[grid_name].update({
        "filename": grid_name,
        "integrated_path": integrated_path,
        "integrated_format": "hdf5",
        "axes": ["teff", "logg", "feh", "rv", "av"],
        "supports_feh": True,
        "supports_rv": True,
        "info": (
            "PHOENIX NewEra V3 LowRes alpha=0 integrated grid with explicit "
            "Rv and Av axes; default build keeps [Fe/H] >= -2.5."
        ),
    })
    with desc_path.open("w") as handle:
        yaml.safe_dump(desc, handle, sort_keys=False)


def _print_summary(args, spectra, flux, avs, rvs, photbands, output_dir):
    nspec = len(spectra)
    nav = len(avs)
    nrv = len(rvs)
    nfilter = len(photbands)
    wave_gib = flux.nbytes / 2 ** 30
    chunk_gib = nspec * nav * nfilter * np.dtype("f4").itemsize / 2 ** 30
    logical_gib = chunk_gib * nrv

    print("NewEra alpha=0 Rv grid")
    print(f"  spectra: {nspec}")
    print(f"  teff: {np.min(spectra['teff']):g}..{np.max(spectra['teff']):g} K")
    print(f"  logg: {np.min(spectra['logg']):g}..{np.max(spectra['logg']):g}")
    print(f"  feh: {np.min(spectra['feh']):g}..{np.max(spectra['feh']):g}")
    print(f"  Av points: {nav} ({avs[0]:g}..{avs[-1]:g})")
    print(f"  Rv points: {nrv} ({rvs[0]:g}..{rvs[-1]:g})")
    print(f"  filters: {nfilter}")
    print(f"  loaded spectra array: {wave_gib:.2f} GiB")
    print(f"  one Rv chunk, uncompressed: {chunk_gib:.2f} GiB")
    print(f"  full virtual grid, logical size: {logical_gib:.2f} GiB")
    print(f"  law/case1: {args.law}/case{args.case1}")
    print(f"  threads: {args.threads}")
    print(f"  output: {output_dir}")
    print("  one numerical thread per worker enforced by environment")


def parse_args():
    root = _default_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--newera-dir",
        type=Path,
        default=Path(os.environ.get("NEWERA_DIR", root / "NewEra")),
        help="Directory containing PHOENIX-NewEraV3-LowRes-SPECTRA.tar.gz.",
    )
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--package-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--grid-name", default="newera_alpha0_rv")
    parser.add_argument("--threads", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=96)
    parser.add_argument("--law", default="WC2019")
    parser.add_argument("--case1", type=int, default=1)
    parser.add_argument("--av-min", type=float, default=0.0)
    parser.add_argument("--av-max", type=float, default=4.0)
    parser.add_argument("--av-small-max", type=float, default=1.0)
    parser.add_argument("--av-small-step", type=float, default=0.005)
    parser.add_argument("--av-mid-max", type=float, default=3.0)
    parser.add_argument("--av-mid-step", type=float, default=0.02)
    parser.add_argument("--av-large-step", type=float, default=0.05)
    parser.add_argument("--rv-min", type=float, default=2.0)
    parser.add_argument("--rv-max", type=float, default=5.0)
    parser.add_argument("--rv-step", type=float, default=0.01)
    parser.add_argument("--feh-min", type=float, default=-2.5)
    parser.add_argument("--feh-max", type=float, default=0.5)
    parser.add_argument("--teff-min", type=float, default=-np.inf)
    parser.add_argument("--teff-max", type=float, default=np.inf)
    parser.add_argument("--logg-min", type=float, default=-np.inf)
    parser.add_argument("--logg-max", type=float, default=np.inf)
    parser.add_argument("--min-filter-coverage", type=float, default=0.99)
    parser.add_argument(
        "--responses",
        default="auto",
        help="Comma-separated response systems/passbands, or 'auto'/'all'.",
    )
    parser.add_argument("--max-spectra", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Build only the Rv endpoints; combine with --max-spectra for quick tests.",
    )
    parser.add_argument("--update-grid-description", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.resolve()
    model_dir = (args.model_dir or root / "sed_models").resolve()
    package_dir = (args.package_dir or root / "sedforge").resolve()
    output_dir = (args.output_dir or model_dir / args.grid_name).resolve()
    chunks_dir = output_dir / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))
    os.environ["SEDFORGE_MODELS"] = str(model_dir)

    avs = integrate_grid.default_av_grid(
        av_min=args.av_min,
        av_max=args.av_max,
        small_max=args.av_small_max,
        small_step=args.av_small_step,
        mid_max=args.av_mid_max,
        mid_step=args.av_mid_step,
        large_step=args.av_large_step,
    )
    rvs = _rv_axis(args.rv_min, args.rv_max, args.rv_step, args.smoke)

    t0 = time.time()
    wave, spectra, flux, photbands = _load_newera_spectra(args)
    weights, invalid = integrate_grid._response_weight_matrix(wave, photbands)
    weights = np.asarray(weights, dtype=np.float32)
    invalid = np.asarray(invalid, dtype=np.int64)

    _print_summary(args, spectra, flux, avs, rvs, photbands, output_dir)

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

    grid_path = output_dir / f"{args.grid_name}_grid.h5"
    _write_virtual_grid(
        grid_path,
        spectra,
        avs,
        rvs,
        photbands,
        chunk_files,
        args.grid_name,
        attrs={
            "teff_min": float(np.min(spectra["teff"])),
            "teff_max": float(np.max(spectra["teff"])),
            "logg_min": float(np.min(spectra["logg"])),
            "logg_max": float(np.max(spectra["logg"])),
            "feh_min": float(np.min(spectra["feh"])),
            "feh_max": float(np.max(spectra["feh"])),
            "requested_feh_min": float(args.feh_min),
            "requested_feh_max": float(args.feh_max),
            "law": str(args.law),
            "case1": int(args.case1),
            "extaxis": "Av",
            "source": "PHOENIX NewEra V3 LowRes alpha=0",
            "created_by": Path(__file__).name,
        },
    )
    print(f"Wrote virtual grid: {grid_path}")

    if args.update_grid_description:
        _update_grid_description(model_dir, args.grid_name, grid_path)
        print(f"Updated {Path(model_dir) / 'grid_description.yaml'}")

    print(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
