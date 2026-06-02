#!/usr/bin/env python
"""Build NewEra alpha=0 integrated grids and plotting spectra from LSR memmaps."""

import argparse
import csv
import gc
import json
import math
import multiprocessing as mp
import os
import re
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from astropy.table import Table

from sedforge import filters, integrate_grid, model, reddening, spectral_cache


FLUX_MEMMAP = None
WAVE = None
ROWS = None
KERNELS = None
CACHE_INDEX = None
CACHE_FRAC = None
CACHE_NWAVE = None

DEFAULT_EXCLUDED_TEFF = (3350.0, 5770.0, 6050.0, 6060.0, 9602.0)
DEFAULT_EXCLUDED_LOGG = (4.15, 4.95)


def lsr_logflux_to_flambda(logflux):
    """Convert NewEra HDF5 log10(F_lambda per cm) to F_lambda per Angstrom."""
    return (np.power(10.0, np.asarray(logflux, dtype=np.float32)) * 1.0e-8).astype(np.float32)


def parse_newera_filename(filename):
    match = re.match(
        r"lte(?P<teff>\d+)-(?P<logg>[0-9]+(?:\.[0-9]+)?)(?P<feh>[+-][0-9]+(?:\.[0-9]+)?)",
        Path(filename).name,
    )
    if match is None:
        raise ValueError(f"Could not parse NewEra parameters from {filename}")
    feh = float(match.group("feh"))
    if np.isclose(feh, -0.0):
        feh = 0.0
    return float(match.group("teff")), float(match.group("logg")), feh


def read_completed_rows(lsr_dir, max_spectra=None, exclude_teff=None, exclude_logg=None):
    completed = Path(lsr_dir) / "completed.tsv"
    excluded_teff = set(float(value) for value in (exclude_teff or []))
    excluded_logg = set(float(value) for value in (exclude_logg or []))
    rows = []
    excluded_by_teff = []
    excluded_by_logg = []
    seen = set()
    with completed.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            teff, logg, feh = parse_newera_filename(row["filename"])
            key = (teff, logg, feh)
            if key in seen:
                continue
            seen.add(key)
            if any(np.isclose(teff, value) for value in excluded_teff):
                excluded_by_teff.append((teff, logg, feh, row["filename"]))
                continue
            if any(np.isclose(logg, value) for value in excluded_logg):
                excluded_by_logg.append((teff, logg, feh, row["filename"]))
                continue
            rows.append(
                {
                    "alpha_index": int(row["alpha_index"]),
                    "teff": teff,
                    "logg": logg,
                    "feh": feh,
                    "filename": row["filename"],
                }
            )
            if max_spectra is not None and len(rows) >= max_spectra:
                break
    rows.sort(key=lambda item: (item["teff"], item["logg"], item["feh"]))
    if excluded_by_teff:
        counts = {}
        for teff, _logg, _feh, _filename in excluded_by_teff:
            counts[teff] = counts.get(teff, 0) + 1
        parts = [f"{teff:g} K: {counts[teff]}" for teff in sorted(counts)]
        print(
            "Excluded special NewEra Teff points: " + ", ".join(parts),
            flush=True,
        )
    if excluded_by_logg:
        counts = {}
        examples = {}
        for teff, logg, feh, _filename in excluded_by_logg:
            counts[logg] = counts.get(logg, 0) + 1
            examples.setdefault(logg, []).append(f"Teff={teff:g}, [Fe/H]={feh:g}")
        parts = [
            f"logg={logg:g}: {counts[logg]} ({'; '.join(examples[logg])})"
            for logg in sorted(counts)
        ]
        print(
            "Excluded isolated NewEra logg points: " + ", ".join(parts),
            flush=True,
        )
    return rows


def load_lsr_shape(lsr_dir):
    with (Path(lsr_dir) / "shape.json").open() as handle:
        shape = json.load(handle)
    return shape


def _rv_label(rv):
    return f"{float(rv):0.2f}"


def _law_label(law, case1=1):
    if str(law).lower() == "wc2019" and int(case1) != 1:
        return f"{law}_case{int(case1)}"
    return law


def default_integrated_name(grid_name, law, rv, case1):
    return f"i{grid_name}_law{_law_label(law, case1)}_Rv{_rv_label(rv)}.fits"


def select_responses(wave, requested=None):
    if requested:
        responses = []
        for item in requested:
            responses.extend(filters.list_response(item))
        return list(dict.fromkeys(responses))
    return filters.list_response(wave_range=(float(wave[0]), float(wave[-1])))


def compile_filter_kernels(wave, responses, avs, law, rv, case1):
    kernels = []
    for response in responses:
        weights, invalid = integrate_grid._response_weight_matrix(wave, [response])
        if len(invalid):
            raise ValueError(f"Response {response} has invalid integration weights")
        weights = np.asarray(weights[:, 0], dtype=np.float64)
        index = np.flatnonzero(np.isfinite(weights) & (weights != 0.0))
        if len(index) == 0:
            raise ValueError(f"Response {response} has zero useful weights")
        wave_slice = np.asarray(wave[index], dtype=np.float64)
        _, redmag = reddening.get_law(law, wave=wave_slice, norm="Av", Rv=rv, case1=case1)
        transmission = 10 ** (-0.4 * np.asarray(avs, dtype=np.float64)[:, None] * redmag[None, :])
        kernel = transmission * weights[index][None, :]
        kernels.append((index.astype(np.int64), np.asarray(kernel, dtype=np.float32)))
        print(
            f"Compiled {response}: {len(index)} wavelength points "
            f"({wave_slice[0]:.1f}-{wave_slice[-1]:.1f} A)",
            flush=True,
        )
    return kernels


def init_integrated_worker(flux_path, shape, wave_path, rows, kernels):
    global FLUX_MEMMAP, WAVE, ROWS, KERNELS
    FLUX_MEMMAP = np.memmap(
        flux_path,
        dtype=np.dtype(shape["dtype"]),
        mode="r",
        shape=(int(shape["n_spectra"]), int(shape["n_wave"])),
    )
    WAVE = np.load(wave_path, mmap_mode="r")
    ROWS = rows
    KERNELS = kernels


def integrated_chunk(task):
    start, stop, avs = task
    n_av = len(avs)
    n_responses = len(KERNELS)
    block = np.empty((stop - start, n_av, 5 + n_responses), dtype=np.float32)
    wave = np.asarray(WAVE, dtype=np.float64)

    for local_i, row_i in enumerate(range(start, stop)):
        row = ROWS[row_i]
        flux = lsr_logflux_to_flambda(FLUX_MEMMAP[row["alpha_index"]])
        block[local_i, :, 0] = row["teff"]
        block[local_i, :, 1] = row["logg"]
        block[local_i, :, 2] = row["feh"]
        block[local_i, :, 3] = model.luminosity(wave, flux)
        block[local_i, :, 4] = avs
        for response_i, (index, kernel) in enumerate(KERNELS):
            block[local_i, :, 5 + response_i] = kernel.dot(flux[index])

    return start, block.reshape((stop - start) * n_av, 5 + n_responses)


def write_integrated_fits(path, memmap_path, n_rows, column_names, metadata):
    data = np.memmap(memmap_path, dtype=np.float32, mode="r", shape=(n_rows, len(column_names)))
    table = Table(data=data, names=column_names)
    for key, value in metadata.items():
        table.meta[key] = value
    print(f"Writing {path}", flush=True)
    table.write(path, overwrite=True)
    del table
    del data
    gc.collect()


def chunk_ranges(n_items, chunk_size):
    for start in range(0, n_items, chunk_size):
        yield start, min(start + chunk_size, n_items)


def build_integrated(args, shape, wave, rows, responses):
    avs = integrate_grid.default_av_grid(
        av_min=args.av_min,
        av_max=args.av_max,
        small_max=args.av_small_max,
        small_step=args.av_small_step,
        mid_max=args.av_mid_max,
        mid_step=args.av_mid_step,
        large_step=args.av_large_step,
    ).astype(np.float32)
    kernels = compile_filter_kernels(wave, responses, avs, args.law, args.rv, args.case1)
    out_path = Path(args.output_dir) / "integrated" / default_integrated_name(
        args.grid_name, args.law, args.rv, args.case1
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp.dat")
    n_rows = len(rows) * len(avs)
    n_cols = 5 + len(responses)
    out = np.memmap(tmp_path, dtype=np.float32, mode="w+", shape=(n_rows, n_cols))

    tasks = [(start, stop, avs) for start, stop in chunk_ranges(len(rows), args.chunk_size)]
    t0 = time.time()
    done = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=args.processes,
        initializer=init_integrated_worker,
        initargs=(shape["flux_path"], shape, shape["wavelength_path"], rows, kernels),
    ) as pool:
        for start, block in pool.imap_unordered(integrated_chunk, tasks, chunksize=1):
            row0 = start * len(avs)
            out[row0:row0 + len(block)] = block
            done += len(block) // len(avs)
            if done == len(rows) or done % args.progress_every < args.chunk_size:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else math.nan
                print(
                    f"Integrated {done}/{len(rows)} spectra "
                    f"({rate:.2f} spectra/s)",
                    flush=True,
                )
    out.flush()
    del out

    metadata = {
        "GRID": args.grid_name,
        "SOURCE": "PHOENIX NewEra HSR LSR memmap",
        "ALPHA": 0.0,
        "FLUXTYPE": "Flambda",
        "FLUXUNIT": "erg/s/cm2/Angstrom",
        "REDLAW": args.law,
        "RV": float(args.rv),
        "CASE1": int(args.case1),
        "EXTAXIS": "Av",
        "NAV": len(avs),
        "AVMIN": float(np.nanmin(avs)),
        "AVMAX": float(np.nanmax(avs)),
        "NSPEC": len(rows),
        "NWAVEIN": int(shape["n_wave"]),
        "EXCLTEFF": ",".join(f"{value:g}" for value in args.exclude_teff),
        "EXCLLOGG": ",".join(f"{value:g}" for value in args.exclude_logg),
    }
    write_integrated_fits(
        out_path,
        tmp_path,
        n_rows,
        ["teff", "logg", "feh", "Labs", "av"] + responses,
        metadata,
    )
    if not args.keep_tmp:
        tmp_path.unlink()
    return out_path


def cache_interp_indices(source_wave, target_wave):
    pos = np.searchsorted(source_wave, target_wave, side="left")
    pos = np.clip(pos, 1, len(source_wave) - 1)
    left = pos - 1
    right = pos
    denom = source_wave[right] - source_wave[left]
    frac = np.where(denom > 0, (target_wave - source_wave[left]) / denom, 0.0)
    return left.astype(np.int64), frac.astype(np.float32)


def make_cache_wave(args, source_wave):
    wave_min = max(float(args.cache_wave_min), float(np.nanmin(source_wave)))
    wave_max = min(float(args.cache_wave_max), float(np.nanmax(source_wave)))
    short_step = float(args.cache_wave_step)
    long_start = args.cache_long_wave_start
    long_step = args.cache_long_wave_step

    if args.cache_wave_grid == "log":
        log_min = np.log10(wave_min)
        log_max = np.log10(wave_max)
        n_intervals = max(1, int(round((log_max - log_min) / float(args.cache_log10_step))))
        log_wave = np.linspace(log_min, log_max, n_intervals + 1, dtype=np.float64)
        return (10 ** log_wave).astype(np.float32)

    if (args.cache_wave_grid == "linear" or long_start is None
            or long_step is None or long_step <= short_step or wave_max <= long_start):
        return np.arange(
            wave_min,
            wave_max + 0.5 * short_step,
            short_step,
            dtype=np.float32,
        )

    break_wave = max(wave_min, min(float(long_start), wave_max))
    short_wave = np.arange(
        wave_min,
        break_wave + 0.5 * short_step,
        short_step,
        dtype=np.float32,
    )
    long_wave = np.arange(
        break_wave + float(long_step),
        wave_max + 0.5 * float(long_step),
        float(long_step),
        dtype=np.float32,
    )
    return np.concatenate([short_wave, long_wave]).astype(np.float32)


def init_cache_worker(flux_path, shape, rows, index, frac, n_wave):
    global FLUX_MEMMAP, ROWS, CACHE_INDEX, CACHE_FRAC, CACHE_NWAVE
    FLUX_MEMMAP = np.memmap(
        flux_path,
        dtype=np.dtype(shape["dtype"]),
        mode="r",
        shape=(int(shape["n_spectra"]), int(shape["n_wave"])),
    )
    ROWS = rows
    CACHE_INDEX = index
    CACHE_FRAC = frac
    CACHE_NWAVE = n_wave


def cache_chunk(task):
    start, stop = task
    flux_block = np.empty((stop - start, CACHE_NWAVE), dtype=np.float32)
    param_block = np.empty((stop - start, 3), dtype=np.float32)
    index = CACHE_INDEX
    frac = CACHE_FRAC
    for local_i, row_i in enumerate(range(start, stop)):
        row = ROWS[row_i]
        flux = lsr_logflux_to_flambda(FLUX_MEMMAP[row["alpha_index"]])
        left = flux[index]
        right = flux[index + 1]
        flux_block[local_i] = left * (1.0 - frac) + right * frac
        param_block[local_i] = (row["teff"], row["logg"], row["feh"])
    return start, param_block, flux_block


def build_spectral_cache(args, shape, wave, rows):
    out_path = Path(args.output_dir) / "spectral_cache" / f"{args.grid_name}_plot_spectra.fits"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target_wave = make_cache_wave(args, wave)
    print(
        f"Cache wavelength grid: {len(target_wave)} points "
        f"({target_wave[0]:.1f}-{target_wave[-1]:.1f} A)",
        flush=True,
    )
    index, frac = cache_interp_indices(np.asarray(wave, dtype=np.float64), target_wave.astype(np.float64))
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp.dat")
    flux_out = np.memmap(tmp_path, dtype=np.float32, mode="w+", shape=(len(rows), len(target_wave)))
    params = np.empty((len(rows), 3), dtype=np.float32)

    tasks = list(chunk_ranges(len(rows), args.cache_chunk_size))
    t0 = time.time()
    done = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=args.processes,
        initializer=init_cache_worker,
        initargs=(shape["flux_path"], shape, rows, index, frac, len(target_wave)),
    ) as pool:
        for start, param_block, flux_block in pool.imap_unordered(cache_chunk, tasks, chunksize=1):
            flux_out[start:start + len(flux_block)] = flux_block
            params[start:start + len(param_block)] = param_block
            done += len(flux_block)
            if done == len(rows) or done % args.progress_every < args.cache_chunk_size:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else math.nan
                print(f"Cached {done}/{len(rows)} spectra ({rate:.2f} spectra/s)", flush=True)

    flux_out.flush()
    flux_in = np.memmap(tmp_path, dtype=np.float32, mode="r", shape=(len(rows), len(target_wave)))
    print(f"Writing {out_path}", flush=True)
    spectral_cache.write_cache(
        out_path,
        {"teff": params[:, 0], "logg": params[:, 1], "feh": params[:, 2]},
        target_wave,
        flux_in,
        metadata={
            "grid": args.grid_name,
            "source": "NewEra HSR LSR",
            "alpha": 0.0,
            "wgrid": args.cache_wave_grid,
            "step": args.cache_wave_step,
            "log10st": args.cache_log10_step,
            "lstart": args.cache_long_wave_start,
            "lstep": args.cache_long_wave_step,
            "exclteff": ",".join(f"{value:g}" for value in args.exclude_teff),
            "excllogg": ",".join(f"{value:g}" for value in args.exclude_logg),
        },
        overwrite=True,
    )
    del flux_in
    del flux_out
    if not args.keep_tmp:
        tmp_path.unlink()
    return out_path


def update_grid_description(args):
    import yaml

    path = Path(args.output_dir) / "grid_description.yaml"
    if path.is_file():
        with path.open() as handle:
            desc = yaml.safe_load(handle) or {}
    else:
        desc = {}
    desc.setdefault(args.grid_name, {})
    desc[args.grid_name].update(
        {
            "filename": args.grid_name,
            "integrated_subdir": "integrated",
            "supports_feh": True,
            "spectral_cache": f"spectral_cache/{args.grid_name}_plot_spectra.fits",
            "info": "PHOENIX NewEra alpha=0 LSR grid",
        }
    )
    with path.open("w") as handle:
        yaml.safe_dump(desc, handle, sort_keys=False)
    print(f"Updated {path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lsr-dir", default=os.environ.get("NEWERA_LSR_DIR", "NewEra/lsr"))
    parser.add_argument("--output-dir", default="newera_lsr_products")
    parser.add_argument("--grid-name", default="newera_alpha0")
    parser.add_argument("--processes", type=int, default=40)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--cache-chunk-size", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--law", default="WC2019")
    parser.add_argument("--rv", type=float, default=3.1)
    parser.add_argument("--case1", type=int, default=1)
    parser.add_argument("--av-min", type=float, default=0.0)
    parser.add_argument("--av-max", type=float, default=6.2)
    parser.add_argument("--av-small-max", type=float, default=1.0)
    parser.add_argument("--av-small-step", type=float, default=0.005)
    parser.add_argument("--av-mid-max", type=float, default=3.0)
    parser.add_argument("--av-mid-step", type=float, default=0.02)
    parser.add_argument("--av-large-step", type=float, default=0.05)
    parser.add_argument("--responses", nargs="*", default=None)
    parser.add_argument("--cache-wave-grid", choices=("linear", "piecewise", "log"), default="log")
    parser.add_argument("--cache-wave-step", type=float, default=2.0)
    parser.add_argument("--cache-log10-step", type=float, default=0.00192)
    parser.add_argument("--cache-wave-min", type=float, default=1300.0)
    parser.add_argument("--cache-wave-max", type=float, default=286000.0)
    parser.add_argument("--cache-long-wave-start", type=float, default=None)
    parser.add_argument("--cache-long-wave-step", type=float, default=None)
    parser.add_argument("--exclude-teff", nargs="*", type=float,
                        default=list(DEFAULT_EXCLUDED_TEFF),
                        help="NewEra Teff values to exclude from the main alpha=0 grid.")
    parser.add_argument("--exclude-logg", nargs="*", type=float,
                        default=list(DEFAULT_EXCLUDED_LOGG),
                        help="Isolated NewEra logg values to exclude from the main alpha=0 grid.")
    parser.add_argument("--max-spectra", type=int, default=None)
    parser.add_argument("--skip-integrated", action="store_true")
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--update-grid-description", action="store_true")
    parser.add_argument("--keep-tmp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    shape = load_lsr_shape(args.lsr_dir)
    wave = np.load(shape["wavelength_path"], mmap_mode="r")
    rows = read_completed_rows(
        args.lsr_dir,
        max_spectra=args.max_spectra,
        exclude_teff=args.exclude_teff,
        exclude_logg=args.exclude_logg,
    )
    responses = select_responses(wave, args.responses)
    print(f"Loaded {len(rows)} completed spectra", flush=True)
    print(f"Using {len(responses)} response curves", flush=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if not args.skip_integrated:
        build_integrated(args, shape, wave, rows, responses)
    if not args.skip_cache:
        build_spectral_cache(args, shape, wave, rows)
    if args.update_grid_description:
        update_grid_description(args)


if __name__ == "__main__":
    main()
