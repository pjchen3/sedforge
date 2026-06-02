#!/usr/bin/env python
"""
Build an integrated SED grid from PHOENIX NewEra V3 LowRes spectra.

The NewEra low-resolution release stores many spectra in Gaia DR4 text format:
one header record followed by one flux record. This script streams the spectra
without modifying the original NewEra download directory and writes only the
alpha=0 integrated photometry grid.
"""

import argparse
import os
import re
import tarfile
import time
from pathlib import Path

import numpy as np
import yaml
from astropy.table import Table

from sedforge import filters, integrate_grid, model


FLUX_W_M2_NM_TO_CGS_A = 100.0


def _rv_label(rv):
    return f"{float(rv):0.2f}"


def _law_label(law, case1=1):
    if str(law).lower() == 'wc2019' and int(case1) != 1:
        return f"{law}_case{int(case1)}"
    return law


def default_output_name(grid_name, law, rv, case1):
    return os.path.join(
        "integrated",
        f"i{grid_name}_law{_law_label(law, case1)}_Rv{_rv_label(rv)}.fits",
    )


def parse_mh_from_name(name):
    match = re.search(r"SPECTRA\.Z([+-]\d+(?:\.\d+)?)", str(name))
    if match is None:
        raise ValueError(f"Could not parse metallicity from {name}")
    value = float(match.group(1))
    if np.isclose(value, -0.0):
        value = 0.0
    return value


def is_alpha0_member(name):
    base = os.path.basename(str(name))
    return (
        base.startswith("PHOENIX-NewEraV3")
        and "LowRes-SPECTRA.Z" in base
        and base.endswith(".txt")
        and ".alpha=" not in base
    )


def parse_header(line, source_name):
    parts = line.strip().split()
    if len(parts) < 31 or parts[0] != "star":
        raise ValueError(f"Malformed NewEra header in {source_name}: {line[:120]}")
    return {
        "ntot": int(parts[8]),
        "lambda0_nm": float(parts[9]),
        "lambda_end_nm": float(parts[10]),
        "dlambda_nm": float(parts[11]),
        "teff": float(parts[12]),
        "logg": float(parts[13]),
        "alpha": float(parts[19]),
        "mass": float(parts[27]),
    }


def _is_eof(line):
    return line == b"" or line == ""


def _as_text(line):
    if isinstance(line, bytes):
        return line.decode("ascii")
    return line


def wavelength_from_header(header):
    wave_nm = header["lambda0_nm"] + np.arange(header["ntot"]) * header["dlambda_nm"]
    return wave_nm * 10.0


def read_flux_record(handle, ntot, source_name):
    chunks = []
    size = 0
    while size < ntot:
        line = handle.readline()
        if _is_eof(line):
            raise EOFError(f"Unexpected EOF while reading flux record from {source_name}")
        values = np.fromstring(_as_text(line), sep=" ", dtype=np.float64)
        if len(values) == 0:
            continue
        chunks.append(values)
        size += len(values)
    flux = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
    if len(flux) != ntot:
        raise ValueError(
            f"Flux length mismatch in {source_name}: expected {ntot}, got {len(flux)}"
        )
    return flux * FLUX_W_M2_NM_TO_CGS_A


def iter_gaia_text_spectra(fileobj, source_name, mh):
    while True:
        line = fileobj.readline()
        if _is_eof(line):
            break
        line = _as_text(line)
        if not line.strip():
            continue
        header = parse_header(line, source_name)
        flux = read_flux_record(fileobj, header["ntot"], source_name)
        if not np.isclose(header["alpha"], 0.0, atol=1e-8):
            continue
        yield header, mh, flux


def iter_alpha0_spectra(newera_dir):
    newera_dir = Path(newera_dir)
    tar_path = newera_dir / "PHOENIX-NewEraV3-LowRes-SPECTRA.tar.gz"
    add_path = newera_dir / "PHOENIX-NewEraV3-add001-LowRes-SPECTRA.Z+0.5.txt"

    if not tar_path.is_file():
        raise FileNotFoundError(tar_path)

    with tarfile.open(tar_path, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile() or not is_alpha0_member(member.name):
                continue
            mh = parse_mh_from_name(member.name)
            print(f"Reading {member.name} ([M/H]={mh:+.1f})", flush=True)
            fileobj = tar.extractfile(member)
            if fileobj is None:
                continue
            yield from iter_gaia_text_spectra(fileobj, member.name, mh)

    if add_path.is_file():
        mh = parse_mh_from_name(add_path.name)
        print(f"Reading {add_path} ([M/H]={mh:+.1f})", flush=True)
        with add_path.open("rb") as fileobj:
            yield from iter_gaia_text_spectra(fileobj, add_path.name, mh)


def expected_alpha0_count(newera_dir):
    coverage = Path(newera_dir) / "newera_v3_lowres_coverage_by_mh_alpha.tsv"
    if not coverage.is_file():
        return None
    total = 0
    with coverage.open() as handle:
        header = handle.readline().strip().split("\t")
        idx_alpha = header.index("alpha")
        idx_n = header.index("n_spectra")
        for line in handle:
            parts = line.strip().split("\t")
            if not parts:
                continue
            if np.isclose(float(parts[idx_alpha]), 0.0, atol=1e-8):
                total += int(parts[idx_n])
    return total or None


def throughput_coverage(photband, wave_min, wave_max):
    wave, trans = filters.get_response(photband)
    denom = np.trapz(trans * wave, x=wave)
    if denom <= 0 or not np.isfinite(denom):
        return 0.0
    inside = (wave_min <= wave) & (wave <= wave_max)
    if np.count_nonzero(inside) < 2:
        return 0.0
    return np.trapz(trans[inside] * wave[inside], x=wave[inside]) / denom


def select_responses(wave, requested=None, min_coverage=0.99):
    if requested:
        responses = []
        for item in requested:
            responses.extend(filters.list_response(item))
        responses = list(dict.fromkeys(responses))
    else:
        responses = filters.list_response(wave_range=(wave[0], wave[-1]))

    kept = []
    dropped = []
    for response in responses:
        coverage = throughput_coverage(response, wave[0], wave[-1])
        if coverage >= min_coverage:
            kept.append(response)
        else:
            dropped.append((response, coverage))

    print(f"Selected {len(kept)} response curves", flush=True)
    if dropped:
        print("Dropped response curves with incomplete wavelength coverage:", flush=True)
        for name, coverage in dropped:
            print(f"  {name}: coverage={coverage:.3f}", flush=True)
    print(", ".join(kept), flush=True)
    return kept


def allocate_output(expected_spectra, avs, n_responses):
    if expected_spectra is None:
        return None
    n_rows = int(expected_spectra) * len(avs)
    n_cols = 5 + n_responses
    return np.empty((n_rows, n_cols), dtype=np.float32)


def ensure_capacity(data, used_rows, add_rows, n_cols):
    if data is None:
        n_rows = max(4096, add_rows * 16)
        return np.empty((n_rows, n_cols), dtype=np.float32)
    if used_rows + add_rows <= len(data):
        return data
    new_rows = max(used_rows + add_rows, int(len(data) * 1.5))
    new_data = np.empty((new_rows, n_cols), dtype=np.float32)
    new_data[:used_rows] = data[:used_rows]
    return new_data


def update_grid_description(model_dir, grid_name):
    path = Path(model_dir) / "grid_description.yaml"
    if path.is_file():
        with path.open() as handle:
            desc = yaml.safe_load(handle) or {}
    else:
        desc = {}
    desc.setdefault(grid_name, {})
    desc[grid_name].update({
        "filename": grid_name,
        "integrated_subdir": "integrated",
        "supports_feh": True,
        "info": "PHOENIX NewEra V3 LowRes alpha=0 integrated grid",
    })
    with path.open("w") as handle:
        yaml.safe_dump(desc, handle, sort_keys=False)


def build(args):
    avs = integrate_grid.default_av_grid(
        av_min=args.av_min,
        av_max=args.av_max,
        small_max=args.av_small_max,
        small_step=args.av_small_step,
        mid_max=args.av_mid_max,
        mid_step=args.av_mid_step,
        large_step=args.av_large_step,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / (
        args.outfile or default_output_name(args.grid_name, args.law, args.rv, args.case1)
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)

    expected = expected_alpha0_count(args.newera_dir)
    if args.max_spectra is not None:
        expected = min(expected or args.max_spectra, args.max_spectra)

    data = None
    row0 = 0
    n_spectra = 0
    responses = None
    weight_cache = {}
    reddening_cache = {}
    seen = set()
    t0 = time.time()

    for header, mh, flux in iter_alpha0_spectra(args.newera_dir):
        key = (
            round(header["teff"], 6),
            round(header["logg"], 6),
            round(mh, 6),
        )
        if key in seen:
            continue
        seen.add(key)

        wave = wavelength_from_header(header)
        if responses is None:
            responses = select_responses(
                wave,
                requested=args.responses,
                min_coverage=args.min_filter_coverage,
            )
            data = allocate_output(expected, avs, len(responses))

        fluxes = integrate_grid._integrated_fluxes_fast(
            wave,
            flux,
            avs,
            responses,
            args.law,
            args.rv,
            args.case1,
            weight_cache,
            reddening_cache,
        )
        labs = model.luminosity(wave, flux)

        n_rows = len(avs)
        n_cols = 5 + len(responses)
        data = ensure_capacity(data, row0, n_rows, n_cols)
        block = data[row0:row0 + n_rows]
        block[:, 0] = header["teff"]
        block[:, 1] = header["logg"]
        block[:, 2] = mh
        block[:, 3] = labs
        block[:, 4] = fluxes[:, 0]
        block[:, 5:] = fluxes[:, 1:]
        row0 += n_rows
        n_spectra += 1

        if n_spectra % args.progress_every == 0:
            elapsed = time.time() - t0
            rate = n_spectra / elapsed if elapsed > 0 else np.nan
            print(
                f"Integrated {n_spectra} spectra, {row0} rows, "
                f"{rate:.2f} spectra/s",
                flush=True,
            )

        if args.max_spectra is not None and n_spectra >= args.max_spectra:
            break

    if responses is None or row0 == 0:
        raise RuntimeError("No alpha=0 NewEra spectra were integrated.")

    column_names = ["teff", "logg", "feh", "Labs", "av"] + responses
    table = Table(data=data[:row0], names=column_names)
    table.meta["GRID"] = (args.grid_name, "name of the model grid")
    table.meta["SOURCE"] = ("PHOENIX NewEra V3 LowRes", "source model grid")
    table.meta["ALPHA"] = (0.0, "alpha enhancement")
    table.meta["FLUXTYPE"] = ("Flambda", "units of the flux")
    table.meta["FLUXUNIT"] = ("erg/s/cm2/Angstrom", "converted from W/m2/nm")
    table.meta["REDLAW"] = (args.law, "interstellar reddening law")
    table.meta["RV"] = (args.rv, "interstellar reddening parameter")
    table.meta["CASE1"] = (args.case1, "WC2019 case1 branch")
    table.meta["EXTAXIS"] = ("Av", "integrated-grid extinction axis")
    table.meta["NSPEC"] = (n_spectra, "number of spectra integrated")

    print(f"Writing {outfile}", flush=True)
    table.write(outfile, overwrite=True)
    print(f"Wrote {outfile} with {row0} rows and {len(column_names)} columns", flush=True)

    if args.update_grid_description:
        update_grid_description(args.output_dir, args.grid_name)
        print(f"Updated {Path(args.output_dir) / 'grid_description.yaml'}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--newera-dir", default=os.environ.get("NEWERA_DIR", "NewEra"))
    parser.add_argument("--output-dir", default=os.environ.get("SEDFORGE_MODELS", "sed_models"))
    parser.add_argument("--grid-name", default="newera_alpha0")
    parser.add_argument("--outfile", default=None)
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
    parser.add_argument("--min-filter-coverage", type=float, default=0.99)
    parser.add_argument("--responses", nargs="*", default=None)
    parser.add_argument("--max-spectra", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--update-grid-description", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
