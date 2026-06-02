#!/usr/bin/env python
"""Refresh local filter response curves from the SVO FPS service."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import time

import numpy as np
import requests
from astropy.io import ascii
from astropy.io.votable import parse

from sedforge._compat import trapezoid

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "sedforge"
MAP_PATH = PACKAGE_DIR / "filter_svo_map.dat"
CURVE_DIR = PACKAGE_DIR / "transmission_curves"
FILTER_INFO = PACKAGE_DIR / "filter_info.dat"
VEGA_PATH = PACKAGE_DIR / "vega.dat"
SVO_URL = "http://svo2.cab.inta-csic.es/theory/fps/fps.php"


def _response_type(detector_type):
    detector_type = str(detector_type).strip()
    if detector_type == "1":
        return "photon"
    if detector_type == "0":
        return "energy"
    raise ValueError(f"Unknown SVO DetectorType={detector_type!r}")


def _load_vega():
    table = np.loadtxt(VEGA_PATH, comments="#")
    return table[:, 0], table[:, 1]


def _download_svo_curve(svo_id):
    response = requests.get(SVO_URL, params={"ID": svo_id}, timeout=60)
    response.raise_for_status()
    votable = parse(BytesIO(response.content))
    table = votable.resources[0].tables[0]
    params = {param.name: param.value for param in table.params}
    data = table.to_table()
    wave = np.asarray(data["Wavelength"], dtype=float)
    trans = np.asarray(data["Transmission"], dtype=float)
    good = np.isfinite(wave) & np.isfinite(trans)
    wave = wave[good]
    trans = trans[good]
    order = np.argsort(wave)
    return wave[order], trans[order], params


def _integration_weight(response_type, wave):
    return wave if response_type == "photon" else np.ones_like(wave)


def _eff_wave(wave, trans, response_type, vega_wave, vega_flux):
    vega = np.interp(wave, vega_wave, vega_flux, left=np.nan, right=np.nan)
    valid = np.isfinite(vega) & np.isfinite(trans) & (trans > 0)
    if np.any(valid):
        wave = wave[valid]
        trans = trans[valid]
        vega = vega[valid]
        weight = vega * trans * _integration_weight(response_type, wave)
    else:
        weight = trans * _integration_weight(response_type, wave)
    denom = trapezoid(weight, x=wave)
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    return trapezoid(wave * weight, x=wave) / denom


def _bandwidth(wave, trans):
    max_trans = np.nanmax(trans)
    if max_trans <= 0 or not np.isfinite(max_trans):
        return np.nan
    return trapezoid(trans, x=wave) / max_trans


def _write_curve(path, photband, svo_id, response_type, params, wave, trans):
    detector_type = str(params.get("DetectorType", "")).strip()
    components = params.get("components", "")
    reference = params.get("ProfileReference", "")
    with path.open("w") as handle:
        handle.write(f"# photband: {photband}\n")
        handle.write("# source: SVO Filter Profile Service\n")
        handle.write(f"# svo_id: {svo_id}\n")
        handle.write(f"# detector_type: {detector_type}\n")
        handle.write(f"# response_type: {response_type}\n")
        handle.write(f"# components: {components}\n")
        handle.write(f"# profile_reference: {reference}\n")
        handle.write("# wavelength_unit: Angstrom\n")
        handle.write("# columns: wavelength transmission\n")
        np.savetxt(handle, np.column_stack([wave, trans]), fmt="%.10e")


def main():
    mapping = ascii.read(MAP_PATH, comment="#")
    vega_wave, vega_flux = _load_vega()
    rows = []

    for row in mapping:
        photband = str(row["photband"])
        svo_id = str(row["svo_id"])
        wave, trans, params = _download_svo_curve(svo_id)
        response_type = _response_type(params.get("DetectorType"))
        _write_curve(
            CURVE_DIR / f"{photband}.dat",
            photband,
            svo_id,
            response_type,
            params,
            wave,
            trans,
        )
        rows.append(
            (
                photband,
                _eff_wave(wave, trans, response_type, vega_wave, vega_flux),
                _bandwidth(wave, trans),
                response_type,
                svo_id,
            )
        )
        print(f"{photband:22s} {svo_id:28s} {response_type}")
        time.sleep(0.05)

    rows.sort(key=lambda item: item[0])
    with FILTER_INFO.open("w") as handle:
        handle.write("# Filter metadata generated from SVO curves and vega.dat\n")
        handle.write("# eff_wave: Vega-weighted effective wavelength in Angstrom, using response_type-specific integration weights\n")
        handle.write("# bandwidth: rectangular throughput width, int(T dlambda) / max(T), Angstrom\n")
        handle.write("# response_type: convention of the local response curve; photon uses an extra lambda weight, energy does not\n")
        handle.write("# svo_id: SVO Filter Profile Service identifier used for the local response curve\n")
        handle.write("photband eff_wave bandwidth response_type svo_id\n")
        for photband, eff_wave, bandwidth, response_type, svo_id in rows:
            handle.write(
                f"{photband} {eff_wave:.8f} {bandwidth:.8f} "
                f"{response_type} {svo_id}\n"
            )


if __name__ == "__main__":
    main()
