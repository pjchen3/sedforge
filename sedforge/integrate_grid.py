import os
import sys
import time

import numpy as np

from astropy.io import fits
from astropy.table import Table

from multiprocessing import cpu_count, get_context

from sedforge import model, reddening, filters

_WORKER_AVS = None
_WORKER_RESPONSES = None
_WORKER_LAW = None
_WORKER_RV = None
_WORKER_CASE1 = None
_WORKER_WEIGHT_CACHE = None
_WORKER_REDDENING_CACHE = None


def get_responses(responses=None, wave=(0, np.inf)):
    """
    Get a list of response functions and their information.

    You can specify bandpass systems (e.g. GAIA3E) and then all matching
    filters will be collected. You can also specify a single passband
    (e.g. GAIA3E_G), in which case only that one will be returned.

    You can also set responses to C{None} and give a wavelength range for filter
    selection.

    Example input for C{responses} are:

    >>> responses = ['GAIA3E_G', '2MASS_Ks']
    >>> responses = ['GAIA3E', '2MASS_Ks']
    >>> responses = None

    @param responses: a list of filter systems of passbands
    @type responses: list of str
    @param wave: wavelength range
    @type wave: tuple (float,float)
    """

    auto_select = responses is None

    # -- if no responses are given, select using wavelength range
    if responses is None:
        responses = filters.list_response(wave_range=(wave[0], wave[-1]))
    else:
        responses_ = []
        for resp in responses:
            print('... subselection: {}'.format(resp))
            responses_ += filters.list_response(resp)
        responses = responses_
    # -- get information on the responses
    if auto_select:
        responses = [resp for resp in responses if not (
                ('ACS' in resp) or ('WFPC' in resp) or ('STIS' in resp) or ('ISOCAM' in resp) or ('NICMOS' in resp))]

    print('Selected response curves: {}'.format(', '.join(responses)))

    return responses


def get_threads(threads, max=np.inf):
    """
    Reads the threadcount, and returns an integer
    accepts: <integer>, 'max', 'half', 'safe'
       max: all cpus are used
       half: half of the cpus are used
       safe: all but one of the cpus are used
    """
    if threads == 'max':
        threads = cpu_count()
    elif threads == 'half':
        threads = cpu_count() / 2
    elif threads == 'safe':
        threads = cpu_count() - 1
    threads = int(threads)

    if threads > max:
        threads = max

    return threads

def _header_float(header, *names):
    for name in names:
        if name in header:
            return float(header[name])
    return None


def _member_matches_header(member, header):
    if 'he_mass' in member:
        he_mass = _header_float(header, 'HEMASS', 'HE_MASS', 'HE', 'YHE')
        return he_mass is not None and np.isclose(
            he_mass, float(member['he_mass']), rtol=0, atol=1e-6
        )
    return True


def _grid_members(grid):
    if isinstance(grid, dict) and 'members' in grid:
        return model._normalise_grid_members(grid)
    if (isinstance(grid, str) and grid in model.grid_description
            and 'members' in model.grid_description[grid]):
        return model._normalise_grid_members(model.grid_description[grid])
    if isinstance(grid, (list, tuple)):
        return model._normalise_grid_members({'members': grid})
    return model._normalise_grid_members(grid)


def _law_label(law, case1=1):
    if str(law).lower() == 'wc2019' and int(case1) != 1:
        return f"{law}_case{int(case1)}"
    return law


def _output_name(grid, first_gridfile, law, Rv, outfile=None, case1=1):
    law_label = _law_label(law, case1)
    if outfile is not None:
        return outfile
    if isinstance(grid, dict):
        for key in ('outfile', 'output', 'filename'):
            if key in grid:
                filename = grid[key]
                root, ext = os.path.splitext(filename)
                if ext:
                    return filename
                return f"i{filename}_law{law_label}_Rv{Rv:.2f}.fits"
    if isinstance(grid, str) and grid in model.grid_description:
        desc = model.grid_description[grid]
        filename = desc.get('filename')
        if filename:
            root, ext = os.path.splitext(filename)
            if ext:
                filename = root
            output = f"i{filename}_law{law_label}_Rv{Rv:.2f}.fits"
            subdir = desc.get('integrated_subdir')
            if subdir:
                directory = model.defaults.get('directory')
                if directory:
                    return os.path.join(directory, subdir, output)
                return os.path.join(subdir, output)
            return output
    outfile = 'i{0}'.format(os.path.basename(first_gridfile))
    outfile = os.path.splitext(outfile)
    return outfile[0] + '_law{0}_Rv{1:.2f}'.format(law_label, Rv) + outfile[1]


def _trapz_weights(x):
    x = np.asarray(x, dtype=float)
    weights = np.zeros_like(x)
    if len(x) == 1:
        return weights
    dx = np.diff(x)
    weights[0] = dx[0] / 2.0
    weights[-1] = dx[-1] / 2.0
    if len(x) > 2:
        weights[1:-1] = (x[2:] - x[:-2]) / 2.0
    return weights


def _add_interp_weights(source_wave, target_wave, target_weights, out):
    source_wave = np.asarray(source_wave, dtype=float)
    for wave_i, weight in zip(target_wave, target_weights):
        if weight == 0:
            continue
        pos = np.searchsorted(source_wave, wave_i, side='left')
        if pos < len(source_wave) and source_wave[pos] == wave_i:
            out[pos] += weight
        elif pos == 0:
            out[0] += weight
        elif pos >= len(source_wave):
            out[-1] += weight
        else:
            left = source_wave[pos - 1]
            right = source_wave[pos]
            frac = (wave_i - left) / (right - left)
            out[pos - 1] += weight * (1.0 - frac)
            out[pos] += weight * frac


def _response_weight_matrix(wave, responses):
    """
    Compile all filter integrations for one model wavelength grid.

    The returned matrix converts a flux sampled on ``wave`` directly into the
    Band-averaged Flambda values used by ``filters.synthetic_flux``.
    """
    wave = np.asarray(wave, dtype=float)
    matrix = np.zeros((len(wave), len(responses)), dtype=float)
    invalid = []

    for i, photband in enumerate(responses):
        waver, transr = filters.get_response(photband)
        waver = np.asarray(waver, dtype=float)
        transr = np.asarray(transr, dtype=float)
        region = ((waver[0] - 0.4 * waver[0]) <= wave) & (wave <= (2 * waver[-1]))
        source_index = np.where(region)[0]
        if not len(source_index):
            invalid.append(i)
            continue

        source_wave = wave[source_index]
        if (np.searchsorted(source_wave, waver[-1])
                - np.searchsorted(source_wave, waver[0])) < 5:
            int_wave = np.sort(np.hstack([source_wave, waver]))
        else:
            int_wave = source_wave

        trans = np.interp(int_wave, waver, transr, left=0, right=0)
        response_weight = filters.integration_weight(photband, int_wave)
        denom = np.trapz(trans * response_weight, x=int_wave)
        if denom <= 0 or not np.isfinite(denom):
            invalid.append(i)
            continue

        int_weights = _trapz_weights(int_wave) * trans * response_weight / denom
        if len(int_wave) == len(source_wave) and np.all(int_wave == source_wave):
            matrix[source_index, i] = int_weights
        else:
            source_weights = np.zeros(len(source_wave), dtype=float)
            _add_interp_weights(source_wave, int_wave, int_weights, source_weights)
            matrix[source_index, i] = source_weights

    return matrix, np.array(invalid, dtype=int)


def _wave_cache_key(wave):
    wave = np.asarray(wave, dtype=float)
    if not len(wave):
        return (0,)
    return (
        len(wave),
        float(wave[0]),
        float(wave[len(wave) // 2]),
        float(wave[-1]),
        float(np.sum(wave)),
        float(np.sum(wave * wave)),
    )


def default_av_grid(av_min=0.0, av_max=6.2,
                    small_max=1.0, small_step=0.005,
                    mid_max=3.0, mid_step=0.02,
                    large_step=0.05):
    """
    Return the default integrated-grid extinction axis in A(V).

    The grid is intentionally denser at low extinction, where most nearby
    targets live and where the posterior often changes quickly.
    """
    av_min = float(av_min)
    av_max = float(av_max)
    if av_max < av_min:
        raise ValueError("av_max must be >= av_min.")

    small_max = float(small_max)
    mid_max = float(mid_max)
    small_step = float(small_step)
    mid_step = float(mid_step)
    large_step = float(large_step)
    if min(small_step, mid_step, large_step) <= 0:
        raise ValueError("Av grid steps must be positive.")

    segments = []
    if av_min <= min(small_max, av_max):
        segments.append(
            np.arange(av_min, min(small_max, av_max) + 0.5 * small_step, small_step)
        )
    if av_max > small_max:
        start = max(av_min, small_max + mid_step)
        stop = min(mid_max, av_max)
        if start <= stop:
            segments.append(np.arange(start, stop + 0.5 * mid_step, mid_step))
    if av_max > mid_max:
        start = max(av_min, mid_max + large_step)
        if start <= av_max:
            segments.append(np.arange(start, av_max + 0.5 * large_step, large_step))

    avs = np.concatenate(segments) if segments else np.array([av_min])
    avs = avs[(avs >= av_min - 1e-10) & (avs <= av_max + 1e-10)]
    avs = np.unique(np.round(avs, 6))
    if av_min not in avs:
        avs = np.r_[av_min, avs]
    if av_max not in avs:
        avs = np.r_[avs, av_max]
    return np.unique(np.round(avs, 6))


def _reddening_transmission_grid(wave, avs, law, Rv, case1):
    _, redmag = reddening.get_law(law, wave=wave, norm='Av', Rv=Rv, case1=case1)
    return 10 ** (-0.4 * np.asarray(avs, dtype=float)[:, None] * redmag[None, :])


def _integrated_fluxes_fast(wave, flux, avs, responses, law, Rv, case1,
                            weight_cache, reddening_cache):
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    avs = np.asarray(avs, dtype=float)
    key = _wave_cache_key(wave)

    if key not in weight_cache:
        if len(weight_cache) > 8:
            weight_cache.clear()
        weight_cache[key] = _response_weight_matrix(wave, responses)
    weights, invalid = weight_cache[key]

    redkey = (key, str(law).lower(), float(Rv), int(case1), tuple(avs))
    if redkey not in reddening_cache:
        if len(reddening_cache) > 8:
            reddening_cache.clear()
        reddening_cache[redkey] = _reddening_transmission_grid(
            wave, avs, law, Rv, case1
        )

    fluxes = (reddening_cache[redkey] * flux[None, :]).dot(weights)
    if len(invalid):
        fluxes[:, invalid] = np.nan
    return np.column_stack((avs, fluxes))


def _init_integrated_worker(avs, responses, law, Rv, case1):
    global _WORKER_AVS, _WORKER_RESPONSES, _WORKER_LAW, _WORKER_RV, _WORKER_CASE1
    global _WORKER_WEIGHT_CACHE, _WORKER_REDDENING_CACHE
    _WORKER_AVS = np.asarray(avs, dtype=float)
    _WORKER_RESPONSES = list(responses)
    _WORKER_LAW = law
    _WORKER_RV = float(Rv)
    _WORKER_CASE1 = int(case1)
    _WORKER_WEIGHT_CACHE = {}
    _WORKER_REDDENING_CACHE = {}


def _integrate_hdu_worker(task):
    member, hdu_index, include_feh = task
    gridfile = model.get_grid_file(integrated=False, grid=member['grid'])
    with fits.open(gridfile, memmap=True) as ff:
        hdu = ff[hdu_index]
        teff = float(hdu.header['TEFF'])
        logg = float(hdu.header['LOGG'])
        feh = member.get('feh', _header_float(hdu.header, 'FEH', 'M_H', 'MH', 'Z'))
        if include_feh and feh is None:
            raise ValueError(
                f"Grid member {member['grid']} needs a feh value to "
                "be combined into a metallicity grid."
            )
        wave = hdu.data['wavelength']
        flux = hdu.data['flux']
        Labs = model.luminosity(wave, flux)
        arr = _integrated_fluxes_fast(
            wave,
            flux,
            _WORKER_AVS,
            _WORKER_RESPONSES,
            _WORKER_LAW,
            _WORKER_RV,
            _WORKER_CASE1,
            _WORKER_WEIGHT_CACHE,
            _WORKER_REDDENING_CACHE,
        )
    prefix = [teff, logg]
    if include_feh:
        prefix.append(float(feh))
    prefix.extend([Labs])
    block_prefix = np.repeat(np.asarray(prefix, dtype=float)[None, :], len(arr), axis=0)
    return np.column_stack((block_prefix, arr))


def calc_integrated_grid(threads=1, avs=None, ebvs=None, law='WC2019',
                         Rv=3.1, responses=None, grid=None, outfile=None,
                         case1=1):
    """
    Integrate an entire SED grid over all passbands and save to a FITS file.

    The output file can be used to fit SEDs more efficiently, since integration
    over the passbands has already been carried out.

    WARNING: this function can take a long time to compute!

    Extra keywords can be used to specify the grid.

    :param threads: number of threads
    :type threads; integer, 'max', 'half' or 'safe'
    :param avs: A(V) values to include on the extinction axis
    :type avs: numpy array
    :param law: interstellar reddening law to use
    :type law: string (valid law name, see C{reddening.py})
    :param Rv: Rv value for reddening law
    :type Rv: float
    :param case1: WC2019 branch to use
    :type case1: integer
    :param responses: respons curves to add (if None, add all)
    :type responses: list of strings
    """

    members = _grid_members(grid)
    first_gridfile = model.get_grid_file(integrated=False, grid=members[0]['grid'])

    # -- Use the first spectrum only to decide which response curves fit.
    with fits.open(first_gridfile) as first_ff:
        wave0 = first_ff[1].data['wavelength']
        responses = get_responses(responses=responses, wave=wave0)
        raw_feh = _header_float(first_ff[1].header, 'FEH', 'M_H', 'MH', 'Z')

    include_feh = any('feh' in member for member in members) or raw_feh is not None

    if avs is not None and ebvs is not None:
        raise ValueError("Use either avs or legacy ebvs, not both.")
    if avs is None:
        if ebvs is None:
            avs = default_av_grid()
        else:
            avs = np.asarray(ebvs, dtype=float) * float(Rv)
    rows = []
    avs = np.asarray(avs, dtype=float)
    weight_cache = {}
    reddening_cache = {}

    # also track possible errors
    exceptions = 0
    exceptions_logs = []

    tasks = []
    for member in members:
        gridfile = model.get_grid_file(integrated=False, grid=member['grid'])
        with fits.open(gridfile) as ff:
            for hdu_index, hdu in enumerate(ff[1:], start=1):
                if _member_matches_header(member, hdu.header):
                    tasks.append((member, hdu_index, include_feh))
    total_tables = len(tasks)
    print('Total number of tables: %i ' % total_tables)

    done = 0
    c0 = time.time()
    n_threads = get_threads(threads, max=max(1, total_tables))
    print('Using %i worker processes' % n_threads)
    if n_threads <= 1:
        for member, hdu_index, _include_feh in tasks:
            gridfile = model.get_grid_file(integrated=False, grid=member['grid'])
            with fits.open(gridfile) as ff:
                hdu = ff[hdu_index]
                teff = float(hdu.header['TEFF'])
                logg = float(hdu.header['LOGG'])
                feh = member.get('feh', _header_float(hdu.header, 'FEH', 'M_H', 'MH', 'Z'))
                if include_feh and feh is None:
                    raise ValueError(
                        f"Grid member {member['grid']} needs a feh value to "
                        "be combined into a metallicity grid."
                    )
                if done > 0:
                    et = (time.time() - c0) / done * (total_tables - done)
                    print('%s %s %s %s: ET %d seconds' % (teff, logg, done, total_tables, et))

                wave, flux = hdu.data['wavelength'], hdu.data['flux']
                Labs = model.luminosity(wave, flux)

                try:
                    arr = _integrated_fluxes_fast(
                        wave, flux, avs, responses, law, Rv, case1,
                        weight_cache, reddening_cache,
                    )
                    for row in arr:
                        prefix = [teff, logg]
                        if include_feh:
                            prefix.append(float(feh))
                        prefix.extend([Labs, row[0]])
                        rows.append(np.concatenate((np.array(prefix, dtype=float), row[1:])))
                except Exception:
                    print('Exception in calculating Teff=%f, logg=%f' % (teff, logg))
                    print('Exception: %s' % (sys.exc_info()[1]))
                    exceptions = exceptions + 1
                    exceptions_logs.append(sys.exc_info()[1])
                done += 1
    else:
        ctx = get_context("fork")
        with ctx.Pool(
            processes=n_threads,
            initializer=_init_integrated_worker,
            initargs=(avs, responses, law, Rv, case1),
        ) as pool:
            for block in pool.imap(_integrate_hdu_worker, tasks, chunksize=1):
                rows.extend(block)
                done += 1
                if done > 0:
                    et = (time.time() - c0) / done * (total_tables - done)
                    print('%s/%s: ET %d seconds' % (done, total_tables, et))

    # -- create Table object and add header info to the table
    column_names = ['teff', 'logg']
    if include_feh:
        column_names.append('feh')
    column_names.extend(['Labs', 'av'])
    if not rows:
        raise ValueError('No integrated grid rows were produced.')
    output = Table(data=np.array(rows), names=column_names + responses)
    output.meta['gridfile'] = (os.path.basename(first_gridfile), 'first original model file')
    output.meta['GRID'] = (str(grid), 'name of the model grid')
    output.meta['fluxtype'] = ('Flambda', 'units of the flux')
    output.meta['redlaw'] = (law, 'interstellar reddening law')
    output.meta['rv'] = (Rv, 'interstellar reddening parameter')
    output.meta['case1'] = (case1, 'WC2019 case1 branch')
    output.meta['extaxis'] = ('Av', 'integrated-grid extinction axis')

    # -- create the name of the output file and safe to disk
    outfile = _output_name(grid, first_gridfile, law, Rv, outfile=outfile,
                           case1=case1)
    outdir = os.path.dirname(outfile)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    output.write(outfile, overwrite=True)

    # # -- make FITS columns
    # output = output.T
    # cols = [fits.Column(name='teff', format='E', array=output[0]),
    #         fits.Column(name='logg', format='E', array=output[1]),
    #         fits.Column(name='av', format='E', array=output[3]),
    #         fits.Column(name='Labs', format='E', array=output[2])]
    # for i, photband in enumerate(responses):
    #     cols.append(fits.Column(name=photband, format='E', array=output[4 + i]))
    #
    # # -- make FITS extension and write grid/reddening specifications to header
    # table = fits.TableHDU.from_columns(fits.ColDefs(cols))
    # table.header.update(gridfile=os.path.basename(gridfile))
    # table.header.update(GRID=(grid, 'name of the model grid'),
    #                     FLUXTYPE=('Flambda', 'units of the flux'),
    #                     REDLAW=(law, 'interstellar reddening law'),
    #                     RV=(Rv, 'interstellar reddening parameter'))
    # # -- make/update complete FITS file
    # if os.path.isfile(outfile):
    #     os.remove(outfile)
    #     print('Removed existing file: %s' % (outfile))

    # hdulist = fits.HDUList([])
    # hdulist.append(fits.PrimaryHDU(np.array([[0, 0]])))
    # hdulist.append(table)
    # hdulist.writeto(outfile)
    print("Written output to %s" % outfile)

    print('Encountered %s exceptions!' % exceptions)
    for i in exceptions_logs:
        print('ERROR\n', i)

    return outfile


def check_grid(grid):
    """
    Check if the grid with integrated photometry calculated with calc_integrated_grid
    has all header information, and if all models succeeded.
    """
    print('Checking grid: {}'.format(grid))

    hdulist = fits.open(grid, mode='update')
    names = hdulist[1].columns.names
    for i, name in enumerate(names):
        if name.lower() in ['teff', 'logg', 'av', 'ebv', 'feh', 'labs', 'vrad', 'rv', 'z']:
            names[i] = name.lower()
    cols = [fits.Column(name=name, format='E', array=hdulist[1].data.field(name)) for name in names]
    N = len(hdulist[1].data)

    keys = [key.lower() for key in hdulist[1].header.keys()]

    if 'z' not in hdulist[1].header:
        hdulist[1].header['Z'] = 0.0
        print('Adding metallicity (Z={}) to header!'.format(hdulist[1].header['Z']))

    if 'Rv' not in hdulist[1].header:
        hdulist[1].header['Rv'] = 3.1
        print('Adding Rv (Rv={}) to header!'.format(hdulist[1].header['Rv']))

    if 'feh' not in names and 'z' not in names:
        z = hdulist[1].header.get('z', 0.0)
        print('Adding metallicity from header {}'.format(z))
        cols.append(fits.Column(name='feh', format='E', array=np.ones(N) * z))
    else:
        print("Metallicity already in there")
    if 'vrad' not in names:
        vrad = 0.
        print('Adding radial velocity {}'.format(vrad))
        cols.append(fits.Column(name='vrad', format='E', array=np.ones(N) * vrad))
    else:
        print("Radial velocity already in there")

    fix_rv = False
    if 'rv' not in names:
        if 'rv' in keys:
            rv = hdulist[1].header['Rv']
            print("Adding interstellar Rv from header {}".format(rv))
        else:
            rv = 3.1
            print("Adding default interstellar Rv {}".format(rv))
        cols.append(fits.Column(name='rv', format='E', array=np.ones(N) * rv))
    elif not hdulist[1].header['Rv'] == hdulist[1].data.field('rv')[0]:
        rv = hdulist[1].header['Rv']
        fix_rv = rv
        print('Correcting interstellar Rv with {}'.format(rv))
    else:
        print("Interstellar Rv already in there")

    table = fits.TableHDU.from_columns(fits.ColDefs(cols))
    if fix_rv:
        table.data.field('rv')[:] = rv
    fake_keys = [key.lower() for key in table.header.keys()]

    # make sure all keywords set above are also in the new table header
    table.header.update(hdulist[1].header)

    # now update header keywords related to the created table
    for key in hdulist[1].header.keys():
        if key.lower() not in fake_keys:
            if len(key) > 8:
                key = 'HIERARCH ' + key
            table.header.update(key=hdulist[1].header[key])
    hdulist[1] = table

    logstring = "Axis:\n"
    for name in hdulist[1].columns.names:
        if name.islower() and not name == 'labs':
            ax = np.unique(hdulist[1].data.field(name))
            logstring += "\t\t{} {} {} {}\n".format(name, len(ax), min(ax), max(ax))
    print(logstring)

    keep = hdulist[1].data.field('teff') > 0
    print('Removing {}/{} false entries'.format(sum(~keep), len(keep)))
    hdulist[1].data = hdulist[1].data[keep]
    hdulist.flush()
    hdulist.close()
