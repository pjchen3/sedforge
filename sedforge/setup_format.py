"""Helpers for writing human-readable SED fitting setup YAML files."""

import re


def _format_scalar(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if value is None:
        return 'null'
    return str(value)


def _format_flow(value):
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(_format_flow(item) for item in value) + ']'
    return _format_scalar(value)


def _add_mapping(lines, key, value, indent=0):
    pad = ' ' * indent
    if isinstance(value, dict):
        if not value:
            lines.append(f'{pad}{key}: {{}}')
            return
        lines.append(f'{pad}{key}:')
        for child_key, child_value in value.items():
            _add_mapping(lines, child_key, child_value, indent + 2)
    elif isinstance(value, (list, tuple)):
        lines.append(f'{pad}{key}: {_format_flow(value)}')
    else:
        lines.append(f'{pad}{key}: {_format_scalar(value)}')


def _add_section(lines, title):
    if lines:
        lines.append('')
    lines.append('# ' + '=' * 72)
    lines.append(f'# {title}')
    lines.append('# ' + '=' * 72)


def _plot_keys(setup):
    def index(key):
        match = re.fullmatch(r'plot(\d+)', key)
        return int(match.group(1)) if match else 999

    return sorted(
        (key for key in setup if re.fullmatch(r'plot\d+', key)),
        key=index,
    )


def setup_to_readable_yaml(setup):
    """
    Convert an SED fitting setup dictionary to a commented YAML string.

    The comments are intentionally kept outside the data model, so the result
    can still be read with yaml.safe_load.
    """
    lines = [
        '# SED fitting setup file',
        '# Photometry format: photband mag mag_err system',
        '# Optional photometry columns: mag_type mag_zp_offset',
        '# Magnitudes are converted internally to band-averaged Flambda.',
        '# Extra flux columns are checks and are ignored when mag columns exist.',
    ]

    _add_section(lines, 'Target And Photometry')
    for key in ('objectname', 'photometryfile', 'photband_include', 'photband_exclude'):
        if key in setup:
            _add_mapping(lines, key, setup[key])

    _add_section(lines, 'Model Grids And Extinction')
    for key in ('grids', 'grid_variables', 'reddening_law', 'reddening_Rv', 'reddening_case1'):
        if key in setup:
            _add_mapping(lines, key, setup[key])

    _add_section(lines, 'Fitted Parameters')
    if 'pnames' in setup:
        _add_mapping(lines, 'pnames', setup['pnames'])
    if 'limits' in setup:
        lines.append('limits:')
        pnames = setup.get('pnames', [])
        for i, limit in enumerate(setup['limits']):
            comment = f'  # {pnames[i]}' if i < len(pnames) else ''
            lines.append(f'  - {_format_flow(limit)}{comment}')

    _add_section(lines, 'Fixed Parameters')
    lines.append('# fixed parameters are held constant and are not sampled by MCMC')
    lines.append('# use feh for a shared binary metallicity, or feh/feh2 for separate values')
    _add_mapping(lines, 'fixed', setup.get('fixed', {}))

    _add_section(lines, 'Priors')
    lines.append('# Gaussian priors on sampled parameters only: [value, sigma] or [value, -sigma, +sigma]')
    _add_mapping(lines, 'priors', setup.get('priors', {}))

    if 'error_model' in setup:
        _add_section(lines, 'Error Model')
        lines.append('# optional group-level fractional jitter by filter system')
        _add_mapping(lines, 'error_model', setup['error_model'])

    _add_section(lines, 'MCMC Sampler')
    for key in ('nwalkers', 'nsteps', 'nrelax', 'a', 'percentiles'):
        if key in setup:
            _add_mapping(lines, key, setup[key])

    _add_section(lines, 'Outputs')
    for key in ('resultfile', 'datafile'):
        if key in setup:
            _add_mapping(lines, key, setup[key])
    for key in _plot_keys(setup):
        _add_mapping(lines, key, setup[key])

    return '\n'.join(lines) + '\n'
