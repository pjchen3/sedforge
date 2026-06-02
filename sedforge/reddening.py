import numpy as np
from scipy.interpolate import interp1d

from sedforge import filters, model

#{ Main interface

def get_law(name,norm='E(B-V)',photbands=None,**kwargs):
    """
    Retrieve an interstellar reddening law.
    
    Parameter C{name} must be the function name of one of the laws defined in
    this module.
    
    By default, the law will be interpolated on a grid from 100 angstrom to
    10 micron in steps of 10 angstrom. This can be adjusted with the parameter
    C{wave} (array), which B{must} be in angstrom. You can change the units
    ouf the returned wavelength array via C{wave_units}.
    
    By default, the curve is normalised with respect to E(B-V) (you get
    A(l)/E(B-V)). You can set the C{norm} keyword to Av if you want A(l)/Av.
    Remember that
    
    A(V) = Rv * E(B-V)
    
    The parameter C{Rv} is by default 3.1, other reasonable values lie between
    2.0 and 5.1
    
    Extra accepted keywords depend on the type of reddening law used.
    
    Example usage:
    
    >>> wave = np.r_[1e3:1e5:10]
    >>> wave, mag = get_law('WC2019', wave=wave, Rv=3.1)
    
    @param name: name of the interstellar law
    @type name: str, one of the functions defined here
    @param norm: type of normalisation of the curve
    @type norm: str (one of E(B-V), Av)
    @param photbands: list of photometric passbands
    @type photbands: list of strings
    @keyword wave: wavelength array to interpolate the law on
    @type wave: ndarray
    @return: wavelength, reddening magnitude
    @rtype: (ndarray,ndarray)
    """
    #-- get the inputs
    wave_ = kwargs.pop('wave',None)
    Rv = kwargs.setdefault('Rv',3.1)
    
    #-- get the curve
    key = str(name).lower()
    if key not in {'wc2019', 'rvcodeg'}:
        raise ValueError("Only the WC2019 extinction law is supported.")
    wave,mag = globals()[key](**kwargs)
    
    #-- interpolate on user defined grid
    if wave_ is not None:
        mag = np.interp(wave_,wave,mag,right=0)
        wave = wave_
           
    #-- pick right normalisation: convert to A(lambda)/Av if needed
    if norm.lower()=='e(b-v)':
        mag *= Rv
    elif norm.lower() == 'av':
        pass
    else:
        raise ValueError("Only E(B-V) and Av reddening normalisations are supported.")
    
    #-- maybe we want the curve in photometric filters
    if photbands is not None:
        mag = model.synthetic_flux(wave, mag, photbands)
        wave = filters.get_info(photbands)['eff_wave']
    
    return wave,mag


def redden(flux,wave=None,photbands=None,ebv=None,av=None,rtype='flux',law='WC2019',**kwargs):
    """
    Redden fluxes.
    
    The preferred reddening parameter C{av} means A(V).  The legacy C{ebv}
    parameter means E(B-V).
    
    If it is negative, we B{deredden}.
    
    If you give the keyword C{wave}, it is assumed that you want to (de)redden
    a B{model}, i.e. a spectral energy distribution.
    
    If you give the keyword C{photbands}, it is assumed that you want to (de)redden
    B{photometry}, i.e. integrated fluxes.
    
    @param flux: fluxes to (de)redden
    @type flux: ndarray (floats)
    @param wave: wavelengths matching the fluxes (or give C{photbands})
    @type wave: ndarray (floats)
    @param photbands: photometry bands matching the fluxes (or give C{wave})
    @type photbands: ndarray of str
    @param ebv: reddening parameter E(B-V)
    @type ebv: float
    @param av: reddening parameter A(V)
    @type av: float
    @param rtype: retained for compatibility; only 'flux' is supported
    @type rtype: str
    @return: (de)reddened flux
    @rtype: ndarray (floats)
    """
    if photbands is not None:
        wave = filters.get_info(photbands)['eff_wave']
        
    old_settings =  np.seterr(all='ignore')
    if av is not None and ebv is not None:
        np.seterr(**old_settings)
        raise ValueError("Use either av or ebv for reddening, not both.")
    if av is not None:
        norm = 'Av'
        reddening_value = av
    else:
        norm = 'E(B-V)'
        reddening_value = 0.0 if ebv is None else ebv

    wave, reddeningMagnitude = get_law(law,wave=wave,norm=norm,**kwargs)

    if rtype != 'flux':
        np.seterr(**old_settings)
        raise ValueError("Only flux reddening is supported.")

    flux_reddened = flux / 10**(reddeningMagnitude*reddening_value/2.5)
    np.seterr(**old_settings)
    return flux_reddened

def deredden(flux,wave=None,photbands=None,ebv=None,av=None,rtype='flux',**kwargs):
    """
    Deredden fluxes.
    
    @param flux: fluxes to (de)redden
    @type flux: ndarray (floats)
    @param wave: wavelengths matching the fluxes (or give C{photbands})
    @type wave: ndarray (floats)
    @param photbands: photometry bands matching the fluxes (or give C{wave})
    @type photbands: ndarray of str
    @param ebv: reddening parameter E(B-V)
    @type ebv: float
    @param rtype: retained for compatibility; only 'flux' is supported
    @type rtype: str
    @return: (de)reddened flux
    @rtype: ndarray (floats)
    """
    if av is not None and ebv is not None:
        raise ValueError("Use either av or ebv for reddening, not both.")
    if av is not None:
        return redden(flux,wave=wave,photbands=photbands,av=-av,rtype=rtype,**kwargs)
    ebv = 0.0 if ebv is None else ebv
    return redden(flux,wave=wave,photbands=photbands,ebv=-ebv,rtype=rtype,**kwargs)
    

#}

#{ Curve definitions
def wc2019(Rv=3.1, num=10000, case1=1, **kwargs):
    """
    WC2019 Rv-dependent extinction curve.

    Returns wavelengths in Angstrom and A(lambda)/Av, matching the convention
    used by the other reddening laws in this module. ``case1`` keeps the same
    meaning as in the supplied implementation.
    """
    if num < 2:
        raise ValueError('num must be at least 2')

    step = (np.log10(30) - np.log10(0.1)) / (num - 1)
    ccc = np.log10(0.1) + np.arange(num) * step
    wave = 10 ** ccc

    X = np.arange(1.1, 3.3, 0.0001)
    X1 = np.arange(0.33, 1.00, 0.0001)
    X2 = np.arange(3.3, 8.0, 0.0001)
    X3 = np.arange(0.02, 0.3, 0.0001)
    X4 = np.arange(8.0, 10.0, 0.0001)
    X5 = np.arange(10.0, 15.0, 0.0001)

    cX = np.arange(0.63, 2.2, 0.0001)
    cX1 = np.arange(0.12, 0.63, 0.0001)

    Y = X - 1.82

    if case1 == 2:
        A = 1 + 0.17699 * Y - 0.50447 * Y**2 - 0.02427 * Y**3 + 0.72085 * Y**4 \
            + 0.01979 * Y**5 - 0.77530 * Y**6 + 0.32999 * Y**7
        B = 1.41338 * Y + 2.28305 * Y**2 + 1.07233 * Y**3 - 5.38434 * Y**4 \
            - 0.62251 * Y**5 + 5.30260 * Y**6 - 2.09002 * Y**7
    elif case1 == 3:
        A = 1
        B = 2.659 * (-1.857 + 1.040 / cX)
    else:
        A = 1 + 0.7499 * Y - 0.1086 * Y**2 - 0.08909 * Y**3 + 0.02905 * Y**4 \
            + 0.01069 * Y**5 + 0.001707 * Y**6 - 0.001002 * Y**7
        B = (1.41338 * Y + 2.28305 * Y**2 + 1.07233 * Y**3 - 5.38434 * Y**4 \
             - 0.62251 * Y**5 + 5.30260 * Y**6 - 2.09002 * Y**7) * (1 - Rv / 3.1)

    result = A + B / Rv

    if case1 == 2:
        Y1 = X1**1.61
        A1 = 0.574 * Y1
        B1 = -0.527 * Y1
    elif case1 == 3:
        A1 = 1
        B1 = 2.659 * (-2.156 + 1.509 / cX1 - 0.198 / cX1**2 + 0.011 / cX1**3)
    else:
        Y1 = X1**2.07
        A1 = 0.3722 * Y1
        B1 = -0.5182 * Y1 * (1 - Rv / 3.1)

    result1 = A1 + B1 / Rv

    A2 = 1.752 - 0.316 * X2 - 0.104 / ((X2 - 4.67)**2 + 0.341)
    B2 = -3.090 + 1.825 * X2 + 1.206 / ((X2 - 4.62)**2 + 0.263)
    sel2 = X2 > 5.9
    A2[sel2] -= 0.04473 * (X2[sel2] - 5.9)**2 - 0.009779 * (X2[sel2] - 5.9)**3
    B2[sel2] += 0.2130 * (X2[sel2] - 5.9)**2 + 0.1207 * (X2[sel2] - 5.9)**3

    result2 = A2 + B2 / Rv

    if case1 == 2:
        Y3 = (X3 / 0.3)**2 * (0.3**1.61)
        A3 = 0.574 * Y3
        B3 = -0.527 * Y3
    elif case1 == 3:
        A3 = 0
        B3 = 0
    else:
        Y3 = (X3 / 0.3)**2 * (0.3**2.07)
        A3 = 0.3722 * Y3
        B3 = -0.5182 * Y3 * (1 - Rv / 3.1)

    result3i = A3 + B3 / Rv

    DIB = np.array([9.7e4, 2.5e4, 6.9e3])
    DIB1 = np.array([18.e4, 5.0e4, 6.0e3])
    WAV = 1.e4 / X3
    FAC = (0.574 - 0.527 / Rv) / (0.574 - 0.527 / 3.1)

    result3 = result3i + 0.63622 * FAC * (DIB[2] / Rv) * DIB[1] / (DIB[1]**2 + (WAV - DIB[0]**2 / WAV)**2) \
              + 0.63622 * FAC * (DIB1[2] / Rv) * DIB1[1] / (DIB1[1]**2 + (WAV - DIB1[0]**2 / WAV)**2)

    Y4 = X4 - 8
    A4 = -1.073 - 0.628 * Y4 + 0.137 * Y4**2 - 0.070 * Y4**3
    B4 = 13.670 + 4.257 * Y4 - 0.420 * Y4**2 + 0.374 * Y4**3

    result4 = A4 + B4 / Rv

    Y5 = X5 - 10
    A5 = -2.341 - 0.92 * Y5
    B5 = 23.496 + 7.065 * Y5

    result5 = A5 + B5 / Rv

    if case1 == 3:
        zongx = np.concatenate([cX1, cX])
        zongy = np.concatenate([result1, result])
        sort_idx = np.argsort(zongx)
        zongx1 = zongx[sort_idx]
        zongy1 = zongy[sort_idx]
        oo = zongy1 < 0
        tt = np.min(zongx1[oo]) if np.any(oo) else np.inf
        mask = (ccc > np.log10(0.12)) & (ccc < np.log10(2.2)) & (ccc < np.log10(tt))
        wave = wave[mask]
        ccc1 = ccc[mask]
        positive = zongy1 > 0
        logAWAV = interp1d(np.log10(zongx1[positive]), np.log10(zongy1[positive]), kind='cubic')(ccc1)
        AW_AV = 10 ** logAWAV
    else:
        zongx = np.concatenate([1 / X, 1 / X1, 1 / X2, 1 / X3, 1 / X4, 1 / X5])
        zongy = np.concatenate([result, result1, result2, result3, result4, result5])
        sort_idx = np.argsort(zongx)
        zongx1 = zongx[sort_idx]
        zongy1 = zongy[sort_idx]
        positive = zongy1 > 0
        logAWAV = interp1d(np.log10(zongx1[positive]), np.log10(zongy1[positive]), kind='cubic')(ccc)
        AW_AV = 10 ** logAWAV

    return wave * 1e4, AW_AV


def rvcodeg(**kwargs):
    """Backward-compatible alias for the WC2019 extinction curve."""
    return wc2019(**kwargs)
