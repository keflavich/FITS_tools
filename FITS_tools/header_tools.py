from astropy import wcs
import numpy as np
from astropy import units as u
from astropy.wcs import WCSSUB_CELESTIAL

def header_to_platescale(header):
    """
    Attempt to determine the spatial platescale from a `~astropy.io.fits.Header`
    """
    w = wcs.WCS(header)
    return wcs_to_platescale(w.wcs)

def wcs_to_platescale(mywcs, assert_square=True, use_units=False):
    """
    Attempt to determine the spatial plate scale from a `~astropy.wcs.WCS`
    """

    # Code adapted from APLpy

    mywcs = mywcs.sub([WCSSUB_CELESTIAL])
    cdelt = np.matrix(mywcs.wcs.get_cdelt())
    pc = np.matrix(mywcs.wcs.get_pc())
    scale = np.array(cdelt * pc)

    if assert_square:
        try:
            np.testing.assert_almost_equal(abs(cdelt[0,0]), abs(cdelt[0,1]))
            np.testing.assert_almost_equal(abs(pc[0,0]), abs(pc[1,1]))
            np.testing.assert_almost_equal(abs(scale[0,0]), abs(scale[0,1]))
        except AssertionError:
            raise ValueError("Non-square pixels.  Please resample data.")

    if use_units:
        return abs(scale[0,0]) * u.Unit(mywcs.wcs.cunit[0])
    else:
        return abs(scale[0,0])
