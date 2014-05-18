from astropy import wcs
import numpy as np
from astropy import units as u
from astropy.wcs import WCSSUB_CELESTIAL

from .load_header import get_cd

def header_to_platescale(header, **kwargs):
    """
    Attempt to determine the spatial platescale from a
    `~astropy.io.fits.Header`

    Parameters
    ----------
    header : `~astropy.io.fits.Header`
        The FITS header to extract the platescale from
    kwargs : dict
        Passed to `wcs_to_platescale`.  See that function for more details

    Returns
    -------
    platescale : float or `~astropy.units.Quantity`
        The platescale in degrees with attached units if `use_units` is True
    """
    w = wcs.WCS(header)
    return wcs_to_platescale(w, **kwargs)

def wcs_to_platescale(mywcs, assert_square=True, use_units=False):
    """
    Attempt to determine the spatial plate scale from a `~astropy.wcs.WCS`

    Parameters
    ----------
    mywcs : :class:`~astropy.wcs.WCS`
        The WCS instance to examine
    assert_square : bool
        Fail if the pixels are non-square
    use_units : bool
        Return a `~astropy.units.Quantity` if True

    Returns
    -------
    platescale : float or `~astropy.units.Quantity`
        The platescale in degrees with attached units if `use_units` is True
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

def smoothing_kernel_size(hdr_from, hdr_to):
    """
    Determine the smoothing kernel size needed to convolve header_from to
    header_to without losing signal.  Operates only in the spatial dimensions
    """

    if not isinstance(hdr_from, wcs.WCS):
        w_from = wcs.WCS(hdr_from).sub(wcs.WCSSUB_CELESTIAL)
    if not isinstance(hdr_to, wcs.WCS):
        w_to = wcs.WCS(hdr_to).sub(wcs.WCSSUB_CELESTIAL)

    widths = []

    for ii in (1,2):
        cd_from = get_cd(w_from,ii)
        cd_to = get_cd(w_to,ii)

        if np.abs(cd_to) < np.abs(cd_from):
            # upsampling: no convolution
            widths[ii-1] = 1e-8
        else:
            # downsampling: smooth with fwhm = pixel size ratio
            widths[ii] = np.abs(cd_to/cd_from) / np.sqrt(8*np.log(2))

    return widths
