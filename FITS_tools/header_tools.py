from astropy import wcs
import numpy as np
from astropy import units as u
from astropy import coordinates
from astropy.wcs import WCSSUB_CELESTIAL
from astropy import log

from .load_header import get_cd
from .hcongrid import _ctype_to_csys

__all__ = ["enclosing_header", "header_to_platescale", "smoothing_kernel_size",
           "wcs_to_platescale",]

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

def enclosing_header(header1, header2, wrapangle=180*u.deg):
    """
    Find the smallest header that encloses both header1 and header2 in the
    frame of header2
    """
    log.warning("'Enclosing Header' does not work exactly - there is at least "
                "a few-pixel offset in the result.  Need a good test suite.")
    w1 = wcs.WCS(header1).sub([WCSSUB_CELESTIAL])
    w2 = wcs.WCS(header2).sub([WCSSUB_CELESTIAL])

    pedges1 = np.array([(1, header1['NAXIS2']),
                        (header1['NAXIS1'], header1['NAXIS2']),
                        (header1['NAXIS1'], 1),
                        (1, 1),])
    pedges2 = np.array([(1,header2['NAXIS2']),
                        (header2['NAXIS1'], header2['NAXIS2']),
                        (header2['NAXIS1'], 1),
                        (1, 1),])
    edges1 = w1.wcs_pix2world(pedges1, 1)
    edges2 = w2.wcs_pix2world(pedges2, 1)

    cedges2 = coordinates.SkyCoord(edges2*u.deg,
                                   frame=_ctype_to_csys(w1.wcs))

    cedges1 = coordinates.SkyCoord(edges1*u.deg,
                                   frame=_ctype_to_csys(w2.wcs)).transform_to(cedges2.frame)

    #(a,b), = w2.wcs_pix2world([[1,1]],1)
    #lbref = coordinates.SkyCoord(a*u.deg, b*u.deg,
    #                             frame=_ctype_to_csys(w2.wcs))

    wa = wrapangle
    if hasattr(cedges2, 'l'):
        llow = min(cedges1.l.wrap_at(wa).min(), cedges2.l.wrap_at(wa).min())
        lhi  = max(cedges1.l.wrap_at(wa).max(), cedges2.l.wrap_at(wa).max())
        blow = min(cedges1.b.wrap_at(wa).min(), cedges2.b.wrap_at(wa).min())
        bhi  = max(cedges1.b.wrap_at(wa).max(), cedges2.b.wrap_at(wa).max())
        #lref, bref = lbref.l, lbref.b
    elif hasattr(cedges2, 'ra'):
        llow = min(cedges1.ra.min(),  cedges2.ra.min())
        lhi  = max(cedges1.ra.max(),  cedges2.ra.max())
        blow = min(cedges1.dec.min(), cedges2.dec.min())
        bhi  = max(cedges1.dec.max(), cedges2.dec.max())
        #lref, bref = lbref.ra, lbref.dec
    else:
        raise ValueError("Invalid coordinates.")


    (xlow,ylow), = w2.wcs_world2pix([[llow.deg,blow.deg]], 1)
    (xhi ,yhi ), = w2.wcs_world2pix([[lhi.deg ,bhi.deg ]], 1)

    (xref, yref), = w2.wcs_world2pix([[(max(llow,lhi)).deg,
                                       (min(blow,bhi)).deg]], 0)

    header = header2.copy()
    header['NAXIS1'] = int(np.ceil(np.abs(xhi-xlow)))
    header['NAXIS2'] = int(np.ceil(np.abs(yhi-ylow)))
    header['CRPIX1'] = header2['CRPIX1'] - xref #min(xlow, xhi)
    header['CRPIX2'] = header2['CRPIX2'] - yref #min(ylow, yhi)

    return header
