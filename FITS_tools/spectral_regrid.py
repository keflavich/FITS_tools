from astropy import units as u
import numpy as np
from astropy.wcs import WCS

def get_spectral_mapping(header1, header2, specaxis1=None, specaxis2=None):
    """
    Determine the mapping from header1 pixel units to header2 pixel units
    """

    wcs1 = WCS(header1)
    wcs2 = WCS(header2)
    if specaxis1 is None:
        specaxis1 = wcs1.wcs.spec
    if specaxis2 is None:
        specaxis2 = wcs2.wcs.spec

    u1 = u.Unit(header1['CUNIT%i' % (specaxis1+1)])
    u2 = u.Unit(header2['CUNIT%i' % (specaxis2+1)])

    # Functions to give the spectral coordinate from each FITS header
    def w1(x):
        coords=list(wcs1.wcs.crpix)
        coords[specaxis1] = x+1
        coords = list(np.broadcast(*coords))
        return wcs1.wcs_pix2world(coords,1)[:,specaxis1]*u1
    def w2(x):
        coords=list(wcs2.wcs.crpix)
        coords[specaxis2] = x+1
        coords = list(np.broadcast(*coords))
        return wcs2.wcs_pix2world(coords,1)[:,specaxis2]*u2

    # specaxis2 indexed from 0, naxis indexed from 1
    outshape = header2['NAXIS%i' % (specaxis2+1)]
    xx_out = np.arange(outshape)
    world_out = w2(xx_out)

    # There is no invert function, so we need to determine where the grid points fall
    inshape = header1['NAXIS%i' % (specaxis1+1)]
    xx_in = np.arange(inshape)
    world_in = w1(xx_in)

    # Convert to common unit frame
    wo = world_out.to(world_in.unit).value

    # np.interp requires that the xp points be increasing, but world_in could be decreasing
    sortinds = np.argsort(world_in)
    wi = world_in.to(world_in.unit).value[sortinds]

    grid = np.interp(wo,wi,xx_in[sortinds],left=np.nan,right=np.nan)

    if all(np.isnan(grid)):
        raise ValueError("No overlap between input & output header.")

    return grid
