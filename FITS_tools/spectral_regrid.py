from astropy import units as u
import numpy as np
from astropy.wcs import WCS

def spec_pix_to_world(pixel, wcs, axisnumber, unit=None):
    """
    Given a WCS, an axis ID, and a pixel ID, return the WCS spectral value at a
    pixel location
    
    .. TODO:: refactor to use wcs.sub
    """
    coords = list(wcs.wcs.crpix)
    coords[axisnumber] = pixel+1
    coords = list(np.broadcast(*coords))
    if unit is None:
        return wcs.wcs_pix2world(coords,1)[:,axisnumber]
    else:
        return wcs.wcs_pix2world(coords,1)[:,axisnumber]*unit

def spec_world_to_pix(worldunit, wcs, axisnumber, unit):
    """
    Given a WCS, an axis ID, and WCS location, return the pixel index value at a
    pixel location
    
    .. TODO:: refactor to use wcs.sub
    """
    coords = list(wcs.wcs.crpix)
    coords[axisnumber] = worldunit.to(unit).value
    coords = list(np.broadcast(*coords))
    return wcs.wcs_world2pix(coords,0)[:,axisnumber]


def get_spectral_mapping(header1, header2, specaxis1=None, specaxis2=None):
    """
    Determine the mapping from header1 pixel units to header2 pixel units along
    their respective spectral axes
    """

    wcs1 = WCS(header1)
    wcs2 = WCS(header2)
    if specaxis1 is None:
        specaxis1 = wcs1.wcs.spec
    if specaxis2 is None:
        specaxis2 = wcs2.wcs.spec

    # Functions to give the spectral coordinate from each FITS header
    w1 = lambda x: spec_pix_to_world(x, wcs1, specaxis1, unit=wcs1.wcs.cunit[specaxis1])
    w2 = lambda x: spec_pix_to_world(x, wcs2, specaxis2, unit=wcs1.wcs.cunit[specaxis2])

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
