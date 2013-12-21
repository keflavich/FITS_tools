from specutils.io import fits as spfits
import numpy as np

def get_spectral_mapping(header1, header2, specaxis1=0, specaxis2=0):
    """
    Determine the mapping from header1 pixel units to header2 pixel units
    """

    h1 = spfits.FITSWCSSpectrum(header1)
    h2 = spfits.FITSWCSSpectrum(header2)
    w1 = spfits.read_fits_wcs_linear1d(h1,spectral_axis=specaxis1)
    w2 = spfits.read_fits_wcs_linear1d(h2,spectral_axis=specaxis2)

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
    wi = world_in.to(world_in.unit).value

    grid = np.interp(wo,wi,xx_in,left=np.nan,right=np.nan)

    return grid
