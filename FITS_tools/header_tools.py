from astropy import wcs
import numpy as np

def header_to_platescale(header):
    w = wcs.WCS(header)
    return wcs_to_platescale(w.wcs)

def wcs_to_platescale(wcs):
    cdelt = np.matrix(wcs.get_cdelt())
    pc = np.matrix(wcs.get_pc())
    scale = np.array(cdelt * pc)[0,:]
    # this may be wrong in perverse cases
    pixscale = np.abs(scale[0])
    return pixscale
