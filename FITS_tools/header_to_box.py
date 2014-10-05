import numpy as np
from astropy import wcs

def header_to_box(header):
    """
    Given a flat header (2 dimensions in world coordinates),
    return a box defined by xcenter, ycenter, height, width
    """
    w = wcs.WCS(header).sub([wcs.WCSSUB_CELESTIAL])

    cd1 = header.get('CDELT1') or header.get('CD1_1')
    cd2 = header.get('CDELT2') or header.get('CD2_2')

    height = cd2 * header['NAXIS2']
    width = cd1 * header['NAXIS1']

    #xcenter = header['CRVAL1'] + ((header['NAXIS1']-1)/2 + header['CRPIX1'])*cd1
    #ycenter = header['CRVAL2'] + ((header['NAXIS2']-1)/2 + header['CRPIX2'])*cd2
    (xcenter,ycenter), = w.wcs_pix2world([[(header['NAXIS1']-1)/2.,
                                           (header['NAXIS2']-1)/2.]], 0)

    return xcenter,ycenter,height,width

def box_to_header(xcenter,ycenter,height,width,cd1,cd2):
    raise NotImplementedError

def header_to_ds9reg(header):
    return "box({0}, {1}, {3}, {2}, 0)".format(*header_to_box(header))
