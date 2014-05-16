import numpy as np

def header_to_box(header):
    """
    Given a flat header (2 dimensions in world coordinates),
    return a box defined by xcenter, ycenter, height, width
    """

    cd1 = header.get('CDELT1') or header.get('CD1_1')
    cd2 = header.get('CDELT2') or header.get('CD2_2')

    height = cd2 * header['NAXIS2']
    width = cd1 * header['NAXIS1']

    xcenter = header['CRVAL1'] + ((header['NAXIS1']-1)/2 + header['CRPIX1'])*cd1
    ycenter = header['CRVAL2'] + ((header['NAXIS2']-1)/2 + header['CRPIX2'])*cd2

    return xcenter,ycenter,height,width

def box_to_header(xcenter,ycenter,height,width,cd1,cd2):
    raise NotImplementedError
