import numpy as np
import scipy.ndimage
from .spectral_regrid import get_spectral_mapping
from .hcongrid import get_pixel_mapping
from .strip_headers import flatten_header,speccen_header

def regrid_cube_hdu(hdu, outheader,**kwargs):
    return regrid_cube(hdu.data,hdu.header,outheader,**kwargs)

def regrid_cube(cubedata, cubeheader, targetheader, preserve_bad_pixels=True, **kwargs):
    """
    Attempt to reproject a cube onto another cube's header.
    Uses interpolation via scipy.ndimage.map_coordinates

    Assumptions:
    
     * Both the cube and the target are 3-dimensional, with lon/lat/spectral axest
     * Both cube and header use CD/CDELT rather than PC

    kwargs will be passed to `scipy.ndimage.map_coordinates`

    Parameters
    ----------
    cubedata : ndarray
        A two-dimensional image 
    cubeheader : `pyfits.Header` or `pywcs.WCS`
        The header or WCS corresponding to the image
    targetheader : `pyfits.Header` or `pywcs.WCS`
        The header or WCS to interpolate onto
    preserve_bad_pixels: bool
        Try to set NAN pixels to NAN in the zoomed image.  Otherwise, bad
        pixels will be set to zero
    """

    grid = get_cube_mapping(cubeheader, targetheader)

    bad_pixels = np.isnan(cubedata) + np.isinf(cubedata)

    cubedata[bad_pixels] = 0

    newcubedata = scipy.ndimage.map_coordinates(cubedata, grid, **kwargs)

    if preserve_bad_pixels:
        newbad = scipy.ndimage.map_coordinates(bad_pixels, grid, order=0, mode='constant', cval=np.nan)
        newcubedata[newbad] = np.nan
    
    return newcubedata

def get_cube_mapping(header1, header2):
    """
    Determine the pixel mapping from Header 1 to Header 2

    Assumptions are spelled out in regrid_cube
    """
    specgrid = get_spectral_mapping(header1,header2,specaxis1=2,specaxis2=2)
    pixgrid = get_pixel_mapping(flatten_header(header1),flatten_header(header2))
    
    return np.meshgrid(pixgrid[0,:,0],pixgrid[1,0,:],specgrid)
