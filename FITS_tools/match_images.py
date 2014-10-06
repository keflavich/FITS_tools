import numpy as np
import astropy.io.fits as fits
import astropy.wcs as wcs
from .strip_headers import flatten_header
from .cube_regrid import regrid_cube_hdu
from .load_header import load_header

__all__ = ['match_fits', 'match_fits_cubes', 'project_to_header']


def project_to_header(fitsfile, header, use_montage=True, quiet=True,
                      **kwargs):
    """
    Light wrapper of montage with hcongrid as a backup

    kwargs will be passed to `~hcongrid.hcongrid` if ``use_montage==False``

    Parameters
    ----------
    fitsfile : string
        a FITS file name
    header : `~astropy.io.fits.Header`
        A fits Header instance with valid WCS to project to
    use_montage : bool
        Use montage or hcongrid (based on `~scipy.ndimage.interpolation.map_coordinates`)
    quiet : bool
        Silence Montage's output

    Returns
    -------
    image : `~numpy.ndarray`
        image projected to header's coordinates

    """
    try:
        import montage
        montageOK=True
    except ImportError:
        montageOK=False
    try:
        from hcongrid import hcongrid
        hcongridOK=True
    except ImportError:
        hcongridOK=False
    import tempfile

    if montageOK and use_montage:
        temp_headerfile = tempfile.NamedTemporaryFile()
        header.toTxtFile(temp_headerfile.name)

        outfile = tempfile.NamedTemporaryFile()
        montage.wrappers.reproject(fitsfile, outfile.name,
                temp_headerfile.name, exact_size=True,
                silent_cleanup=quiet)
        image = fits.getdata(outfile.name)
        
        outfile.close()
        temp_headerfile.close()
    elif hcongridOK:
        # only works for 2D images
        image = hcongrid(fits.getdata(fitsfile).squeeze(),
                         flatten_header(fits.getheader(fitsfile)),
                         header,
                         **kwargs)

    return image

def match_fits(fitsfile1, fitsfile2, header=None, sigma_cut=False,
               return_header=False, **kwargs):
    """
    Project one FITS file into another's coordinates.  If ``sigma_cut`` is
    used, will try to find only regions that are significant in both images
    using the standard deviation, masking out other signal

    Parameters
    ----------
    fitsfile1 : str
        Offset fits file name
    fitsfile2 : str
        Reference fits file name
    header : `~astropy.io.fits.Header`
        Optional - can pass a header that both input images will be projected
        to match
    sigma_cut : bool or int
        Perform a sigma-cut on the returned images at this level

    Returns
    -------
    image1,image2,[header] : `~numpy.ndarray`, `~numpy.ndarray`, `~astropy.io.fits.Header`
        Two images projected into the same space, and optionally
        the header used to project them

    See Also
    --------
    match_fits_cubes : this function, but for cubes
    """

    if header is None:
        header = flatten_header(fits.getheader(fitsfile2))
        image2 = fits.getdata(fitsfile2).squeeze()
    else: # project image 1 to input header coordinates
        image2 = project_to_header(fitsfile2, header, **kwargs)

    # project image 1 to image 2 coordinates
    image1_projected = project_to_header(fitsfile1, header, **kwargs)

    if image1_projected.shape != image2.shape:
        raise ValueError("Failed to reproject images to same shape.")

    if sigma_cut:
        corr_image1 = image1_projected*(image1_projected > image1_projected.std()*sigma_cut)
        corr_image2 = image2*(image2 > image2.std()*sigma_cut)
        OK = (corr_image1==corr_image1)*(corr_image2==corr_image2)
        if (corr_image1[OK]*corr_image2[OK]).sum() == 0:
            print "Could not use sigma_cut of %f because it excluded all valid data" % sigma_cut
            corr_image1 = image1_projected
            corr_image2 = image2
    else:
        corr_image1 = image1_projected
        corr_image2 = image2

    returns = corr_image1, corr_image2
    if return_header:
        returns = returns + (header,)
    return returns

def match_fits_cubes(fitsfile1, fitsfile2, header=None, sigma_cut=False,
                     return_header=False, smooth=False, **kwargs):
    """
    Project one FITS file representing a data cube into another's coordinates.

    Parameters
    ----------
    fitsfile1 : str
        FITS file name to reproject
    fitsfile2 : str
        Reference FITS file name.  If ``header`` is specified, 
    smooth : bool
        Smooth the HDUs to match resolution?
        Kernel size is determined using `cube_regrid.smoothing_kernel_size`

        .. WARNING:: Smoothing is done in 3D to be maximally general.
                     This can be exceedingly slow!

    header : `~astropy.io.fits.Header`
        Optional - can pass a header that both input images will be projected
        to match

    Raises
    ------
    ValueError : 
        Will raise an error if the axes are not consistent with a FITS cube,
        i.e.  two spatial and one spectral axis.

    Returns
    -------
    image1,image2,[header] : `~numpy.ndarray`, `~numpy.ndarray`, `~astropy.io.fits.Header`
        Two images projected into the same space, and optionally
        the header used to project them

    See Also
    --------
    cube_regrid.regrid_fits_cube : regrid a single cube
        This function performs a similar purpose and does the underlying work
        for `match_fits_cubes`, but it has a different call specification and
        returns an HDU

    """
    
    header1 = load_header(fitsfile1)
    header2 = load_header(fitsfile2)
    wcs1 = wcs.WCS(header1)
    wcs2 = wcs.WCS(header2)

    if wcs1.wcs.naxis != 3:
        raise ValueError("First input file is not a cube.")
    if wcs2.wcs.naxis != 3:
        raise ValueError("Second input file is not a cube.")

    if header is not None:
        wcs3 = wcs.WCS(header)
        if wcs3.wcs.naxis != 3:
            raise ValueError("Input header is not a cube header.")

        if wcs2 != wcs3:
            image2 = regrid_cube_hdu(hdu=fits.open(fitsfile2)[0],
                                     outheader=header,
                                     smooth=smooth,
                                     **kwargs).data
            header2 = header

    else:
        image2 = fits.getdata(fitsfile2)

    
    image1 = regrid_cube_hdu(hdu=fits.open(fitsfile1)[0],
                             outheader=header2,
                             smooth=smooth,
                             **kwargs).data

    if return_header:
        return image1,image2,header2
    else:
        return image1,image2
