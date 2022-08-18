import numpy as np
import scipy.ndimage
from astropy.io import fits
from astropy.convolution import convolve,convolve_fft
from astropy import wcs
import warnings
from astropy.convolution import Gaussian2DKernel,Gaussian1DKernel
import itertools
import copy
import contextlib
try:
    import __builtin__ as builtins
except ImportError:
    import builtins

from .downsample import downsample_axis
from .spectral_regrid import get_spectral_mapping
from .hcongrid import get_pixel_mapping
from .strip_headers import flatten_header
from .load_header import load_header,get_cd

__all__ = ["downsample_cube", "get_cube_mapping", "gsmooth_cube",
           "regrid_cube", "regrid_cube_hdu", "regrid_fits_cube",
           "smoothing_kernel_size", "spatial_smooth_cube",
           'spectral_smooth_cube', 'find_grid_limits']

def regrid_fits_cube(cubefilename, outheader, hdu=0, outfilename=None,
                     clobber=False, **kwargs):
    """
    Regrid a FITS file to a target header.  Requires that the FITS file and the
    target header have spectral and spatial overlap.

    See `regrid_cube_hdu` and `regrid_cube` for additional details or
    alternative input options.

    Parameters
    ----------
    cubefilename : str
        FITS file name containing the cube to be reprojected
    outheader : `~astropy.io.fits.Header`
        A target Header to project to
    hdu : int
        The hdu to project to the target header
    outfilename : str
        The output filename to save to
    clobber : bool
        Overwrite the output file if it exists?
    kwargs : dict
        Passed to `~FITS_tools.cube_regrid.regrid_cube_hdu`

    Returns
    -------
    rgcube : `~astropy.io.fits.PrimaryHDU`
        The regridded cube HDU
    """
    cube_hdu = fits.open(cubefilename)[hdu]
    rgcube = regrid_cube_hdu(cube_hdu, outheader)

    if outfilename:
        rgcube.writeto(outfilename, clobber=clobber)

    return rgcube

def regrid_cube_hdu(hdu, outheader, smooth=False, **kwargs):
    """
    Regrid a FITS HDU to a target header.  Requires that the FITS object and
    the target header have spectral and spatial overlap.  See `~regrid_cube` for
    additional details or alternative input options.

    Parameters
    ----------
    hdu : `~astropy.io.fits.PrimaryHDU`
        FITS HDU (not HDUlist) containing the cube to be reprojected
    outheader : `~astropy.io.fits.Header`
        A target Header to project to
    smooth : bool
        Smooth the HDUs to match resolution?
        Kernel size is determined using `smoothing_kernel_size`

        .. WARNING:: Smoothing is done in 3D to be maximally general.
                     This can be exceedingly slow!

    Returns
    -------
    rgcube : `~astropy.io.fits.PrimaryHDU`
        The regridded cube HDU
    """

    outheader = load_header(outheader)

    if smooth:
        kw = smoothing_kernel_size(hdu.header, outheader)
        data = gsmooth_cube(hdu.data, kw)
    else:
        data = hdu.data

    cubedata = regrid_cube(data,hdu.header,outheader,**kwargs)
    newhdu = fits.PrimaryHDU(data=cubedata, header=outheader)
    return newhdu

def regrid_cube(cubedata, cubeheader, targetheader, preserve_bad_pixels=True,
                order=1, mode='constant', out_of_bounds=np.nan, **kwargs):
    """
    Attempt to reproject a cube onto another cube's header.  Uses interpolation
    via `~scipy.ndimage.interpolation.map_coordinates`

    Assumptions:
    
     * Both the cube and the target are 3-dimensional, with lon/lat/spectral axes
     * Both cube and header use CD/CDELT rather than PC

    ``kwargs`` will be passed to `~scipy.ndimage.interpolation.map_coordinates`

    Parameters
    ----------
    cubedata : `~numpy.ndarray`
        A three-dimensional data cube
    cubeheader : `~astropy.io.fits.Header` or `~astropy.wcs.WCS`
        The header or WCS corresponding to the image
    targetheader : `~astropy.io.fits.Header` or `~astropy.wcs.WCS`
        The header or WCS to interpolate onto
    preserve_bad_pixels : bool
        Try to set NAN pixels to NAN in the zoomed image.  Otherwise, bad
        pixels will be set to zero
    out_of_bounds : float or np.nan
        Pixels in the output image that were out of bounds in the old image
        will be set to this value *if* mode='constant'
    mode : str
        The mode of treatment of out-of-bounds pixels; can be 'constant',
        'wrap', 'reflect', or 'nearest'.  See
        `~scipy.ndimage.interpolation.map_coordinates` for details.

    Returns
    -------
    newcubedata : `~numpy.ndarray`
        The regridded cube
    """

    if cubedata.ndim != 3:
        cubedata = cubedata.squeeze()
        if cubedata.ndim != 3:
            raise ValueError("Dataset has %i dimensions, so it's not a cube." % cubedata.ndim)

    full_grid = get_cube_mapping(cubeheader, targetheader)
    grid_limits = find_grid_limits(full_grid)
    # subtract off the lower coordinate of each grid:
    # this is a performance optimization to only use a subset of the 
    # cube if gridding a large cube onto a smaller space
    # This is therefore done IF AND ONLY IF that coordinate is positive!
    grid = [g - m[0] if m[0] > 0 else g
            for g,m in zip(full_grid, grid_limits)]

    if grid[0].ndim != full_grid[0].ndim:
        raise ValueError("Grid dropped a dimension.  This is not possible.")

    # Truncate the input cube on both sides if the input cube is larger than
    # the output grid.  If the input cube is smaller, m[1]+1 should be (at
    # least) the "right side" of the frame and therefore all data should be
    # included.  The +1 is because of python indexing conventions
    limit_slice = tuple(slice((m[0]) if m[0]>0 else 0, (m[1]+1))
                   for m in grid_limits)
    cubedata = cubedata[limit_slice]

    bad_pixels = np.isnan(cubedata) + np.isinf(cubedata)

    cubedata[bad_pixels] = 0

    dtype_in = cubedata.dtype

    # There appears to be a bug in scipy that affects only large arrays:
    # cubedata will be forcibly converted to float64 unless order == 1
    # but maybe it has to be float64?
    if order == 1:
        newcubedata = scipy.ndimage.map_coordinates(cubedata.astype(np.float64),
                                                    grid, order=order, mode=mode,
                                                    cval=out_of_bounds, **kwargs)
    else:
        newcubedata = scipy.ndimage.map_coordinates(cubedata, grid,
                                                    order=order, mode=mode,
                                                    cval=out_of_bounds,
                                                    **kwargs)

    if preserve_bad_pixels:
        # Set any pixels that had a nearest-neighbor which was 'bad' and any
        # out-of-bounds pixels to NaN
        newbad = scipy.ndimage.map_coordinates(bad_pixels, grid, order=0,
                                               mode='constant', cval=True)
        newcubedata[newbad.astype('bool')] = np.nan
    
    return newcubedata.astype(dtype_in)

def find_grid_limits(grid):
    """
    Determine the min/max of each dimension along a grid returned by
    `get_cube_mapping`.  These parameters can be used to slice a cube prior to
    passing it to, e.g., `scipy.ndimage.map_coordinates` to reduce the memory
    use
    """
    return [(int(np.floor(np.nanmin(g))),int(np.ceil(np.nanmax(g)))) for g in grid]

def get_cube_mapping(header1, header2):
    """
    Determine the pixel mapping from Header 1 to Header 2

    Assumptions are spelled out in `~regrid_cube`
    """
    specgrid = get_spectral_mapping(header1,header2,specaxis1=2,specaxis2=2)
    # pixgrid is returned in the order y,x, which is correct for np array indexing
    pixgrid = get_pixel_mapping(flatten_header(header1),flatten_header(header2))
    
    # spec, lat, lon
    # copy=False results in a *huge* speed gain on large arrays
    # indexing='ij' is necessary to prevent weird, arbitrary array shape swappings
    # (indexing='xy' is the "cartesian grid" convention, which numpy doesn't obey...)
    # grid = np.meshgrid(specgrid,pixgrid[0,:,0],pixgrid[1,0,:],copy=False,indexing='ij')
    grid = np.broadcast_arrays(specgrid.reshape(specgrid.size,1,1),
                               pixgrid[0][np.newaxis,:,:],
                               pixgrid[1][np.newaxis,:,:],)

    return grid

def gsmooth_cube(cube, kernelsize, use_fft=True, psf_pad=False, fft_pad=False,
                 kernelsize_mult=8, **kwargs):
    """
    Smooth a cube with a gaussian in 3d
    
    Because even a tiny cube can become enormous if you have, say, a 1024x32x32
    cube, padding is off by default
    """
    if cube.ndim != 3:
        raise ValueError("Wrong number of dimensions for a data cube")
    
    #z,y,x = np.indices(cube.shape)
    # use an odd kernel size for non-fft, even kernel size for fft
    ks = (np.array(kernelsize)*kernelsize_mult).astype('int')
    if np.any(ks % 2 == 0) and not use_fft:
        ks[ks % 2 == 0] += 1

    z,y,x = np.indices(ks)
    kernel = np.exp(-((x-x.max()/2.)**2 / (2.*(kernelsize[2])**2) +
                      (y-y.max()/2.)**2 / (2.*(kernelsize[1])**2) +
                      (z-z.max()/2.)**2 / (2.*(kernelsize[0])**2)))

    if use_fft:
        return convolve_fft(cube, kernel, normalize_kernel=True,
                            psf_pad=psf_pad, fft_pad=fft_pad, **kwargs)
    else:
        return convolve(cube, kernel, normalize_kernel=True, **kwargs)

def smoothing_kernel_size(hdr_from, hdr_to):
    """
    Determine the smoothing kernel size needed to convolve a cube before
    downsampling it to retain signal.
    """

    w_from = wcs.WCS(hdr_from)
    w_to = wcs.WCS(hdr_to)

    widths = []

    for ii in (1,2,3):
        cd_from = get_cd(w_from,ii)
        cd_to = get_cd(w_to,ii)

        if np.abs(cd_to) < np.abs(cd_from):
            # upsampling: no convolution
            widths[ii-1] = 1e-8
        else:
            # downsampling: smooth with fwhm = pixel size ratio
            widths[ii] = np.abs(cd_to/cd_from) / np.sqrt(8*np.log(2))

    return widths


def _gsmooth_img(args):
    """
    HELPER FUNCTION: private!
    Smooth an image with a gaussian in 2d
    """
    img,kernel,use_fft,kwargs = args

    if use_fft:
        return convolve_fft(img, kernel, normalize_kernel=True, **kwargs)
    else:
        return convolve(img, kernel, normalize_kernel=True, **kwargs)

def spatial_smooth_cube(cube, kernelwidth, kernel=Gaussian2DKernel, cubedim=0,
                        numcores=None, use_fft=True, **kwargs):
    """
    Parallelized spatial smoothing

    Parameters
    ----------
    cube : `~numpy.ndarray`
        A data cube, with ndim=3
    kernelwidth : float
        Width of the kernel.  Defaults to Gaussian.
    kernel : `~astropy.convolution.Kernel2D`
        A 2D kernel from astropy
    cubedim : int
        The axis to map across.  If you have a normal FITS data cube with
        AXIS1=RA, AXIS2=Dec, and AXIS3=wavelength, for example, cubedim is 0
        (because axis3 -> 0, axis2 -> 1, axis1 -> 2)
    numcores : int
        Number of cores to use in parallel-processing.
    use_fft : bool
        Use `astropy.convolution.convolve_fft` or `astropy.convolution.convolve`?
    kwargs : dict
        Passed to `~astropy.convolution.convolve`
    """
        
    if cubedim != 0:
        cube = cube.swapaxes(0,cubedim)

    cubelist = [cube[ii,:,:] for ii in xrange(cube.shape[0])]

    kernel = kernel(kernelwidth)

    with _map_context(numcores) as map:
        smoothcube = np.array(map(_gsmooth_img,
                                  zip(cubelist,
                                      itertools.cycle([kernel]),
                                      itertools.cycle([use_fft]),
                                      itertools.cycle([kwargs]))
                                  )
                              )

    if cubedim != 0:
        smoothcube = smoothcube.swapaxes(0,cubedim)

    return smoothcube

def _gsmooth_spectrum(args):
    """
    HELPER FUNCTION: private!
    Smooth a spectrum with a gaussian in 1d
    """
    spec,kernel,use_fft,kwargs = args

    if use_fft:
        return convolve_fft(spec, kernel, normalize_kernel=True, **kwargs)
    else:
        return convolve(spec, kernel, normalize_kernel=True, **kwargs)


def spectral_smooth_cube(cube, kernelwidth, kernel=Gaussian1DKernel, cubedim=0,
                         parallel=True, numcores=None, use_fft=False,
                         **kwargs):
    """
    Parallelized spectral smoothing

    Parameters
    ----------
    cube : `~numpy.ndarray`
        A data cube, with ndim=3
    kernelwidth : float
        Width of the kernel.  Defaults to Gaussian.
    kernel : `~astropy.convolution.Kernel1D`
        A 1D kernel from astropy
    cubedim : int
        The axis *NOT* to map across, i.e. the spectral axis.  If you have a
        normal FITS data cube with AXIS1=RA, AXIS2=Dec, and AXIS3=wavelength,
        for example, cubedim is 0 (because axis3 -> 0, axis2 -> 1, axis1 -> 2)
    numcores : int
        Number of cores to use in parallel-processing.
    use_fft : bool
        Use convolve_fft or convolve?
    kwargs : dict
        Passed to astropy.convolution.convolve
    """

    if cubedim != 0:
        cube = cube.swapaxes(0,cubedim)

    shape = cube.shape

    cubelist = [cube[:,jj,ii]
                for jj in xrange(cube.shape[1])
                for ii in xrange(cube.shape[2])]

    kernel = kernel(kernelwidth)

    with _map_context(numcores) as map:
        smoothcube = np.array(map(_gsmooth_spectrum,
                                  zip(cubelist,
                                      itertools.cycle([kernel]),
                                      itertools.cycle([use_fft]),
                                      itertools.cycle([kwargs]))
                                  )
                              )

    # empirical: need to swapaxes to get shape right
    # cube = np.arange(6*5*4).reshape([4,5,6]).swapaxes(0,2)
    # cubelist.T.reshape(cube.shape) == cube
    smoothcube = smoothcube.T.reshape(shape)
    
    if cubedim != 0:
        smoothcube = smoothcube.swapaxes(0,cubedim)

    return smoothcube

def downsample_cube(cubehdu, factor, spectralaxis=None):
    """
    Downsample a cube along the spectral axis

    Parameters
    ----------
    cubehdu : `~astropy.io.fits.PrimaryHDU`
        The cube to downsample
    factor : int
        The factor by which the cube should be downsampled
    spectralaxis : int
        The 0-indexed ID fo the spectral axis.  Will be determined
        automatically if not specified
    """

    
    header = copy.copy(cubehdu.header)

    mywcs = wcs.WCS(header)
    if spectralaxis is None:
        spectralaxis = mywcs.wcs.spec

    avg = downsample_axis(cubehdu.data, factor, axis=spectralaxis)

    mywcs.wcs.cdelt[spectralaxis] *= factor
    crpix = mywcs.wcs.crpix[spectralaxis]

    scalefactor = 1./factor
    crpix_new = (crpix-1)*scalefactor+0.5+scalefactor/2.
    mywcs.wcs.crpix[spectralaxis] = crpix_new

    whdr = mywcs.to_header()
    for key in whdr:
        header[key] = whdr[key]

    hdu = fits.PrimaryHDU(data=avg, header=header)

    return hdu


@contextlib.contextmanager
def _map_context(numcores):
    if numcores is not None and numcores > 1:
        try:
            import multiprocessing
            p = multiprocessing.Pool(processes=numcores)
            map = p.map
            parallel = True
        except ImportError:
            map = builtins.map
            warnings.warn("Could not import multiprocessing.  map will be non-parallel.")
            parallel = False
    else:
        parallel = False
        map = builtins.map

    try:
        yield map
    finally:
        if parallel:
            p.close()
