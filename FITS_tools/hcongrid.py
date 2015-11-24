import numpy as np
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
from astropy import coordinates
from astropy import units as u
import scipy.ndimage

__doctest_skip__ = ['hcongrid']


def hcongrid(image, header1, header2, preserve_bad_pixels=True, **kwargs):
    """
    Interpolate an image from one FITS header onto another

    kwargs will be passed to `~scipy.ndimage.interpolation.map_coordinates`

    Parameters
    ----------
    image : `~numpy.ndarray`
        A two-dimensional image
    header1 : `~astropy.io.fits.Header` or `~astropy.wcs.WCS`
        The header or WCS corresponding to the image
    header2 : `~astropy.io.fits.Header` or `~astropy.wcs.WCS`
        The header or WCS to interpolate onto
    preserve_bad_pixels : bool
        Try to set NAN pixels to NAN in the zoomed image.  Otherwise, bad
        pixels will be set to zero

    Returns
    -------
    newimage : `~numpy.ndarray`
        ndarray with shape defined by header2's naxis1/naxis2

    Raises
    ------
    TypeError if either is not a Header or WCS instance
    Exception if image1's shape doesn't match header1's naxis1/naxis2

    Examples
    --------
    >>> fits1 = pyfits.open('test.fits')
    >>> target_header = pyfits.getheader('test2.fits')
    >>> new_image = hcongrid(fits1[0].data, fits1[0].header, target_header)

    """

    _check_header_matches_image(image, header1)

    grid1 = get_pixel_mapping(header1, header2)

    bad_pixels = np.isnan(image) + np.isinf(image)

    image[bad_pixels] = 0

    newimage = scipy.ndimage.map_coordinates(image, grid1, **kwargs)

    if preserve_bad_pixels:
        newbad = scipy.ndimage.map_coordinates(bad_pixels, grid1, order=0,
                                               mode='constant',
                                               cval=np.nan)
        newimage[newbad] = np.nan

    return newimage


def _load_wcs_from_header(header):
    if issubclass(pywcs.WCS, header.__class__):
        wcs = header
    else:
        try:
            wcs = pywcs.WCS(header)
        except:
            raise TypeError("header must either be a astropy.io.fits.Header "
                            " or pywcs.WCS instance")

        if not hasattr(wcs, 'naxis1'):
            wcs.naxis1 = header['NAXIS1']
        if not hasattr(wcs, 'naxis2'):
            wcs.naxis2 = header['NAXIS2']

    return wcs


def _check_header_matches_image(image, header):

    wcs = _load_wcs_from_header(header)

    # wcs.naxis attributes are deprecated, we perform this check conditionally
    if ((hasattr(wcs, 'naxis1') and hasattr(wcs, 'naxis2')) and not
            (wcs.naxis1 == image.shape[1] and wcs.naxis2 == image.shape[0])):
        raise Exception("Image shape must match header shape.")


def get_pixel_mapping(header1, header2):
    """
    Determine the mapping from pixel coordinates in header1 to pixel
    coordinates in header2

    Parameters
    ----------
    header1 : `~astropy.io.fits.Header` or `~astropy.wcs.WCS`
        The header or WCS corresponding to the image to be transformed
    header2 : `~astropy.io.fits.Header` or `~astropy.wcs.WCS`
        The header or WCS to interpolate onto

    Returns
    -------
    grid : `~numpy.ndarray`
        ndarray describing a grid of y,x pixel locations in the input
        header's pixel units but the output header's world units

    Raises
    ------
    TypeError :
        If either header is not a Header or WCS instance
        NotImplementedError if the CTYPE in the header is not recognized
    """

    wcs1 = _load_wcs_from_header(header1)
    wcs2 = _load_wcs_from_header(header2)

    if not all([w1 == w2 for w1, w2 in zip(wcs1.wcs.ctype, wcs2.wcs.ctype)]):
        allowed_coords = ('GLON', 'GLAT', 'RA', 'DEC')
        if all([(any(word in w1 for word in allowed_coords) and
                 any(word in w2 for word in allowed_coords))
                for w1, w2 in zip(wcs1.wcs.ctype, wcs2.wcs.ctype)]):
            csys1 = _ctype_to_csys(wcs1.wcs)
            csys2 = _ctype_to_csys(wcs2.wcs)
            convert_coordinates = True
        else:
            # do unit conversions
            raise NotImplementedError("Unit conversions between {0} and {1} "
                                      "have not yet been implemented."
                                      .format(wcs1.wcs.ctype, wcs2.wcs.ctype))
    else:
        convert_coordinates = False

    # sigh... why does numpy use matrix convention?  Makes everything so
    # much harder...
    # WCS has naxis attributes because it is loaded with
    # _load_wcs_from_header
    outshape = [wcs2.naxis2, wcs2.naxis1]
    yy2, xx2 = np.indices(outshape)

    # get the world coordinates of the output image
    lon2, lat2 = wcs2.wcs_pix2world(xx2, yy2, 0)

    if convert_coordinates:
        # transform the world coordinates from the output image into the
        # coordinate system of the input image
        C2 = coordinates.SkyCoord(lon2, lat2, unit=(u.deg, u.deg), frame=csys2)
        C1 = C2.transform_to(csys1)
        lon2, lat2 = C1.spherical.lon.deg, C1.spherical.lat.deg

    xx1, yy1 = wcs1.wcs_world2pix(lon2, lat2, 0)
    grid = np.array([yy1.reshape(outshape), xx1.reshape(outshape)])

    return grid


def _ctype_to_csys(wcs):
    ctype = wcs.ctype[0]
    if 'RA' in ctype or 'DEC' in ctype:
        if wcs.equinox == 2000:
            return 'fk5'
        elif wcs.equinox == 1950:
            return 'fk4'
        else:
            raise NotImplementedError("Non-fk4/fk5 equinoxes are not allowed")
    elif 'GLON' in ctype or 'GLAT' in ctype:
        return 'galactic'


def hcongrid_hdu(hdu_in, header, **kwargs):
    """
    Wrapper of hcongrid to work on HDUs

    See `hcongrid` for details
    """

    reproj_image = hcongrid(hdu_in.data, hdu_in.header, header, **kwargs)

    return pyfits.PrimaryHDU(data=reproj_image, header=header)


def zoom_fits(fitsfile, scalefactor, preserve_bad_pixels=True, **kwargs):
    """
    Zoom in on a FITS image by interpolating using `~scipy.ndimage.interpolation.zoom`

    Parameters
    ----------
    fitsfile : str
        FITS file name
    scalefactor : float
        Zoom factor along all axes
    preserve_bad_pixels : bool
        Try to set NAN pixels to NAN in the zoomed image.  Otherwise, bad
        pixels will be set to zero
    """

    arr = pyfits.getdata(fitsfile)
    h = pyfits.getheader(fitsfile)

    h['CRPIX1'] = (h['CRPIX1']-1)*scalefactor + scalefactor/2. + 0.5
    h['CRPIX2'] = (h['CRPIX2']-1)*scalefactor + scalefactor/2. + 0.5
    if 'CD1_1' in h:
        for ii in (1, 2):
            for jj in (1, 2):
                k = "CD%i_%i" % (ii, jj)
                if k in h:  # allow for CD1_1 but not CD1_2
                    h[k] = h[k]/scalefactor
    elif 'CDELT1' in h:
        h['CDELT1'] = h['CDELT1']/scalefactor
        h['CDELT2'] = h['CDELT2']/scalefactor

    bad_pixels = np.isnan(arr) + np.isinf(arr)

    arr[bad_pixels] = 0

    upscaled = scipy.ndimage.zoom(arr, scalefactor, **kwargs)

    if preserve_bad_pixels:
        bp_up = scipy.ndimage.zoom(bad_pixels, scalefactor,
                                   mode='constant', cval=np.nan, order=0)
        upscaled[bp_up] = np.nan

    up_hdu = pyfits.PrimaryHDU(data=upscaled, header=h)

    return up_hdu

# hastrom and hcongrid are basically the same...?
# http://idlastro.gsfc.nasa.gov/ftp/pro/astrom/hastrom.pro
# http://idlastro.gsfc.nasa.gov/ftp/pro/astrom/hcongrid.pro
hastrom = hcongrid
hastrom_hdu = hcongrid_hdu


def wcsalign(hdu_in, header, outname=None, clobber=False):
    """
    Align one FITS image to a specified header

    Requires pyast.

    Parameters
    ----------
    hdu_in : `~astropy.io.fits.PrimaryHDU`
        The HDU to reproject (must have header & data)
    header : `~astropy.io.fits.Header`
        The target header to project to
    outname : str (optional)
        The filename to write to.
    clobber : bool
        Overwrite the file ``outname`` if it exists

    Returns
    -------
    The reprojected fits.PrimaryHDU

    Credits
    -------
    Written by David Berry and adapted to functional form by Adam Ginsburg
    (adam.g.ginsburg@gmail.com)
    """
    try:
        import starlink.Ast as Ast
        import starlink.Atl as Atl
    except ImportError:
        raise ImportError("starlink could not be imported, wcsalign is not "
                          "available")

    #  Create objects that will transfer FITS header cards between an AST
    #  FitsChan and the fits header describing the primary HDU of the
    #  supplied FITS file.
    adapter_in = Atl.PyFITSAdapter(hdu_in)
    hdu_ref = pyfits.PrimaryHDU(header=header)
    adapter_ref = Atl.PyFITSAdapter(hdu_ref)

    #  Create a FitsChan for each and use the above adapters to copy all the
    #  header cards into it.
    fitschan_in = Ast.FitsChan(adapter_in, adapter_in)
    fitschan_ref = Ast.FitsChan(adapter_ref, adapter_ref)

    #  Get the flavour of FITS-WCS used by the header cards currently in the
    #  input FITS file. This is so that we can use the same flavour when we
    #  write out the modified WCS.
    encoding = fitschan_in.Encoding

    #  Read WCS information from the two FitsChans. Additionally, this removes
    #  all WCS information from each FitsChan. The returned wcsinfo object
    #  is an AST FrameSet, in which the current Frame describes WCS coordinates
    #  and the base Frame describes pixel coodineates. The FrameSet includes a
    #  Mapping that specifies the transformation between the two Frames.
    wcsinfo_in = fitschan_in.read()
    wcsinfo_ref = fitschan_ref.read()

    #  Check that the input FITS header contained WCS in a form that can be
    #  understood by AST.
    if wcsinfo_in is None:
        raise ValueError("Failed to read WCS information from {0}"
                         .format(hdu_in))

    #  This is restricted to 2D arrays, so check theinput FITS file has 2
    #  pixel axes (given by Nin) and 2 WCS axes (given by Nout).
    elif wcsinfo_in.Nin != 2 or wcsinfo_in.Nout != 2:
        raise ValueError("{0} is not 2-dimensional".format(hdu_in))

    #  Check the reference FITS file in the same way.
    elif wcsinfo_ref is None:
        raise ValueError("Failed to read WCS information from {0}"
                         .format(hdu_ref))

    elif wcsinfo_ref.Nin != 2 or wcsinfo_ref.Nout != 2:
        raise ValueError("{0} is not 2-dimensional".format(hdu_ref))

    #  Proceed if the WCS information was read OK.

    #  Attempt to get a mapping from pixel coords in the input FITS file to
    #  pixel coords in the reference fits file, with alignment occuring by
    #  preference in the current WCS frame. Since the pixel coordinate frame
    #  will be the base frame in each Frameset, we first invert the
    #  FrameSets. This is because the Convert method aligns current Frames,
    #  not base frames.
    wcsinfo_in.invert()
    wcsinfo_ref.invert()
    alignment_fs = wcsinfo_in.convert(wcsinfo_ref)

    #  Invert them again to put them back to their original state (i.e.
    #  base frame = pixel coords, and current Frame = WCS coords).
    wcsinfo_in.invert()
    wcsinfo_ref.invert()

    #  Check alignment was possible.
    if alignment_fs is None:
        raise Exception("Cannot find a common coordinate system shared by "
                        "{0} and {1}".format(hdu_in, hdu_ref))

    else:
        #  Get the lower and upper bounds of the input image in pixel
        #  indices.  All FITS arrays by definition have lower pixel bounds of
        #  [1, 1] (unlike NDFs). Note, unlike fits AST uses FITS ordering for
        #  storing pixel axis values in an array (i.e. NAXIS1 first, NAXIS2
        #  second, etc).
        lbnd_in = [1, 1]
        ubnd_in = [fitschan_in["NAXIS1"], fitschan_in["NAXIS2"]]

        #  Find the pixel bounds of the input image within the pixel coordinate
        #  system of the reference fits file.
        (lb1, ub1, xl, xu) = alignment_fs.mapbox(lbnd_in, ubnd_in, 1)
        (lb2, ub2, xl, xu) = alignment_fs.mapbox(lbnd_in, ubnd_in, 2)

        #  Calculate the bounds of the output image.
        lbnd_out = [int(lb1), int(lb2)]
        ubnd_out = [int(ub1), int(ub2)]

        #  Unlike NDFs, FITS images cannot have an arbitrary pixel origin so
        #  we need to ensure that the bottom left corner of the input image
        #  gets mapped to pixel [1,1] in the output. To do this we, extract the
        #  mapping from the alignment FrameSet and add on a ShiftMap (a mapping
        #  that just applies a shift to each axis).
        shift = [1 - lbnd_out[0],
                 1 - lbnd_out[1]]

        alignment_mapping = alignment_fs.getmapping()
        shiftmap = Ast.ShiftMap(shift)
        total_map = Ast.CmpMap(alignment_mapping, shiftmap)

        #  Modify the pixel bounds of the output image to take account of this
        #  shift of origin.
        lbnd_out[0] += shift[0]
        lbnd_out[1] += shift[1]
        ubnd_out[0] += shift[0]
        ubnd_out[1] += shift[1]

        #  Get the value used to represent missing pixel values
        if "BLANK" in fitschan_in:
            badval = fitschan_in["BLANK"]
            flags = Ast.USEBAD
        else:
            badval = 0
            flags = 0

        # Resample the data array using the above mapping.
        # total_map was pixmap; is this right?
        (npix, out, out_var) = total_map.resample(lbnd_in, ubnd_in,
                                                  hdu_in.data, None,
                                                  Ast.LINEAR, None, flags,
                                                  0.05, 1000, badval, lbnd_out,
                                                  ubnd_out, lbnd_out, ubnd_out)

        #  Store the aligned data in the primary HDU, and update the NAXISi
        #  keywords to hold the number of pixels along each edge of the
        #  rotated image.
        hdu_in.data = out
        fitschan_in["NAXIS1"] = ubnd_out[0] - lbnd_out[0] + 1
        fitschan_in["NAXIS2"] = ubnd_out[1] - lbnd_out[1] + 1

    #  The WCS to store in the output is the same as the reference WCS
    #  except for the extra shift of origin. So use the above shiftmap to
    #  remap the pixel coordinate frame in the reference WCS FrameSet. We
    #  can then use this FrameSet as the output FrameSet.
    wcsinfo_ref.remapframe(Ast.BASE, shiftmap)

    #  Attempt to write the modified WCS information to the primary HDU (i.e.
    #  convert the FrameSet to a set of FITS header cards stored in the
    #  FITS file). Indicate that we want to use original flavour of FITS-WCS.
    fitschan_in.Encoding = encoding
    fitschan_in.clear('Card')

    if fitschan_in.write(wcsinfo_ref) == 0:
        raise Exception("Failed to convert the aligned WCS to Fits-WCS")

    #  If successful, force the FitsChan to copy its contents into the
    #  fits header, then write the changed data and header to the output
    #  FITS file.
    else:
        fitschan_in.writefits()

    if outname is not None:
        hdu_in.writeto(outname, clobber=clobber)

    return hdu_in
