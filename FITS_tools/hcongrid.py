import numpy as np
try:
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
except ImportError:
    import pyfits
    import pywcs
try:
    import scipy.ndimage

    def hcongrid(image, header1, header2, preserve_bad_pixels=True, **kwargs):
        """
        Interpolate an image from one FITS header onto another

        kwargs will be passed to `scipy.ndimage.map_coordinates`

        Parameters
        ----------
        image : ndarray
            A two-dimensional image 
        header1 : `pyfits.Header` or `pywcs.WCS`
            The header or WCS corresponding to the image
        header2 : `pyfits.Header` or `pywcs.WCS`
            The header or WCS to interpolate onto
        preserve_bad_pixels: bool
            Try to set NAN pixels to NAN in the zoomed image.  Otherwise, bad
            pixels will be set to zero

        Returns
        -------
        ndarray with shape defined by header2's naxis1/naxis2

        Raises
        ------
        TypeError if either is not a Header or WCS instance
        Exception if image1's shape doesn't match header1's naxis1/naxis2

        Examples
        --------
        (not written with >>> because test.fits/test2.fits do not exist)
        fits1 = pyfits.open('test.fits')
        target_header = pyfits.getheader('test2.fits')
        new_image = hcongrid(fits1[0].data, fits1[0].header, target_header)

        """

        if issubclass(pywcs.WCS, header1.__class__):
            wcs1 = header1
        else:
            try:
                wcs1 = pywcs.WCS(header1)
            except:
                raise TypeError("Header1 must either be a pyfits.Header or pywcs.WCS instance")

        if not (wcs1.naxis1 == image.shape[1] and wcs1.naxis2 == image.shape[0]):
            raise Exception("Image shape must match header shape.")

        if issubclass(pywcs.WCS, header2.__class__):
            wcs2 = header2
        else:
            try:
                wcs2 = pywcs.WCS(header2)
            except:
                raise TypeError("Header2 must either be a pyfits.Header or pywcs.WCS instance")

        if not all([w1==w2 for w1,w2 in zip(wcs1.wcs.ctype,wcs2.wcs.ctype)]):
            # do unit conversions
            raise NotImplementedError("Unit conversions have not yet been implemented.")

        # sigh... why does numpy use matrix convention?  Makes everything so much harder...
        outshape = [wcs2.naxis2,wcs2.naxis1]
        yy2,xx2 = np.indices(outshape)
        lon2,lat2 = wcs2.wcs_pix2sky(xx2, yy2, 0)
        xx1,yy1 = wcs1.wcs_sky2pix(lon2, lat2, 0)
        grid1 = np.array([yy1.reshape(outshape),xx1.reshape(outshape)])

        bad_pixels = np.isnan(image) + np.isinf(image)

        image[bad_pixels] = 0

        newimage = scipy.ndimage.map_coordinates(image, grid1, **kwargs)

        if preserve_bad_pixels:
            newbad = scipy.ndimage.map_coordinates(bad_pixels, grid1, order=0, mode='constant', cval=np.nan)
            newimage[newbad] = np.nan
        
        return newimage

    def zoom_fits(fitsfile, scalefactor, preserve_bad_pixels=True, **kwargs):
        """
        Zoom in on a FITS image by interpolating using scipy.ndimage.zoom

        Parameters
        ----------
        fitsfile: str
            FITS file name
        scalefactor: float
            Zoom factor along all axes
        preserve_bad_pixels: bool
            Try to set NAN pixels to NAN in the zoomed image.  Otherwise, bad
            pixels will be set to zero
        """

        arr = pyfits.getdata(fitsfile)
        h = pyfits.getheader(fitsfile)

        h['CRPIX1'] = (h['CRPIX1']-1)*scalefactor + scalefactor/2. + 0.5
        h['CRPIX2'] = (h['CRPIX2']-1)*scalefactor + scalefactor/2. + 0.5
        if 'CD1_1' in h:
            for ii in (1,2):
                for jj in (1,2):
                    k = "CD%i_%i" % (ii,jj)
                    if k in h: # allow for CD1_1 but not CD1_2
                        h[k] = h[k]/scalefactor
        elif 'CDELT1' in h:
            h['CDELT1'] = h['CDELT1']/scalefactor
            h['CDELT2'] = h['CDELT2']/scalefactor

        bad_pixels = np.isnan(arr) + np.isinf(arr)

        arr[bad_pixels] = 0

        upscaled = scipy.ndimage.zoom(arr,scalefactor,**kwargs)

        if preserve_bad_pixels:
            bp_up = scipy.ndimage.zoom(bad_pixels,scalefactor,mode='constant',cval=np.nan,order=0)
            upscaled[bp_up] = np.nan

        up_hdu = pyfits.PrimaryHDU(data=upscaled, header=h)
        
        return up_hdu

except ImportError:
    # needed to do this to get travis-ci tests to pass, even though scipy is installed...
    def hcongrid(*args, **kwargs):
        raise ImportError("scipy.ndimage could not be imported; hcongrid is not available")

def wcsalign(hdu_in, header, outname=None):
    """
    Align one FITS image to a specified header
    """
    import starlink.Ast as Ast
    import starlink.Atl as Atl
  
    #  Create objects that will transfer FITS header cards between an AST
    #  FitsChan and the fits header describing the primary HDU of the
    #  supplied FITS file.
    adapter_in = Atl.PyFITSAdapter(hdu_in)
    hdu_ref = pyfits.PrimaryHDU(header=header)
    adapter_ref = Atl.PyFITSAdapter(hdu_ref)
     
    #  Create a FitsChan for each and use the above adapters to copy all the header
    #  cards into it.
    fitschan_in = Ast.FitsChan(adapter_in, adapter_in)
    fitschan_ref = Ast.FitsChan(adapter_ref, adapter_ref)
     
    #  Get the flavour of FITS-WCS used by the header cards currently in the
    #  input FITS file. This is so that we can use the same flavour when we write
    #  out the modified WCS.
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
        raise ValueError("Failed to read WCS information from {0}".format(hdu_in))
     
    #  This is restricted to 2D arrays, so check theinput  FITS file has 2 pixel
    #  axes (given by Nin) and 2 WCS axes (given by Nout).
    elif wcsinfo_in.Nin != 2 or wcsinfo_in.Nout != 2:
        raise ValueError("{0} is not 2-dimensional".format(hdu_in))
     
    #  Check the reference FITS file in the same way.
    elif wcsinfo_ref is None:
        raise ValueError("Failed to read WCS information from {0}".format(hdu_ref))
     
    elif wcsinfo_ref.Nin != 2 or wcsinfo_ref.Nout != 2:
        raise ValueError("{0} is not 2-dimensional".format(hdu_ref))
     
    #  Proceed if the WCS information was read OK.
 
    #  Attempt to get a mapping from pixel coords in the input FITS file to pixel
    #  coords in the reference fits file, with alignment occuring by preference in
    #  the current WCS frame. Since the pixel coordinate frame will be the base frame
    #  in each Frameset, we first invert the FrameSets. This is because the Convert method
    #  aligns current Frames, not base frames.
    wcsinfo_in.invert()
    wcsinfo_ref.invert()
    alignment_fs = wcsinfo_in.convert(wcsinfo_ref)

    #  Invert them again to put them back to their original state (i.e.
    #  base frame = pixel coords, and current Frame = WCS coords).
    wcsinfo_in.invert()
    wcsinfo_ref.invert()

    #  Check alignment was possible.
    if alignment_fs is None:
        raise Exception("Cannot find a common coordinate system shared by {0} and {1}".format(hdu_in,hdu_ref))
      
    else:
        #  Get the lower and upper bounds of the input image in pixel indices.
        #  All FITS arrays by definition have lower pixel bounds of [1,1] (unlike
        #  NDFs). Note, unlike fits AST uses FITS ordering for storing pixel axis
        #  values in an array (i.e. NAXIS1 first, NAXIS2 second, etc).
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
                                                  hdu_in[0].data, None,
                                                  Ast.LINEAR, None, flags,
                                                  0.05, 1000, badval, lbnd_out,
                                                  ubnd_out, lbnd_out, ubnd_out)
 
        #  Store the aligned data in the primary HDU, and update the NAXISi keywords
        #  to hold the number of pixels along each edge of the rotated image.
        hdu_in[0].data = out
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
        hdu_in.writeto(outname, clobber=True)
    
    return hdu_in
