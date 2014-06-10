"""
Regridding example:

    Take two non-overlapping GRS cubes and extract a chunk that overlaps with
    both

    GRS data are available here: http://grunt.bu.edu/grs-stitch/source/
"""
import numpy as np
import os

from astropy import wcs
from astropy.io import fits

import FITS_tools

if not os.path.exists('grs-48-cube.fits') and os.path.exists('grs-50-cube.fits'):
    raise IOError("The data cubes don't exist.  Either copy them to the present "
                  "directory or download them from http://grunt.bu.edu/grs-stitch/source/."
                  "\nwget http://grunt.bu.edu/grs-stitch/source/grs-50-cube.fits"
                  "\nwget http://grunt.bu.edu/grs-stitch/source/grs-48-cube.fits")

outsize = [100,100,100]

# Create a WCS to project to
# Warning: wcslib automatically converts km/s to m/s, so you need
# to specify units in m/s here
w = wcs.WCS(naxis=3)
w.wcs.ctype = ['GLON-CAR','GLAT-CAR','VRAD']
w.wcs.crval = [49,0,25000]
w.wcs.cdelt = [15/3600., 15/3600., 1000]
w.wcs.crpix = np.array(outsize)/2.+1
w.wcs.cunit = ['deg','deg','m/s']

# WCS objects can't store information about how big the resulting cube/image
# will be, so we need to add that information to the header object
header = w.to_header()
header['NAXIS1'] = 100
header['NAXIS2'] = 100
header['NAXIS3'] = 100

# optionally, one could now change the units specified above to km/s
# header['CDELT3'] = 1.0
# header['CUNIT3'] = 'km/s'
# header['CRVAL3'] = 25

# Extract the regions matching the target header from each cube
im1, im2 = FITS_tools.match_images.match_fits_cubes('grs-48-cube.fits',
                                                    'grs-50-cube.fits', header)

# Merge the two:
im_merged = im1 + im2

hdu = fits.PrimaryHDU(data=im_merged, header=header)
# The header is, unfortunately, created in the wrong order, and there's no
# simple way around that.  astropy will fix the order if told, though
hdu.writeto('grs-subcube-merged.fits', output_verify='fix')

# Just to test that the image-to-image version works:
im1b, im2b = FITS_tools.match_images.match_fits_cubes('grs-48-cube.fits',
                                                      'grs-subcube-merged.fits')

# This will raise an AssertionError if im1 and im1b are not identical
np.testing.assert_array_almost_equal(im1, im1b)
