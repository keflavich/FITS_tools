import numpy as np
from astropy.io import fits
from astropy.tests.helper import pytest
import warnings
warnings.filterwarnings(action='error', category=DeprecationWarning)

header1 = """
SIMPLE  =                    T / conforms to FITS standard
BITPIX  =                  -64 / array data type
NAXIS   =                    3 / number of array dimensions
NAXIS1  =                   64
NAXIS2  =                   64
NAXIS3  =                   64
EQUINOX = 2000.0
CRVAL1  =                  0.0 / Value at ref. pixel on axis 1
CRVAL2  =                  0.0 / Value at ref. pixel on axis 2
CTYPE1  = 'GLON-CAR'           / Type of co-ordinate on axis 1
CTYPE2  = 'GLAT-CAR'           / Type of co-ordinate on axis 2
CRPIX1  =                 33.0 / Reference pixel on axis 1
CRPIX2  =                 33.0 / Reference pixel on axis 2
CDELT1  =      -0.005555555556 / Pixel size on axis 1
CDELT2  =       0.005555555556 / Pixel size on axis 2
CTYPE3  = 'VELO-LSR'
CDELT3  = 1.0
CRPIX3  = 33.0
CRVAL3  = 0.0
CUNIT3  = 'km/s'
SPECSYS = 'LSRK'
END
""".strip().lstrip()

header2 = """
SIMPLE  =                    T / conforms to FITS standard
BITPIX  =                  -64 / array data type
NAXIS   =                    3 / number of array dimensions
NAXIS1  =                   65
NAXIS2  =                   65
NAXIS3  =                   65
EQUINOX = 2000.0
CRVAL1  =        266.416816625 / Value at ref. pixel on axis 1
CRVAL2  =        -29.007824972 / Value at ref. pixel on axis 2
CTYPE1  = 'RA---TAN'           / Type of co-ordinate on axis 1
CTYPE2  = 'DEC--TAN'           / Type of co-ordinate on axis 2
CRPIX1  =                 33.0 / Reference pixel on axis 1
CRPIX2  =                 33.0 / Reference pixel on axis 2
CDELT1  =      -0.005555555556 / Pixel size on axis 1
CDELT2  =       0.005555555556 / Pixel size on axis 2
CTYPE3  = 'VELO-LSR'
CDELT3  = 0.5
CRPIX3  = 33.0
CRVAL3  = 0.0
CUNIT3  = 'km/s'
SPECSYS = 'LSRK'
END
""".strip().lstrip()

header3 = """
SIMPLE  =                    T / conforms to FITS standard
BITPIX  =                  -64 / array data type
NAXIS   =                    3 / number of array dimensions
NAXIS1  =                   63
NAXIS2  =                   63
NAXIS3  =                   63
EQUINOX = 2000.0
CRVAL1  =        266.416816625 / Value at ref. pixel on axis 1
CRVAL2  =        -29.007824972 / Value at ref. pixel on axis 2
CTYPE1  = 'RA---TAN'           / Type of co-ordinate on axis 1
CTYPE2  = 'DEC--TAN'           / Type of co-ordinate on axis 2
CRPIX1  =                 33.0 / Reference pixel on axis 1
CRPIX2  =                 33.0 / Reference pixel on axis 2
CDELT1  =             -0.00225 / Pixel size on axis 1
CDELT2  =              0.00225 / Pixel size on axis 2
CTYPE3  = 'VELO-LSR'
CDELT3  = 0.25
CRPIX3  = 33.0
CRVAL3  = 0.0
CUNIT3  = 'km/s'
SPECSYS = 'LSRK'
END
""".strip().lstrip()

from ..cube_regrid import regrid_cube

@pytest.mark.parametrize(('h1','h2'),zip((header1,header2,header3),(header2,header3,header1)))
def test_wcsalign_gaussian_smallerpix(h1,h2):
    """
    Reproject different coordinate systems
    """

    hdr1 = fits.Header().fromstring(h1,'\n')
    nax1,nax2,nax3 = [hdr1['NAXIS%i' % n] for n in range(1,4)]

    x,y,z = np.mgrid[:nax1,:nax2,:nax3]
    r2 = ((x-(nax1-1)/2.)**2 + (y-(nax2-1)/2.)**2 + (z-(nax3-1)/2.)**2)
    e = np.exp(-r2/(2.*10.**2))

    hdu_in = fits.PrimaryHDU(data=e, header=hdr1)
    hdr2 = fits.Header().fromstring(h2,'\n')

    hdu_out = regrid_cube(hdu_in.data, hdu_in.header, hdr2)

    return hdu_out
