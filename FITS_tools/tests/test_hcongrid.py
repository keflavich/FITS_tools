import numpy as np
from astropy.io import fits
from astropy.tests.helper import pytest

try:
    import starlink
    HAS_STARLINK = True
except ImportError:
    HAS_STARLINK = False


header1 = """
SIMPLE  =                    T / conforms to FITS standard
BITPIX  =                  -64 / array data type
NAXIS   =                    2 / number of array dimensions
NAXIS1  =                  128
NAXIS2  =                  128
CRVAL1  =                  0.0 / Value at ref. pixel on axis 1
CRVAL2  =                  0.0 / Value at ref. pixel on axis 2
CTYPE1  = 'GLON-CAR'           / Type of co-ordinate on axis 1
CTYPE2  = 'GLAT-CAR'           / Type of co-ordinate on axis 2
CRPIX1  =                 65.0 / Reference pixel on axis 1
CRPIX2  =                 65.0 / Reference pixel on axis 2
CDELT1  =      -0.005555555556 / Pixel size on axis 1
CDELT2  =       0.005555555556 / Pixel size on axis 2
END
""".strip().lstrip()

header2 = """
SIMPLE  =                    T / conforms to FITS standard
BITPIX  =                  -64 / array data type
NAXIS   =                    2 / number of array dimensions
NAXIS1  =                  128
NAXIS2  =                  128
CRVAL1  =        266.416816625 / Value at ref. pixel on axis 1
CRVAL2  =        -29.007824972 / Value at ref. pixel on axis 2
CTYPE1  = 'RA---TAN'           / Type of co-ordinate on axis 1
CTYPE2  = 'DEC--TAN'           / Type of co-ordinate on axis 2
CRPIX1  =                 65.0 / Reference pixel on axis 1
CRPIX2  =                 65.0 / Reference pixel on axis 2
CDELT1  =      -0.005555555556 / Pixel size on axis 1
CDELT2  =       0.005555555556 / Pixel size on axis 2
END
""".strip().lstrip()

header3 = """
SIMPLE  =                    T / conforms to FITS standard
BITPIX  =                  -64 / array data type
NAXIS   =                    2 / number of array dimensions
NAXIS1  =                  128
NAXIS2  =                  128
CRVAL1  =        266.416816625 / Value at ref. pixel on axis 1
CRVAL2  =        -29.007824972 / Value at ref. pixel on axis 2
CTYPE1  = 'RA---TAN'           / Type of co-ordinate on axis 1
CTYPE2  = 'DEC--TAN'           / Type of co-ordinate on axis 2
CRPIX1  =                 65.0 / Reference pixel on axis 1
CRPIX2  =                 65.0 / Reference pixel on axis 2
CDELT1  =             -0.00225 / Pixel size on axis 1
CDELT2  =              0.00225 / Pixel size on axis 2
END
""".strip().lstrip()

from ..hcongrid import wcsalign


@pytest.mark.skipif('not HAS_STARLINK')
@pytest.mark.parametrize(('h1', 'h2'), zip((header1, header2, header3),
                                           (header1, header2, header3)))
def test_wcsalign_gaussian_smallerpix(h1, h2):
    """
    Reproject different coordinate systems
    """

    x,y = np.mgrid[:128, :128]
    r = ((x-63.5)**2 + (y-63.5)**2)**0.5
    e = np.exp(-r**2/(2.*10.**2))

    hdr1 = fits.Header().fromstring(h1, '\n')
    hdu_in = fits.PrimaryHDU(data=e, header=hdr1)
    hdr2 = fits.Header().fromstring(h2, '\n')

    hdu_out = wcsalign(hdu_in, hdr2)

    return hdu_out


@pytest.mark.skipif('not HAS_STARLINK')
def test_hcongrid_gaussian_smallerpix():
    """
    Reproject RA/Dec -> RA/Dec with smaller pixels
    """

    x,y = np.mgrid[:128, :128]
    r = ((x-63.5)**2 + (y-63.5)**2)**0.5
    e = np.exp(-r**2/(2.*10.**2))

    hdr1 = fits.Header().fromstring(header2, '\n')
    hdu_in = fits.PrimaryHDU(data=e, header=hdr1)
    hdr2 = fits.Header().fromstring(header3, '\n')

    hdu_out = wcsalign(hdu_in, hdr2)

    return hdu_out
