# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This is an Astropy affiliated package.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from .match_images import project_to_header,match_fits
    from .fits_overlap import fits_overlap,header_overlap
    from . import hcongrid,spectral_regrid
    from .cube_regrid import regrid_fits_cube,regrid_cube_hdu,regrid_cube
    from . import header_tools
    from . import spectral_regrid
