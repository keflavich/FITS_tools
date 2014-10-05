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

    import os
    from warnings import warn
    from astropy import config

    # add these here so we only need to cleanup the namespace at the end
    config_dir = None

    if not os.environ.get('ASTROPY_SKIP_CONFIG_UPDATE', False):
        config_dir = os.path.dirname(__file__)
        try:
            config.configuration.update_default_config(__package__, config_dir)
        except config.configuration.ConfigurationDefaultMissingError as e:
            wmsg = (e.args[0] + " Cannot install default profile. If you are "
                    "importing from source, this is expected.")
            warn(config.configuration.ConfigurationDefaultMissingWarning(wmsg))
            del e

    del os, warn, config_dir  # clean up namespace

    from match_images import project_to_header,match_fits
    from fits_overlap import fits_overlap,header_overlap
    import hcongrid,spectral_regrid
    from cube_regrid import regrid_fits_cube,regrid_cube_hdu,regrid_cube
    import header_tools
    import spectral_regrid

