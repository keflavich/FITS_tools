Tools
=====

Image Regridding
----------------

`FITS_tools.hcongrid.hcongrid` is meant to replicate `hcongrid
<http://idlastro.gsfc.nasa.gov/ftp/pro/astrom/hcongrid.pro>`_ and `hastrom
<http://idlastro.gsfc.nasa.gov/ftp/pro/astrom/hastrom.pro>`_.  It uses scipy's
interpolation routines.

`FITS_tools.hcongrid.wcsalign` does the same thing as `hcongrid` but
uses `pyast <https://pypi.python.org/pypi/starlink-pyast/>`_ as its
backend.

Cube Regridding
---------------

`FITS_tools.cube_regrid.regrid_fits_cube` reprojects a cube to a new
grid using scipy's interpolation routines.

`FITS_tools.match_images.match_fits_cubes` takes two cubes, and reprojects the
first to the coordinates of the second (it's a wrapper)

For a flux-conserving (but slower) approach, there is a `wrapper of montage
<http://montage-wrapper.readthedocs.org/en/v0.9.5/_generated/montage_wrapper.wrappers.reproject_cube.html>`_
in `python-montage <http://www.astropy.org/montage-wrapper/>`_.

Reference/API
=============

.. automodapi:: FITS_tools
    :no-inheritance-diagram:

.. automodapi:: FITS_tools.cube_regrid
    :no-inheritance-diagram:
    
.. automodapi:: FITS_tools.downsample
    :no-inheritance-diagram:

.. automodapi:: FITS_tools.spectral_regrid
    :no-inheritance-diagram:

.. automodapi:: FITS_tools.hcongrid
    :no-inheritance-diagram:

.. automodapi:: FITS_tools.match_images
    :no-inheritance-diagram:

.. automodapi:: FITS_tools.strip_headers
    :no-inheritance-diagram:

.. automodapi:: FITS_tools.load_header
    :no-inheritance-diagram:

.. automodapi:: FITS_tools.header_tools
    :no-inheritance-diagram:
