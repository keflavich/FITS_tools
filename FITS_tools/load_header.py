from astropy.io import fits

def load_header(header):
    """
    Attempt to load a header specified as a header, a string pointing to a FITS
    file, or a string pointing to a Header text file, or a string that contains
    the actual header
    """
    if hasattr(header,'get'):
        return fits.Header(header)
    try:
        # assume fits file first
        return fits.getheader(header)
    except IOError:
        # assume header textfile
        try:
            return fits.Header().fromtextfile(header)
        except IOError:
            return fits.Header().fromstring(header,'\n')
