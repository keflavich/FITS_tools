import astropy.io.fits as pyfits

def flatten_header(header,delete=False):
    """
    Attempt to turn an N-dimensional fits header into a 2-dimensional header
    Turns all CRPIX[>2] etc. into new keywords with prefix 'A'

    header must be a `~astropy.io.fits.Header` instance
    """

    # TODO: Determine bad axes by examining CTYPE and excise them

    if not isinstance(header,pyfits.Header):
        raise Exception("flatten_header requires a pyfits.Header instance")

    newheader = header.copy()

    for key in newheader.keys():
        try:
            if delete and int(key[-1]) >= 3 and key[:2] in ['CD','CR','CT','CU','NA']:
                newheader.pop(key)
            elif (int(key[-1]) >= 3 or int(key[2])>=3) and key[:2] in ['CD','CR','CT','CU','NA','PC']:
                newheader.rename_keyword(key,'A'+key,force=True)
            if delete and (int(key[3]) >= 3 or int(key[7]) >= 3) and key[:2]=='PC' and key in newheader:
                # for PC03_04 etc
                newheader.pop(key)
            elif (int(key[3]) >= 3 or int(key[7]) >= 3) and key[:2]=='PC' and key in newheader:
                newheader.rename_keyword(key,'A'+key[1:],force=True)
        except ValueError:
            # if key[-1] is not an int
            pass
        except IndexError:
            # if len(key) < 2
            pass
    newheader['NAXIS'] = 2
    if header.get('WCSAXES'):
        newheader['WCSAXES'] = 2

    return newheader

def speccen_header(header,lon=None,lat=None):
    """
    Turn a cube header into a spectrum header, retaining RA/Dec vals where possible
    (speccen is like flatten; spec-ify would be better but, specify?  nah)

    Assumes 3rd axis is velocity
    """
    newheader = header.copy()
    newheader.set('CRVAL1',header.get('CRVAL3'))
    newheader.set('CRPIX1',header.get('CRPIX3'))
    if 'CD1_1' in header: newheader.rename_keyword('CD1_1','OLDCD1_1')
    elif 'CDELT1' in header: newheader.rename_keyword('CDELT1','OLDCDEL1')
    if 'CD3_3' in header: newheader.set('CDELT1',header.get('CD3_3'))
    elif 'CDELT3' in header: newheader.set('CDELT1',header.get('CDELT3'))
    newheader.set('CTYPE1','VRAD')
    if header.get('CUNIT3'): newheader.set('CUNIT1',header.get('CUNIT3'))
    else: 
        print "Assuming CUNIT3 is km/s in speccen_header"
        newheader.set('CUNIT1','km/s')
    newheader.set('CRPIX2',1)
    newheader.set('CTYPE2','RA---TAN')
    newheader.set('CRPIX3',1)
    newheader.set('CTYPE3','DEC--TAN')

    if lon is not None: newheader.set('CRVAL2',lon)
    if lat is not None: newheader.set('CRVAL3',lat)

    if 'CD2_2' in header: newheader.rename_keyword('CD2_2','OLDCD2_2')
    if 'CD3_3' in header: newheader.rename_keyword('CD3_3','OLDCD3_3')

    return newheader
