import numpy as np
import fitsio
import astropy.io.fits as fits
from astropy.table import Table


def getGalaxies(minGR, maxGR, deepFiles, measBalrFile, balrFile, writeFile):
    '''
    This method serves to get and store the necessary balrog galaxy data for further use. It checks
    which balrog objects were originally deep field galaxies according to the KNN classification and
    then stores their necessary data. All files need to be .fits files.
    '''
    
    # All I need from the deep fields is the ID numbers and original KNN classification.
    deepCols  = ['KNN_CLASS', 'ID']
    deepID = []
    deepClass= []
    
    for deepFile in deepFiles:
        deepData = fitsio.read(deepFile, columns = deepCols)
        deepID.extend(deepData['ID'])
        deepClass.extend(deepData['KNN_CLASS'])
    
    deepID = np.array(deepID)
    deepClass = np.array(deepClass)
    
    # This serves to make it easier to check the original classification of an object.
    # This way I can simply check the classification by indexing to an ID number minus
    # the minimum ID number to find the classification. This prevented having an overly
    # large array but still has the speed advantage of indexing.
    minID = np.min(deepID)
    deepGalID = np.zeros(np.max(deepID) - minID + 1)
    deepGalID[deepID - minID] = deepClass
    
    measBalrData = fits.open(measBalrFile)[1].data
    
    # This will be used to match galaxies to their deep field counterparts.
    ID = measBalrData['true_id']
    
    # This is the Balrog Object ID
    BALR_ID = measBalrData['bal_id']
    
    # These are some of the data points I will be storing for valid data.
    RA = measBalrData['meas_ra']
    DEC = measBalrData['meas_dec']
    CLASS = measBalrData['meas_EXTENDED_CLASS_SOF']
    GMAG_CM = measBalrData['meas_cm_mag'][:,0]
    RMAG_CM = measBalrData['meas_cm_mag'][:,1]
    SIZE = measBalrData['meas_cm_T']
    
    sortInds = BALR_ID.argsort()
    BALR_ID = BALR_ID[sortInds[::1]]
    ID = ID[sortInds[::1]]
    RA = RA[sortInds[::1]]
    DEC = DEC[sortInds[::1]]
    GMAG_CM = GMAG_CM[sortInds[::1]]
    RMAG_CM = RMAG_CM[sortInds[::1]]
    CLASS = CLASS[sortInds[::1]]
    SIZE = SIZE[sortInds[::1]]
    
    # This is from the detected dataset
    balrData = fits.open(balrFile)[1].data
    
    # This is the Balrog Object ID for all data, not just matches
    FLAG_ID = measBalrData['bal_id']

    # These are the data values useful for cropping the data.
    FOREGROUND = balrData['flags_foreground']
    BADREGIONS = balrData['flags_badregions']
    FOOTPRINT = balrData['flags_footprint']
    ARCSECONDS = balrData['match_flag_1.5_asec']
    
    sortInds = FLAG_ID.argsort()
    FLAG_ID = FLAG_ID[sortInds[::1]]
    FOREGROUND = FOREGROUND[sortInds[::1]]
    BADREGIONS = BADREGIONS[sortInds[::1]]
    FOOTPRINT = FOOTPRINT[sortInds[::1]]
    ARCSECONDS = ARCSECONDS[sortInds[::1]]
    
    # This will serve to align the flags with their measured counterpart.
    # Try sorting both with an argsort and then running this method.
    
    parallelInd = 0
    cropInds = []
    for i in np.arange(len(FLAG_ID)):
        if parallelInd >= len(BALR_ID):
            break
        if BALR_ID[parallelInd] == FLAG_ID[i]:
            cropInds.append(i)
            parallelInd += 1
            
    FOREGROUND = FOREGROUND[cropInds]
    BADREGIONS = BADREGIONS[cropInds]
    FOOTPRINT = FOOTPRINT[cropInds]
    ARCSECONDS = ARCSECONDS[cropInds]

    cutIndices = np.where(# Quality cuts
                          (FOREGROUND == 0) & 
                          (BADREGIONS < 2) & 
                          (FOOTPRINT == 1) & 
                          (ARCSECONDS < 2) &
                          (CLASS >= 0) &
                          # Color cuts
                          (GMAG_CM - RMAG_CM >= minGR) &
                          (GMAG_CM - RMAG_CM <= maxGR))[0]

    ID = ID[cutIndices]
    RA = RA[cutIndices]
    DEC = DEC[cutIndices]
    GMAG_CM = GMAG_CM[cutIndices]
    RMAG_CM = RMAG_CM[cutIndices]
    CLASS = CLASS[cutIndices]
    SIZE = SIZE[cutIndices]

    # This makes it so I only store objects that were initially labeled as galaxies in the deep fields.
    # It also performs some cuts on extreme values.
    trueClass = deepGalID[ID - minID]
    useInds = np.where((trueClass == 1) & 
                       (GMAG_CM < 37) &
                       (RMAG_CM < 37))[0]

    RA = RA[useInds]
    DEC = DEC[useInds]
    GMAG_CM = GMAG_CM[useInds]
    RMAG_CM = RMAG_CM[useInds]
    CLASS = CLASS[useInds]
    SIZE = SIZE[useInds]
    
    my_table = Table()
    my_table['RA'] = RA
    my_table['DEC'] = DEC
    my_table['GMAG'] = GMAG_CM
    my_table['RMAG'] = RMAG_CM
    my_table['CLASS'] = CLASS
    my_table['SIZE'] = SIZE
    my_table.write(writeFile, overwrite = True)
    

def getMatStars(minGR, maxGR, measStarFile, starFile, writeFile):
    '''
    This method is used to get and store the necessary balrog delta star information for later use.
    It applies basic quality cuts to the data and then stores it. Both files given need to be .fits files.
    This method only looks at stars that were detected so that a measured magnitude exists. 
    '''
    
    # This reads in all of the data. Most of these are just flags, but the g magnitude, ra, dec, detected
    # flag, and others will be stored for later use.
    measStarData = fitsio.read(measStarFile, columns = ['meas_ra', 'meas_dec', 'meas_EXTENDED_CLASS_SOF',
                                                'meas_psf_mag', 'bal_id', 'meas_cm_T'])
    
    starData = fitsio.read(starFile, columns = ['bal_id', 'flags_foreground', 'flags_badregions', 
                                                'flags_footprint', 'match_flag_1.5_asec'])
    

    RA = measStarData['meas_ra']
    DEC = measStarData['meas_dec']
    # PSF Magnitudes
    GMAG_PSF = measStarData['meas_psf_mag'][:,0]
    RMAG_PSF = measStarData['meas_psf_mag'][:,1]
    # This is the class that the object was measured as.
    CLASS = measStarData['meas_EXTENDED_CLASS_SOF']
    # This is the ID from the measured catalog.
    MEAS_ID  = measStarData['bal_id']
    SIZE = measStarData['meas_cm_T']
    
    sortInds = MEAS_ID.argsort()
    MEAS_ID = MEAS_ID[sortInds[::1]]
    RA = RA[sortInds[::1]]
    DEC = DEC[sortInds[::1]]
    GMAG_PSF = GMAG_PSF[sortInds[::1]]
    RMAG_PSF = RMAG_PSF[sortInds[::1]]
    CLASS = CLASS[sortInds[::1]]
    SIZE = SIZE[sortInds[::1]]
    
    # Everything from here on out is simply used in order to filter the data
    FOREGROUND = starData['flags_foreground']
    BADREGIONS = starData['flags_badregions']
    FOOTPRINT = starData['flags_footprint']
    ARCSECONDS = starData['match_flag_1.5_asec']
    FLAG_ID = starData['bal_id']
    
    sortInds = FLAG_ID.argsort()
    FLAG_ID = FLAG_ID[sortInds[::1]]
    FOREGROUND = FOREGROUND[sortInds[::1]]
    BADREGIONS = BADREGIONS[sortInds[::1]]
    FOOTPRINT = FOOTPRINT[sortInds[::1]]
    ARCSECONDS = ARCSECONDS[sortInds[::1]]
    
    # This will serve to align the flags with their measured counterpart.
    parallelInd = 0
    cropInds = []
    for i in np.arange(len(FLAG_ID)):
        if parallelInd >= len(MEAS_ID):
            break
        if MEAS_ID[parallelInd] == FLAG_ID[i]:
            cropInds.append(i)
            parallelInd += 1
            
    FOREGROUND = FOREGROUND[cropInds]
    BADREGIONS = BADREGIONS[cropInds]
    FOOTPRINT = FOOTPRINT[cropInds]
    ARCSECONDS = ARCSECONDS[cropInds]

    # This is used to filter out any injections that had flags raised.
    cutIndices = np.where((FOREGROUND == 0) & 
                          (BADREGIONS < 2) & 
                          (FOOTPRINT == 1) & 
                          (ARCSECONDS < 2) &
                          (CLASS >= 0) &
                          # Color cuts
                          (GMAG_PSF - RMAG_PSF >= minGR) &
                          (GMAG_PSF - RMAG_PSF <= maxGR))[0]

    # This reduced the data down to the actually valid pixels.
    RA = RA[cutIndices]
    DEC = DEC[cutIndices]
    GMAG_PSF = GMAG_PSF[cutIndices]
    RMAG_PSF = RMAG_PSF[cutIndices]
    CLASS = CLASS[cutIndices]
    SIZE = SIZE[cutIndices]
    
    useInds = np.where((GMAG_PSF < 37) & 
                       (RMAG_PSF < 37))[0]

    RA = RA[useInds]
    DEC = DEC[useInds]
    GMAG_PSF = GMAG_PSF[useInds]
    RMAG_PSF = RMAG_PSF[useInds]
    CLASS = CLASS[useInds]
    SIZE = SIZE[useInds]
    
    my_table = Table()
    my_table['RA'] = RA
    my_table['DEC'] = DEC
    my_table['GMAG'] = GMAG_PSF
    my_table['RMAG'] = RMAG_PSF
    my_table['CLASS'] = CLASS
    my_table['SIZE'] = SIZE
    my_table.write(writeFile, overwrite = True)
    
    
def getDetStar(minGR, maxGR, starFile, writeFile):
    '''
    This method is used to get and store the necessary balrog delta star information for later use.
    It applies basic quality cuts to the data and then stores it. Both files given need to be .fits files.
    '''
    
    # This reads in all of the data. Most of these are just flags, but the g magnitude, ra, dec, and detected
    # flag will be stored for later use..
    starData = fitsio.read(starFile, columns = ['detected', 'true_ra', 'true_dec',
                                                'flags_foreground', 'flags_badregions', 'flags_footprint',
                                                'match_flag_1.5_asec', 'true_g_Corr', 'true_gr_Corr'])

    RA = starData['true_ra']
    DEC = starData['true_dec']
    # This is used for detection rates, each point is either a 0 (no detection) or a 1 (detection)
    DETECTED = starData['detected']
    # Everything from here on out is simply used in order to filter the data
    FOREGROUND = starData['flags_foreground']
    BADREGIONS = starData['flags_badregions']
    FOOTPRINT = starData['flags_footprint']
    ARCSECONDS = starData['match_flag_1.5_asec']
    # Magnitudes are used for color cuts.
    GMAG = starData['true_g_Corr']
    RMAG = starData['true_g_Corr'] - starData['true_gr_Corr']

    # This is used to filter out any injections that had flags raised.
    cutIndices = np.where((FOREGROUND == 0) & 
                          (BADREGIONS < 2) & 
                          (FOOTPRINT == 1) & 
                          (ARCSECONDS < 2) &
                          # Color cuts
                          (GMAG - RMAG >= minGR) &
                          (GMAG - RMAG <= maxGR) &
                          (GMAG < 37) &
                          (RMAG < 37))[0]

    # This reduced the data down to the actually valid pixels.
    DETECTED = DETECTED[cutIndices]
    RA = RA[cutIndices]
    DEC = DEC[cutIndices]
    GMAG = GMAG[cutIndices]
    
    my_table = Table()
    my_table['RA'] = RA
    my_table['DEC'] = DEC
    my_table['GMAG'] = GMAG
    my_table['DETECTED'] = DETECTED
    my_table.write(writeFile, overwrite = True)
