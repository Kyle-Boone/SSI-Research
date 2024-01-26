# Four functions:
# First: Get all data for matched delta stars.
# Second: Get all positions for injected delta stars that pass quality cuts.
# Third: Get all data for matched balrog galaxies.
# Fourth: Get all positions for injected galaxies that pass quality cuts.

# Cuts: valid measured class, valid pixel, flag cuts, isochrone cut.

import numpy as np
import fitsio
import healpy as hp
from astropy.table import Table
from matplotlib.path import Path


def getMatStars(path, mu, matStarFile, detStarFile, validPixFile, writeFile, gCut, classCutoff):
    '''
    This method is used to get and store the necessary balrog delta star information for later use.
    It applies basic quality cuts to the data and then stores it. Both files given need to be .fits files.
    This method only looks at stars that were detected so that a measured magnitude exists. These first two
    files should be for balrog objects.
    '''
    
    # This reads in all of the data. Most of these are just flags, but the g magnitude, ra, dec, detected
    # flag, and others will be stored for later use.
    matStarData = fitsio.read(matStarFile, columns = ['true_RA_new', 'true_DEC_new', 
                                                        'meas_EXTENDED_CLASS_SOF',
                                                        'meas_psf_mag', 'meas_cm_mag', 'bal_id'])
    
    detStarData = fitsio.read(detStarFile, columns = ['bal_id', 'flags_foreground', 'flags_badregions', 
                                                'flags_footprint', 'match_flag_1.5_asec'])
    

    RA = matStarData['true_RA_new']
    DEC = matStarData['true_DEC_new']
    # PSF Magnitudes
    GMAG_PSF = matStarData['meas_psf_mag'][:,0]
    RMAG_PSF = matStarData['meas_psf_mag'][:,1]
    
    GMAG_CM = matStarData['meas_cm_mag'][:,0]
    RMAG_CM = matStarData['meas_cm_mag'][:,1]
    # This is the class that the object was measured as.
    CLASS = matStarData['meas_EXTENDED_CLASS_SOF']
    # This is the ID from the measured catalog.
    MAT_ID  = matStarData['bal_id']
    
    GMAG = np.copy(GMAG_PSF)
    RMAG = np.copy(RMAG_PSF)
    
    GMAG[np.where((CLASS >= classCutoff) & (CLASS <= 3))] = GMAG_CM[np.where((CLASS >= classCutoff) & (CLASS <= 3))]
    RMAG[np.where((CLASS >= classCutoff) & (CLASS <= 3))] = RMAG_CM[np.where((CLASS >= classCutoff) & (CLASS <= 3))]
    
    sortInds = MAT_ID.argsort()
    MAT_ID = MAT_ID[sortInds[::1]]
    RA = RA[sortInds[::1]]
    DEC = DEC[sortInds[::1]]
    GMAG = GMAG[sortInds[::1]]
    RMAG = RMAG[sortInds[::1]]
    CLASS = CLASS[sortInds[::1]]
    
    # Everything from here on out is simply used in order to filter the data
    FOREGROUND = detStarData['flags_foreground']
    BADREGIONS = detStarData['flags_badregions']
    FOOTPRINT = detStarData['flags_footprint']
    ARCSECONDS = detStarData['match_flag_1.5_asec']
    FLAG_ID = detStarData['bal_id']
    
    # This cut shouldn't really cut anything. It's more of a sanity thing.
    sanityCut = np.isin(MAT_ID, FLAG_ID)
    MAT_ID = MAT_ID[sanityCut]
    RA = RA[sanityCut]
    DEC = DEC[sanityCut]
    GMAG = GMAG[sanityCut]
    RMAG = RMAG[sanityCut]
    CLASS = CLASS[sanityCut]
    
    sortInds = FLAG_ID.argsort()
    FLAG_ID = FLAG_ID[sortInds[::1]]
    FOREGROUND = FOREGROUND[sortInds[::1]]
    BADREGIONS = BADREGIONS[sortInds[::1]]
    FOOTPRINT = FOOTPRINT[sortInds[::1]]
    ARCSECONDS = ARCSECONDS[sortInds[::1]]
    
    # This will serve to align the flags with their measured counterpart.
    
    cropInds = np.isin(FLAG_ID, MAT_ID)
            
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
                          (CLASS <= 3) &
                          # Maximum Magnitude Cut
                          (GMAG < gCut))[0]

    # This reduced the data down to the actually valid pixels.
    RA = RA[cutIndices]
    DEC = DEC[cutIndices]
    GMAG = GMAG[cutIndices]
    RMAG = RMAG[cutIndices]
    CLASS = CLASS[cutIndices]
    MG = GMAG - mu
    GR = GMAG - RMAG
    
    # Isochrone Cut
    filterSelection=Path.contains_points(path,np.vstack((GR,MG)).T)
    
    RA = RA[filterSelection]
    DEC = DEC[filterSelection]
    GMAG = GMAG[filterSelection]
    RMAG = RMAG[filterSelection]
    CLASS = CLASS[filterSelection]
    
    # Valid Pixel Cut
    validPix = fitsio.read(validPixFile)['PIXEL']
    
    currentPix = hp.ang2pix(4096, RA, DEC, lonlat = True, nest = True)
    
    pixCut = np.isin(currentPix, validPix)
    
    RA = RA[pixCut]
    DEC = DEC[pixCut]
    GMAG = GMAG[pixCut]
    RMAG = RMAG[pixCut]
    CLASS = CLASS[pixCut]
    
    my_table = Table()
    my_table['RA'] = RA
    my_table['DEC'] = DEC
    my_table['GMAG'] = GMAG
    my_table['RMAG'] = RMAG
    my_table['CLASS'] = CLASS
    my_table.write(writeFile, overwrite = True)
    
    
def getDetStarPositions(detStarFile, validPixFile, writeFile):
    """
    The goal of this is to get the position of every injection that had a valid pixel value and had
    no flags.
    """
    
    detStarData = fitsio.read(detStarFile, columns = ['true_ra', 'true_dec',
                                                      'flags_foreground', 'flags_badregions', 
                                                      'flags_footprint', 'match_flag_1.5_asec'])
    
    validPix = fitsio.read(validPixFile)['PIXEL']
    
    RA = detStarData['true_ra']
    DEC = detStarData['true_dec']
    FOREGROUND = detStarData['flags_foreground']
    BADREGIONS = detStarData['flags_badregions']
    FOOTPRINT = detStarData['flags_footprint']
    ARCSECONDS = detStarData['match_flag_1.5_asec']
    
    flagCut = np.where((FOREGROUND == 0) & 
                       (BADREGIONS < 2) & 
                       (FOOTPRINT == 1) & 
                       (ARCSECONDS < 2))[0]
    
    RA = RA[flagCut]
    DEC = DEC[flagCut]
    
    PIX = hp.ang2pix(4096, RA, DEC, lonlat = True, nest = True)
    
    pixCut = np.isin(PIX, validPix)
    
    PIX = PIX[pixCut]
    
    my_table = Table()
    my_table['PIXEL'] = PIX
    my_table.write(writeFile, overwrite = True)
    
    
def getMatGalas(path, mu, deepFiles, matGalaFile, detGalaFile, validPixFile, writeFile, gCut, classCutoff):
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
    
    wideCols = ['true_id', 'bal_id', 'true_ra', 'true_dec', 'meas_EXTENDED_CLASS_SOF', 'meas_cm_mag', 'meas_psf_mag']
    matBalrData = fitsio.read(matGalaFile, columns = wideCols)
    
    # This will be used to match galaxies to their deep field counterparts.
    ID = matBalrData['true_id']
    
    # This is the Balrog Object ID
    BALR_ID = matBalrData['bal_id']
    
    # These are some of the data points I will be storing for valid data.
    RA = matBalrData['true_ra']
    DEC = matBalrData['true_dec']
    CLASS = matBalrData['meas_EXTENDED_CLASS_SOF']
    GMAG_CM = matBalrData['meas_cm_mag'][:,0]
    RMAG_CM = matBalrData['meas_cm_mag'][:,1]
    
    GMAG_PSF = matBalrData['meas_psf_mag'][:,0]
    RMAG_PSF = matBalrData['meas_psf_mag'][:,1]
    
    GMAG = np.copy(GMAG_CM)
    RMAG = np.copy(RMAG_CM)
    
    GMAG[np.where((CLASS <= classCutoff) & (CLASS >= 0))] = GMAG_PSF[np.where((CLASS <= classCutoff) & (CLASS >= 0))]
    RMAG[np.where((CLASS <= classCutoff) & (CLASS >= 0))] = RMAG_PSF[np.where((CLASS <= classCutoff) & (CLASS >= 0))]
    
    sortInds = BALR_ID.argsort()
    BALR_ID = BALR_ID[sortInds[::1]]
    ID = ID[sortInds[::1]]
    RA = RA[sortInds[::1]]
    DEC = DEC[sortInds[::1]]
    GMAG = GMAG[sortInds[::1]]
    RMAG = RMAG[sortInds[::1]]
    CLASS = CLASS[sortInds[::1]]
    
    # This is from the detected dataset
    detBalrData = fitsio.read(detGalaFile, columns = ['bal_id', 'flags_foreground', 'flags_badregions', 'flags_footprint', 'match_flag_1.5_asec'])
    
    # This is the Balrog Object ID for all data, not just matches
    FLAG_ID = detBalrData['bal_id']
    
    # This cut shouldn't really cut anything. It's more of a sanity thing.
    sanityCut = np.isin(BALR_ID, FLAG_ID)
    BALR_ID = BALR_ID[sanityCut]
    ID = ID[sanityCut]
    RA = RA[sanityCut]
    DEC = DEC[sanityCut]
    GMAG = GMAG[sanityCut]
    RMAG = RMAG[sanityCut]
    CLASS = CLASS[sanityCut]

    # These are the data values useful for cropping the data.
    FOREGROUND = detBalrData['flags_foreground']
    BADREGIONS = detBalrData['flags_badregions']
    FOOTPRINT = detBalrData['flags_footprint']
    ARCSECONDS = detBalrData['match_flag_1.5_asec']
    
    sortInds = FLAG_ID.argsort()
    FLAG_ID = FLAG_ID[sortInds[::1]]
    FOREGROUND = FOREGROUND[sortInds[::1]]
    BADREGIONS = BADREGIONS[sortInds[::1]]
    FOOTPRINT = FOOTPRINT[sortInds[::1]]
    ARCSECONDS = ARCSECONDS[sortInds[::1]]
    
    # This will serve to align the flags with their measured counterpart.
    # Try sorting both with an argsort and then running this method.
    
    cropInds = np.isin(FLAG_ID, BALR_ID)
            
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
                          # Isochrone cuts
                          (GMAG < gCut))[0]

    ID = ID[cutIndices]
    RA = RA[cutIndices]
    DEC = DEC[cutIndices]
    GMAG = GMAG[cutIndices]
    RMAG = RMAG[cutIndices]
    CLASS = CLASS[cutIndices]

    # This makes it so I only store objects that were initially labeled as galaxies in the deep fields.
    # It also performs some cuts on extreme values.
    trueClass = deepGalID[ID - minID]
    useInds = np.where((trueClass == 1))[0]

    RA = RA[useInds]
    DEC = DEC[useInds]
    GMAG = GMAG[useInds]
    RMAG = RMAG[useInds]
    CLASS = CLASS[useInds]
    
    MG = GMAG - mu
    GR = GMAG - RMAG
    
    filterSelection=Path.contains_points(path,np.vstack((GR,MG)).T)
    
    RA = RA[filterSelection]
    DEC = DEC[filterSelection]
    GMAG = GMAG[filterSelection]
    RMAG = RMAG[filterSelection]
    CLASS = CLASS[filterSelection]
    
    # Valid Pixel Cut
    validPix = fitsio.read(validPixFile)['PIXEL']
    
    currentPix = hp.ang2pix(4096, RA, DEC, lonlat = True, nest = True)
    
    pixCut = np.isin(currentPix, validPix)
    
    RA = RA[pixCut]
    DEC = DEC[pixCut]
    GMAG = GMAG[pixCut]
    RMAG = RMAG[pixCut]
    CLASS = CLASS[pixCut]
    
    my_table = Table()
    my_table['RA'] = RA
    my_table['DEC'] = DEC
    my_table['GMAG'] = GMAG
    my_table['RMAG'] = RMAG
    my_table['CLASS'] = CLASS
    my_table.write(writeFile, overwrite = True)
    
    
def getDetGalaPositions(deepFiles, detGalaFile, validPixFile, writeFile):
    """
    Getting all positions of galaxy injections.
    """
    
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
    
    detGalaData = fitsio.read(detGalaFile, columns = ['true_id', 'true_ra', 'true_dec',
                                                      'flags_foreground', 'flags_badregions', 
                                                      'flags_footprint', 'match_flag_1.5_asec'])
    
    validPix = fitsio.read(validPixFile)['PIXEL']
    
    ID = detGalaData['true_id']
    RA = detGalaData['true_ra']
    DEC = detGalaData['true_dec']
    FOREGROUND = detGalaData['flags_foreground']
    BADREGIONS = detGalaData['flags_badregions']
    FOOTPRINT = detGalaData['flags_footprint']
    ARCSECONDS = detGalaData['match_flag_1.5_asec']
    
    flagCut = np.where((FOREGROUND == 0) & 
                       (BADREGIONS < 2) & 
                       (FOOTPRINT == 1) & 
                       (ARCSECONDS < 2))[0]
    
    ID = ID[flagCut]
    RA = RA[flagCut]
    DEC = DEC[flagCut]
    
    idCut = np.where((deepGalID[ID - minID] == 1))[0]
    
    RA = RA[idCut]
    DEC = DEC[idCut]
    
    PIX = hp.ang2pix(4096, RA, DEC, lonlat = True, nest = True)
    
    pixCut = np.isin(PIX, validPix)
    
    PIX = PIX[pixCut]
    
    my_table = Table()
    my_table['PIXEL'] = PIX
    my_table.write(writeFile, overwrite = True)
    