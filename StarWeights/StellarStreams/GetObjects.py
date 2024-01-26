import numpy as np
import fitsio
from astropy.table import Table
from matplotlib.path import Path


def getMatStars(path, mu, measStarFile, starFile, writeFile, gCut, classCutoff):
    '''
    This method is used to get and store the necessary balrog delta star information for later use.
    It applies basic quality cuts to the data and then stores it. Both files given need to be .fits files.
    This method only looks at stars that were detected so that a measured magnitude exists. 
    '''
    
    # This reads in all of the data. Most of these are just flags, but the g magnitude, ra, dec, detected
    # flag, and others will be stored for later use.
    measStarData = fitsio.read(measStarFile, columns = ['meas_ra', 'meas_dec', 'meas_EXTENDED_CLASS_SOF',
                                                'meas_psf_mag', 'meas_cm_mag', 'bal_id'])
    
    starData = fitsio.read(starFile, columns = ['bal_id', 'flags_foreground', 'flags_badregions', 
                                                'flags_footprint', 'match_flag_1.5_asec'])
    

    RA = measStarData['meas_ra']
    DEC = measStarData['meas_dec']
    # PSF Magnitudes
    GMAG_PSF = measStarData['meas_psf_mag'][:,0]
    RMAG_PSF = measStarData['meas_psf_mag'][:,1]
    
    GMAG_CM = measStarData['meas_cm_mag'][:,0]
    RMAG_CM = measStarData['meas_cm_mag'][:,1]
    # This is the class that the object was measured as.
    CLASS = measStarData['meas_EXTENDED_CLASS_SOF']
    # This is the ID from the measured catalog.
    MEAS_ID  = measStarData['bal_id']
    
    GMAG = np.copy(GMAG_PSF)
    RMAG = np.copy(RMAG_PSF)
    
    GMAG[np.where((CLASS >= classCutoff) & (CLASS <= 3))] = GMAG_CM[np.where((CLASS >= classCutoff) & (CLASS <= 3))]
    RMAG[np.where((CLASS >= classCutoff) & (CLASS <= 3))] = RMAG_CM[np.where((CLASS >= classCutoff) & (CLASS <= 3))]
    
    sortInds = MEAS_ID.argsort()
    MEAS_ID = MEAS_ID[sortInds[::1]]
    RA = RA[sortInds[::1]]
    DEC = DEC[sortInds[::1]]
    GMAG = GMAG[sortInds[::1]]
    RMAG = RMAG[sortInds[::1]]
    CLASS = CLASS[sortInds[::1]]
    
    # Everything from here on out is simply used in order to filter the data
    FOREGROUND = starData['flags_foreground']
    BADREGIONS = starData['flags_badregions']
    FOOTPRINT = starData['flags_footprint']
    ARCSECONDS = starData['match_flag_1.5_asec']
    FLAG_ID = starData['bal_id']
    
    # This cut shouldn't really cut anything. It's more of a sanity thing.
    sanityCut = np.isin(MEAS_ID, FLAG_ID)
    MEAS_ID = MEAS_ID[sanityCut]
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
    
    cropInds = np.isin(FLAG_ID, MEAS_ID)
            
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
                          # Isochrone cuts
                          (GMAG < gCut))[0]

    # This reduced the data down to the actually valid pixels.
    RA = RA[cutIndices]
    DEC = DEC[cutIndices]
    GMAG = GMAG[cutIndices]
    RMAG = RMAG[cutIndices]
    CLASS = CLASS[cutIndices]
    MG = GMAG - mu
    GR = GMAG - RMAG
    
    filterSelection=Path.contains_points(path,np.vstack((GR,MG)).T)
    
    RA = RA[filterSelection]
    DEC = DEC[filterSelection]
    GMAG = GMAG[filterSelection]
    RMAG = RMAG[filterSelection]
    CLASS = CLASS[filterSelection]
    
    my_table = Table()
    my_table['RA'] = RA
    my_table['DEC'] = DEC
    my_table['GMAG'] = GMAG
    my_table['RMAG'] = RMAG
    my_table['CLASS'] = CLASS
    my_table.write(writeFile, overwrite = True)


def getDetStar(path, mu, starFile, writeFile, gCut):
    '''
    This method is used to get and store the necessary balrog delta star information for later use.
    It applies basic quality cuts to the data and then stores it. Both files given need to be .fits files.
    '''
    
    # This reads in all of the data. Most of these are just flags, but the g magnitude, ra, dec, and detected
    # flag will be stored for later use..
    starData = fitsio.read(starFile, columns = ['detected', 'true_ra', 'true_dec',
                                                'flags_foreground', 'flags_badregions', 'flags_footprint',
                                                'match_flag_1.5_asec', 'true_g_Corr', 'true_gr_Corr',
                                                'meas_EXTENDED_CLASS_SOF'])

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
    CLASS = starData['meas_EXTENDED_CLASS_SOF']

    # This is used to filter out any injections that had flags raised.
    cutIndices = np.where((FOREGROUND == 0) & 
                          (BADREGIONS < 2) & 
                          (FOOTPRINT == 1) & 
                          (ARCSECONDS < 2) & 
                          # Isochrone Cut
                          (GMAG < gCut))[0]

    # This reduced the data down to the actually valid pixels.
    DETECTED = DETECTED[cutIndices]
    RA = RA[cutIndices]
    DEC = DEC[cutIndices]
    GMAG = GMAG[cutIndices]
    RMAG = RMAG[cutIndices]
    CLASS = CLASS[cutIndices]
    MG = GMAG - mu
    GR = GMAG - RMAG
    
    filterSelection=Path.contains_points(path,np.vstack((GR,MG)).T)
    
    DETECTED = DETECTED[filterSelection]
    RA = RA[filterSelection]
    DEC = DEC[filterSelection]
    GMAG = GMAG[filterSelection]
    RMAG = RMAG[filterSelection]
    CLASS = CLASS[filterSelection]
    
    my_table = Table()
    my_table['RA'] = RA
    my_table['DEC'] = DEC
    my_table['GMAG'] = GMAG
    my_table['RMAG'] = RMAG
    my_table['CLASS'] = CLASS
    my_table['DETECTED'] = DETECTED
    my_table.write(writeFile, overwrite = True)
    
    
def getDetStarsMeasData(path, mu, measStarFile, starFile, writeFile, writeAllPosFile, gCut, classCutoff):
    '''
    This method is used with the N/<N> method. The general idea is to get stars that are detected as galaxies and stars that are detected as stars. For these objects we get the measured magnitude in whichever method is used for these objects (so cm for objects called galaxies and psf for objects called stars). We also get an overall idea of the locations of every star injection so that we know which pixels must be used for our <N> number.
    '''
    
    # This reads in all of the data. Most of these are just flags, but the g magnitude, ra, dec, detected
    # flag, and others will be stored for later use.
    starData = fitsio.read(starFile, columns = ['bal_id', 'flags_foreground', 'flags_badregions', 
                                                'flags_footprint', 'match_flag_1.5_asec', 'true_ra', 'true_dec'])
    FOREGROUND = starData['flags_foreground']
    BADREGIONS = starData['flags_badregions']
    FOOTPRINT = starData['flags_footprint']
    ARCSECONDS = starData['match_flag_1.5_asec']

    ALL_STAR_RA = starData['true_ra']
    ALL_STAR_DEC = starData['true_dec']

    qualityCut = np.where((FOREGROUND == 0) & 
                          (BADREGIONS < 2) & 
                          (FOOTPRINT == 1) & 
                          (ARCSECONDS < 2))[0]

    ALL_STAR_RA = ALL_STAR_RA[qualityCut]
    ALL_STAR_DEC = ALL_STAR_DEC[qualityCut]

    measStarData = fitsio.read(measStarFile, columns = ['meas_ra', 'meas_dec', 'meas_EXTENDED_CLASS_SOF',
                                                'meas_psf_mag', 'meas_cm_mag', 'bal_id'])

    RA = measStarData['meas_ra']
    DEC = measStarData['meas_dec']
    # PSF Magnitudes
    GMAG_PSF = measStarData['meas_psf_mag'][:,0]
    RMAG_PSF = measStarData['meas_psf_mag'][:,1]
    # CM Magnitudes
    GMAG_CM = measStarData['meas_cm_mag'][:,0]
    RMAG_CM = measStarData['meas_cm_mag'][:,1]
    # This is the class that the object was measured as.
    CLASS = measStarData['meas_EXTENDED_CLASS_SOF']
    # This is the ID from the measured catalog.
    MEAS_ID  = measStarData['bal_id']
    
    GMAG = np.copy(GMAG_PSF)
    RMAG = np.copy(RMAG_PSF)
    
    GMAG[np.where((CLASS >= classCutoff) & (CLASS <= 3))] = GMAG_CM[np.where((CLASS >= classCutoff) & (CLASS <= 3))]
    RMAG[np.where((CLASS >= classCutoff) & (CLASS <= 3))] = RMAG_CM[np.where((CLASS >= classCutoff) & (CLASS <= 3))]

    sortInds = MEAS_ID.argsort()
    MEAS_ID = MEAS_ID[sortInds[::1]]
    RA = RA[sortInds[::1]]
    DEC = DEC[sortInds[::1]]
    GMAG = GMAG[sortInds[::1]]
    RMAG = RMAG[sortInds[::1]]
    CLASS = CLASS[sortInds[::1]]
    
    # Everything from here on out is simply used in order to filter the data
    FLAG_ID = starData['bal_id']
    
    # This cut shouldn't really cut anything. It's more of a sanity thing.
    sanityCut = np.isin(MEAS_ID, FLAG_ID)
    MEAS_ID = MEAS_ID[sanityCut]
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

    cropInds = np.isin(FLAG_ID, MEAS_ID)

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
                          # Isochrone cuts
                          (GMAG < gCut))[0]

    # This reduced the data down to the actually valid pixels.
    RA = RA[cutIndices]
    DEC = DEC[cutIndices]
    GMAG = GMAG[cutIndices]
    RMAG = RMAG[cutIndices]
    CLASS = CLASS[cutIndices]
    
    MG = GMAG - mu
    GR = GMAG - RMAG
    
    filterSelection=Path.contains_points(path,np.vstack((GR,MG)).T)
    
    RA = RA[filterSelection]
    DEC = DEC[filterSelection]
    GMAG = GMAG[filterSelection]
    RMAG = RMAG[filterSelection]
    CLASS = CLASS[filterSelection]
    
    my_table = Table()
    my_table['RA'] = RA
    my_table['DEC'] = DEC
    my_table['GMAG'] = GMAG
    my_table['RMAG'] = RMAG
    my_table['CLASS'] = CLASS
    my_table.write(writeFile, overwrite = True)
    
    my_table = Table()
    my_table['RA'] = ALL_STAR_RA
    my_table['DEC'] = ALL_STAR_DEC
    my_table.write(writeAllPosFile, overwrite = True)
    
    
    
def getMatGalas(path, mu, deepFiles, measBalrFile, balrFile, writeFile, gCut, classCutoff):
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
    
    wideCols = ['true_id', 'bal_id', 'meas_ra', 'meas_dec', 'meas_EXTENDED_CLASS_SOF', 'meas_cm_mag', 'meas_psf_mag']
    measBalrData = fitsio.read(measBalrFile, columns = wideCols)
    
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
    
    GMAG_PSF = measBalrData['meas_psf_mag'][:,0]
    RMAG_PSF = measBalrData['meas_psf_mag'][:,1]
    
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
    balrData = fitsio.read(balrFile, columns = ['bal_id', 'flags_foreground', 'flags_badregions', 'flags_footprint', 'match_flag_1.5_asec'])
    
    # This is the Balrog Object ID for all data, not just matches
    FLAG_ID = balrData['bal_id']
    
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
    
    my_table = Table()
    my_table['RA'] = RA
    my_table['DEC'] = DEC
    my_table['GMAG'] = GMAG
    my_table['RMAG'] = RMAG
    my_table['CLASS'] = CLASS
    my_table.write(writeFile, overwrite = True)
    

def getDetGalas(path, mu, deepFiles, balrFile, writeFile, gCut):
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
    
    wideCols = ['true_id', 'true_ra', 'true_dec', 'meas_EXTENDED_CLASS_SOF', 'true_bdf_mag_deredden', 'flags_foreground', 'flags_badregions', 'flags_footprint', 'match_flag_1.5_asec', 'detected']
    
    balrData = fitsio.read(balrFile, columns = wideCols)
    
    ID = balrData['true_id']
    RA = balrData['true_ra']
    DEC = balrData['true_dec']
    CLASS = balrData['meas_EXTENDED_CLASS_SOF']
    DETECTED = balrData['detected']
    GMAG = balrData['true_bdf_mag_deredden'][:,0]
    RMAG = balrData['true_bdf_mag_deredden'][:,1]
    FOREGROUND = balrData['flags_foreground']
    BADREGIONS = balrData['flags_badregions']
    FOOTPRINT = balrData['flags_footprint']
    ARCSECONDS = balrData['match_flag_1.5_asec']
    
    trueClass = deepGalID[ID - minID]
    
    # This is used to filter out any injections that had flags raised.
    cutIndices = np.where((FOREGROUND == 0) & 
                          (BADREGIONS < 2) & 
                          (FOOTPRINT == 1) & 
                          (ARCSECONDS < 2) &
                          (trueClass == 1) &
                          # Isochrone Cut
                          (GMAG < gCut))[0]
    
    # This reduced the data down to the actually valid pixels.
    DETECTED = DETECTED[cutIndices]
    RA = RA[cutIndices]
    DEC = DEC[cutIndices]
    GMAG = GMAG[cutIndices]
    RMAG = RMAG[cutIndices]
    CLASS = CLASS[cutIndices]
    MG = GMAG - mu
    GR = GMAG - RMAG
    
    filterSelection=Path.contains_points(path,np.vstack((GR,MG)).T)
    
    DETECTED = DETECTED[filterSelection]
    RA = RA[filterSelection]
    DEC = DEC[filterSelection]
    GMAG = GMAG[filterSelection]
    RMAG = RMAG[filterSelection]
    CLASS = CLASS[filterSelection]
    
    my_table = Table()
    my_table['RA'] = RA
    my_table['DEC'] = DEC
    my_table['GMAG'] = GMAG
    my_table['RMAG'] = RMAG
    my_table['CLASS'] = CLASS
    my_table['DETECTED'] = DETECTED
    my_table.write(writeFile, overwrite = True)
    
    
def getDetGalasMeasData(path, mu, deepFiles, measBalrFile, balrFile, writeFile, writeAllPosFile, gCut, classCutoff):
    '''
    This method aims to get stars but include only measured magnitude data for those galaxies which are recovered.
    The idea is that when looking at relative detection rates, it is fine to use measured magnitudes since we don't
    really need magnitudes for every object.
    
    This method will work as follows:
    
    First, I will get the measured data, following a previous method's example.
    
    Secondly, I will get detected data of objects. For these, all I really need is the position, since I will not be using
    anything else. However, I will perform an isochrone cut on them based on magnitude to reduce the total number of 
    objects.
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

    # This is from the detected dataset
    balrData = fitsio.read(balrFile, columns = ['true_ra', 'true_dec', 'true_id', 'bal_id', 'flags_foreground', 'flags_badregions', 'flags_footprint', 'match_flag_1.5_asec'])

    # These are the data values useful for cropping the data.
    FOREGROUND = balrData['flags_foreground']
    BADREGIONS = balrData['flags_badregions']
    FOOTPRINT = balrData['flags_footprint']
    ARCSECONDS = balrData['match_flag_1.5_asec']

    ALL_GALA_ID = balrData['true_id']
    ALL_GALA_RA = balrData['true_ra']
    ALL_GALA_DEC = balrData['true_dec']

    qualityCut = np.where((FOREGROUND == 0) & 
                          (BADREGIONS < 2) & 
                          (FOOTPRINT == 1) & 
                          (ARCSECONDS < 2) &
                          (deepGalID[ALL_GALA_ID - minID] == 1))[0]

    ALL_GALA_RA = ALL_GALA_RA[qualityCut]
    ALL_GALA_DEC = ALL_GALA_DEC[qualityCut]

    wideCols = ['true_id', 'bal_id', 'meas_ra', 'meas_dec', 'meas_EXTENDED_CLASS_SOF', 'meas_psf_mag', 'meas_cm_mag']
    measBalrData = fitsio.read(measBalrFile, columns = wideCols)

    # This will be used to match galaxies to their deep field counterparts.
    ID = measBalrData['true_id']

    # This is the Balrog Object ID
    BALR_ID = measBalrData['bal_id']

    # These are some of the data points I will be storing for valid data.
    RA = measBalrData['meas_ra']
    DEC = measBalrData['meas_dec']
    CLASS = measBalrData['meas_EXTENDED_CLASS_SOF']

    GMAG_PSF = measBalrData['meas_psf_mag'][:,0]
    RMAG_PSF = measBalrData['meas_psf_mag'][:,1]
    
    GMAG_CM = measBalrData['meas_cm_mag'][:,0]
    RMAG_CM = measBalrData['meas_cm_mag'][:,1]
    
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

    # This is the Balrog Object ID for all data, not just matches
    FLAG_ID = balrData['bal_id']
    
    # This cut shouldn't really cut anything. It's more of a sanity thing.
    sanityCut = np.isin(BALR_ID, FLAG_ID)
    BALR_ID = BALR_ID[sanityCut]
    ID = ID[sanityCut]
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
    # Try sorting both with an argsort and then running this method.

    cropInds = np.isin(FLAG_ID, BALR_ID)

    # parallelInd = 0
    # cropInds = []
    # for i in np.arange(len(FLAG_ID)):
    #     if parallelInd >= len(BALR_ID):
    #         break
    #     if BALR_ID[parallelInd] == FLAG_ID[i]:
    #         cropInds.append(i)
    #         parallelInd += 1

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
    
    my_table = Table()
    my_table['RA'] = RA
    my_table['DEC'] = DEC
    my_table['GMAG'] = GMAG
    my_table['RMAG'] = RMAG
    my_table['CLASS'] = CLASS
    my_table.write(writeFile, overwrite = True)
    
    my_table = Table()
    my_table['RA'] = ALL_GALA_RA
    my_table['DEC'] = ALL_GALA_DEC
    my_table.write(writeAllPosFile, overwrite = True)
    