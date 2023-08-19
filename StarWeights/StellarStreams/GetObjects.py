import numpy as np
import fitsio
from astropy.table import Table
from matplotlib.path import Path


def getDetStar(path, mu, starFile, writeFile, gCut):
    '''
    This method is used to get and store the necessary balrog delta star information for later use.
    It applies basic quality cuts to the data and then stores it. Both files given need to be .fits files.
    '''
    
    # This reads in all of the data. Most of these are just flags, but the g magnitude, ra, dec, and detected
    # flag will be stored for later use..
    starData = fitsio.read(starFile, columns = ['detected', 'true_ra', 'true_dec',
                                                'flags_foreground', 'flags_badregions', 'flags_footprint',
                                                'match_flag_1.5_asec', 'true_g_Corr', 'true_gr_Corr', 'true_gi_Corr',
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
    