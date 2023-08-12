# This file will contain methods relating to finding valid pixels and cropping data accordingly.
# These methods will need to have resolution as a parameter that can be adjusted.
# Another parameter for these methods would be the name of the file to write the data out to.

import fitsio
import numpy as np
import Config
import healpy as hp
from astropy.table import Table


def validPixCropData(res, condFiles, stelFile, pixWriteFile, cropWriteFiles, perCovered):
    '''
    This method serves to get a list of valid pixels and then crop the survey conditions to said pixels.
    All files given should be .fits files. cropWriteFiles includes the file to write the stellar density
    to so it should have one more entry than condFiles.
    '''
    
    # This will be used to check where valid pixels are.
    validPix = np.full(12*(4096**2), True, dtype = bool)
    
    for file in condFiles:
        condData = fitsio.read(file) # This reads in the data.
        condSigExt = np.full(12*(4096**2), -1.6375e+30) # Gives a default value.
        condSigExt[condData['PIXEL']] = condData['SIGNAL'] # Changes all valid pixels to their corresponding signals.
        validPix[np.where(condSigExt < -100)] = False # An invalid survey property marks this pixel as false.
    
    stelDensExt = fitsio.read(stelFile)['I'].flatten() # Loads in the stellar density.
    validPix[np.where(stelDensExt < -100)] = False # Any non valid pixel has its value changed to False.

    # This reduces the healpixel resolution to the requested size.
    validPixRes = np.full(12*(4096**2), 0.0)
    validPixRes[validPix] = 1.0
    validPixRes = hp.ud_grade(validPixRes, res, order_in = 'NESTED', order_out = 'NESTED')

    # This provides a cutoff for what percent of the superpixel has to be covered to be considered valid.
    PIX = np.where(validPixRes >= perCovered)[0]
    
    # This stores the valid pixels in a fits file.
    my_table = Table()
    my_table['PIXEL'] = PIX
    my_table.write(pixWriteFile, overwrite = True)
    
    for i in range(len(condFiles)):
        condData = fitsio.read(condFiles[i]) # This reads in the data.
        condSigExt = np.full(12*(4096**2), -1.6375e+30) # Gives a default value.
        condSigExt[condData['PIXEL']] = condData['SIGNAL'] # Changes all valid pixels to their corresponding signals.
        condSigExt = hp.ud_grade(condSigExt, res, order_in = 'NESTED', order_out = 'NESTED') # Degrades resolution.
        condSig = condSigExt[PIX] # Crops to the desired pixels.
        # Writes data.
        my_table = Table()
        my_table['SIGNAL'] = condSig
        my_table.write(cropWriteFiles[i], overwrite = True)
        
    # Same as block above but for stellar density.
    stelDensExt = fitsio.read(stelFile)['I'].flatten() # Loads in the stellar density
    stelDensExt = hp.ud_grade(stelDensExt, res, order_in = 'NESTED', order_out = 'NESTED')
    condSig = stelDensExt[PIX]
    my_table = Table()
    my_table['SIGNAL'] = condSig
    my_table.write(cropWriteFiles[-1], overwrite = True)