# This file will contain methods relating to finding valid pixels and cropping data accordingly.
# These methods will need to have resolution as a parameter that can be adjusted.
# Another parameter for these methods would be the name of the file to write the data out to.

import fitsio
import numpy as np
import Config
import healpy as hp
from astropy.table import Table


def validPixCropData(condFiles, stelFile, pixWriteFile, cropWriteFiles):
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

    # This gives actual pixel numbers.
    PIX = np.where(validPix)[0]
    
    # This stores the valid pixels in a fits file.
    pix_table = Table()
    pix_table['PIXEL'] = PIX
    pix_table.write(pixWriteFile, overwrite = True)
    
    for i in range(len(condFiles)):
        condData = fitsio.read(condFiles[i]) # This reads in the data.
        condSigExt = np.full(12*(4096**2), -1.6375e+30) # Gives a default value.
        condSigExt[condData['PIXEL']] = condData['SIGNAL'] # Changes all pixels to their corresponding signals.
        condSig = condSigExt[PIX] # Crops to the desired valid pixels.
        # Writes data.
        cond_table = Table()
        cond_table['SIGNAL'] = condSig
        cond_table.write(cropWriteFiles[i], overwrite = True)
        
    # Same as block above but for stellar density.
    stelDensExt = fitsio.read(stelFile)['I'].flatten() # Loads in the stellar density
    condSig = stelDensExt[PIX]
    stel_table = Table()
    stel_table['SIGNAL'] = condSig
    stel_table.write(cropWriteFiles[-1], overwrite = True)