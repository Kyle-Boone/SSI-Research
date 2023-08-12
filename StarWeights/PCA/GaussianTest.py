import numpy as np
from numpy import savetxt
from numpy import loadtxt
import fitsio
import healpy as hp
import matplotlib.pyplot as plt
import Config
from scipy import interpolate as inter
from astropy.table import Table

res = 4096 # Resolution of the heal pixels
sigma = 0.5
# widthBins = 0.05 # Width of the bins
numBins = 100 # Number of bins to use
perVar = 0.98 # Percent of the variance to be captured
perMap = 0.625 # Percent of the PC maps to use, adjust this later

# This is the actual file containing all of the data
starFile = '/afs/hep.wisc.edu/bechtol-group/MegansThings/balrog_detection_catalog_sof_run2_stars_v1.4_avg_added_match_flags.fits'
# This reads in all of the data. Most of these are just flags, the only pieces that get used much outside
# of filtering are detected, true_ra and true_dec which get used to convert into healPixels.
starData = fitsio.read(starFile, columns = ['detected', 'true_ra', 'true_dec',
                                            'flags_foreground', 'flags_badregions', 'flags_footprint',
                                            'meas_FLAGS_GOLD_SOF_ONLY', 'match_flag_1.5_asec'])

# These are in degrees which is why lonlat is set to True in the next cell.
RA = starData['true_ra']
DEC = starData['true_dec']
# This is used for detection rates, each point is either a 0 (no detection) or a 1 (detection)
DETECTED = starData['detected']
# Everything from here on out is simply used in order to filter the data
FOREGROUND = starData['flags_foreground']
BADREGIONS = starData['flags_badregions']
FOOTPRINT = starData['flags_footprint']
GOLDSOF = starData['meas_FLAGS_GOLD_SOF_ONLY']
ARCSECONDS = starData['match_flag_1.5_asec']

# This is used to filter out any injections that either weren't detected or had flags raised.
cutIndices = np.where((FOREGROUND == 0) & 
                      (BADREGIONS < 2) & 
                      (FOOTPRINT == 1) & 
                      (ARCSECONDS < 2))[0]

# This reduced the data down to the actually valid pixels.
DETECTED = DETECTED[cutIndices]
RA = RA[cutIndices]
DEC = DEC[cutIndices]

# This converts the RA and DEC values from above to healpixels so we can compare to the sky condition.
starPixels = hp.ang2pix(res, RA, DEC, lonlat = True, nest = True)

# This sorts by the pixel in order to make following methods more efficient.
sortInds = starPixels.argsort()
starPix = starPixels[sortInds[::1]]
DET = DETECTED[sortInds[::1]]

# These are indices that will be looping through the pixStar and starPix arrays in parallel.
uniqInd = 0
starInd = 0

# This will be used to store the number of stars at each pixel.
pixStar = np.unique(starPix) # The unique pixels, with no repeats.
detStar = np.zeros_like(pixStar)
injStar = np.zeros_like(pixStar)

while starInd < len(starPix):
    if pixStar[uniqInd] == starPix[starInd]: # If the pixels match up in the arrays.
        detStar[uniqInd] += DET[starInd]     # Add one if there was a detection at this location.
        injStar[uniqInd] += 1                # Add one to the corresponding spot in the balStar array.
        starInd += 1                         # Add one to the starInd to see if the next index in starPix is also the same.
        # Since the last index of pixStar and starPix are the same, starInd will increase the last time through the loop,
        # making this the index that we must restrict in the while loop.
    else:
        uniqInd += 1 # If the pixels are no longer the same, increase the index you check in the pixStar array.
        
# This loads in all of the file names of the survey conditions
condFiles = Config.files
condMapsExt = []

# This loops over every condition file except for stellar density which has a different format
for i in range(len(condFiles) - 1):
    condData = fitsio.read(condFiles[i]) # This reads in the data
    condSigExt = np.full(12*(4096**2), -1.6375e+30) # Gives a default value
    condSigExt[condData['PIXEL']] = condData['SIGNAL'] # Changes all valid pixels to their corresponding signals
    condSigExt[np.where(condSigExt == -1.6375e+30)[0]] = hp.UNSEEN # Masks all non valid pixels
    if res != 4096:
        condSigExt=hp.ud_grade(condSigExt, res) # Only degrades if necessary (this is very time consuming)
    condMapsExt.append(condSigExt[pixStar]) # Only stores the values that are in pixels with injections
    
stelDensExt = fitsio.read(condFiles[-1])['I'].flatten() # Loads in the stellar density
stelDensExt[np.where(stelDensExt < 0)[0]] = hp.UNSEEN # Masks all non valid pixels
if res != 4096:
    stelDensExt=hp.ud_grade(stelDensExt, res) # Degrades if necessary
condMapsExt.append(stelDensExt[pixStar])

condMapsExt = np.array(condMapsExt, dtype = object) # Converts to an array

validIndices = np.full(len(pixStar), True, dtype = bool)
# The only valid indices are ones where every survey property is unmasked
for cond in condMapsExt:
    tempValidIndices = np.full(len(pixStar), True, dtype = bool)
    tempValidIndices[np.where(cond < -1000000000)[0]] = False
    validIndices = validIndices & tempValidIndices
    
condMaps = []
# Degrades all of the values to a common set of pixels
pixStar = pixStar[validIndices]
detStar = detStar[validIndices]
injStar = injStar[validIndices]

for cond in condMapsExt:
    condMaps.append(cond[validIndices])
    
condMaps = np.array(condMaps)

stanMaps = []
averages = []
stanDevs = []
# This standardizes every map as a first step of PCA
for cond in condMaps:
    averages.append(np.average(cond))
    stanDevs.append(np.std(cond))
    stanMaps.append((cond - np.average(cond)) / np.std(cond))
    
stanMaps = np.array(stanMaps)

# This gives the covariance matrix of the standardized maps
# Bias is true since the variance of each individual standardized map should be 1
cov = np.cov(stanMaps.astype(float), bias = True)

# This gives the eigenvalues and vectors of the covariance matrix
evalues, evectors = np.linalg.eig(cov)

# This cuts after the specified percentage of the variance has been achieved
for i in range(len(evalues)):
    if np.sum(evalues[0:i+1]) / np.sum(evalues) >= perVar:
        cutoff = i + 1
        break
featVec = evectors[0:cutoff]

redMaps = np.matmul(featVec, stanMaps) # Reduces the maps to PCA maps

redStds = []
for i in np.arange(len(redMaps)):
    redStds.append(np.std(redMaps[i]))
    redMaps[i] = redMaps[i]/np.std(redMaps[i])
    
# Stores the original data for later comparisons
originalDetStar = detStar
originalInjStar = injStar
aveEff = np.sum(originalDetStar) / np.sum(originalInjStar)

# Goal of this method is to find the index of the map that has the largest impact on detection rates.
def mostSigPCMap(redMaps, detStar, injStar = injStar, sigma = sigma, numBins = 100):
    
    maxAdjustment = []

    for i in range(len(redMaps)):
        
        onePC = redMaps[i] # Load up a PC map

        x = np.linspace(-3, 3, 100)
        y = []
        
        for xi in x:
            totDet = np.sum(detStar * np.exp(-1*(((onePC.astype(float) - xi) / sigma)**2)))
            totInj = np.sum(injStar * np.exp(-1*(((onePC.astype(float) - xi) / sigma)**2)))
            y.append((totDet / totInj) / aveEff)

        y = np.array(y)
        
        # Make the error the sum of the squared difference between the binned values and 1.
        maxAdjustment.append(np.sum((y - 1)**2))
        
    maxAdjustment = np.array(maxAdjustment)
    
    mostSigIndex = np.where(maxAdjustment == np.max(maxAdjustment))[0]
    
    return mostSigIndex[0] # Return wherever the error is the largest

trimRedMaps = np.copy(redMaps)

iterations = int(perMap * len(redMaps))

for num in np.arange(iterations):
    
    print(num)
    
    index = mostSigPCMap(trimRedMaps, detStar)
    
    onePC = trimRedMaps[index]
    
    x = np.linspace(-3, 3, 100)
    y = []

    for xi in x:
        totDet = np.sum(detStar * np.exp(-1*(((onePC.astype(float) - xi) / sigma)**2)))
        totInj = np.sum(injStar * np.exp(-1*(((onePC.astype(float) - xi) / sigma)**2)))
        y.append((totDet / totInj) / aveEff)

    y = np.array(y)

    f = inter.interp1d(x, y, bounds_error = False, fill_value = (y[0], y[-1]))
    
    # TODO Flag extrapolated data.

    correction = f(trimRedMaps[index].astype('float'))

    correction = 1 / correction

    detStar = detStar * correction

    pcMapCutoff = np.full(len(trimRedMaps), True, dtype = bool)
    pcMapCutoff[index] = False
    trimRedMaps = trimRedMaps[pcMapCutoff]
    
file = '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/Kyle_Stuff/training/'
my_table = Table()
my_table['PIXEL'] = pixStar
my_table['SIGNAL'] = detStar
my_table.write(file + 'Gaussian_detStar_Bins_100_Sig_0.5', overwrite = True)
