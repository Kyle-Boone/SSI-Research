import numpy as np
from numpy import savetxt
from numpy import loadtxt
import fitsio
import healpy as hp
import Config
from scipy import interpolate as inter
from astropy.table import Table


def mostSigPCMap(redMaps, Cor, All, sigma, numBins, aveAcc):
    '''
    The purpose of this map is to find which PC map has the largest systematic variation from the average probability of 
    detection.
    '''
    
    maxAdjustment = []

    for i in range(len(redMaps)):
        
        onePC = redMaps[i] # Load up a PC map

        x = np.linspace(-3, 3, numBins) # xValues for plot, goes out to 3 standard deviation.
        y = []
        
        for xi in x:
            # Gaussian weighting the values close by to each x value.
            totCor = np.sum(Cor * np.exp(-1*(((onePC.astype(float) - xi) / sigma)**2)))
            totAll = np.sum(All * np.exp(-1*(((onePC.astype(float) - xi) / sigma)**2)))
            y.append((totCor / totAll) / aveAcc)

        y = np.array(y)
        
        # Make the error the sum of the squared difference between the binned values and 1.
        maxAdjustment.append(np.sum((y - 1)**2))
        
    maxAdjustment = np.array(maxAdjustment)
    
    mostSigIndex = np.where(maxAdjustment == np.max(maxAdjustment))[0]
    
    return mostSigIndex[0] # Return wherever the error is the largest


def singleCorrectionTrain(objectFile, condFiles, pixFile, magBins, trainDataFiles, prevDataFiles, sigma, perMap, perVar, numBins, res, isStar, isDet, iterations = 5):
    '''
    This will train a single overall correction to the data. objectFile contains the object properties. condFiles contains
    the locations of survey property maps. pixFile contains a list of valid pixels. trainDataFile is where all the 
    information necessary for performing the correction on new data is stored. sigma is the standard deviation of the 
    Gaussian kernel used in fitting. perMap is the percent of PC maps used in fitting. perVar is the percent of variance
    to capture when performing PCA. numBins is the number of data points to approximate in the fitting. res is the
    resolution of the data in healpixels. isStar is a boolean corresponding to whether the objects are stars or not.
    isDet is whether this is a det based one or not. iterations can be filled with a new value if desired. fullSkyProbFiles
    is where probabilities will be stores, extrFiles are where extrapolations will be stored.
    '''
    
    # Classes corresponding to a correct classification.
    if isStar:
        lowClass = 0
        highClass = 1
    else:
        lowClass = 2
        highClass = 3
        
    validPix = fitsio.read(pixFile)['PIXEL']
    # Boolean alternative to validPix allows for some things to be easier.
    pixCheck = np.full(12*(res**2), False, dtype = bool)
    pixCheck[validPix] = True
    
    objectData = fitsio.read(objectFile)
    
    # Use these to get pixel locations for every object.
    origRA = objectData['RA']
    origDEC = objectData['DEC']
    
    # These are the object properties that will be used.
    GMAG = objectData['GMAG']
    
    # This determines whether the classification is correct or not.
    if isDet:
        origCLASS = objectData['DETECTED']
    else:
        EXTENDED_CLASS = objectData['CLASS']
        origCLASS = np.zeros_like(EXTENDED_CLASS)
        origCLASS[np.where((EXTENDED_CLASS == lowClass) | (EXTENDED_CLASS == highClass))[0]] = 1
        
    minMag = 0 # These two lines set default values to avoid warnings in the program. They will not be used.
    maxMag = 0
    
    for i in range(len(magBins) + 1):
        
        prevIndices = loadtxt(prevDataFiles[i][0:-5] + '_Indices.csv', delimiter=',').astype(int)
        prevYValues = fitsio.read(prevDataFiles[i])
        
        # This defines magnitude cuts in accordance with the magnitude bins.
        if i == 0:
            maxGMAG = magBins[i]
            magCut = np.where(GMAG <= maxGMAG)[0]
        elif i == len(magBins):
            minGMAG = magBins[i - 1]
            magCut = np.where(GMAG > minGMAG)[0]
        else:
            minGMAG = magBins[i - 1]
            maxGMAG = magBins[i]
            magCut = np.where((GMAG <= maxGMAG) & (GMAG > minGMAG))[0]
            
        RA = origRA[magCut]
        DEC = origDEC[magCut]
        CLASSRepeats = origCLASS[magCut]
        
        # This gives corresponding pixel numbers.
        PIXRepeats = hp.ang2pix(res, RA, DEC, lonlat = True, nest = True)
        
        # This sorts according to pixel.
        sortInds = PIXRepeats.argsort()
        PIXRepeats = PIXRepeats[sortInds[::1]]
        CLASSRepeats = CLASSRepeats[sortInds[::1]]
        
        uniqInd = 0
        objInd = 0
        
        # This will be used to store the number of objects at each pixel.
        PIX = np.unique(PIXRepeats) # The unique pixels, with no repeats.
        COR = np.zeros_like(PIX) # Correct classifications or detected object.
        TOT = np.zeros_like(PIX) # All objects per pixel.

        while objInd < len(PIXRepeats):
            if PIX[uniqInd] == PIXRepeats[objInd]: # If the pixels match up in the arrays.
                COR[uniqInd] += CLASSRepeats[objInd] # Add one if there was a correct object at this location.
                TOT[uniqInd] += 1                # Add one to the corresponding spot in the TOT.
                objInd += 1                         # Add one to the objInd to see if the next index in Pix is same.
                # Since the last index of Pix and PixRepeats are the same, objInd will increase 
                # the last time through the loop, making this the index that we must restrict in the while loop.
            else:
                uniqInd += 1 # If the pixels are no longer the same, increase the index you check in the Pix array.
                
        # Restricts to valid pixels.
        COR = COR[pixCheck[PIX]]
        TOT = TOT[pixCheck[PIX]]
        PIX = PIX[pixCheck[PIX]]
        
        condMaps = []

        # This loops over every condition file
        for condFile in condFiles:
            condData = fitsio.read(condFile) # This reads in the data
            condSigExt = np.full(12*(res**2), -1.6375e+30) # Gives a default value
            condSigExt[validPix] = condData['SIGNAL'] # Changes all valid pixels to their corresponding signals
            condMaps.append(condSigExt[PIX]) # Only stores the values that are in pixels with injections

        condMaps = np.array(condMaps, dtype = object) # Converts to an array
        
        origCOR = COR
        origTOT = TOT
        aveAcc = np.sum(COR) / np.sum(TOT)
    
        means = []
        stds = []
        # This will hold every standardized map
        stanMaps = []
        # This standardizes every map as a first step of PCA
        for j in range(len(condMaps)):
            # Store mean and std dev for later use.
            means.append(np.average(condMaps[j]))
            stds.append(np.std(condMaps[j]))
            stanMaps.append((condMaps[j] - np.average(condMaps[j])) / np.std(condMaps[j]))

        stanMaps = np.array(stanMaps)
    
        # This gives the covariance matrix of the standardized maps
        # Bias is true since the variance of each individual standardized map should be 1
        cov = np.cov(stanMaps.astype(float), bias = True)

        # This gives the eigenvalues and vectors of the covariance matrix
        evalues, evectors = np.linalg.eig(cov)

        # This cuts after the specified percentage of the variance has been achieved
        for j in range(len(evalues)):
            if np.sum(evalues[0:j+1]) / np.sum(evalues) >= perVar:
                cutoff = j + 1
                break
        featVec = evectors[0:cutoff]

        redMaps = np.matmul(featVec, stanMaps) # Reduces the maps to PCA maps

        # Standard deviations will once more be stored for later use.
        # Maps are reduced to standard deviation of 1 for consistent x values in the following steps.
        redStds = []
        for j in np.arange(len(redMaps)):
            redStds.append(np.std(redMaps[j]))
            redMaps[j] = redMaps[j]/np.std(redMaps[j])

        yValues = []
        corrIndices = []

        trimRedMaps = np.copy(redMaps)

        initMapCutoff = np.full(len(trimRedMaps), True, dtype = bool)
        for ind in prevIndices:
            initMapCutoff[ind] = False    
        trimRedMaps = trimRedMaps[initMapCutoff]
        
        for j in range(len(prevIndices)):
            yValues.append(prevYValues[str(prevIndices[j])])
            
        x = np.linspace(-3, 3, numBins)
        
        # This is applying the corrections that have already occured
        for j in range(len(prevIndices)):
            f = inter.interp1d(x, prevYValues[str(prevIndices[j])], bounds_error = False, fill_value = (prevYValues[str(prevIndices[j])][0], prevYValues[str(prevIndices[j])][-1]))
        
            correction = f(redMaps[prevIndices[j]].astype('float'))

            correction = 1 / correction

            # Apply correction
            COR = COR * correction
        
        print('Iterations: ' + str(iterations))
    
        timeThrough = 0
    
        for _ in np.arange(iterations):

            timeThrough += 1
            print(timeThrough)

            # Figure out the most significant map.
            index = mostSigPCMap(trimRedMaps, COR, TOT, sigma, numBins, aveAcc)

            # Store this index for later use.
            corrIndices.append(index)

            # Use this map to generate values.
            onePC = trimRedMaps[index]
            
            y = []

            for xi in x:
                # Gaussian weight the values when determining y Values.
                totCor = np.sum(COR * np.exp(-1*(((onePC.astype(float) - xi) / sigma)**2)))
                totAll = np.sum(TOT * np.exp(-1*(((onePC.astype(float) - xi) / sigma)**2)))
                y.append((totCor / totAll) / aveAcc)

            y = np.array(y)

            yValues.append(y)

            # Generate an interpolation function with constant extrapolation around the ends.
            f = inter.interp1d(x, y, bounds_error = False, fill_value = (y[0], y[-1]))

            correction = f(trimRedMaps[index].astype('float'))

            correction = 1 / correction

            # Apply correction and remove whichever principal component was used.
            COR = COR * correction

            pcMapCutoff = np.full(len(trimRedMaps), True, dtype = bool)
            pcMapCutoff[index] = False
            trimRedMaps = trimRedMaps[pcMapCutoff]

        actualCorrIndices = []
        originalIndices = np.arange(len(redMaps))
        indCrop = np.full(len(originalIndices), True, dtype = bool)

        for index in prevIndices:
            actualCorrIndices.append(index)
            indCrop[index] = False
        originalIndices = originalIndices[indCrop]

        for index in corrIndices:
            actualCorrIndices.append(originalIndices[index])
            originalIndices = np.delete(originalIndices, index)

        actualCorrIndices = np.array(actualCorrIndices)

        # Store data for later use
        savetxt(trainDataFiles[i][0:-5] + '_Means.csv', means, delimiter=',')
        savetxt(trainDataFiles[i][0:-5] + '_Stds.csv', stds, delimiter=',')
        savetxt(trainDataFiles[i][0:-5] + '_Red_Stds.csv', redStds, delimiter=',')
        savetxt(trainDataFiles[i][0:-5] + '_Indices.csv', actualCorrIndices, delimiter=',')
        savetxt(trainDataFiles[i][0:-5] + '_Ave_Acc.csv', np.array([aveAcc]), delimiter=',')
        savetxt(trainDataFiles[i][0:-5] + '_Feat_Vec.csv', featVec, delimiter='\t', fmt='%s')

        my_table = Table()
        for j in np.arange(len(actualCorrIndices)):
            my_table[str(actualCorrIndices[j])] = yValues[j]
        my_table.write(trainDataFiles[i], overwrite = True)