"""
Goals for this are to calculate out classification probabilities. This will also include extrapolating these maps to the full sky. Functions will be approximated by a product of one dimensional corrections. This file is being split from the detection rate variation file due to slight differences throughout each piece of the method which could potenitially cause confusions. One large difference is that detection rate variations are typically calculated through N/<N> variations, while for classification rates we have exact counts on each pixel, so we don't need to make the assumption of uniform injection, which not only can fail on small scales, but would also not be a valid assumption since some pixels will have lower overall detection rates leading to less objects to use.
"""

import numpy as np
from numpy import savetxt
from numpy import loadtxt
import fitsio
import healpy as hp
import Config
from scipy import interpolate as inter
from astropy.table import Table


def mostSigInd(y, cutOffPercent):
    maxSquaredDiff = 0
    index = -1
    
    maxSingError = np.max(np.abs(y - 1))
    
    if maxSingError <= cutOffPercent:
        return index
    
    for i in range(len(y)):
        yi = y[i]
        
        diff = np.sum((yi - 1)**2)
        
        if diff > maxSquaredDiff:
            maxSquaredDiff = diff
            index = i
            
    return index


def singleCorrectionTrain(objectFile, condFiles, pixFile, magBins, trainDataFiles, fullSkyProbFiles, extrFiles, numBins, res, isStar, classCut, binNum, cutOffPercent):
    '''
    This will train a single overall correction to the data. objectFile contains the object properties. condFiles contains the locations of survey property maps. pixFile contains a list of valid pixels. trainDataFile is where all the information necessary for performing the correction on new data is stored. sigma is the standard deviation of the Gaussian kernel used in fitting. perMap is the percent of PC maps used in fitting. perVar is the percent of variance to capture when performing PCA. numBins is the number of data points to approximate in the fitting. res is the resolution of the data in healpixels. isStar is a boolean corresponding to whether the objects are stars or not. isDet is whether this is a det based one or not. iterations can be filled with a new value if desired. fullSkyProbFiles is where probabilities will be stores, extrFiles are where extrapolations will be stored. alreadyTrained corresponds to the number of maps already trained. Just useful if an error occurs and stops code halfway through execution.
    '''
    
    # Classes corresponding to a correct classification.
    if isStar:
        lowClass = 0
        highClass = classCut
    else:
        lowClass = classCut
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
    RMAG = objectData['RMAG']
    
    EXTENDED_CLASS = objectData['CLASS']
    origCLASS = np.zeros_like(EXTENDED_CLASS)
    origCLASS[np.where((EXTENDED_CLASS >= lowClass) & (EXTENDED_CLASS <= highClass))[0]] = 1
    
    for i in range(len(magBins) - 1):
        
        # This defines magnitude cuts in accordance with the magnitude bins.
        # If below is uncommented again, include -10 in mag cut for RMAG
        minRMAG = magBins[i]
        maxRMAG = magBins[i + 1]
        magCut = np.where(((RMAG <= maxRMAG) & (RMAG > minRMAG)))[0]
            
        RA = origRA[magCut]
        DEC = origDEC[magCut]
        CLASSRepeats = origCLASS[magCut]
        
        # This gives corresponding pixel numbers.
        PIXRepeats = hp.ang2pix(res, RA, DEC, lonlat = True, nest = True)
        CORPIXRepeats = PIXRepeats[np.where(CLASSRepeats == 1)[0]]
        
        # This will be used to store the number of objects at each pixel.
        PIX, TOT = np.unique(PIXRepeats, return_counts = True)
        _, COR = np.unique(np.append(PIX, CORPIXRepeats), return_counts = True)
        COR = COR - 1
        
        condMaps = []

        numSurveyProps = len(condFiles)
        # This loops over every condition file
        for condFile in condFiles:
            condData = fitsio.read(condFile) # This reads in the data
            condSigExt = np.full(12*(res**2), -1.6375e+30) # Gives a default value
            condSigExt[validPix] = condData['SIGNAL'] # Changes all valid pixels to their corresponding signals
            condMaps.append(condSigExt[PIX]) # Only stores the values that are in pixels with injections

        condMaps = np.array(condMaps, dtype = object) # Converts to an array
        
        aveAcc = np.sum(COR) / np.sum(TOT)

        yValues = []
        corrIndices = []
        
        sortInds = []
        for j in range(len(condMaps)):
            sortInds.append(condMaps[j].argsort())
        sortInds = np.array(sortInds)

        binIndLims = [0]

        for j in range(binNum):
            binIndLims.append(int((len(condMaps[0]) - binIndLims[-1]) / (binNum - j)) + (binIndLims[-1]))

        xBins = []

        for j in range(len(condMaps)):
            cond_Map_Sort = condMaps[j][sortInds[j][::1]]
            condBins = []
            for k in range(binNum):
                condBins.append(cond_Map_Sort[binIndLims[k]:binIndLims[k+1]])
            indXBin = []

            for k in range(binNum):
                indXBin.append(np.sum(condBins[k]) / len(condBins[k]))

            xBins.append(np.array(indXBin))

        xBins = np.array(xBins)
        
        while(True):

            yBins = []
            for j in range(len(condMaps)):
                corSort = COR[sortInds[j][::1]]
                totSort = TOT[sortInds[j][::1]]
                corBins = []
                totBins = []
                for k in range(binNum):
                    corBins.append(corSort[binIndLims[k]:binIndLims[k+1]])
                    totBins.append(totSort[binIndLims[k]:binIndLims[k+1]])
                indYBin = []

                for k in range(binNum):
                    indYBin.append((np.sum(corBins[k]) / np.sum(totBins[k])) / aveAcc)

                yBins.append(np.array(indYBin))

            yBins = np.array(yBins)

            index = mostSigInd(yBins, cutOffPercent)
            if index == -1:
                break
            else:
                corrIndices.append(index)
                yValues.append(yBins[index])

            corrFunc = inter.interp1d(xBins[index], yBins[index], bounds_error = False, fill_value = (yBins[index][0], yBins[index][-1]))

            COR = COR / (corrFunc(condMaps[index].astype('float')))

            COR = COR * aveAcc / (np.sum(COR) / np.sum(TOT))

        storeCorrIndices = []
        
        for j in range(len(corrIndices)):
            storeCorrIndices.append(corrIndices[j] + 92*j)

        storeCorrIndices = np.array(storeCorrIndices)

        # Store data for later use
        savetxt(trainDataFiles[i][0:-5] + '_Indices.csv', storeCorrIndices, delimiter=',')
        savetxt(trainDataFiles[i][0:-5] + '_Ave_Acc.csv', np.array([aveAcc]), delimiter=',')

        y_table = Table()
        for j in np.arange(len(storeCorrIndices)):
            y_table[str(storeCorrIndices[j])] = yValues[j]
        y_table.write(trainDataFiles[i], overwrite = True)
        
        x_table = Table()
        for j in np.arange(len(xBins)):
            x_table[str(j)] = xBins[j]
        x_table.write(trainDataFiles[i][0:-5] + '_X_Values.fits', overwrite = True)
        
    # Extend this to the full sky.
    fullSky(pixFile, condFiles, trainDataFiles, fullSkyProbFiles, extrFiles, res, numBins)
    
    
def fullSky(pixFile, condFiles, trainDataFiles, fullSkyProbFiles, extrFiles, res, numBins):
    '''
    This function serves to extend the corrections dependend on principal component values to the full sky.
    '''
    
    validPix = fitsio.read(pixFile)['PIXEL']
    
    for i in range(len(trainDataFiles)):
        # This reads in the training data that was just gathered.
        yValues = fitsio.read(trainDataFiles[i])
        xValues = fitsio.read(trainDataFiles[i][0:-5] + '_X_Values.fits')
        aveAcc = loadtxt(trainDataFiles[i][0:-5] + '_Ave_Acc.csv', delimiter=',')
        print(aveAcc)
        indices = loadtxt(trainDataFiles[i][0:-5] + '_Indices.csv', delimiter=',').astype(int)
        
        # This makes sure we only read in 1% of the full sky at a time to avoid overflow.
        binLims = [0]
        for j in range(numBins):
            binLims.append(int((len(validPix) - binLims[-1]) / (numBins - j)) + (binLims[-1]))
            
        # This will be used to record extrapolations.
        extrMap = []
        for j in range(len(indices)):
            extrMap.append([])
        
        # This will be the probabilities map.
        probMap = []

        for j in range(len(binLims) - 1):
            # if j <= 75:
            #     continue
            # if j > 75:
            #     continue
            print(j)

            condMaps = []
            # This loops over every condition file.
            for condFile in condFiles:
                condMaps.append(fitsio.read(condFile, rows = np.arange(binLims[j], binLims[j + 1]))['SIGNAL'])
                # This reads in the data.
            condMaps = np.array(condMaps, dtype = object)

            sectProb = np.ones(len(condMaps[0]))
            # This is the corrections for this specific section of pixels read in.

            for k in range(len(indices)):

                # Finds all places where extrapolation is necessary and marks them
                extrapolation = np.zeros(len(condMaps[indices[k] % 92]))
                
                # Get x and y values
                x = xValues[str(indices[k] % 92)]
                y = yValues[str(indices[k])]
                
                extrapolation[np.where((condMaps[indices[k] % 92] > x[-1]) | (condMaps[indices[k] % 92] < x[0]))[0]] = 1

                # Generates the function via extrapolation
                f = inter.interp1d(x, y, bounds_error = False, fill_value = (y[0], y[-1]))

                # Generates the relative probability
                corr = f(condMaps[indices[k] % 92].astype('float'))

                # Applies the probability to this section.
                sectProb = sectProb * corr

                # For this index, expands out the extrapolation map.
                extrMap[k].extend(extrapolation)

            # Multiplies the relative probability by the average probability.
            probMap.extend(sectProb * aveAcc)

        probMap = np.array(probMap)
        
        # This collapses all of the extrapolations down into one map.
        fullExtrMap = np.zeros_like(probMap)
        for j in range(len(extrMap)):
            fullExtrMap = fullExtrMap + np.array((extrMap[j]))
            
#         oldProbs = fitsio.read(fullSkyProbFiles[i])['SIGNAL']
#         oldExtrs = fitsio.read(extrFiles[i])['EXTRAPOLATIONS']
        
#         probMap = np.append(oldProbs, probMap)
#         fullExtrMap = np.append(oldExtrs, fullExtrMap)
        
        # This stores the probabilities and extrapolations.
        my_table = Table()
        my_table['SIGNAL'] = probMap
        my_table.write(fullSkyProbFiles[i], overwrite = True) 
        
        ext_table = Table()
        ext_table['EXTRAPOLATIONS'] = fullExtrMap
        ext_table.write(extrFiles[i], overwrite = True)
