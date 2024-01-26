import numpy as np
from numpy import savetxt
from numpy import loadtxt
import fitsio
import healpy as hp
import Config
from scipy import interpolate as inter
from astropy.table import Table


def mostSigPCMap(redMaps, Cor, allData, sigma, numBins, aveAcc):
    '''
    The purpose of this map is to find which PC map has the largest systematic variation from the average probability of 
    detection.
    '''
    
    maxAdjustment = []

    for i in range(len(redMaps)):
        
        onePC = redMaps[i] # Load up a PC map
        
        totAll = allData[i]

        x = np.linspace(-3, 3, numBins) # xValues for plot, goes out to 3 standard deviation.
        y = []
        
        for j in range(len(x)):
            # Gaussian weighting the values close by to each x value.
            totCor = np.sum(Cor * np.exp(-1*(((onePC.astype(float) - x[j]) / sigma)**2)))
            # totAll = np.sum(All * np.exp(-1*(((onePC.astype(float) - xi) / sigma)**2)))
            y.append((totCor / totAll[j]) / aveAcc)

        y = np.array(y)
        
        # Make the error the sum of the squared difference between the binned values and 1.
        maxAdjustment.append(np.sum((y - 1)**2))
        
    maxAdjustment = np.array(maxAdjustment)
    
    mostSigIndex = np.where(maxAdjustment == np.max(maxAdjustment))[0]
    
    return mostSigIndex[0] # Return wherever the error is the largest


# def mostSigPCMap(redMaps, Cor, All, sigma, numBins, aveAcc):
#     '''
#     The purpose of this map is to find which PC map has the largest systematic variation from the average probability of 
#     detection.
#     '''
    
#     maxAdjustment = []

#     for i in range(len(redMaps)):
        
#         onePC = redMaps[i] # Load up a PC map

#         x = np.linspace(-3, 3, numBins) # xValues for plot, goes out to 3 standard deviation.
#         y = []
        
#         for xi in x:
#             # Gaussian weighting the values close by to each x value.
#             totCor = np.sum(Cor * np.exp(-1*(((onePC.astype(float) - xi) / sigma)**2)))
#             totAll = np.sum(All * np.exp(-1*(((onePC.astype(float) - xi) / sigma)**2)))
#             y.append((totCor / totAll) / aveAcc)

#         y = np.array(y)
        
#         # Make the error the sum of the squared difference between the binned values and 1.
#         maxAdjustment.append(np.sum((y - 1)**2))
        
#     maxAdjustment = np.array(maxAdjustment)
    
#     mostSigIndex = np.where(maxAdjustment == np.max(maxAdjustment))[0]
    
#     return mostSigIndex[0] # Return wherever the error is the largest


def fullSky(pixFile, condFiles, trainDataFiles, fullSkyProbFiles, extrFiles, res, numBins):
    '''
    This function serves to extend the corrections dependend on principal component values to the full sky.
    '''
    
    validPix = fitsio.read(pixFile)['PIXEL']
    
    x = np.linspace(-3, 3, numBins)
    
    for i in range(len(trainDataFiles)):
        # This reads in the training data that was just gathered.
        yValues = fitsio.read(trainDataFiles[i])
        aveAcc = loadtxt(trainDataFiles[i][0:-5] + '_Ave_Acc.csv', delimiter=',')
        print(aveAcc)
        indices = loadtxt(trainDataFiles[i][0:-5] + '_Indices.csv', delimiter=',').astype(int)
        condMeans = loadtxt(trainDataFiles[i][0:-5] + '_Means.csv', delimiter=',')
        condStds = loadtxt(trainDataFiles[i][0:-5] + '_Stds.csv', delimiter=',')
        redStds = loadtxt(trainDataFiles[i][0:-5] + '_Red_Stds.csv', delimiter=',')
        
        # Ths reads in the feature vector and converts it to floating points.
        strFeatVec = loadtxt(trainDataFiles[i][0:-5] + '_Feat_Vec.csv', delimiter = '\t', dtype = object)
        featVec = []
        for j in range(len(strFeatVec)):

            indVec = []

            for k in range(len(strFeatVec[j])):

                indVec.append(float(strFeatVec[j][k]))

            indVec = np.array(indVec)
            featVec.append(indVec)

        featVec = np.array(featVec, dtype = object)
        
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

            condMaps = []
            # This loops over every condition file.
            for condFile in condFiles:
                condMaps.append(fitsio.read(condFile, rows = np.arange(binLims[j], binLims[j + 1]))['SIGNAL'])
                # This reads in the data.
            condMaps = np.array(condMaps, dtype = object)    

            stanMaps = []
            for k in range(len(condMaps)):
                stanMaps.append((condMaps[k] - condMeans[k]) / condStds[k])
                # Create the standardized maps, using the means and stds from last time.
            stanMaps = np.array(stanMaps, dtype = object)

            redMaps = np.matmul(featVec, stanMaps)
            # This created the reduced maps using PCA and then standardizes them with standard deviations from training.
            for k in range(len(redMaps)):
                redMaps[k] = redMaps[k] / redStds[k]

            sectProb = np.ones(len(stanMaps[0]))
            # This is the corrections for this specific section of pixels read in.

            for k in range(len(indices)):

                # Finds all places where extrapolation is necessary and marks them
                extrapolation = np.zeros(len(redMaps[indices[k]]))
                extrapolation[np.where((redMaps[indices[k]] > 3) | (redMaps[indices[k]] < -3))[0]] = 1

                # Generates y values
                y = yValues[str(indices[k])]

                # Generates the function via extrapolation
                f = inter.interp1d(x, y, bounds_error = False, fill_value = (y[0], y[-1]))

                # Generates the relative probability
                corr = f(redMaps[indices[k]].astype('float'))

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
            
        # This stores the probabilities and extrapolations.
        my_table = Table()
        my_table['SIGNAL'] = probMap
        my_table.write(fullSkyProbFiles[i], overwrite = True) 
        
        ext_table = Table()
        ext_table['EXTRAPOLATIONS'] = fullExtrMap
        ext_table.write(extrFiles[i], overwrite = True)
        
        
def singleCorrectionTrainNMethod(objectFile, condFiles, pixFile, magBins, allPosFile, trainDataFiles, fullSkyProbFiles, extrFiles, sigma, perMap, perVar, numBins, res, isStar, classCut, iterations = -1, alreadyTrained = 0):
    
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
    origPIX = hp.ang2pix(res, origRA, origDEC, nest = True, lonlat = True)
    
    # These are the object properties that will be used.
    RMAG = objectData['RMAG']
    
    EXTENDED_CLASS = objectData['CLASS']
    CLASS_CUT = np.where((EXTENDED_CLASS >= lowClass) & (EXTENDED_CLASS <= highClass))[0]
    
    origPIX = origPIX[CLASS_CUT]
    RMAG = RMAG[CLASS_CUT]
    
    allObjectData = fitsio.read(allPosFile)
    allRA = allObjectData['RA']
    allDEC = allObjectData['DEC']
    allPIX = np.unique(hp.ang2pix(res, allRA, allDEC, nest = True, lonlat = True))
    allPIX = allPIX[np.isin(allPIX, validPix)]
    
    RMAG = RMAG[np.isin(origPIX, allPIX)]
    origPIX = origPIX[np.isin(origPIX, allPIX)]
    
    condMaps = []

    # This loops over every condition file
    for condFile in condFiles:
        condData = fitsio.read(condFile) # This reads in the data
        condSigExt = np.full(12*(res**2), -1.6375e+30) # Gives a default value
        condSigExt[validPix] = condData['SIGNAL'] # Changes all valid pixels to their corresponding signals
        condMaps.append(condSigExt[allPIX]) # Only stores the values that are in pixels with injections

    condMaps = np.array(condMaps, dtype = object) # Converts to an array

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
        
    x = np.linspace(-3, 3, numBins)
        
    origTotAllData = []
    for i in np.arange(len(redMaps)):
        indTotData = np.array([])
        for xi in x:
            indTotData = np.append(indTotData, np.sum(np.exp(-1*(((redMaps[i].astype(float) - xi) / sigma)**2))))
        origTotAllData.append(indTotData)
    origTotAllData = np.array(origTotAllData)
        
    for i in range(len(magBins)):
        
        totAllData = np.copy(origTotAllData)
        
        if i < alreadyTrained:
            continue
        
        # This defines magnitude cuts in accordance with the magnitude bins.
        # If below is uncommented again, include -10 in mag cut for RMAG
        if i == 0:
            maxRMAG = magBins[i]
            magCut = np.where((RMAG <= maxRMAG))[0]
        else:
            minRMAG = magBins[i - 1]
            maxRMAG = magBins[i]
            magCut = np.where(((RMAG <= maxRMAG) & (RMAG > minRMAG)))[0]
            
        magPix = origPIX[magCut]
        
        _, magPixCounts = np.unique(np.append(allPIX, magPix), return_counts = True)
        magPixCounts = magPixCounts - 1
        
        # magPixCounts = np.zeros(12*res**2)
        # for pix in magPix:
        #     magPixCounts[pix] += 1
        # magPixCounts = magPixCounts[np.unique(allPIX)]
        
        yValues = []
        corrIndices = []

        trimRedMaps = np.copy(redMaps)

        # Iterate however many times is called for.
        if iterations < 0:
            iterations = int(perMap * len(redMaps))
    
        print('Iterations: ' + str(iterations))
    
        timeThrough = 0
        
        aveAcc = np.sum(magPixCounts) / len(magPixCounts)
        
        for _ in np.arange(iterations):

            timeThrough += 1
            print(timeThrough)

            # Figure out the most significant map.
            index = mostSigPCMap(trimRedMaps, magPixCounts, totAllData, sigma, numBins, aveAcc)

            # Store this index for later use.
            corrIndices.append(index)

            # Use this map to generate values.
            onePC = trimRedMaps[index]
            y = []
            
            totAll = totAllData[index]

            for j in range(len(x)):
                # Gaussian weight the values when determining y Values.
                totCor = np.sum(magPixCounts * np.exp(-1*(((onePC.astype(float) - x[j]) / sigma)**2)))
                y.append((totCor / totAll[j]) / aveAcc)

            y = np.array(y)

            yValues.append(y)

            # Generate an interpolation function with constant extrapolation around the ends.
            f = inter.interp1d(x, y, bounds_error = False, fill_value = (y[0], y[-1]))

            correction = f(trimRedMaps[index].astype('float'))

            correction = 1 / correction

            # Apply correction and remove whichever principal component was used.
            magPixCounts = magPixCounts * correction

            pcMapCutoff = np.full(len(trimRedMaps), True, dtype = bool)
            pcMapCutoff[index] = False
            trimRedMaps = trimRedMaps[pcMapCutoff]
            totAllData = totAllData[pcMapCutoff]

        # This is used to find th original indices used accounting for the fact that maps were removed throughout.
        actualCorrIndices = []
        originalIndices = np.arange(len(redMaps))

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
        
    # Extend this to the full sky.
    fullSky(pixFile, condFiles, trainDataFiles, fullSkyProbFiles, extrFiles, res, numBins)



def singleCorrectionTrain(objectFile, condFiles, pixFile, magBins, trainDataFiles, fullSkyProbFiles, extrFiles, sigma, perMap, perVar, numBins, res, isStar, classCut, iterations = -1, alreadyTrained = 0):
    '''
    This will train a single overall correction to the data. objectFile contains the object properties. condFiles contains
    the locations of survey property maps. pixFile contains a list of valid pixels. trainDataFile is where all the 
    information necessary for performing the correction on new data is stored. sigma is the standard deviation of the 
    Gaussian kernel used in fitting. perMap is the percent of PC maps used in fitting. perVar is the percent of variance
    to capture when performing PCA. numBins is the number of data points to approximate in the fitting. res is the
    resolution of the data in healpixels. isStar is a boolean corresponding to whether the objects are stars or not.
    isDet is whether this is a det based one or not. iterations can be filled with a new value if desired. fullSkyProbFiles
    is where probabilities will be stores, extrFiles are where extrapolations will be stored. alreadyTrained corresponds
    to the number of maps already trained. Just useful if an error occurs and stops code halfway through execution.
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
    
    for i in range(len(magBins)):
        
        if i < alreadyTrained:
            continue
        
        # This defines magnitude cuts in accordance with the magnitude bins.
        # If below is uncommented again, include -10 in mag cut for RMAG
        if i == 0:
            maxRMAG = magBins[i]
            magCut = np.where((RMAG <= maxRMAG))[0]
        else:
            minRMAG = magBins[i - 1]
            maxRMAG = magBins[i]
            magCut = np.where(((RMAG <= maxRMAG) & (RMAG > minRMAG)))[0]
        
        # Measured magnitude det variations, use entire list, but only keep the valid mags as having the class flag.
        # Edit out this and differing magnitudes to go back to original setup.
        # if np.min(RMAG) == -10:
        #     RA = np.copy(origRA)
        #     DEC = np.copy(origDEC)
        #     boolMagCut = np.full(len(origCLASS), False, dtype = bool)
        #     boolMagCut[magCut] = True
        #     CLASSRepeats = np.copy(origCLASS)
        #     CLASSRepeats[~boolMagCut] = 0
        # else:
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

        # Iterate however many times is called for.
        if iterations < 0:
            iterations = int(perMap * len(redMaps))
    
        print('Iterations: ' + str(iterations))
        
        x = np.linspace(-3, 3, numBins)
        
        totAllData = []
        for j in np.arange(len(redMaps)):
            indTotData = np.array([])
            for xi in x:
                indTotData = np.append(indTotData, np.sum(TOT * np.exp(-1*(((redMaps[j].astype(float) - xi) / sigma)**2))))
            totAllData.append(indTotData)
        totAllData = np.array(totAllData)
    
        timeThrough = 0
    
        for _ in np.arange(iterations):

            timeThrough += 1
            print(timeThrough)

            # Figure out the most significant map.
            index = mostSigPCMap(trimRedMaps, COR, totAllData, sigma, numBins, aveAcc)

            # Store this index for later use.
            corrIndices.append(index)

            # Use this map to generate values.
            onePC = trimRedMaps[index]

            x = np.linspace(-3, 3, numBins)
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
            totAllData = totAllData[pcMapCutoff]

        # This is used to find th original indices used accounting for the fact that maps were removed throughout.
        actualCorrIndices = []
        originalIndices = np.arange(len(redMaps))

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
        
    # Extend this to the full sky.
    fullSky(pixFile, condFiles, trainDataFiles, fullSkyProbFiles, extrFiles, res, numBins)
    
    
def singleCorrectionTrainNMethodDeRes(objectFile, condFiles, pixFile, deResPixFile, magBins, allPosFile, trainDataFiles, fullSkyProbFiles, extrFiles, sigma, perMap, perVar, numBins, origRes, deRes, isStar, classCut, iterations = -1, alreadyTrained = 0):
    
    # Classes corresponding to a correct classification.
    if isStar:
        lowClass = 0
        highClass = classCut
    else:
        lowClass = classCut
        highClass = 3
        
    validPix = fitsio.read(pixFile)['PIXEL']
    # Boolean alternative to validPix allows for some things to be easier.
    pixCheck = np.full(12*(origRes**2), False, dtype = bool)
    pixCheck[validPix] = True
    
    deResValidPix = fitsio.read(deResPixFile)['PIXEL']
    
    objectData = fitsio.read(objectFile)
    
    # Use these to get pixel locations for every object.
    origRA = objectData['RA']
    origDEC = objectData['DEC']
    origPIX = hp.ang2pix(deRes, origRA, origDEC, nest = True, lonlat = True)
    
    # These are the object properties that will be used.
    RMAG = objectData['RMAG']
    
    EXTENDED_CLASS = objectData['CLASS']
    CLASS_CUT = np.where((EXTENDED_CLASS >= lowClass) & (EXTENDED_CLASS <= highClass))[0]
    
    origPIX = origPIX[CLASS_CUT]
    RMAG = RMAG[CLASS_CUT]
    
    allObjectData = fitsio.read(allPosFile)
    allRA = allObjectData['RA']
    allDEC = allObjectData['DEC']
    allPIX = np.unique(hp.ang2pix(deRes, allRA, allDEC, nest = True, lonlat = True))
    allPIX = allPIX[np.isin(allPIX, deResValidPix)]
    
    RMAG = RMAG[np.isin(origPIX, allPIX)]
    origPIX = origPIX[np.isin(origPIX, allPIX)]
    
    condMaps = []

    # This loops over every condition file
    for condFile in condFiles:
        condData = fitsio.read(condFile) # This reads in the data
        condSigExt = np.full(12*(origRes**2), -1.6375e+30) # Gives a default value
        condSigExt[validPix] = condData['SIGNAL'] # Changes all valid pixels to their corresponding signals
        condMaps.append(condSigExt[validPix]) # Only stores the values that are in pixels with injections in that 512 pixel

    condMaps = np.array(condMaps, dtype = object) # Converts to an array

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
        
    for j in np.arange(len(redMaps)):
        
        fullRedMap = np.full(12*(origRes**2), hp.UNSEEN)
        fullRedMap[validPix] = redMaps[j]
        fullRedMap = hp.ud_grade(fullRedMap, deRes, order_in = 'NESTED', order_out = 'NESTED')
        validRedMap = fullRedMap[deResValidPix]
        
        if j == 0:
            deGradeRedMaps = np.atleast_2d(validRedMap)
        else:
            deGradeRedMaps = np.append(deGradeRedMaps, np.atleast_2d(validRedMap), axis = 0)
            
    redMaps = deGradeRedMaps
        
    x = np.linspace(-3, 3, numBins)
        
    origTotAllData = []
    for i in np.arange(len(redMaps)):
        indTotData = np.array([])
        for xi in x:
            indTotData = np.append(indTotData, np.sum(np.exp(-1*(((redMaps[i].astype(float) - xi) / sigma)**2))))
        origTotAllData.append(indTotData)
    origTotAllData = np.array(origTotAllData)
        
    for i in range(len(magBins)):
        
        totAllData = np.copy(origTotAllData)
        
        if i < alreadyTrained:
            continue
        
        # This defines magnitude cuts in accordance with the magnitude bins.
        # If below is uncommented again, include -10 in mag cut for RMAG
        if i == 0:
            maxRMAG = magBins[i]
            magCut = np.where((RMAG <= maxRMAG))[0]
        else:
            minRMAG = magBins[i - 1]
            maxRMAG = magBins[i]
            magCut = np.where(((RMAG <= maxRMAG) & (RMAG > minRMAG)))[0]
            
        magPix = origPIX[magCut]
        
        _, magPixCounts = np.unique(np.append(allPIX, magPix), return_counts = True)
        magPixCounts = magPixCounts - 1
        
        # magPixCounts = np.zeros(12*res**2)
        # for pix in magPix:
        #     magPixCounts[pix] += 1
        # magPixCounts = magPixCounts[np.unique(allPIX)]
        
        yValues = []
        corrIndices = []

        trimRedMaps = np.copy(redMaps)

        # Iterate however many times is called for.
        if iterations < 0:
            iterations = int(perMap * len(redMaps))
    
        print('Iterations: ' + str(iterations))
    
        timeThrough = 0
        
        aveAcc = np.sum(magPixCounts) / len(magPixCounts)
        
        for _ in np.arange(iterations):

            timeThrough += 1
            print(timeThrough)

            # Figure out the most significant map.
            index = mostSigPCMap(trimRedMaps, magPixCounts, totAllData, sigma, numBins, aveAcc)

            # Store this index for later use.
            corrIndices.append(index)

            # Use this map to generate values.
            onePC = trimRedMaps[index]
            y = []
            
            totAll = totAllData[index]

            for j in range(len(x)):
                # Gaussian weight the values when determining y Values.
                totCor = np.sum(magPixCounts * np.exp(-1*(((onePC.astype(float) - x[j]) / sigma)**2)))
                y.append((totCor / totAll[j]) / aveAcc)

            y = np.array(y)

            yValues.append(y)

            # Generate an interpolation function with constant extrapolation around the ends.
            f = inter.interp1d(x, y, bounds_error = False, fill_value = (y[0], y[-1]))

            correction = f(trimRedMaps[index].astype('float'))

            correction = 1 / correction

            # Apply correction and remove whichever principal component was used.
            magPixCounts = magPixCounts * correction

            pcMapCutoff = np.full(len(trimRedMaps), True, dtype = bool)
            pcMapCutoff[index] = False
            trimRedMaps = trimRedMaps[pcMapCutoff]
            totAllData = totAllData[pcMapCutoff]

        # This is used to find th original indices used accounting for the fact that maps were removed throughout.
        actualCorrIndices = []
        originalIndices = np.arange(len(redMaps))

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
        
    # Extend this to the full sky.
    fullSky(pixFile, condFiles, trainDataFiles, fullSkyProbFiles, extrFiles, origRes, numBins)
