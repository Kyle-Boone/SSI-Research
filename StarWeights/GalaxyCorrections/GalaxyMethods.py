import numpy as np
from numpy import savetxt
from numpy import loadtxt
import fitsio
import healpy as hp
import Config
from scipy import interpolate as inter
from astropy.table import Table


def mostSigPCMap(redMaps, corGALA, galaALL, sigma, numBins, aveAcc):
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
            totCor = np.sum(corGALA * np.exp(-1*(((onePC.astype(float) - xi) / sigma)**2)))
            totAll = np.sum(galaALL * np.exp(-1*(((onePC.astype(float) - xi) / sigma)**2)))
            y.append((totCor / totAll) / aveAcc)

        y = np.array(y)
        
        # Make the error the sum of the squared difference between the binned values and 1.
        maxAdjustment.append(np.sum((y - 1)**2))
        
    maxAdjustment = np.array(maxAdjustment)
    
    mostSigIndex = np.where(maxAdjustment == np.max(maxAdjustment))[0]
    
    return mostSigIndex[0] # Return wherever the error is the largest


def galaExtend(pixFile, condFiles, trainDataFiles, fullSkyProbFiles, extrFiles, res, numBins, numIndBins = 100):
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
        for j in range(numIndBins):
            binLims.append(int((len(validPix) - binLims[-1]) / (numIndBins - j)) + (binLims[-1]))
            
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


def galaTrain(galaFile, condFiles, pixFile, magBins, trainDataFiles, fullSkyProbFiles, extrFiles, sigma, perMap, perVar, numBins, res):
    '''
    This method will serve to train an interpolation model which will predict the probability of misclassification of a 
    galaxy based on survey properties. This method will then call a method to expand this interpolation model to the full 
    sky. All files are assumed to be fits files. The length of trainDataFiles and fullSkyProbFiles should be one more than 
    the length of magBins.
    '''
    
    validPix = fitsio.read(pixFile)['PIXEL']
    # Boolean alternative to validPix allows for some things to be easier.
    pixCheck = np.full(12*(res**2), False, dtype = bool)
    pixCheck[validPix] = True
    
    galaData = fitsio.read(galaFile)
    
    origGalaRA = galaData['RA']
    origGalaDEC = galaData['DEC']
    galaGMAG = galaData['GMAG']
    origGalaSOFCLASSRepeats = galaData['CLASS']
    
    minMag = 0 # These two lines set default values to avoid warnings in the program. They will not be used.
    maxMag = 0
    
    for i in range(len(magBins) + 1):
        
        # This defines magnitude cuts in accordance with the magnitude bins.
        if i == 0:
            maxGMAG = magBins[i]
            magCut = np.where(galaGMAG <= maxGMAG)[0]
        elif i == len(magBins):
            minGMAG = magBins[i - 1]
            magCut = np.where(galaGMAG > minGMAG)[0]
        else:
            minGMAG = magBins[i - 1]
            maxGMAG = magBins[i]
            magCut = np.where((galaGMAG <= maxGMAG) & (galaGMAG > minGMAG))[0]
            
        # Trims down the data to the points we want.
        galaRA = origGalaRA[magCut]
        galaDEC = origGalaDEC[magCut]
        galaSOFCLASSRepeats = origGalaSOFCLASSRepeats[magCut]
        
        # A 1 denotes the correct classification, a 0 denotes an incorrect classification.
        galaCLASSRepeats = np.zeros_like(galaSOFCLASSRepeats)
        galaCLASSRepeats[np.where(galaSOFCLASSRepeats >= 2)[0]] = 1
        
        # This gives corresponding pixel numbers.
        galaPIXRepeats = hp.ang2pix(res, galaRA, galaDEC, lonlat = True, nest = True)
        
        # This sorts according to pixel.
        sortInds = galaPIXRepeats.argsort()
        galaPIXRepeats = galaPIXRepeats[sortInds[::1]]
        galaCLASSRepeats = galaCLASSRepeats[sortInds[::1]]
        
        uniqInd = 0
        galaInd = 0
        
        # This will be used to store the number of stars at each pixel.
        galaPIX = np.unique(galaPIXRepeats) # The unique pixels, with no repeats.
        galaCOR = np.zeros_like(galaPIX) # Correctly classified galaxies per pixel
        galaALL = np.zeros_like(galaPIX) # All galaxy injections per pixel.

        while galaInd < len(galaPIXRepeats):
            if galaPIX[uniqInd] == galaPIXRepeats[galaInd]: # If the pixels match up in the arrays.
                galaCOR[uniqInd] += galaCLASSRepeats[galaInd] # Add one if there was a galaxy at this location.
                galaALL[uniqInd] += 1                # Add one to the corresponding spot in the galaAll.
                galaInd += 1                         # Add one to the starInd to see if the next index in starPix is same.
                # Since the last index of galaPix and galaPixRepeats are the same, galaInd will increase 
                # the last time through the loop, making this the index that we must restrict in the while loop.
            else:
                uniqInd += 1 # If the pixels are no longer the same, increase the index you check in the galaPix array.
                
        # Restricts to valid pixels.
        galaCOR = galaCOR[pixCheck[galaPIX]]
        galaALL = galaALL[pixCheck[galaPIX]]
        galaPIX = galaPIX[pixCheck[galaPIX]]
        
        balrCondMaps = []

        # This loops over every condition file
        for condFile in condFiles:
            condData = fitsio.read(condFile) # This reads in the data
            condSigExt = np.full(12*(res**2), -1.6375e+30) # Gives a default value
            condSigExt[validPix] = condData['SIGNAL'] # Changes all valid pixels to their corresponding signals
            condSigExt[np.where(condSigExt == -1.6375e+30)[0]] = hp.UNSEEN # Masks all non valid pixels
            balrCondMaps.append(condSigExt[galaPIX]) # Only stores the values that are in pixels with injections

        balrCondMaps = np.array(balrCondMaps, dtype = object) # Converts to an array

        # Stores the original data for later comparisons
        originalGalaCOR = galaCOR
        originalGalaALL = galaALL
        aveAcc = np.sum(originalGalaCOR) / np.sum(originalGalaALL) # Average accuracy of detections.
        
        condMeans = []
        condStds = []
        balrStanMaps = []
        # This standardizes every map as a first step of PCA
        for j in range(len(balrCondMaps)):
            # Store mean and std dev for later use.
            condMeans.append(np.average(balrCondMaps[j]))
            condStds.append(np.std(balrCondMaps[j]))
            balrStanMaps.append((balrCondMaps[j] - np.average(balrCondMaps[j])) / np.std(balrCondMaps[j]))

        balrStanMaps = np.array(balrStanMaps)
        
        # This gives the covariance matrix of the standardized maps
        # Bias is true since the variance of each individual standardized map should be 1
        cov = np.cov(balrStanMaps.astype(float), bias = True)

        # This gives the eigenvalues and vectors of the covariance matrix
        evalues, evectors = np.linalg.eig(cov)

        # This cuts after the specified percentage of the variance has been achieved
        for j in range(len(evalues)):
            if np.sum(evalues[0:j+1]) / np.sum(evalues) >= perVar:
                cutoff = j + 1
                break
        featVec = evectors[0:cutoff]
        
        balrRedMaps = np.matmul(featVec, balrStanMaps) # Reduces the maps to PCA maps
        
        # Standard deviations will once more be stored for later use.
        # Maps are reduced to standard deviation of 1 for consistent x values in the following steps.
        redStds = []
        for j in np.arange(len(balrRedMaps)):
            redStds.append(np.std(balrRedMaps[j]))
            balrRedMaps[j] = balrRedMaps[j]/np.std(balrRedMaps[j])
            
        galaCOR = originalGalaCOR
        yValues = []
        corrIndices = []
        
        trimBalrRedMaps = np.copy(balrRedMaps)

        # Iterate however many times is called for.
        iterations = int(perMap * len(balrRedMaps))

        for _ in np.arange(iterations):

            # Figure out the most significant map.
            index = mostSigPCMap(trimBalrRedMaps, galaCOR, galaALL, sigma, numBins, aveAcc)

            # Store this index for later use.
            corrIndices.append(index)

            # Use this map to generate values.
            onePC = trimBalrRedMaps[index]

            x = np.linspace(-3, 3, numBins)
            y = []

            for xi in x:
                # Gaussian weight the values when determining y Values.
                totCor = np.sum(galaCOR * np.exp(-1*(((onePC.astype(float) - xi) / sigma)**2)))
                totAll = np.sum(galaALL * np.exp(-1*(((onePC.astype(float) - xi) / sigma)**2)))
                y.append((totCor / totAll) / aveAcc)

            y = np.array(y)

            yValues.append(y)

            # Generate an interpolation function with constant extrapolation around the ends.
            f = inter.interp1d(x, y, bounds_error = False, fill_value = (y[0], y[-1]))

            correction = f(trimBalrRedMaps[index].astype('float'))

            correction = 1 / correction

            # Apply correction and remove whichever principal component was used.
            galaCOR = galaCOR * correction

            pcMapCutoff = np.full(len(trimBalrRedMaps), True, dtype = bool)
            pcMapCutoff[index] = False
            trimBalrRedMaps = trimBalrRedMaps[pcMapCutoff]
            
        # This is used to find th original indices used accounting for the fact that maps were removed throughout.
        actualCorrIndices = []
        originalIndices = np.arange(len(balrRedMaps))

        for index in corrIndices:
            actualCorrIndices.append(originalIndices[index])
            originalIndices = np.delete(originalIndices, index)

        actualCorrIndices = np.array(actualCorrIndices)
        
        # Store data for later use
        savetxt(trainDataFiles[i][0:-5] + '_Means.csv', condMeans, delimiter=',')
        savetxt(trainDataFiles[i][0:-5] + '_Stds.csv', condStds, delimiter=',')
        savetxt(trainDataFiles[i][0:-5] + '_Red_Stds.csv', redStds, delimiter=',')
        savetxt(trainDataFiles[i][0:-5] + '_Indices.csv', actualCorrIndices, delimiter=',')
        savetxt(trainDataFiles[i][0:-5] + '_Ave_Acc.csv', np.array([aveAcc]), delimiter=',')
        savetxt(trainDataFiles[i][0:-5] + '_Feat_Vec.csv', featVec, delimiter='\t', fmt='%s')
        
        my_table = Table()
        for j in np.arange(len(actualCorrIndices)):
            my_table[str(actualCorrIndices[j])] = yValues[j]
        my_table.write(trainDataFiles[i], overwrite = True)
        
    # Extend this to the full sky.
    galaExtend(pixFile, condFiles, trainDataFiles, fullSkyProbFiles, extrFiles, res, numBins)
        
        