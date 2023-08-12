import numpy as np
from numpy import savetxt
from numpy import loadtxt
import fitsio
import healpy as hp
import Config
from scipy import interpolate as inter
from astropy.table import Table


def mostSigPCMap(redMaps, corOBJECT, sigma, numBins, aveAcc):
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
            totCor = np.sum(corOBJECT * np.exp(-1*(((onePC.astype(float) - xi) / sigma)**2)))
            totAll = np.sum(np.exp(-1*(((onePC.astype(float) - xi) / sigma)**2)))
            y.append((totCor / totAll) / aveAcc)

        y = np.array(y)
        
        # Make the error the sum of the squared difference between the binned values and 1.
        maxAdjustment.append(np.sum((y - 1)**2))
        
    maxAdjustment = np.array(maxAdjustment)
    
    mostSigIndex = np.where(maxAdjustment == np.max(maxAdjustment))[0]
    
    return mostSigIndex[0] # Return wherever the error is the largest


def singleCorrectionTrain(objectFile, condFiles, pixFile, trainDataFile, sigma, perMap, perVar, numBins, res, isStar):
    '''
    This will train a single overall correction to the data. objectFile contains the object properties. condFiles contains
    the locations of survey property maps. pixFile contains a list of valid pixels. trainDataFile is where all the 
    information necessary for performing the correction on new data is stored. sigma is the standard deviation of the 
    Gaussian kernel used in fitting. perMap is the percent of PC maps used in fitting. perVar is the percent of variance
    to capture when performing PCA. numBins is the number of data points to approximate in the fitting. res is the
    resolution of the data in healpixels. isStar is a boolean corresponding to whether the objects are stars or not.
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
    RA = objectData['RA']
    DEC = objectData['DEC']
    PIX = hp.ang2pix(res, RA, DEC, lonlat = True, nest = True)
    
    # These are the object properties that will be used.
    GMAG_PSF = objectData['GMAG_PSF']
    RMAG_PSF = objectData['RMAG_PSF']
    IMAG_PSF = objectData['IMAG_PSF']
    ZMAG_PSF = objectData['ZMAG_PSF']
    GMAG_CM = objectData['GMAG_CM']
    RMAG_CM = objectData['RMAG_CM']
    IMAG_CM = objectData['IMAG_CM']
    ZMAG_CM = objectData['ZMAG_CM']
    SIZE = objectData['SIZE']
    SIZE_ERR = objectData['SIZE_ERR']
    
    # This determines whether the classification is correct or not.
    EXTENDED_CLASS = objectData['CLASS']
    CLASS = np.zeros_like(EXTENDED_CLASS)
    CLASS[np.where((EXTENDED_CLASS == lowClass) | (EXTENDED_CLASS == highClass))[0]] = 1
    
    # This will crop to validPixels
    pixCrop = np.where(pixCheck[PIX])[0]
    
    PIX = PIX[pixCrop]
    GMAG_PSF = GMAG_PSF[pixCrop]
    RMAG_PSF = RMAG_PSF[pixCrop]
    IMAG_PSF = IMAG_PSF[pixCrop]
    ZMAG_PSF = ZMAG_PSF[pixCrop]
    GMAG_CM = GMAG_CM[pixCrop]
    RMAG_CM = RMAG_CM[pixCrop]
    IMAG_CM = IMAG_CM[pixCrop]
    ZMAG_CM = ZMAG_CM[pixCrop]
    SIZE = SIZE[pixCrop]
    SIZE_ERR = SIZE_ERR[pixCrop]
    CLASS = CLASS[pixCrop]
    
    # Store each property being used
    propMaps = []
    propMaps.append(GMAG_PSF)
    propMaps.append(RMAG_PSF)
    propMaps.append(IMAG_PSF)
    propMaps.append(ZMAG_PSF)
    propMaps.append(GMAG_CM)
    propMaps.append(RMAG_CM)
    propMaps.append(IMAG_CM)
    propMaps.append(ZMAG_CM)
    propMaps.append(SIZE)
    propMaps.append(SIZE_ERR)

    # This loops over every condition file
    for condFile in condFiles:
        condData = fitsio.read(condFile) # This reads in the data
        condSigExt = np.full(12*(res**2), -1.6375e+30) # Gives a default value
        condSigExt[validPix] = condData['SIGNAL'] # Changes all valid pixels to their corresponding signals
        propMaps.append(condSigExt[PIX]) # Only stores the values that are in pixels with injections

    propMaps = np.array(propMaps, dtype = object) # Converts to an array
    
    origCLASS = np.copy(CLASS)
    aveAcc = np.sum(CLASS) / len(CLASS)
    
    means = []
    stds = []
    # This will hold every standardized map
    stanMaps = []
    # This standardizes every map as a first step of PCA
    for j in range(len(propMaps)):
        # Store mean and std dev for later use.
        means.append(np.average(propMaps[j]))
        stds.append(np.std(propMaps[j]))
        stanMaps.append((propMaps[j] - np.average(propMaps[j])) / np.std(propMaps[j]))

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
    iterations = int(perMap * len(redMaps))

    print(iterations)
    
    timeThrough = 0
    
    for _ in np.arange(iterations):
        
        timeThrough += 1
        print(timeThrough)

        # Figure out the most significant map.
        index = mostSigPCMap(trimRedMaps, CLASS, sigma, numBins, aveAcc)

        # Store this index for later use.
        corrIndices.append(index)

        # Use this map to generate values.
        onePC = trimRedMaps[index]

        x = np.linspace(-3, 3, numBins)
        y = []

        for xi in x:
            # Gaussian weight the values when determining y Values.
            totCor = np.sum(CLASS * np.exp(-1*(((onePC.astype(float) - xi) / sigma)**2)))
            totAll = np.sum(np.exp(-1*(((onePC.astype(float) - xi) / sigma)**2)))
            y.append((totCor / totAll) / aveAcc)

        y = np.array(y)

        yValues.append(y)

        # Generate an interpolation function with constant extrapolation around the ends.
        f = inter.interp1d(x, y, bounds_error = False, fill_value = (y[0], y[-1]))

        correction = f(trimRedMaps[index].astype('float'))

        correction = 1 / correction

        # Apply correction and remove whichever principal component was used.
        CLASS = CLASS * correction

        pcMapCutoff = np.full(len(trimRedMaps), True, dtype = bool)
        pcMapCutoff[index] = False
        trimRedMaps = trimRedMaps[pcMapCutoff]

    # This is used to find th original indices used accounting for the fact that maps were removed throughout.
    actualCorrIndices = []
    originalIndices = np.arange(len(redMaps))

    for index in corrIndices:
        actualCorrIndices.append(originalIndices[index])
        originalIndices = np.delete(originalIndices, index)

    actualCorrIndices = np.array(actualCorrIndices)

    # Store data for later use
    savetxt(trainDataFile[0:-5] + '_Means.csv', means, delimiter=',')
    savetxt(trainDataFile[0:-5] + '_Stds.csv', stds, delimiter=',')
    savetxt(trainDataFile[0:-5] + '_Red_Stds.csv', redStds, delimiter=',')
    savetxt(trainDataFile[0:-5] + '_Indices.csv', actualCorrIndices, delimiter=',')
    savetxt(trainDataFile[0:-5] + '_Ave_Acc.csv', np.array([aveAcc]), delimiter=',')
    savetxt(trainDataFile[0:-5] + '_Feat_Vec.csv', featVec, delimiter='\t', fmt='%s')

    my_table = Table()
    for j in np.arange(len(actualCorrIndices)):
        my_table[str(actualCorrIndices[j])] = yValues[j]
    my_table.write(trainDataFile, overwrite = True)
    
    
def singleObjectCorrection(allProps, starTrainFile, galaTrainFile, isStar, starY, starAcc, starInds, starMeans, starSTDs, starRedSTDs, starFeatVec,  galaY, galaAcc, galaInds, galaMeans, galaSTDs, galaRedSTDs, galaFeatVec, numBins = 100):
    '''
    allProps contains all the properties for the object and conditions at a certain spot in the sky for one object. It
    should be a one dimensional array of values. starTrainFile and galaTrainFile correspond to the training files for
    stars and galaxies respectively. isStar is a boolean which says whether or not the object is a star, which will
    determine the multiplicative correction returned.
    '''
    
    x = np.linspace(-3, 3, numBins)
    
    # First calculation will be for P(O_S|T_S)
    
    yValues = starY
    aveAcc = starAcc
    indices = starInds
    means = starMeans
    stds = starSTDs
    redStds = starRedSTDs
    featVec = starFeatVec
    
    POSTS = aveAcc # Multiplicative factors will later be applied to this, starts out as average probability.
    
    stanProps = np.atleast_2d((allProps - means) / stds).T # This needs to be 2D for matrix multiplication.
    
    redProps = np.matmul(featVec, stanProps).T[0] # This reduces the maps and makes it a 1D array
    
    redProps = redProps / redStds # Normalize the reduced maps
    
    for i in range(len(indices)):

        # Generates y values
        y = yValues[str(indices[i])]

        # Generates the function via extrapolation
        f = inter.interp1d(x, y, bounds_error = False, fill_value = (y[0], y[-1]))

        # Generates the relative probability
        corr = f(redProps[indices[i]])

        # Applies the probability to this section.
        POSTS = POSTS * corr
        
    # Second Calculation for P(O_G|T_G)
    
    yValues = galaY
    aveAcc = galaAcc
    indices = galaInds
    means = galaMeans
    stds = galaSTDs
    redStds = galaRedSTDs
    featVec = galaFeatVec
    
    POGTG = aveAcc # Multiplicative factors will later be applied to this, starts out as average probability.
    
    stanProps = np.atleast_2d((allProps - means) / stds).T # This needs to be 2D for matrix multiplication.
    
    redProps = np.matmul(featVec, stanProps).T[0] # This reduces the maps and makes it a 1D array
    
    redProps = redProps / redStds # Normalize the reduced maps
    
    for i in range(len(indices)):

        # Generates y values
        y = yValues[str(indices[i])]

        # Generates the function via extrapolation
        f = inter.interp1d(x, y, bounds_error = False, fill_value = (y[0], y[-1]))

        # Generates the relative probability
        corr = f(redProps[indices[i]])

        # Applies the probability to this section.
        POGTG = POGTG * corr
        
    retval = 0
    if isStar:
        retval = POGTG / (POSTS + POGTG - 1)
    else:
        retval = (POGTG - 1) / (POSTS + POGTG - 1)
    return retval
    
