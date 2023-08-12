import numpy as np
from numpy import savetxt
from numpy import loadtxt
import fitsio
import healpy as hp
import Config
from scipy import interpolate as inter
from astropy.table import Table


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



