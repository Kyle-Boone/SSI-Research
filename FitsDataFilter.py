import numpy as np
import astropy.io.fits as fits
from numpy.random import default_rng

rng = default_rng()
contractionScale = int(input("Take 1 data point out of every: "))
inputFile = input("Enter input file path: ")
keyValue = input("Enter a property of each data entry in the fits file: ")
outputFile = input("Enter output file path: ")
newFile = bool(input("Does this file already exist? Enter \"True\" for true or \"False\" for false: "))

data = fits.open(inputFile)[1].data

restrictionIndices = rng.choice(len(data[keyValue]), size = int(len(data["bal_id"])/contractionScale), replace = False)

fits.writeto(outputFile, data[restrictionIndices], overwrite = newFile)