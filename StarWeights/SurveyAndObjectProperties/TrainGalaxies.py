import numpy as np
import matplotlib.pyplot as plt
import fitsio
import astropy.io.fits as fits
from astropy.table import Table
import healpy as hp
from GetObjectProperties import *
from TrainAndExtend import *
from SegmentedGalaxyTrain import *
import Config

minGR = -0.3 # Minimum G-R color
maxGR = 1 # Maximum G-R color
res = 512 # Healpixel resolution
perCovered = 0.6 # Percent of healpixel that must have valid survey properties at the 4096 scale
sigma = 0.5 # Sigma used for gaussian weighting
perVar = 0.98 # Percent of the variance to be captured
perMap = 0.625 # Percent of the PC maps to use, adjust this later
numBins = 100 # Number of points in interpolation

conditions = Config.conditions
oldGalaDir = '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/Kyle_Stuff/GalaxyContamination/'

deepFiles = ['/hdfs/bechtol/balrog/y3_deep_fields/y3_deep_fields_catalog/deepfields_000001.fits', 
             '/hdfs/bechtol/balrog/y3_deep_fields/y3_deep_fields_catalog/deepfields_000002.fits', 
             '/hdfs/bechtol/balrog/y3_deep_fields/y3_deep_fields_catalog/deepfields_000003.fits',
             '/hdfs/bechtol/balrog/y3_deep_fields/y3_deep_fields_catalog/deepfields_000004.fits',
             '/hdfs/bechtol/balrog/y3_deep_fields/y3_deep_fields_catalog/deepfields_000005.fits']

newGalaDir = '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/Kyle_Stuff/ObjectPropertyGalaxyContamination/'

balrFile = '/hdfs/bechtol/balrog/y3/balrog_detection_catalog_sof_y3-merged_v1.2.fits'
measBalrFile = '/hdfs/bechtol/balrog/y3/balrog_matched_catalog_sof_y3-merged_v1.2.fits'
galaFile = newGalaDir + 'Blue_Galaxies.fits'

balrStarFile = '/afs/hep.wisc.edu/bechtol-group/MegansThings/balrog_detection_catalog_sof_run2_stars_v1.4_avg_added_match_flags.fits'
measBalrStarFile = '/afs/hep.wisc.edu/bechtol-group/MegansThings/balrog_matched_catalog_sof_run2_stars_v1.4.fits'
starFile = newGalaDir + 'Blue_Stars.fits'

origCondFiles = Config.files[:-1]
stelFile = Config.files[-1]
pixFile = oldGalaDir + 'Valid_'+str(res)+'_Pixels.fits'
condFiles = []
for cond in conditions:
    condFiles.append(oldGalaDir + str(res) + '_' + cond + '.fits')
condFiles = np.array(condFiles)

starTrainFile = newGalaDir + 'Star_Train.fits'
prevGalaTrainFile = newGalaDir + 'Gala_Train_1-6.fits'
newGalaTrainFile = newGalaDir + 'Gala_Train_1-7.fits'

laterCorrectionTrain(galaFile, condFiles, pixFile, newGalaTrainFile, prevGalaTrainFile, sigma, perMap, perVar, numBins, res)