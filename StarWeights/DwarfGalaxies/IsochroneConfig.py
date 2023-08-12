import fitsio
import numpy as np
import yaml
import simple.survey
import Config
from os import listdir

res = 512 # Resolution for corrections.
perCovered = 0.6 # Percent of healpixel that must have valid survey properties at the 4096 scale.
sigma = 0.5 # Sigma used for gaussian weighting.
perVar = 0.98 # Percent of the variance to be captured.
perMap = 0.625 # Percent of the PC maps to use, adjust this later.
numBins = 100 # Number of points in interpolation.
classCutoff = 2.5 # Distinction between stars and galaxies.
distMod = 18.5 # Distance modulus for the isochrone.
numMagBins = 5 # Number of magnitude bins. Binning will be done based on r-band magnitude.

qualityCuts = [24.5, 24.25] # r and i band cuts respectively

configFile = '/afs/hep.wisc.edu/home/kkboone/software/simple/config.yaml' # Configuration for isochrone.

# Read config file and set up "survey" object
with open(configFile) as fname:
    config = yaml.safe_load(fname)
    survey = simple.survey.Survey(config)

# Create isochrone
iso = survey.get_isochrone(distance_modulus=distMod)

isoDir = '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/Kyle_Stuff/MaximumLikelihood/Isochrone_DistMod' + str(distMod) + '_Class' + str(classCutoff) + '_Res' + str(res) + '/'

detBalrStarFile = '/afs/hep.wisc.edu/bechtol-group/MegansThings/balrog_detection_catalog_sof_run2_stars_v1.4_avg_added_match_flags.fits'
detStarFile = isoDir + 'InjectionData/Det_Stars.fits'

matBalrStarFile = '/afs/hep.wisc.edu/bechtol-group/MegansThings/balrog_matched_catalog_sof_run2_stars_v1.4.fits'
matStarFile = isoDir + 'InjectionData/Mat_Stars.fits'

deepFiles = ['/hdfs/bechtol/balrog/y3_deep_fields/y3_deep_fields_catalog/deepfields_000001.fits', 
             '/hdfs/bechtol/balrog/y3_deep_fields/y3_deep_fields_catalog/deepfields_000002.fits', 
             '/hdfs/bechtol/balrog/y3_deep_fields/y3_deep_fields_catalog/deepfields_000003.fits',
             '/hdfs/bechtol/balrog/y3_deep_fields/y3_deep_fields_catalog/deepfields_000004.fits',
             '/hdfs/bechtol/balrog/y3_deep_fields/y3_deep_fields_catalog/deepfields_000005.fits']

deepCols = ['KNN_CLASS', 'RA', 'DEC', 'MASK_FLAGS', 'MASK_FLAGS_NIR']

detBalrFile = '/hdfs/bechtol/balrog/y3/balrog_detection_catalog_sof_y3-merged_v1.2.fits'
matBalrFile = '/hdfs/bechtol/balrog/y3/balrog_matched_catalog_sof_y3-merged_v1.2.fits'
galaFile = isoDir + 'InjectionData/Galaxies.fits'

conditions = Config.conditions
origCondFiles = Config.files[:-1]
stelFile = Config.files[-1]
pixFile = isoDir + 'PixAndConds/Valid_Pixels.fits'
condFiles = []
for cond in conditions:
    condFiles.append(isoDir + 'PixAndConds/' + cond + '.fits')
condFiles = np.array(condFiles)

galaDir = isoDir + 'Galaxies/'

galaExtrFiles = []
galaTrainFiles =  []
galaProbFiles = []
for i in np.arange(numMagBins):
    galaExtrFiles.append(galaDir + 'Gala_Extr_Bin' + str(i+1) + '.fits')
    galaTrainFiles.append(galaDir + 'Gala_Train_Bin' + str(i+1) + '.fits')
    galaProbFiles.append(galaDir + 'Gala_Prob_Bin' + str(i+1) + '.fits')
    
starDir = isoDir + 'Stars/'

starExtrFiles = []
starTrainFiles =  []
starProbFiles = []
for i in np.arange(numMagBins):
    starExtrFiles.append(starDir + 'Star_Extr_Bin' + str(i+1) + '.fits')
    starTrainFiles.append(starDir + 'Star_Train_Bin' + str(i+1) + '.fits')
    starProbFiles.append(starDir + 'Star_Prob_Bin' + str(i+1) + '.fits')
    
starDetExtrFiles = []
starDetTrainFiles =  []
starDetProbFiles = []
for i in np.arange(numMagBins):
    starDetExtrFiles.append(starDir + 'Star_Det_Extr_Bin' + str(i+1) + '.fits')
    starDetTrainFiles.append(starDir + 'Star_Det_Train_Bin' + str(i+1) + '.fits')
    starDetProbFiles.append(starDir + 'Star_Det_Prob_Bin' + str(i+1) + '.fits')
    
obsStarDir = isoDir + 'GoldObjects/Stars/'
obsGalaDir = isoDir + 'GoldObjects/Galaxies/'

obsStarFiles = []
obsGalaFiles = []
for i in np.arange(numMagBins):
    obsStarFiles.append(obsStarDir + 'Bin' + str(i+1) + '.fits')
    obsGalaFiles.append(obsGalaDir + 'Bin' + str(i+1) + '.fits')
    
# These should not be changed
goldObjectsDir = '/hdfs/bechtol/balrog/y3/y3a2_gold_v2p2_skim/healpixel2/'
goldObjectsFiles = listdir(goldObjectsDir)
goldCols = ['FLAGS_FOREGROUND', 'FLAGS_BADREGIONS', 'FLAGS_FOOTPRINT', 'EXTENDED_CLASS_SOF', 'SOF_PSF_MAG_G', 'SOF_PSF_MAG_R', 'SOF_PSF_MAG_I', 'SOF_CM_MAG_G', 'SOF_CM_MAG_R', 'SOF_CM_MAG_I', 'RA', 'DEC']

correctionFile = isoDir + 'Correction/MultiplicativeCorrections.fits'

noCorrectionFile = isoDir + 'Correction/NoCorrections.fits'

caliDir = isoDir + 'Calibration/'

starPosFiles = []
galaPosFiles = []
calibrationFile = caliDir + 'Calibrations.fits'
for i in np.arange(numMagBins):
    starPosFiles.append(caliDir + 'StarPos_Bin' + str(i+1) + '.fits')
    galaPosFiles.append(caliDir + 'GalaPos_Bin' + str(i+1) + '.fits')
    
countsFile = isoDir + 'Correction/StellarCounts.fits'
OGBinsCountFile = isoDir + 'Correction/ObservedGalaxyCountBins.fits'
TSBinsCountFile = isoDir + 'Correction/TrueStarCountBins.fits'
