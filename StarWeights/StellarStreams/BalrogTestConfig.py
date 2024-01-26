import fitsio
import numpy as np
import Config
from os import listdir

# All tests

testDir = '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/Kyle_Stuff/MaximumLikelihood/BalrogTests/'

sigma = 0.5 # Sigma used for gaussian weighting.
perVar = 0.98 # Percent of the variance to be captured.
perMap = 0.625 # Percent of the PC maps to use, adjust this later.
numBins = 100 # Number of points in interpolation.
perCovered = 0.6 # Percent of healpixel that must have valid survey properties at the 4096 scale.

detBalrStarFile = '/afs/hep.wisc.edu/bechtol-group/MegansThings/balrog_detection_catalog_sof_run2_stars_v1.4_avg_added_match_flags.fits'
matBalrStarFile = '/afs/hep.wisc.edu/bechtol-group/MegansThings/balrog_matched_catalog_sof_run2_stars_v1.4.fits'
detBalrGalaFile = '/hdfs/bechtol/balrog/y3/balrog_detection_catalog_sof_y3-merged_v1.2.fits'

deepFiles = ['/hdfs/bechtol/balrog/y3_deep_fields/y3_deep_fields_catalog/deepfields_000001.fits', 
             '/hdfs/bechtol/balrog/y3_deep_fields/y3_deep_fields_catalog/deepfields_000002.fits', 
             '/hdfs/bechtol/balrog/y3_deep_fields/y3_deep_fields_catalog/deepfields_000003.fits',
             '/hdfs/bechtol/balrog/y3_deep_fields/y3_deep_fields_catalog/deepfields_000004.fits',
             '/hdfs/bechtol/balrog/y3_deep_fields/y3_deep_fields_catalog/deepfields_000005.fits']

conditions = Config.conditions
origCondFiles = Config.files[:-1]
stelFile = Config.files[-1]

# Test 1

test1Dir = testDir + 'Test1/'

test1Res = 4096
test1MagBin = [22]

test1CondFiles = []
for cond in conditions:
    test1CondFiles.append(test1Dir + 'Conds/' + cond + '.fits')
test1CondFiles = np.array(test1CondFiles)

test1StarFile = test1Dir + 'BalrogObjects.fits'
test1AllPixFile = test1Dir + 'AllBalrogPixels.fits'
test1ValidPixFile = test1Dir + 'ValidPix.fits'

test1TrainDir = test1Dir + 'Training/'
test1ExtrFile = np.array([test1TrainDir + 'Extrapolation.fits'])
test1TrainFile =  np.array([test1TrainDir + 'Train.fits'])
test1ProbFile = np.array([test1TrainDir + 'Prob.fits'])

# Test 1a

test1aDir = testDir + 'Test1a/'

test1aRes = 4096
test1aMagBin = [22]

test1aCondFiles = []
for cond in conditions:
    test1aCondFiles.append(test1aDir + 'Conds/' + cond + '.fits')
test1aCondFiles = np.array(test1aCondFiles)

test1aGalaFile = test1aDir + 'BalrogObjects.fits'
test1aAllPixFile = test1aDir + 'AllBalrogPixels.fits'
test1aValidPixFile = test1aDir + 'ValidPix.fits'

test1aTrainDir = test1aDir + 'Training/'
test1aExtrFile = np.array([test1aTrainDir + 'Extrapolation.fits'])
test1aTrainFile =  np.array([test1aTrainDir + 'Train.fits'])
test1aProbFile = np.array([test1aTrainDir + 'Prob.fits'])

# Test 1b

test1bDir = testDir + 'Test1b/'

test1bRes = 4096
test1bDeRes = 512
test1bMagBin = [22]

test1bCondFiles = []
for cond in conditions:
    test1bCondFiles.append(test1bDir + 'Conds/' + cond + '.fits')
test1bCondFiles = np.array(test1bCondFiles)

test1bStarFile = test1bDir + 'BalrogObjects.fits'
test1bAllPixFile = test1bDir + 'AllBalrogPixels.fits'

test1bValidPixFile = test1bDir + 'ValidPix.fits'
test1bValidPixDeResFile = test1bDir + 'ValidPixDeRes.fits'

test1bTrainDir = test1bDir + 'Training/'
test1bExtrFile = np.array([test1bTrainDir + 'Extrapolation.fits'])
test1bTrainFile =  np.array([test1bTrainDir + 'Train.fits'])
test1bProbFile = np.array([test1bTrainDir + 'Prob.fits'])

# Test 1b

test1cDir = testDir + 'Test1c/'

test1cRes = 4096
test1cDeRes = 512
test1cMagBin = [25]

test1cMagCut = [22, 24]
test1cColorCut = [-0.3, 1]
test1cClassCut = 1.5

test1cCondFiles = []
for cond in conditions:
    test1cCondFiles.append(test1cDir + 'Conds/' + cond + '.fits')
test1cCondFiles = np.array(test1cCondFiles)

test1cStarFile = test1cDir + 'BalrogObjects.fits'
test1cAllPixFile = test1cDir + 'AllBalrogPixels.fits'

test1cValidPixFile = test1cDir + 'ValidPix.fits'
test1cValidPixDeResFile = test1cDir + 'ValidPixDeRes.fits'

test1cTrainDir = test1cDir + 'Training/'
test1cExtrFile = np.array([test1cTrainDir + 'Extrapolation.fits'])
test1cTrainFile =  np.array([test1cTrainDir + 'Train.fits'])
test1cProbFile = np.array([test1cTrainDir + 'Prob.fits'])
