import fitsio
import numpy as np
import Config
from os import listdir
from ugali.analysis.isochrone import factory as isochrone_factory
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection


def mkpol(mu, age=12., z=0.0004, dmu=0.5, C=[0.05, 0.05], E=4., err=None, survey='DECaLS', clip=None):
    if err == None:
        print('Using DES err!')
        err = surveys.surveys['DES_DR1']['err']
    """ Builds ordered polygon for masking """

    iso = isochrone_factory('Bressan2012', survey='des',
                            age=age, distance_modulus=mu, z=z)
    c = iso.color
    m = iso.mag

    # clip=4
    # clip = 3.4
    if clip is not None:
        # Clip for plotting, use gmin otherwise
        # clip abs mag
        cut = (m > clip) & ((m + mu) < 240) & (c > 0) & (c < 1)
        c = c[cut]
        m = m[cut]

    mnear = m + mu - dmu / 2.
    mfar = m + mu + dmu / 2.
    C = np.r_[c + E * err(mfar) + C[1], c[::-1] -  E * err(mnear[::-1]) - C[0]]
    M = np.r_[m, m[::-1]]
    return np.c_[C, M],iso
err=lambda x: (0.0010908679647672335 + np.exp((x - 27.091072029215375) / 1.0904624484538419))

def feh2z( feh):
        # Section 3 of Dotter et al. 2008
        Y_p     = 0.245            # Primordial He abundance (WMAP, 2003)
        c       = 1.54             # He enrichment ratio 

        # This is not well defined...
        #Z_solar/X_solar = 0.0229  # Solar metal fraction (Grevesse 1998)
        ZX_solar = 0.0229
        return (1 - Y_p)/( (1 + c) + (1/ZX_solar) * 10**(-feh))
    

# Hyperparameter setup
res = 512 # Resolution for corrections.
perCovered = 0.6 # Percent of healpixel that must have valid survey properties at the 4096 scale.
sigma = 0.5 # Sigma used for gaussian weighting.
perVar = 0.98 # Percent of the variance to be captured.
perMap = 0.625 # Percent of the PC maps to use, adjust this later.
numBins = 100 # Number of points in interpolation.
classCutoff = 1.5 # Distinction between stars and galaxies.
gCut = 27 # Cutoff for g magnitude
numMagBins = 6 # Number of magnitude bins
    
# Isochrone setup
mu = 16.2
age=12.8
feh=-2.5
z=feh2z(feh)

mk,iso=mkpol(mu,age,z,dmu=0.5,C=[0.01,0.1],E=2,err=err, survey="DES_Y3A2")
path=Path(mk)

# This is an old directory, but it has many files in it already so I don't have to redo all my work.
isoDir = '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/Kyle_Stuff/MaximumLikelihood/Isochrone_DistMod18.5_Class2.5_Res512/'

conditions = Config.conditions
origCondFiles = Config.files[:-1]
stelFile = Config.files[-1]
pixFile = isoDir + 'PixAndConds/Valid_Pixels.fits'
condFiles = []
for cond in conditions:
    condFiles.append(isoDir + 'PixAndConds/' + cond + '.fits')
condFiles = np.array(condFiles)

# This will be the new directory with any new files in it.
stellarDir = '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/Kyle_Stuff/MaximumLikelihood/PhoenixStellarStream_Class'+str(classCutoff)+'/'

detStarFile = stellarDir + 'InjectionData/Det_Stars.fits'
matStarFile = stellarDir + 'InjectionData/Mat_Stars.fits'
detStarMeasDataFile = stellarDir + 'InjectionData/Det_Stars_Meas_Data.fits'
detStarAllPosFile = stellarDir + 'InjectionData/Det_Stars_All_Position_Data.fits'

detGalaFile = stellarDir + 'InjectionData/Det_Galaxies.fits'
matGalaFile = stellarDir + 'InjectionData/Mat_Galaxies.fits'
detGalaMeasDataFile = stellarDir + 'InjectionData/Det_Galaxies_Meas_Data.fits'
detGalaAllPosFile = stellarDir + 'InjectionData/Det_Galaxies_All_Position_Data.fits'

galaDir = stellarDir + 'Galaxies/'

galaExtrFiles = []
galaTrainFiles =  []
galaProbFiles = []
for i in np.arange(numMagBins):
    galaExtrFiles.append(galaDir + 'Gala_Extr_Bin' + str(i+1) + '.fits')
    galaTrainFiles.append(galaDir + 'Gala_Train_Bin' + str(i+1) + '.fits')
    galaProbFiles.append(galaDir + 'Gala_Prob_Bin' + str(i+1) + '.fits')
    
galaDetExtrFiles = []
galaDetTrainFiles =  []
galaDetProbFiles = []
for i in np.arange(numMagBins):
    galaDetExtrFiles.append(galaDir + 'Gala_Det_Extr_Bin' + str(i+1) + '.fits')
    galaDetTrainFiles.append(galaDir + 'Gala_Det_Train_Bin' + str(i+1) + '.fits')
    galaDetProbFiles.append(galaDir + 'Gala_Det_Prob_Bin' + str(i+1) + '.fits')
    
galaDetAsGalaExtrFiles = []
galaDetAsGalaTrainFiles =  []
galaDetAsGalaProbFiles = []
for i in np.arange(numMagBins):
    galaDetAsGalaExtrFiles.append(galaDir + 'Gala_Det_As_Gala_Extr_Bin' + str(i+1) + '.fits')
    galaDetAsGalaTrainFiles.append(galaDir + 'Gala_Det_As_Gala_Train_Bin' + str(i+1) + '.fits')
    galaDetAsGalaProbFiles.append(galaDir + 'Gala_Det_As_Gala_Prob_Bin' + str(i+1) + '.fits')
    
galaDetMeasDataExtrFiles = []
galaDetMeasDataTrainFiles =  []
galaDetMeasDataProbFiles = []
for i in np.arange(numMagBins):
    galaDetMeasDataExtrFiles.append(galaDir + 'Gala_Det_Meas_Data_Extr_Bin' + str(i+1) + '.fits')
    galaDetMeasDataTrainFiles.append(galaDir + 'Gala_Det_Meas_Data_Train_Bin' + str(i+1) + '.fits')
    galaDetMeasDataProbFiles.append(galaDir + 'Gala_Det_Meas_Data_Prob_Bin' + str(i+1) + '.fits')
    
galaDetMeasDataAsGalaExtrFiles = []
galaDetMeasDataAsGalaTrainFiles =  []
galaDetMeasDataAsGalaProbFiles = []
for i in np.arange(numMagBins):
    galaDetMeasDataAsGalaExtrFiles.append(galaDir + 'Gala_Det_Meas_Data_As_Gala_Extr_Bin' + str(i+1) + '.fits')
    galaDetMeasDataAsGalaTrainFiles.append(galaDir + 'Gala_Det_Meas_Data_As_Gala_Train_Bin' + str(i+1) + '.fits')
    galaDetMeasDataAsGalaProbFiles.append(galaDir + 'Gala_Det_Meas_Data_As_Gala_Prob_Bin' + str(i+1) + '.fits')
    
galaDetAsAnyExtrFiles = []
galaDetAsAnyTrainFiles =  []
galaDetAsAnyProbFiles = []
for i in np.arange(numMagBins):
    galaDetAsAnyExtrFiles.append(galaDir + 'Gala_Det_As_Any_Extr_Bin' + str(i+1) + '.fits')
    galaDetAsAnyTrainFiles.append(galaDir + 'Gala_Det_As_Any_Train_Bin' + str(i+1) + '.fits')
    galaDetAsAnyProbFiles.append(galaDir + 'Gala_Det_As_Any_Prob_Bin' + str(i+1) + '.fits')
    
starDir = stellarDir + 'Stars/'

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
    
starDetAsGalaExtrFiles = []
starDetAsGalaTrainFiles =  []
starDetAsGalaProbFiles = []
for i in np.arange(numMagBins):
    starDetAsGalaExtrFiles.append(starDir + 'Star_Det_As_Gala_Extr_Bin' + str(i+1) + '.fits')
    starDetAsGalaTrainFiles.append(starDir + 'Star_Det_As_Gala_Train_Bin' + str(i+1) + '.fits')
    starDetAsGalaProbFiles.append(starDir + 'Star_Det_As_Gala_Prob_Bin' + str(i+1) + '.fits')
    
starDetMeasDataExtrFiles = []
starDetMeasDataTrainFiles =  []
starDetMeasDataProbFiles = []
for i in np.arange(numMagBins):
    starDetMeasDataExtrFiles.append(starDir + 'Star_Det_Meas_Data_Extr_Bin' + str(i+1) + '.fits')
    starDetMeasDataTrainFiles.append(starDir + 'Star_Det_Meas_Data_Train_Bin' + str(i+1) + '.fits')
    starDetMeasDataProbFiles.append(starDir + 'Star_Det_Meas_Data_Prob_Bin' + str(i+1) + '.fits')
    
starDetMeasDataAsGalaExtrFiles = []
starDetMeasDataAsGalaTrainFiles =  []
starDetMeasDataAsGalaProbFiles = []
for i in np.arange(numMagBins):
    starDetMeasDataAsGalaExtrFiles.append(starDir + 'Star_Det_Meas_Data_As_Gala_Extr_Bin' + str(i+1) + '.fits')
    starDetMeasDataAsGalaTrainFiles.append(starDir + 'Star_Det_Meas_Data_As_Gala_Train_Bin' + str(i+1) + '.fits')
    starDetMeasDataAsGalaProbFiles.append(starDir + 'Star_Det_Meas_Data_As_Gala_Prob_Bin' + str(i+1) + '.fits')
    
starDetAsAnyExtrFiles = []
starDetAsAnyTrainFiles =  []
starDetAsAnyProbFiles = []
for i in np.arange(numMagBins):
    starDetAsAnyExtrFiles.append(starDir + 'Star_Det_As_Any_Extr_Bin' + str(i+1) + '.fits')
    starDetAsAnyTrainFiles.append(starDir + 'Star_Det_As_Any_Train_Bin' + str(i+1) + '.fits')
    starDetAsAnyProbFiles.append(starDir + 'Star_Det_As_Any_Prob_Bin' + str(i+1) + '.fits')
    
goldStarDir = stellarDir + 'GoldObjects/Stars/'
goldGalaDir = stellarDir + 'GoldObjects/Galaxies/'

goldStarFiles = []
goldGalaFiles = []
for i in np.arange(numMagBins):
    goldStarFiles.append(goldStarDir + 'Bin' + str(i+1) + '.fits')
    goldGalaFiles.append(goldGalaDir + 'Bin' + str(i+1) + '.fits')
    
goldMoreInfoStarFiles = []
goldMoreInfoGalaFiles = []
for i in np.arange(numMagBins):
    goldMoreInfoStarFiles.append(goldStarDir + 'More_Info_Bin' + str(i+1) + '.fits')
    goldMoreInfoGalaFiles.append(goldGalaDir + 'More_Info_Bin' + str(i+1) + '.fits')
    
correctionFile = stellarDir + 'Correction/MultiplicativeCorrections.fits'

caliDir = stellarDir + 'Calibration/'

starPosFiles = []
galaPosFiles = []
calibrationFile = caliDir + 'Calibrations.fits'
for i in np.arange(numMagBins):
    starPosFiles.append(caliDir + 'StarPos_Bin' + str(i+1) + '.fits')
    galaPosFiles.append(caliDir + 'GalaPos_Bin' + str(i+1) + '.fits')
    
phoenixFile = stellarDir + 'Phoenix_Data.fits'

# Assorted data files.
detBalrStarFile = '/afs/hep.wisc.edu/bechtol-group/MegansThings/balrog_detection_catalog_sof_run2_stars_v1.4_avg_added_match_flags.fits'
matBalrStarFile = '/afs/hep.wisc.edu/bechtol-group/MegansThings/balrog_matched_catalog_sof_run2_stars_v1.4.fits'

detBalrFile = '/hdfs/bechtol/balrog/y3/balrog_detection_catalog_sof_y3-merged_v1.2.fits'
matBalrFile = '/hdfs/bechtol/balrog/y3/balrog_matched_catalog_sof_y3-merged_v1.2.fits'

deepFiles = ['/hdfs/bechtol/balrog/y3_deep_fields/y3_deep_fields_catalog/deepfields_000001.fits', 
             '/hdfs/bechtol/balrog/y3_deep_fields/y3_deep_fields_catalog/deepfields_000002.fits', 
             '/hdfs/bechtol/balrog/y3_deep_fields/y3_deep_fields_catalog/deepfields_000003.fits',
             '/hdfs/bechtol/balrog/y3_deep_fields/y3_deep_fields_catalog/deepfields_000004.fits',
             '/hdfs/bechtol/balrog/y3_deep_fields/y3_deep_fields_catalog/deepfields_000005.fits']

deepCols = ['KNN_CLASS', 'RA', 'DEC', 'MASK_FLAGS', 'MASK_FLAGS_NIR']

goldObjectsDir = '/hdfs/bechtol/balrog/y3/y3a2_gold_v2p2_skim/healpixel2/'
goldObjectsFiles = listdir(goldObjectsDir)
goldCols = ['FLAGS_FOREGROUND', 'FLAGS_BADREGIONS', 'FLAGS_FOOTPRINT', 'EXTENDED_CLASS_SOF', 'SOF_PSF_MAG_G', 'SOF_PSF_MAG_R', 'SOF_CM_MAG_G', 'SOF_CM_MAG_R', 'RA', 'DEC']