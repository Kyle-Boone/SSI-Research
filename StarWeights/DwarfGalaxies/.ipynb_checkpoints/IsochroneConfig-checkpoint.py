import fitsio
import numpy as np
import yaml
import simple.survey
import Config

res = 512 # Resolution for corrections.
perCovered = 0.6 # Percent of healpixel that must have valid survey properties at the 4096 scale.
sigma = 0.5 # Sigma used for gaussian weighting.
perVar = 0.98 # Percent of the variance to be captured.
perMap = 0.625 # Percent of the PC maps to use, adjust this later.
numBins = 100 # Number of points in interpolation.
classCutoff = 2.5 # Distinction between stars and galaxies.
distMod = 18.5 # Distance modulus for the isochrone.
numMagBins = 5 # Number of magnitude bins. Binning will be done based on r-band magnitude.

configFile = '/afs/hep.wisc.edu/home/kkboone/software/simple/config.yaml' # Configuration for isochrone.

# Read config file and set up "survey" object
with open(configFile) as fname:
    config = yaml.safe_load(fname)
    survey = simple.survey.Survey(config)

# Create isochrone
iso = survey.get_isochrone(distance_modulus=18.5)

matBalrStarFile = '/afs/hep.wisc.edu/bechtol-group/MegansThings/balrog_matched_catalog_sof_run2_stars_v1.4.fits'
data = fitsio.read(matBalrStarFile, columns = ['meas_psf_mag'])

mag_g = data['meas_psf_mag'][:,0]
mag_r = data['meas_psf_mag'][:,1]

magCut = np.where((mag_g > 0) & (mag_g < 37) & (mag_r > 0) & (mag_r < 37))[0]

mag_g = mag_g[magCut]
mag_r = mag_r[magCut]

mag_g_err = np.zeros_like(mag_g)
mag_r_err = np.zeros_like(mag_r)

cut = iso.cut_separation('g', 'r', mag_g, mag_r, mag_g_err, mag_r_err)

cutMag = np.sort(mag_r[cut])

splitMags = np.array_split(cutMag, 5)

magBins = [] # This will define the boundary of bins so that the number of objects is constant.
for i in np.arange(len(splitMags) - 1):
    magBins.append(splitMags[i][-1])

isoDir = '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/Kyle_Stuff/MaximumLikelihood/Isochrone_DistMod' + str(distMod) + '_Class' + str(classCutoff) + '_Res' + str(res) + '/'
oldGalaDir = '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/Kyle_Stuff/GalaxyContamination/'

deepFiles = ['/hdfs/bechtol/balrog/y3_deep_fields/y3_deep_fields_catalog/deepfields_000001.fits', 
             '/hdfs/bechtol/balrog/y3_deep_fields/y3_deep_fields_catalog/deepfields_000002.fits', 
             '/hdfs/bechtol/balrog/y3_deep_fields/y3_deep_fields_catalog/deepfields_000003.fits',
             '/hdfs/bechtol/balrog/y3_deep_fields/y3_deep_fields_catalog/deepfields_000004.fits',
             '/hdfs/bechtol/balrog/y3_deep_fields/y3_deep_fields_catalog/deepfields_000005.fits']

detBalrFile = '/hdfs/bechtol/balrog/y3/balrog_detection_catalog_sof_y3-merged_v1.2.fits'
matBalrFile = '/hdfs/bechtol/balrog/y3/balrog_matched_catalog_sof_y3-merged_v1.2.fits'
galaFile = isoDir + 'InjectionData/Galaxies.fits'

detBalrStarFile = '/afs/hep.wisc.edu/bechtol-group/MegansThings/balrog_detection_catalog_sof_run2_stars_v1.4_avg_added_match_flags.fits'
detStarFile = isoDir + 'InjectionData/Det_Stars.fits'

matBalrStarFile = '/afs/hep.wisc.edu/bechtol-group/MegansThings/balrog_matched_catalog_sof_run2_stars_v1.4.fits'
matStarFile = isoDir + 'InjectionData/Mat_Stars.fits'

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
