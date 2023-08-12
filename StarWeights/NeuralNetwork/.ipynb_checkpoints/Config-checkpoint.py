import numpy as np

orgAirFiles = ['/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/airmass/y3a2_g_o.4096_t.32768_AIRMASS.WMEAN_EQU.fits.gz',
               '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/airmass/y3a2_r_o.4096_t.32768_AIRMASS.WMEAN_EQU.fits.gz',
               '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/airmass/y3a2_i_o.4096_t.32768_AIRMASS.WMEAN_EQU.fits.gz',
               '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/airmass/y3a2_z_o.4096_t.32768_AIRMASS.WMEAN_EQU.fits.gz']

orgDensFile = '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/stellar_density/psf_stellar_density_fracdet_binned_256_nside_4096_cel.fits.gz'

orgExpFiles = ['/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/exptime_teff/y3a2_g_o.4096_t.32768_EXPTIME.SUM_EQU.fits.gz',
               '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/exptime_teff/y3a2_r_o.4096_t.32768_EXPTIME.SUM_EQU.fits.gz',
               '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/exptime_teff/y3a2_i_o.4096_t.32768_EXPTIME.SUM_EQU.fits.gz',
               '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/exptime_teff/y3a2_z_o.4096_t.32768_EXPTIME.SUM_EQU.fits.gz']

orgVarFiles = ['/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/skyvar/y3a2_g_o.4096_t.32768_SKYVAR.UNCERTAINTY_EQU.fits.gz',
               '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/skyvar/y3a2_r_o.4096_t.32768_SKYVAR.UNCERTAINTY_EQU.fits.gz',
               '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/skyvar/y3a2_i_o.4096_t.32768_SKYVAR.UNCERTAINTY_EQU.fits.gz',
               '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/skyvar/y3a2_z_o.4096_t.32768_SKYVAR.UNCERTAINTY_EQU.fits.gz']

orgBriFiles = ['/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/skybrite/y3a2_g_o.4096_t.32768_SKYBRITE.WMEAN_EQU.fits.gz',
               '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/skybrite/y3a2_r_o.4096_t.32768_SKYBRITE.WMEAN_EQU.fits.gz',
               '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/skybrite/y3a2_i_o.4096_t.32768_SKYBRITE.WMEAN_EQU.fits.gz',
               '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/skybrite/y3a2_z_o.4096_t.32768_SKYBRITE.WMEAN_EQU.fits.gz']

orgSeeFiles = ['/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/seeing/y3a2_g_o.4096_t.32768_FWHM.WMEAN_EQU.fits.gz',
               '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/seeing/y3a2_r_o.4096_t.32768_FWHM.WMEAN_EQU.fits.gz',
               '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/seeing/y3a2_i_o.4096_t.32768_FWHM.WMEAN_EQU.fits.gz',
               '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/seeing/y3a2_z_o.4096_t.32768_FWHM.WMEAN_EQU.fits.gz']

orgZptFiles = ['/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/zpt_resid/y3a2_g_o.4096_t.32768_SIGMA_MAG_ZERO.QSUM_EQU.fits.gz',
               '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/zpt_resid/y3a2_r_o.4096_t.32768_SIGMA_MAG_ZERO.QSUM_EQU.fits.gz',
               '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/zpt_resid/y3a2_i_o.4096_t.32768_SIGMA_MAG_ZERO.QSUM_EQU.fits.gz',
               '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/zpt_resid/y3a2_z_o.4096_t.32768_SIGMA_MAG_ZERO.QSUM_EQU.fits.gz']

airBins = np.array([[1, 1.4], [1, 1.4], [1, 1.4], [1, 1.4]])
densBin = np.array([1, 6])
expBins = np.array([[100, 600], [100, 600], [100, 600], [100, 600]])
varBins = np.array([[2, 5], [3, 7], [4.5, 12], [9, 22]])
briBins = np.array([[325, 600], [800, 1600], [2000, 5000], [5000, 11000]])
seeBins = np.array([[0.9, 1.4], [0.8, 1.2], [0.775, 1.05], [0.7, 1.1]])
zptBins = np.array([[8, 16], [7, 15], [7, 15], [7, 16]]) / 1000

res = 2048
numBins = 12
aveEffs = 0.7282926019070762