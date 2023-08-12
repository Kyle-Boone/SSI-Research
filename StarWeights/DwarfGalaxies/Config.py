import numpy as np

folder ='/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/'

__files__ = []

bands = ['g', 'r', 'i', 'z']

conditions = ['Airmass Weighted Mean, g Band', 'Airmass Weighted Mean, r Band', 'Airmass Weighted Mean, i Band', 'Airmass Weighted Mean, z Band', 
             'Airmass Minimum, g Band', 'Airmass Minimum, r Band', 'Airmass Minimum, i Band', 'Airmass Minimum, z Band',
             'Airmass Maximum, g Band', 'Airmass Maximum, r Band', 'Airmass Maximum, i Band', 'Airmass Maximum, z Band', 
             'FWHM Weighted Mean, g Band', 'FWHM Weighted Mean, r Band', 'FWHM Weighted Mean, i Band', 'FWHM Weighted Mean, z Band',
             'FWHM Minimum, g Band', 'FWHM Minimum, r Band', 'FWHM Minimum, i Band', 'FWHM Minimum, z Band',
             'FWHM Maximum, g Band', 'FWHM Maximum, r Band', 'FWHM Maximum, i Band', 'FWHM Maximum, z Band',
             'FWHM Fluxrad Weighted Mean, g Band', 'FWHM Fluxrad Weighted Mean, r Band', 'FWHM Fluxrad Weighted Mean, i Band', 'FWHM Fluxrad Weighted Mean, z Band',
             'FWHM Fluxrad Minimum, g Band', 'FWHM Fluxrad Minimum, r Band', 'FWHM Fluxrad Minimum, i Band', 'FWHM Fluxrad Minimum, z Band',
             'FWHM Fluxrad Maximum, g Band', 'FWHM Fluxrad Maximum, r Band', 'FWHM Fluxrad Maximum, i Band', 'FWHM Fluxrad Maximum, z Band',
             'Exposure Time Sum, g Band', 'Exposure Time Sum, r Band', 'Exposure Time Sum, i Band', 'Exposure Time Sum, z Band',
             'Teff Weighted Mean, g Band', 'Teff Weighted Mean, r Band', 'Teff Weighted Mean, i Band', 'Teff Weighted Mean, z Band',
             'Teff Minimum, g Band', 'Teff Minimum, r Band', 'Teff Minimum, i Band', 'Teff Minimum, z Band',
             'Teff Maximum, g Band', 'Teff Maximum, r Band', 'Teff Maximum, i Band', 'Teff Maximum, z Band',
             'Teff Exposure Time Sum, g Band', 'Teff Exposure Time Sum, r Band', 'Teff Exposure Time Sum, i Band', 'Teff Exposure Time Sum, z Band',
             'Skybrite Weighted Mean, g Band', 'Skybrite Weighted Mean, r Band', 'Skybrite Weighted Mean, i Band', 'Skybrite Weighted Mean, z Band',
             'Skyvar Weighted Mean, g Band', 'Skyvar Weighted Mean, r Band', 'Skyvar Weighted Mean, i Band', 'Skyvar Weighted Mean, z Band',
             'Skyvar Minimum, g Band', 'Skyvar Minimum, r Band', 'Skyvar Minimum, i Band', 'Skyvar Minimum, z Band',
             'Skyvar Maximum, g Band', 'Skyvar Maximum, r Band', 'Skyvar Maximum, i Band', 'Skyvar Maximum, z Band',
             'Skyvar Sqrt Weighted Mean, g Band', 'Skyvar Sqrt Weighted Mean, i Band', 'Skyvar Sqrt Weighted Mean, z Band',
             'Skyvar Uncertainty, g Band', 'Skyvar Uncertainty, r Band', 'Skyvar Uncertainty, i Band', 'Skyvar Uncertainty, z Band',
             'Sigma Mag Zero QSum, g Band', 'Sigma Mag Zero QSum, r Band', 'Sigma Mag Zero QSum, i Band', 'Sigma Mag Zero QSum, z Band',
             'FGCM GRY Weighted Mean, g Band', 'FGCM GRY Weighted Mean, r Band', 'FGCM GRY Weighted Mean, i Band', 'FGCM GRY Weighted Mean, z Band',
             'FGCM GRY Minimum, g Band', 'FGCM GRY Minimum, r Band', 'FGCM GRY Minimum, i Band', 'FGCM GRY Minimum, z Band', 'Stellar Density']

# AIRMASS

air = 'airmass/y3a2_'

airmassWMEg, airmassWMEr, airmassWMEi, airmassWMEz = np.arange(4)
airmassWMEext = '_o.4096_t.32768_AIRMASS.WMEAN_EQU.fits.gz'
for band in bands:
    __files__.append(folder + air + band + airmassWMEext)
    
airmassMINg, airmassMINr, airmassMINi, airmassMINz = np.arange(4) + 4
airmassMINext = '_o.4096_t.32768_AIRMASS.MIN_EQU.fits.gz'
for band in bands:
    __files__.append(folder + air + band + airmassMINext)
    
airmassMAXg, airmassMAXr, airmassMAXi, airmassMAXz = np.arange(4) + 8
airmassMAXext = '_o.4096_t.32768_AIRMASS.MAX_EQU.fits.gz'
for band in bands:
    __files__.append(folder + air + band + airmassMAXext)

# FWHM

fwhm = 'seeing/y3a2_'

fwhmWMEg, fwhmWMEr, fwhmWMEi, fwhmWMEz = np.arange(4) + 12
fwhmWMEext = '_o.4096_t.32768_FWHM.WMEAN_EQU.fits.gz'
for band in bands:
    __files__.append(folder + fwhm + band + fwhmWMEext)
    
fwhmMINg, fwhmMINr, fwhmMINi, fwhmMINz = np.arange(4) + 16
fwhmMINext = '_o.4096_t.32768_FWHM.MIN_EQU.fits.gz'
for band in bands:
    __files__.append(folder + fwhm + band + fwhmMINext)

fwhmMAXg, fwhmMAXr, fwhmMAXi, fwhmMAXz = np.arange(4) + 20
fwhmMAXext = '_o.4096_t.32768_FWHM.MAX_EQU.fits.gz'
for band in bands:
    __files__.append(folder + fwhm + band + fwhmMAXext)

fwhmFluxradWMEg, fwhmFluxradWMEr, fwhmFluxradWMEi, fwhmFluxradWMEz = np.arange(4) + 24
fwhmFluxradWMEext = '_o.4096_t.32768_FWHM_FLUXRAD.WMEAN_EQU.fits.gz'
for band in bands:
    __files__.append(folder + fwhm + band + fwhmFluxradWMEext)

fwhmFluxradMINg, fwhmFluxradMINr, fwhmFluxradMINi, fwhmFluxradMINz = np.arange(4) + 28
fwhmFluxradMINext = '_o.4096_t.32768_FWHM_FLUXRAD.MIN_EQU.fits.gz'
for band in bands:
    __files__.append(folder + fwhm + band + fwhmFluxradMINext)

fwhmFluxradMAXg, fwhmFluxradMAXr, fwhmFluxradMAXi, fwhmFluxradMAXz = np.arange(4) + 32
fwhmFluxradMAXext = '_o.4096_t.32768_FWHM_FLUXRAD.MAX_EQU.fits.gz'
for band in bands:
    __files__.append(folder + fwhm + band + fwhmFluxradMAXext)

# EXPOSURE TIME

exp = 'exptime_teff/y3a2_'

exptimeSUMg, exptimeSUMr, exptimeSUMi, exptimeSUMz = np.arange(4) + 36
exptimeSUMext = '_o.4096_t.32768_EXPTIME.SUM_EQU.fits.gz'
for band in bands:
    __files__.append(folder + exp + band + exptimeSUMext)

teffWMEg, teffWMEr, teffWMEi, teffWMEz = np.arange(4) + 40
teffWMEext = '_o.4096_t.32768_T_EFF.WMEAN_EQU.fits.gz'
for band in bands:
    __files__.append(folder + exp + band + teffWMEext)

teffMINg, teffMINr, teffMINi, teffMINz = np.arange(4) + 44
teffMINext = '_o.4096_t.32768_T_EFF.MIN_EQU.fits.gz'
for band in bands:
    __files__.append(folder + exp + band + teffMINext)

teffMAXg, teffMAXr, teffMAXi, teffMAXz = np.arange(4) + 48
teffMAXext = '_o.4096_t.32768_T_EFF.MAX_EQU.fits.gz'
for band in bands:
    __files__.append(folder + exp + band + teffMAXext)

teffExptimeSUMg, teffExptimeSUMr, teffExptimeSUMi, teffExptimeSUMz = np.arange(4) + 52
teffExptimeSUMext = '_o.4096_t.32768_T_EFF_EXPTIME.SUM_EQU.fits.gz'
for band in bands:
    __files__.append(folder + exp + band + teffExptimeSUMext)

# SKYBRITE

bri = 'skybrite/y3a2_'

skybriteWMEg, skybriteWMEr, skybriteWMEi, skybriteWMEz = np.arange(4) + 56
skybriteWMEext = '_o.4096_t.32768_SKYBRITE.WMEAN_EQU.fits.gz'
for band in bands:
    __files__.append(folder + bri + band + skybriteWMEext)

# SKYVARIANCE

var = 'skyvar/y3a2_'

skyvarWMEg, skyvarWMEr, skyvarWMEi, skyvarWMEz = np.arange(4) + 60
skyvarWMEext = '_o.4096_t.32768_SKYVAR_WMEAN_EQU.fits.gz'
for band in bands:
    __files__.append(folder + var + band + skyvarWMEext)

skyvarMINg, skyvarMINr, skyvarMINi, skyvarMINz = np.arange(4) + 64
skyvarMINext = '_o.4096_t.32768_SKYVAR.MIN_EQU.fits.gz'
for band in bands:
    __files__.append(folder + var + band + skyvarMINext)

skyvarMAXg, skyvarMAXr, skyvarMAXi, skyvarMAXz = np.arange(4) + 68
skyvarMAXext = '_o.4096_t.32768_SKYVAR.MAX_EQU.fits.gz'
for band in bands:
    __files__.append(folder + var + band + skyvarMAXext)

skyvarSqrtWMEg, skyvarSqrtWMEi, skyvarSqrtWMEz = np.arange(3) + 72
skyvarSqrtWMEext = '_o.4096_t.32768_SKYVAR_SQRT_WMEAN_EQU.fits.gz'
for i in [0, 2, 3]:
    __files__.append(folder + var + bands[i] + skyvarSqrtWMEext)

skyvarUNCg, skyvarUNCr, skyvarUNCi, skyvarUNCz = np.arange(4) + 75
skyvarUNCext = '_o.4096_t.32768_SKYVAR.UNCERTAINTY_EQU.fits.gz'
for band in bands:
    __files__.append(folder + var + band + skyvarUNCext)

# SIGMA MAG ZERO

zpt = 'zpt_resid/y3a2_'

sigmaMagZeroQSUg, sigmaMagZeroQSUr, sigmaMagZeroQSUi, sigmaMagZeroQSUz = np.arange(4) + 79
sigmaMagZeroQSUext = '_o.4096_t.32768_SIGMA_MAG_ZERO.QSUM_EQU.fits.gz'
for band in bands:
    __files__.append(folder + zpt + band + sigmaMagZeroQSUext)

fgcmGryWMEg, fgcmGryWMEr, fgcmGryWMEi, fgcmGryWMEz = np.arange(4) + 83
fgcmGryWMEext = '_o.4096_t.32768_FGCM_GRY.WMEAN_EQU.fits.gz'
for band in bands:
    __files__.append(folder + zpt + band + fgcmGryWMEext)

fgcmGryMINg, fgcmGryMINr, fgcmGryMINi, fgcmGryMINz = np.arange(4) + 87
fgcmGryMINext = '_o.4096_t.32768_FGCM_GRY.MIN_EQU.fits.gz'
for band in bands:
    __files__.append(folder + zpt + band + fgcmGryMINext)

# STELLAR DENSITY

stellarDens = 91

__files__.append(folder + 'stellar_density/psf_stellar_density_fracdet_binned_256_nside_4096_cel.fits.gz')

files = np.copy(np.array(__files__))

starFile = '/afs/hep.wisc.edu/bechtol-group/MegansThings/balrog_detection_catalog_sof_run2_stars_v1.4_avg_added_match_flags.fits'
