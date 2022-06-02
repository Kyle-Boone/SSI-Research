import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt

data = fits.open("/hdfs/bechtol/balrog/y3/balrog_detection_catalog_sof_y3-merged_v1.2.fits")[1].data

bands = ['g', 'r', 'i', 'z']
colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
binSize = 0.25
thresholdPercent = 0.005
cutoffMagLow = 15
cutoffMagHigh = 30
# reset the plot configurations to default
plt.rcdefaults()
# set the axes color glbally for all plots
plt.rcParams.update({'axes.facecolor':'white'})

def binByMagAndBand(band, binSize = binSize, data = data):
    binBoundaries = np.arange(cutoffMagLow, cutoffMagHigh + binSize, binSize)
    binsByMag = []
    for i in range(len(binBoundaries) - 1):
        validIndices = np.array(np.where((data['true_bdf_mag_deredden'][:,band] >= binBoundaries[i]) & (data['true_bdf_mag_deredden'][:,band] < binBoundaries[i + 1]))[0])
        validData = data[validIndices]
        binsByMag.append(validData)
    return binsByMag

gBinnedData = np.array(binByMagAndBand(0), dtype = object)
rBinnedData = np.array(binByMagAndBand(1), dtype = object)
iBinnedData = np.array(binByMagAndBand(2), dtype = object)
zBinnedData = np.array(binByMagAndBand(3), dtype = object)

gMaxCountsInBin = np.max([len(bin) for bin in gBinnedData])
rMaxCountsInBin = np.max([len(bin) for bin in rBinnedData])
iMaxCountsInBin = np.max([len(bin) for bin in iBinnedData])
zMaxCountsInBin = np.max([len(bin) for bin in zBinnedData])

gIndicesToGraph = np.where([len(bin) for bin in gBinnedData] > (gMaxCountsInBin * thresholdPercent))[0]
rIndicesToGraph = np.where([len(bin) for bin in rBinnedData] > (rMaxCountsInBin * thresholdPercent))[0]
iIndicesToGraph = np.where([len(bin) for bin in iBinnedData] > (iMaxCountsInBin * thresholdPercent))[0]
zIndicesToGraph = np.where([len(bin) for bin in zBinnedData] > (zMaxCountsInBin * thresholdPercent))[0]

gHighCountBins = gBinnedData[gIndicesToGraph]
rHighCountBins = rBinnedData[rIndicesToGraph]
iHighCountBins = iBinnedData[iIndicesToGraph]
zHighCountBins = zBinnedData[zIndicesToGraph]

def percentDetectedByBin(highCountBins):
    percentDetected = []
    for i in range(len(highCountBins)):
        totalEntries = len(highCountBins[i]['bal_id'])
        detectedEntries = len(np.where(highCountBins[i][:]['detected'] == 1)[0])
        percentDetected.append(detectedEntries/totalEntries)
    percentDetected = np.array(percentDetected)
    return percentDetected

gPercentDetectedBins = percentDetectedByBin(gHighCountBins)
rPercentDetectedBins = percentDetectedByBin(rHighCountBins)
iPercentDetectedBins = percentDetectedByBin(iHighCountBins)
zPercentDetectedBins = percentDetectedByBin(zHighCountBins)
percentDetected = np.array([gPercentDetectedBins, rPercentDetectedBins, iPercentDetectedBins, zPercentDetectedBins], dtype = 'object')

gMagnitudePerBin = cutoffMagLow + binSize * gIndicesToGraph
rMagnitudePerBin = cutoffMagLow + binSize * rIndicesToGraph
iMagnitudePerBin = cutoffMagLow + binSize * iIndicesToGraph
zMagnitudePerBin = cutoffMagLow + binSize * zIndicesToGraph
magnitudes = np.array([gMagnitudePerBin, rMagnitudePerBin, iMagnitudePerBin, zMagnitudePerBin], dtype = 'object')

for i in range(len(bands)):
    plt.plot(magnitudes[i], percentDetected[i], label = bands[i] + ' Band', color = colors[i])
plt.xlabel('Magnitude')
plt.ylabel('Percent Detected')
plt.title('Detection Rates as a Function of Magnitude')
plt.legend()
plt.grid()
plt.show()