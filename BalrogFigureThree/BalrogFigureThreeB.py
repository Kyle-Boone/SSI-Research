import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
#import pdb;

data = fits.open("/hdfs/bechtol/balrog/y3/balrog_matched_catalog_sof_y3-merged_v1.2.fits")[1].data

true_i_mag = data['true_bdf_mag_deredden'][:,2]
meas_i_mag = data['meas_cm_mag_deredden'][:,2]
diff_i_mag = meas_i_mag - true_i_mag

first_restriction_indices = np.where(diff_i_mag > -2000)
true_i_mag = true_i_mag[first_restriction_indices]
diff_i_mag = diff_i_mag[first_restriction_indices]

binBoundaries = np.linspace(18, 26, 100, endpoint=True)
binsByMag = []
for i in range(len(binBoundaries) - 1):
    validIndices = np.array(np.where((true_i_mag >= binBoundaries[i]) & (true_i_mag <= binBoundaries[i + 1]))[0])
    validData = diff_i_mag[validIndices]
    binsByMag.append(validData)

mean = []
for i in range(len(binsByMag)):
    mean.append(np.average(binsByMag[i]))
mean = np.array(mean)

restriction_indices = np.where((true_i_mag >= 18) & (true_i_mag <= 26) & (diff_i_mag >= -10) & (diff_i_mag <= 5))[0]
true_i_mag = true_i_mag[restriction_indices]
diff_i_mag = diff_i_mag[restriction_indices]

plt.hist2d(true_i_mag, diff_i_mag, bins = (np.linspace(18,26,100, endpoint=True), np.linspace(-10,5,100, endpoint=True)), norm=LogNorm(),  cmap=plt.cm.viridis)
plt.axhline(y=0, color='k', linestyle='-')

x_for_mean = np.linspace(18,26,99,endpoint=False)
mean_restriction_indices = np.array(np.where(x_for_mean <= 25)[0])
mean = mean[mean_restriction_indices]
x_for_mean = x_for_mean[mean_restriction_indices]
plt.plot(x_for_mean, mean, color='r', linestyle='--', label = 'Mean')

plt.colorbar()
plt.legend()
plt.grid()
plt.xlabel('True i-mag (bdf)')
plt.ylabel('Mesured-True i-mag (cm-bdf; mag)')
plt.show()