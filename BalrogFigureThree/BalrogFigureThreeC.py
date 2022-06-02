import numpy as np
import fitsio
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# /afs/hep.wisc.edu/home/kkboone/data/temp/Balrog_Matched_Filtered.fits
# /hdfs/bechtol/balrog/y3/balrog_matched_catalog_sof_y3-merged_v1.2.fits

filename = '/hdfs/bechtol/balrog/y3/balrog_matched_catalog_sof_y3-merged_v1.2.fits'
data = fitsio.read(filename, columns=['true_bdf_mag_deredden', 'meas_cm_mag_deredden', 'true_bdf_mag', 'meas_cm_mag'])

true_g_mag = data['true_bdf_mag_deredden'][:,0]
meas_g_mag = data['meas_cm_mag_deredden'][:,0]

true_r_mag = data['true_bdf_mag_deredden'][:,1]
meas_r_mag = data['meas_cm_mag_deredden'][:,1]

true_g_mag_red = data['true_bdf_mag'][:,0]
meas_g_mag_red = data['meas_cm_mag'][:,0]

true_r_mag_red = data['true_bdf_mag'][:,1]
meas_r_mag_red = data['meas_cm_mag'][:,1]

first_restriction_indices = np.where((meas_g_mag_red > -9999) & (meas_r_mag_red > -9999) & (true_g_mag_red < 37.5) & (true_r_mag_red < 37.5) & (meas_g_mag_red < 37.5) & (meas_r_mag_red < 37.5))

true_g_mag = true_g_mag[first_restriction_indices]
meas_g_mag = meas_g_mag[first_restriction_indices]
true_r_mag = true_r_mag[first_restriction_indices]
meas_r_mag = meas_r_mag[first_restriction_indices]

true_gr_color = true_g_mag - true_r_mag
meas_gr_color = meas_g_mag - meas_r_mag
diff_gr_color = meas_gr_color - true_gr_color

# binBoundaries = np.linspace(0, 2, 100, endpoint=True)
# binsByMag = []
# for i in range(len(binBoundaries) - 1):
#     validIndices = np.array(np.where((true_gr_color >= binBoundaries[i]) & (true_gr_color <= binBoundaries[i + 1]))[0])
#     validData = diff_gr_color[validIndices]
#     binsByMag.append(validData)

# mean = []
# for i in range(len(binsByMag)):
#     mean.append(np.average(binsByMag[i]))
# mean = np.array(mean)

# median = []
# for i in range(len(binsByMag)):
#     median.append(np.median(binsByMag[i]))
# median = np.array(median)

plt.hist2d(true_gr_color, diff_gr_color, bins = (np.linspace(-20,20,100, endpoint=True), np.linspace(-20,20,100, endpoint=True)), cmap=plt.cm.jet, norm=LogNorm())
plt.axhline(y=0, color='k', linestyle=':')

# x_for_mean = np.linspace(0,2,99,endpoint=False)
# plt.plot(x_for_mean, mean, color='r', linestyle=':', label = 'Mean')
# plt.plot(x_for_mean, median, color='r', linestyle='--', label = 'Median')

# plt.legend()
plt.colorbar()
plt.xlabel('True g-r Color')
plt.ylabel('Measured - True Color')
plt.grid()
plt.show()
