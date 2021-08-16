import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.interpolate import interp1d
from scipy.stats import norm, multivariate_normal
from scipy.interpolate import griddata
from scipy.ndimage.filters import median_filter, gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes

#from skimage.measure import label
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing

from datetime import datetime

# Caculate potential lines function
def _potential_lines(u, v): return .5*( np.cumsum(u, axis = 1) + np.cumsum(v, axis = 0) )

# Caculate stramelines function
def _streamlines(u, v): return .5*( np.cumsum(u, axis = 0) - np.cumsum(v, axis = 1) )

# Calculate vorticity approximating hte veloctiy gradient by numerical diffenciation
def _vorticity(u, v): return np.gradient(u)[1] - np.gradient(v)[0]

# Calculate divergence approximating hte veloctiy gradient by numerical diffenciation
def _divergence(u, v): return np.gradient(u)[1] + np.gradient(v)[0]

# Calculate the magnitude of a vector
def _magnitude(u, v): return np.sqrt(u**2 + v**2)

# # Functions to transform the features selected to histogram counts of groups done by distances to the Sun
# def _calculate_features_statistics(I_segm_, I_, M_, D_, V_, index_, N_y, N_x, dim):
#     # Variables Initialization
#     Xi_  = np.empty((0, 2, dim))
#     idx_ = np.zeros((N_y, N_x))
#     # Loop over the date on each group
#     for idx, g in zip(index_, range(len(index_))):
#         # Iff were pixels selected...
#         if idx.sum() > 0:
#             # Adjusting normal distribution to each group data
#             mu_0, var_0 = norm.fit(I_[idx])
#             mu_1, var_1 = norm.fit(M_[idx])
#             mu_2, var_2 = norm.fit(D_[idx])
#             mu_3, var_3 = norm.fit(V_[idx])
#             # Calculating standard deviation from variance
#             std_0 = np.sqrt(var_0)
#             std_1 = np.sqrt(var_1)
#             std_2 = np.sqrt(var_2)
#             std_3 = np.sqrt(var_3)
#         else:
#             # if not selected statistics are 0.
#             mu_0, std_0 = 0., 0.
#             mu_1, std_1 = 0., 0.
#             mu_2, std_2 = 0., 0.
#             mu_3, std_3 = 0., 0.
#         # Organize on a vector Dx1 the data from each group D = N_groups x dim
#         xi_0 = np.asarray((mu_0, std_0))[:, np.newaxis]
#         xi_1 = np.asarray((mu_1, std_1))[:, np.newaxis]
#         xi_2 = np.asarray((mu_2, std_2))[:, np.newaxis]
#         xi_3 = np.asarray((mu_3, std_3))[:, np.newaxis]
#         # Concatenate Statistics on a tensor for later defining the dataset
#         xi_ = np.concatenate((xi_0, xi_1, xi_2, xi_3), axis = 1)[np.newaxis, ...]
#         Xi_ = np.concatenate((Xi_, xi_), axis = 0)
#         # Representation of sectors by interger labels
#         idx_[idx] = g + 1
#     xi_ = Xi_[..., np.newaxis]
#     return xi_, idx_

# Clouds velocity field features extraction, and remove pixels with no clouds
def _cloud_features(F_, M_, I_segm_):
    # Remove Selected Pixels
    F_[~I_segm_.astype(bool), :] = 0.
    # Calculate Divergence and Vorticity
    M_ = _magnitude( F_[..., 0], F_[..., 1])
    D_ = _divergence(F_[..., 0], F_[..., 1])
    V_ = _vorticity( F_[..., 0], F_[..., 1])
    # Return features
    return F_, M_, D_, V_

# # Finding index of pixels expecting to intercept the Sun for each horizon (k)
# def _sun_intercepting_flow_lines(Phi_, Psi_, x_sun_, A_sun_, g_, N_y, N_x):
#     # Selecting pixels by streamlines and potential lines that intercept the Sun
#     def __select_intercepting_potential_line(Psi_sun):
#         return Psi_ > Psi_sun
#     # Connected components form the Sun streamline alon equipotential streamlines
#     def __select_intercepting_streamline(Phi_, Psi_, x_sun, y_sun, Phi_sun, idx_1_):
#         # Finding connected pixel
#         def ___next_pixel_coordiantes(Phi_, i, j, idx_sun, idx_):
#             # Defining errors matrix
#             E_ = np.zeros((3, 3))
#             # loop over nerbouring pixels
#             for k in [-1, 0, 1]:
#                 for m in [-1, 0, 1]:
#                     c = idx_[i + k - 1: i + k + 2, j + m - 1: j + m + 2].sum()
#                     if idx_[i + k, j + m] or (k == 0 and m == 0) or c > 2:
#                         E_[1 + k, 1 + m] = np.inf
#                     else:
#                         E_[1 + k, 1 + m] = ( Phi_sun - Phi_[i + k, j + m])**2
#             # Unravel error matrix coordiantes of min value error
#             k_, m_ = np.where(E_ == E_.min())
#             # Updating new streamline pixel coordiantes
#             i_new, j_new = i + k_[0] - 1, j + m_[0] - 1
#             return i_new, j_new
#         # Variables initialization
#         i, j, idx_2_ = y_sun, x_sun, np.zeros((N_y, N_x), dtype = int)
#         # Position Initialization
#         count = 1
#         idx_1_[i, j], idx_2_[i, j] = True, count
#         # Loop to the edge of the frame
#         while True:
#             count += 1
#             # Finding next pixel on the streamline
#             i, j = ___next_pixel_coordiantes(Phi_, i, j, Phi_sun, idx_1_)
#             # Updating positon
#             idx_1_[i, j], idx_2_[i, j] = True, count
#             # Finding the edge
#             if (i >= 59 or i <= 0) or (j >= 79 or j <= 0):
#                 break
#         return idx_2_
#     # Initialize List of selected sectors
#     idx_potential_ = []
#     idx_stream_   = []
#     # Loop over groups ..
#     for g in g_:
#         # No incrementes due to camera movements...
#         #x_prime, y_prime = 0., 0.
#         # Sun position absolute increments by elevation and azimuth
#         # Make it robust in case of a day begining or ending
#         if A_sun_.shape[1] - 1 < g:
#             x_prime = A_sun_[1, -1]
#             y_prime = A_sun_[0, -1]
#         else:
#             x_prime = A_sun_[1, g]
#             y_prime = A_sun_[0, g]
#         # Sun position on integer value to use it as indexing value
#         x_sun = int(np.around(x_sun_[0] + x_prime))
#         y_sun = int(np.around(x_sun_[1] - y_prime))
#         # Sun-interceptig streamline
#         Psi_sun = Psi_[y_sun, x_sun]
#         Phi_sun = Phi_[y_sun, x_sun]
#         # Selecting Sun intercepting streamline index
#         idx_potential_.append( __select_intercepting_potential_line(Psi_sun) )
#         idx_stream_.append( __select_intercepting_streamline(Phi_, Psi_, x_sun, y_sun, Phi_sun, idx_potential_[-1]) )
#     return idx_potential_, idx_stream_

# # Finding index of pixels expecting to intercept the Sun for each horizon (k)
# def _pixels_selection(XYZ_, U_, V_, idx_stream_, x_sun_, g_):
#     # Estimate circule center for selection according to estimate arrivale time of a pixel
#     def __estimate_time(XYZ_, U_, V_, Phi_idx_, i_grid):
#         # Initialize variables, identify pixels on the streamline, and proximity sorting index definition
#         Phi_bin_idx_ = Phi_idx_ > 0
#         idx_    = np.argsort(Phi_idx_[Phi_bin_idx_] - 1)
#         # Space distance on the x and y axis on a non-linear-metric
#         x_ = XYZ_[i_grid][Phi_bin_idx_, 0][idx_]
#         y_ = XYZ_[i_grid][Phi_bin_idx_, 1][idx_]
#         z_ = XYZ_[i_grid][Phi_bin_idx_, 2][idx_]
#         # Numerical differenciation on a non-linear frame-metric
#         dx_ = XYZ_[i_grid + 1][Phi_bin_idx_, 0][idx_]
#         dy_ = XYZ_[i_grid + 1][Phi_bin_idx_, 1][idx_]
#         dz_ = XYZ_[i_grid + 1][Phi_bin_idx_, 2][idx_]
#         # Calculate the velocity components for each streamline pixel
#         w_ = np.sqrt(dx_*U_[Phi_bin_idx_][idx_]**2 + (dy_*V_[Phi_bin_idx_][idx_])**2)
#         # Integrating and solving for time for each component
#         t_ = np.cumsum(dz_/(w_ + 1e-25))
#         # Organizing time instants on the matrix
#         idx_   = np.argsort(Phi_idx_.flatten())
#         t_hat_ = np.zeros(idx_.shape)
#         t_hat_[idx_[-t_.shape[0]:]] = t_
#         return t_hat_.reshape(Phi_idx_.shape)

#     # Define patch shape and distance away from the sun for horizon (k)
#     def __sector_center(t_hat_, XYZ_, Phi_idx_, x_sun_, g, i_grid):
#         # Estimate circle center
#         def ___center(XYZ_, ierror_, Phi_bin_idx_):
#             # Variables initialization
#             x_ = XYZ_[Phi_bin_idx_, 0]
#             y_ = XYZ_[Phi_bin_idx_, 1]
#             o_ = np.ones(x_.shape)
#             # Wighted mean ...
#             x_ = np.matmul(ierror_, x_.T)/np.matmul(o_, ierror_.T)
#             y_ = np.matmul(ierror_, y_.T)/np.matmul(o_, ierror_.T)
#             return x_, y_
#         y_sun = x_sun_[1]
#         x_sun = x_sun_[0]
#         #print(y_sun, x_sun)
#         # Finding streamline pixels closest to the horizon (k)
#         Phi_bin_idx_ = Phi_idx_ > 0
#         error_  = np.absolute(t_hat_[Phi_bin_idx_] - g)
#         ierror_ = np.exp( - error_)

#         # Robustness in center when is not possible to calculate
#         if ierror_.sum() != 0:
#             #ierror_ /= ierror_.sum()
#             return ___center(XYZ_[i_grid], ierror_, Phi_bin_idx_)
#         else:
#             return XYZ_[i_grid][y_sun, x_sun, 0], XYZ_[i_grid][y_sun, x_sun, 1]

#     def __get_sectors(x_, i_grid):
#         X_ = np.absolute(XYZ_[i_grid][..., 0] - x_[0])
#         Y_ = np.absolute(XYZ_[i_grid][..., 1] - x_[1])
#         return np.sqrt(X_**2 + Y_**2)

#     h_ = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
#     w_ = [0.06447865, 0.08641122, 0.07388424, 0.08255972, 0.09273319, 0.10092557, 0.11130872,
#           0.11719757, 0.11268813, 0.08855881, 0.05202304, 0.01468343, 0.00254772]

#     # Initialize List of selected sectors
#     idx_sector_ = []
#     idx_line_   = []
#     # Loop over groups ..
#     for g, i in zip(g_, range(len(g_))):
#         Phi_idx_ = idx_stream_[i]
#         # Time estimation for intercepting the Sun for each pixel on the streamline
#         y_hat_bottom_ = __estimate_time(XYZ_, U_, V_, Phi_idx_, i_grid = 0)
#         # Set of pixels selected for each horizon-group (k)
#         x_bottom_ = __sector_center(y_hat_bottom_, XYZ_, Phi_idx_, x_sun_, g, i_grid = 0)
#         Z_bottom_ = __get_sectors(x_bottom_, i_grid = 0)

#         Z_ = np.zeros((U_.shape[0], U_.shape[1], len(h_)))
#         for i in range(len(h_)):
#             Z_[..., i] = np.log(w_[i]) + np.log(Z_bottom_ * h_[i])

#         Z_ = np.sum(Z_, axis = 2)
#         #Z_ /= Z_.sum()

#         #print(Z_.min(), Z_.max())

#         #y_, x_ = np.where(Z_ == np.max(Z_))

#         #print(np.std(Z_.flatten()))
#         idx_ = Z_ < 1.25e-4
#         idx_sector_.append(idx_)
#         idx_ = Phi_idx_ > 0.
#         idx_line_.append(idx_)

#     return idx_sector_, idx_line_

__all__ = ['_potential_lines', '_streamlines', '_vorticity', '_divergence', '_magnitude']
