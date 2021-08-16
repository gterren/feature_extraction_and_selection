import pickle, glob, sys, os
import numpy as np

from matplotlib import pyplot as plt

from scipy.ndimage.filters import median_filter, gaussian_filter
from scipy.special import gamma, gammainc, iv
from scipy.stats import gamma as _Ga
from scipy.stats import vonmises as _Vm
from scipy.stats import beta as _Be
from scipy.stats import mode, multivariate_normal, norm, skew, kurtosis

from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing

# Import All custom functions
from utils import *
from detection_clustering_utils import *
from image_processing_utils import *
from feature_extraction_utils import *
from cloud_segmentation_utils import *

# Transform velocity vectors velocities according to the the average hight of a layer label
def _velocity_vector_prespective_transformation_v1(F_, labels_, dXYZ_, verbose = True):
    W_ = F_.copy()
    # Frames per second
    fps = 1./15.
    # Loop over average pixels height per layer
    for label in np.unique(labels_):
        # Velocity Vectors in the i-th wind flow layer
        idx_ = labels_ == label
        # Transform Velocity vectors from pixels/frame to meters/second
        W_[idx_, 0] = W_[idx_, 0] * dXYZ_[idx_, 0] * fps
        W_[idx_, 1] = W_[idx_, 1] * dXYZ_[idx_, 1] * fps

        if verbose:
            v_x = np.mean(W_[idx_, 0])
            v_y = np.mean(W_[idx_, 1])
            v, a = _cart_to_polar(v_x[np.newaxis], v_y[np.newaxis])
            print(r'Wind Flow Layer = {} Cartesian: v_x = {} [m/s] v_y = {} [m/s] '
                  r'Polar: mag. = {} [m/s] ang. = {} [rad]'.format(label, v_x, v_y, v[0], a[0]))

    return W_, _magnitude(W_[..., 0], W_[..., 1])


def _autoregressive_dataset(X_, Y_, U_, V_, W_, Z_, xy_, uv_, wz_, lag):
    # Lagged list of consecutive vectors
    def __lag_data(x_lag, y_lag, u_lag, v_lag, w_lag, z_lag, xy_, uv_, wz_, lag):
        # Keep the desire number of lags on the list by removing the last and aadding at the bigging
        if len(x_lag) == lag:
            x_lag.pop(0)
            y_lag.pop(0)
            u_lag.pop(0)
            v_lag.pop(0)
            z_lag.pop(0)
            w_lag.pop(0)
        # Keep adding until we have the desired number of lag time stamps
        x_lag.append(xy_[0])
        y_lag.append(xy_[1])
        u_lag.append(uv_[0])
        v_lag.append(uv_[1])
        w_lag.append(wz_[0])
        z_lag.append(wz_[1])
        return x_lag, y_lag, u_lag, v_lag, w_lag, z_lag
    return __lag_data(X_, Y_, U_, V_, W_, Z_, xy_, uv_, wz_, lag)


# Window Artifacts Persistent model
def _window_persistent_model(_tools, I_scatter_, label, n_samples, n_burnt, verbose = False):
    def __update_model(_tools, I_scatter_):
        _tools.W_lag_.append(I_scatter_[..., np.newaxis])
        # Add a new sample but forget the last sample that was added to the list
        if len(_tools.W_lag_) > n_samples:
            # If there are enough clear sky samples make a model of the artifacts in the window
            _tools.W_ = np.median(np.concatenate(_tools.W_lag_, axis = 2)[..., :-n_burnt], axis = 2)
        return _tools
    # Add a new sample if there is clear sky conditions
    if label == 0:
        tools_ = __update_model(_tools, I_scatter_)
    if verbose:
        print('>> Persistent Model Info: label = {} No. Samples = {}'.format(label, len(_tools.W_lag_)))
    return _tools

# Classification of current frame Atmospheric Conditions
def _atmospheric_condition_v1(_tools, i, K_0_, M_lk_, model_, tau = 0.05, n_samples = 50, verbose = False):
    # Compute Statistics
    def __get_stats(X_):
        # Mean, Std, Skew, and Kurtosis
        return np.array((np.mean(X_), np.std(X_), skew(X_), kurtosis(X_)))
    # Regression Output to Classification Label
    def __robust_classification(_SVC, csi_var_now, csi_var_past, X_):
        if csi_var_now != 1. and csi_var_now < tau and csi_var_past < tau:
            return 0
        else:
            return __classification(_SVC, X_)
    # For robustness... Select Most Frequent Label on a labels lagged-list
    def __lag_labels(y, labels_lag_, field_lag):
        # Keep the desire number of lags on the list by removing the last and aadding at the bigging
        if len(labels_lag_) == field_lag:
            labels_lag_.pop(0)
        labels_lag_.append(y)
        return mode(labels_lag_)[0][0].astype(int), labels_lag_
    # Predict
    def __classification(_SVC, X_):
        return _SVC.predict(PolynomialFeatures(degree).fit_transform(X_))[0]

    # Extract model Parameters
    _SVC   = model_[0]
    degree = model_[1]
    # How flat was the CSI?
    csi_var_past = np.absolute(np.sum(_tools.csi_[i - n_samples:i])/n_samples - 1.)
    csi_var_now  = np.absolute(_tools.csi_[i] - 1.)
    # Get weather features
    #w_ = np.array(_tools.pres_[i])[np.newaxis]
    w_ = np.array(_tools.csi_[i])[np.newaxis]
    # get stats of the images
    k_ = __get_stats(K_0_.flatten())
    # Get velocity vectors features
    #m_ = __get_stats(M_.flatten(), _skew = False)
    # Concatenate all selected features
    X_ = np.concatenate((w_, k_), axis = 0)[np.newaxis]
    # Robust SVM Multi-label Classification
    # Robust classification persistent_v7, threshold segmentation tau = 12.12
    y_hat = __robust_classification(_SVC, csi_var_now, csi_var_past, X_)
    # SVM Multi-label Classification
    #y_hat = __classification(_SVC, X_)
    # Return label for the segmentation type for this sky condition
    y_hat, _tools.labels_ =  __lag_labels(y_hat, labels_lag_ = _tools.labels_, field_lag = 3)
    if verbose:
        if y_hat == 0: print('>> Clear Sky ')
        if y_hat == 1: print('>> Cumulus Cloud')
        if y_hat == 2: print('>> Stratus Cloud')
        if y_hat == 3: print('>> Nimbus Cloud')
    return y_hat, _tools

# Classification of current frame Atmospheric Conditions
def _atmospheric_condition(_tools, i, I_, F_, m_, tau = 0.05, n_samples = 50, verbose = False):
    # Compute Statistics
    def __get_stats(X_, _skew = True):
        if _skew:
            # Mean, Std, Skew, and Kurtosis
            return np.array((np.mean(X_), np.std(X_), skew(X_), kurtosis(X_)))
        else:
            return np.array((np.mean(X_), np.std(X_), skew(X_), kurtosis(X_)))
    # Regression Output to Classification Label
    def __robust_classification(_SVC, csi_var_now, csi_var_past, X_):
        if csi_var_now != 1. and csi_var_now < tau and csi_var_past < tau:
            return 0
        else:
            return __classification(_SVC, X_)
    # For robustness... Select Most Frequent Label on a labels lagged-list
    def __lag_labels(y, labels_lag_, field_lag):
        # Keep the desire number of lags on the list by removing the last and aadding at the bigging
        if len(labels_lag_) == field_lag:
            labels_lag_.pop(0)
        labels_lag_.append(y)
        return mode(labels_lag_)[0][0].astype(int), labels_lag_
    # Predict
    def __classification(_SVC, X_):
        return _SVC.predict(PolynomialFeatures(degree).fit_transform(X_))[0]

    K_ = I_/100.
    M_, A_ = _cart_to_polar(F_[..., 0], F_[..., 1])
    # Extract model Parameters
    _SVC   = m_[0]
    degree = m_[1]
    # How flat was the CSI?
    csi_var_past = np.absolute(np.sum(_tools.csi_[i - n_samples:i])/n_samples - 1.)
    csi_var_now  = np.absolute(_tools.csi_[i] - 1.)
    # Get weather features
    w_ = np.array((_tools.pres_[i], _tools.csi_[i]))
    # get stats of the images
    k_ = __get_stats(K_.flatten(), _skew = True)
    # Get velocity vectors features
    m_ = __get_stats(M_.flatten(), _skew = False)
    # Concatenate all selected features
    X_ = np.concatenate((w_, k_, m_), axis = 0)[np.newaxis]
    # Robust SVM Multi-label Classification
    # Robust classification persistent_v7, threshold segmentation tau = 12.12
    y_hat = __robust_classification(_SVC, csi_var_now, csi_var_past, X_)
    # SVM Multi-label Classification
    #y_hat = __classification(_SVC, X_)
    # Return label for the segmentation type for this sky condition
    y_hat, _tools.labels_ =  __lag_labels(y_hat, labels_lag_ = _tools.labels_, field_lag = 3)
    if verbose:
        if y_hat == 0: print('>> Clear Sky ')
        if y_hat == 1: print('>> Cumulus Cloud')
        if y_hat == 2: print('>> Stratus Cloud')
        if y_hat == 3: print('>> Nimbus Cloud')
    return y_hat, _tools

# Segmentation of the VElocity Vectors According to temporal differences
def _velocity_vectors_segmentation(_tools, I_norm_1_, I_norm_2_, I_labels_, x_0_, index_, percentage, method):
    I_segm_ = I_labels_ > 0.
    # Segment Frames Edges or not
    if method == 0:
        Idx_ = np.invert(I_segm_)
    if method == 1:
        Idx_ = np.invert(I_segm_) | np.invert(index_)
    percentage = 100. - percentage
    # Calculate Variance
    I_diff_ = np.sqrt((I_norm_1_ - I_norm_2_)**2)
    I_diff_[Idx_] = 1e-15
    I_diff_ /= I_diff_.sum()

#     plt.figure(figsize = (7.5, 5))
#     plt.title(r'$i_{i,j}^k = \sqrt{( \bar{i}^{k - 1}_{i,j} - \bar{i}^{k}_{i,j} )^2} $', fontsize = 20)
#     plt.imshow(I_diff_, cmap = 'jet')
#     plt.colorbar()
#     plt.ylabel(r'$y-axis$', fontsize = 15)
#     plt.xlabel(r'$x-axis$', fontsize = 15)
#     plt.grid()
#     plt.show()

    # Calculate Accumulated Variance
    i_diff_ = I_diff_.flatten()
    i_diff_ = 100. * i_diff_
    index_  = np.argsort(i_diff_)
    r_ = np.cumsum(i_diff_[index_])
    # Porcentage of Variance Explained
    i_ = r_ > percentage
    if np.sum(i_) == 0:
        r = 0
    else:
        r = i_diff_[index_][i_][0]/100.
    return I_diff_ > r

# Segmentation of Cloudy Pixels
def _cloud_segmentation(I_norm_, I_scatter_, I_diffuse_, K_, H_, M_, m_ = None, idx_var_ = None, idx_shape_ = None):
    # I_2_norm_, M_, I_scatter_, I_diffuse_, K_0_, H_0_, K_1_, H_1_, K_2_, H_2_
    # Concatenate All image files
    X_ = np.concatenate((I_norm_[..., np.newaxis], M_[..., np.newaxis], I_scatter_[..., np.newaxis],
                         I_diffuse_[..., np.newaxis], K_[..., np.newaxis], H_[..., np.newaxis]), axis = 2)
    return _cloud_RRC_segmentation(X_, m_, idx_var_, idx_shape_)

# Select the most probable set of vectors from the clouds velocity field
def _cloud_velocity_vector_selection(X_, Y_, U_, V_, W_, Z_, n_select, n_layers, percentage, method):
    # Vector Selection main function
    def __vector_selection(z_, n_select):
        # Importance Sample Selection
        idx_ = np.arange(WZ_.shape[0])
        if method == 0:
            return idx_[np.argsort(z_)[::-1]][:n_select]
        else:
            return np.random.choice(idx_, size = n_select, p = z_, replace = False)
    # Divide features sources sets in training and test
    def __divide_dataset(XY_, UV_, WZ_, index_, n_select, percentage):
        # Number of samples for training and test
        N_tr = int(n_select * percentage/100.)
        N_ts = n_select - N_tr
        WZ_p_ = np.zeros(WZ_.shape)
        #WZ_ = WZ_**4
        for i in range(len(index_)):
            #w_ = np.argmax(WZ_, axis = 1)
            WZ_ /= np.tile(np.sum(WZ_, axis = 1), (2, 1)).T
            #WZ_p_[index_[i], i] = 1.
            WZ_p_[index_[i], i] = WZ_[index_[i], i]
            #WZ_p_ = WZ_.copy()
        # Combine Index from each distribution
        index_ = np.concatenate(index_, axis = 0)
        np.random.shuffle(index_)
        return [XY_[index_[:N_tr], :], UV_[index_[:N_tr], :], WZ_p_[index_[:N_tr], :]], \
               [XY_[index_[-N_ts:], :], UV_[index_[-N_ts:], :], WZ_p_[index_[-N_ts:], :]]
    # Converting from list of vectors to matrix form the samples of clouds velocity field
    XY_ = np.concatenate((np.hstack(X_)[:, np.newaxis], np.hstack(Y_)[:, np.newaxis]), axis = 1)
    UV_ = np.concatenate((np.hstack(U_)[:, np.newaxis], np.hstack(V_)[:, np.newaxis]), axis = 1)
    WZ_ = np.concatenate((W_[:, np.newaxis], Z_[:, np.newaxis]), axis = 1)
    # Find number of layers
    if XY_.shape[0] < n_select:
        n_select = XY_.shape[0]
    wind_flow_indicator_ = np.sum(WZ_, axis = 0) > 0.
    # Sample Both Distributions Independetly
    index_ = []
    for i in range(n_layers):
        z_ = WZ_[:, i]#**4
        z_ /= z_.sum()
        # Selecting the most likely set of vectors recording over a period of lags
        idx_ = __vector_selection(z_, n_select // n_layers)
        index_.append(idx_)
    # Divide features sources sets in training and test
    X_tr_, X_ts_ = __divide_dataset(XY_, UV_, WZ_, index_, n_select, percentage)

    return X_tr_, X_ts_, wind_flow_indicator_

# Lagged list of consecutive vectors
def _get_autoregressive_dataset(X_, Y_, U_, V_, W_, Z_, xy_, uv_, wz_, lag):
    # Keep the desire number of lags on the list by removing the last and aadding at the bigging
    if len(X_) == lag:
        X_.pop(0)
        Y_.pop(0)
        U_.pop(0)
        V_.pop(0)
        W_.pop(0)
        Z_.pop(0)
    # Keep adding until we have the desired number of lag time stamps
    X_.append(xy_[0])
    Y_.append(xy_[1])
    U_.append(uv_[0])
    V_.append(uv_[1])
    W_.append(wz_[0])
    Z_.append(wz_[1])
    return X_, Y_, U_, V_, W_, Z_

# Compute the resion of interest
def _CI(mse, z = 2.576, ifps = 15.):
    return ifps * np.sqrt(mse) * z

# Processing of the infrared images
def _infrared_images_processing(_tools, I_, x_0_, m_, i_0_max):
    # Geometric coordiantes transformation from cartersian to polar
    _tools.MA_ = _polar_coordinates_transformation(x_0_, _tools.X_, _tools.Y_)
    # Atmospheric Model Evaluation
    I_atmo_, T = _atmospheric_effect_v11(_tools, I_, x_0_, m_, i_0_max, tau = 33000.)

    #fig = plt.figure(figsize = (7.5, 5))
    #ax = fig.gca(projection = '3d')
    #ax.set_title(r'$\mathcal{A} (\mathbf{x}_{i,j}; \mathbf{x}_{0}, \Theta ) $', fontsize = 25)
    #ax.plot_surface(_tools.X_, _tools.Y_, I_atmo_/100., linewidth = 1.,
    #                  antialiased = True, cmap = 'inferno')
    #ax.set_xlabel(r'x-axis', fontsize = 15, labelpad = 10)
    #ax.set_ylabel(r'y-xis', fontsize = 15, labelpad = 10)
    #ax.set_zlabel(r'Temp. $[^\circ K]$', fontsize = 15, labelpad = 10)
    #ax.set_yticks(np.arange(0, 60, 15))
    #ax.set_xticks(np.arange(0, 80, 15))
    #ax.tick_params(labelsize = 15)
    #plt.show()

    # Remove Atmosphere and Sun radiation from the IR images
    I_scatter_ = I_ - I_atmo_
    # Interpolate non-smooth pixels intensity in the circumsolar area
    I_scatter_ = _sun_pixels_interpolation(I_scatter_, _tools.MA_, _tools.X_, _tools.Y_, _tools.N_y, _tools.N_x,
                                           radius = 4.)
    return _tools, I_scatter_, T

# Compute the different Temperatures and Height After Processing the Solar Irradiance
def _extract_radiometric_features(T, ws_, I_global_, I_direct_, I_diffuse_, i_sun_max):
    # Increments of temperature with respect to the Troupopause
    dT_ = I_diffuse_ - T
    # Radiometry Camera Functionality to obtain clouds height using Raw Pixels
    K_0_, H_0_ = _infrared_radiometry(I_global_, ws_, i_sun_max, verbose = False)
    # Radiometry Camera Functionality to obtain clouds height after
    # removing window effects
    K_1_, H_1_ = _infrared_radiometry(I_direct_, ws_, i_sun_max, verbose = False)
    # Radiometry Camera Functionality to obtain clouds height after
    # removing atmospheric effects
    K_2_, H_2_ = _infrared_radiometry(I_diffuse_, ws_, i_sun_max, verbose = False)
    return dT_, K_0_, H_0_, K_1_, H_1_, K_2_, H_2_

__all__ = ['_window_persistent_model', '_atmospheric_condition', '_atmospheric_condition_v1',
           '_velocity_vectors_segmentation', '_cloud_segmentation', '_get_autoregressive_dataset', '_CI',
           '_cloud_velocity_vector_selection', '_infrared_images_processing', '_extract_radiometric_features']
