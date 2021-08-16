import pickle, glob, sys, os, warnings, csv

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from time import time

from scipy.stats import gamma as _Ga
from scipy.stats import vonmises as _Vm
from scipy.stats import beta as _Be
from scipy.stats import multivariate_normal, norm
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture

# Import All custom functions
from detection_clustering_utils import *
from feature_extraction_utils import *
from cloud_velocity_vector_utils import *
from image_processing_utils import *
from cloud_segmentation_utils import *
from wind_velocity_field_utils import *
from lib_motion_vector import _farneback, _lucas_kanade, _pyramidal_weighted_lucas_kanade, _weighted_lucas_kanade
from utils import *
# Import frame-Class, models, and paramters
from config import _frame
# Import functions for a single layer wind velocity field
from extra import *
from files import *

# Do not display warnings in the output file
warnings.filterwarnings('ignore')

def _wind_velocity_field(_tools, X_tr_, X_ts_, XYZ_, dXYZ_, H_, h_stats_, A_0_, x_0_2_, I_norm_2_, wind_flow_indicator_):
    # Unpack Lagged Variables
    X, Y, x_, y_, XY_stdz_, xy_stdz_, _stdz_x, _stdz_y = _tools.get_constants(0)
    p_segment, n_select, p_train, n_layers, lag, step_size = _tools.get_constants(1)
    g_, N_y, N_x, dim = _tools.get_constants(2)
    # Unpack Train and Test Samples
    XY_tr_, UV_tr_, W_tr_ = X_tr_
    XY_ts_, UV_ts_, W_ts_ = X_ts_
    h_mean_, h_med_, h_std_ = h_stats_
    #print(h_mean_)
    #print(h_med_)
    flow_ = []
    # loop over wind layers
    for i in range(n_layers):
        # Weights Regularization
        w_tr_ = W_tr_[:, i]/W_tr_[:, i].sum()
        w_ts_ = W_ts_[:, i]/W_ts_[:, i].sum()
        height = h_med_[i]
        print('>> Training MO-eWSVM-FC', _tools.j)
        # Calculate wind velocity field when there are enough selected vectors
        UV_hat_, theta_ = _wind_velocity_field_svm(_tools, XY_tr_, UV_tr_, w_tr_, XY_ts_, UV_ts_, w_ts_, dXYZ_, XY_stdz_,
                                                   xy_stdz_, _stdz_x, _stdz_y, N_y, N_x, step_size,
                                                   svm = 3, kernel = 'linear', degree = 0, weights = True, CV = False,
                                                   N_grid = 5, N_kfold = 3, N_BO_iterations = 0, display = False)

        #CV_, WMSE_val, WMSE_ts, MSE_ts, WMAE_ts, MAE_ts, D, V = theta_
        CV_, WRMSE_ts, X_WMAE_ts_, Y_WMAE_ts_, D, V = theta_

        M_hat_, A_hat_ = _cart_to_polar(UV_hat_[..., 0], UV_hat_[..., 1])
        m_hat = np.mean(M_hat_.flatten())
        a_hat = np.mean(A_hat_.flatten())
        print(WRMSE_ts, X_WMAE_ts_, Y_WMAE_ts_, m_hat, a_hat, height)

        # Wind velocity field visualization lines
        Phi_ = _streamlines(height*UV_hat_[..., 0]*dXYZ_[..., 1], height*UV_hat_[..., 1]*dXYZ_[..., 0])
        Psi_ = _potential_lines(height*UV_hat_[..., 0]*dXYZ_[..., 0], height*UV_hat_[..., 1]*dXYZ_[..., 1])

        flow_.append([height, UV_hat_, Phi_, Psi_, XYZ_, dXYZ_, [WRMSE_ts, X_WMAE_ts_, Y_WMAE_ts_]])

    return flow_


# Inverence of Multiple Wind Layers on a Frame
def _wind_layer_inference(I_diffuse_, I_segm_2_, H_, index_, n_layers):
    def __proba(x_):
        x_ = x_ - x_.min()
        q_ = np.sum(x_, axis = 2)[..., np.newaxis]
        return np.nan_to_num(x_ / np.concatenate((q_, q_), axis = 2))
    # Label Clouds by layer
    def __cloud_layer_prob(I_diffuse_, I_segm_, H_, index_, n_layers):
        # > 0 Normalization and Set in vector
        t_min = I_diffuse_[index_].min()
        DT_ = (I_diffuse_ - t_min)
        dT_prime_ = DT_.flatten()[:, np.newaxis]
        dT_ = dT_prime_[index_.flatten(), :]
        dt_ = dT_[I_segm_[index_].flatten(), 0][:, np.newaxis]
        # Normalization of the temperatures
        t_max = dt_.max()
        dt_ = dt_/t_max
        # Regularization to avoid infinities
        dt_[dt_ == 1.] -= 1e-5
        dt_[dt_ == 0.] += 1e-5
        # Eval Prbability of all samples
        x_ = (I_diffuse_.flatten()[:, np.newaxis] - t_min)
        x_ = x_/t_max
        x_-= x_.min()/2.

        # Computer Labels and Probabilities for the case of 1 layer
        if n_layers == 1:
            # Fit BeMM for 1 layer and 2 layers
            theta_1_, log_z_1, scores_ = _EM_BeMM(dt_, n_clusters = 1, n_init = 14, tol = 1e-1, epsilon = 0.)
            # One Layer Labelling
            Z_ = np.ones(I_segm_.shape, dtype = int)
            W_ = np.zeros((I_segm_.shape[0], I_segm_.shape[1], 2))
        # Computer Labels and Probabilities for the case of 2 layer
        if n_layers == 2:
            theta_2_, log_z_2, scores_ = _EM_BeMM(dt_, n_clusters = 2, n_init = 14, tol = 1e-1, epsilon = 0.)
            # Clusters mean original scale
            E_0 = t_max*theta_2_[0]/(theta_2_[0] + theta_2_[2])
            E_1 = t_max*theta_2_[1]/(theta_2_[1] + theta_2_[3])
            # Define Beta distribution in the cluster
            _Be_1 = _Be(a = theta_2_[0], loc = 0., b = theta_2_[2])
            _Be_2 = _Be(a = theta_2_[1], loc = 0., b = theta_2_[3])
            # Evaluate Pixels Probabilities
            Z_labels_ = np.concatenate((theta_2_[4] * _Be_1.pdf(x_), theta_2_[5] * _Be_2.pdf(x_)), axis = 1)
            # Make sure that clusters do not take elements from the other cluste
            if E_0 < E_1:
                Z_labels_[dT_prime_[:, 0] < E_0, 1] = 0
                Z_labels_[dT_prime_[:, 0] > E_1, 0] = 0
            else:
                Z_labels_[dT_prime_[:, 0] < E_1, 0] = 0
                Z_labels_[dT_prime_[:, 0] > E_0, 1] = 0
            # Maximum Likelihood Labelling
            Z_ = np.argmax(Z_labels_, axis = 1).reshape(I_segm_.shape) + 1
            W_ = __proba(Z_labels_.reshape(I_segm_.shape[0], I_segm_.shape[1], 2))
        return Z_, W_, scores_

    # Apply Ising model to fine tune segmentation
    def __label_clouds(I_labels_, I_segm_):
        # Unform Image by Field Implementation
        def ___multiclass_ising_model(W_, cliques, beta, n_max_iter = 10):
            # Cliques
            cliques_1_ = [[ 0,  1], [ 0, -1], [1,  0], [-1, 0]]
            cliques_2_ = [[-1, -1], [-1,  1], [1, -1], [ 1, 1]]
            cliques_3_ = [[ 0,  2], [ 0, -2], [2,  0], [-2, 0]]
            cliques_4_ = [[-2, -2], [-2,  2], [2, -2], [ 2, 2]]
            cliques_   = [cliques_1_, cliques_1_ + cliques_2_,
                          cliques_1_ + cliques_2_ + cliques_3_,
                          cliques_1_ + cliques_2_ + cliques_3_ + cliques_4_][cliques]
            # Prior based on neigborhood class
            def __neigborhood(W_, i, j, labels_, cliques_, beta):
                M, N = W_.shape
                class_0_ = 0
                class_1_ = 0
                class_2_ = 0
                class_ = [class_0_, class_1_, class_2_]
                # Loop over neigbors
                for clique_ in cliques_:
                    k = i + clique_[0]
                    m = j + clique_[1]
                    if k < 0 or m < 0 or k >= M or m >= N:
                        continue
                    else:
                        for l in labels_:
                            if W_[k, m] == l:
                                class_[l] += 1
                i_class_ = class_ == np.max(class_)
                if i_class_.sum() < 2:
                    return labels_[i_class_]
                else:
                    if (labels_[i_class_] == W_[i, j]).any():
                        return W_[i, j]
                    if np.random.rand() <= 0.5:
                        return labels_[i_class_][0]
                    else:
                        return labels_[i_class_][1]
            # Constants Init.
            D, N = W_.shape
            Y_hat_ = - np.ones(W_.shape)
            # Variables Init.
            W_init_ = W_.copy()
            W_hat_  = np.zeros(W_.shape)
            e_prev  = np.inf
            labels_ = np.unique(W_)
            # Loop until converge
            while True:
                k = 0
                # Loop Over Pixels in an image
                for i in range(D):
                    for j in range(N):
                        W_hat_[i, j] = __neigborhood(W_init_, i, j, labels_, cliques_, beta)
                # Compute Error
                e_now = np.sum(np.sqrt((W_init_ - W_hat_)**2))
                # Stop if Convergence or reach max. interations
                if e_now >= e_prev or k == n_max_iter:
                    break
                else:
                    # Update for the next interaation
                    W_init_ = W_hat_.copy()
                    e_prev  = e_now.copy()
                    k += 0
                return W_init_

        # Unform Image by Field Implementation
        def ___ising_model(W_, cliques, beta, n_max_iter = 10):
            # Cliques
            cliques_1_ = [[ 0,  1], [ 0, -1], [1,  0], [-1, 0]]
            cliques_2_ = [[-1, -1], [-1,  1], [1, -1], [ 1, 1]]
            cliques_3_ = [[ 0,  2], [ 0, -2], [2,  0], [-2, 0]]
            cliques_4_ = [[-2, -2], [-2,  2], [2, -2], [ 2, 2]]
            cliques_   = [cliques_1_, cliques_1_ + cliques_2_,
                          cliques_1_ + cliques_2_ + cliques_3_,
                          cliques_1_ + cliques_2_ + cliques_3_ + cliques_4_][cliques]
            # Prior based on neigborhood class
            def __neigborhood(W_, i, j, cliques_, beta):
                M, N = W_.shape
                total = 0
                # Loop over neigbors
                for clique_ in cliques_:
                    k = i + clique_[0]
                    m = j + clique_[1]
                    if k < 0 or m < 0 or k >= M or m >= N:
                        continue
                    else:
                        if W_[k, m] == W_[i, j]:
                            total += 1
                        else:
                            total -= 1
                if total >= 0.:
                    return W_[i, j]
                else:
                    return 1 - W_[i, j]
            # Constants Init.
            D, N = W_.shape
            Y_hat_ = - np.ones(W_.shape)
            # Variables Init.
            W_init_ = W_.copy()
            W_hat_  = np.zeros(W_.shape)
            e_prev  = np.inf
            # Loop until converge
            while True:
                k = 0
                # Loop Over Pixels in an image
                for i in range(D):
                    for j in range(N):
                        W_hat_[i, j] = __neigborhood(W_init_, i, j, cliques_, beta)
                # Compute Error
                e_now = np.sum(np.sqrt((W_init_ - W_hat_)**2))
                # Stop if Convergence or reach max. interations
                if e_now >= e_prev or k == n_max_iter:
                    break
                else:
                    # Update for the next interaation
                    W_init_ = W_hat_.copy()
                    e_prev  = e_now.copy()
                    k += 0
                return W_init_

        # Set I_segm_2_ at the top if were detected
        idx_ = I_segm_ == 0
        if idx_.sum() > 0:
            I_labels_[idx_] = 0
        # How many labels are in the image?
        labels_ = np.unique(I_labels_)
        N_labels = labels_.shape[0]
        # Apply issing model if there is more than one layer
        if N_labels == 1:
            return I_labels_
        # Ising Model
        elif N_labels == 2:
            # Is it backgroun in the image?
            if labels_.sum() == 3:
                return ___ising_model(I_labels_ - 1, cliques = 2, beta = 1, n_max_iter = 10) + 1
            else:
                return ___ising_model(I_labels_, cliques = 2, beta = 1, n_max_iter = 10)
        # Multiclass Ising Model
        elif N_labels == 3:
            return ___multiclass_ising_model(I_labels_, cliques = 2, beta = 1, n_max_iter = 10)
    # Labels ordered from highest to lowers
    def __order_cloud_height(I_labels_, H_, W_):
        # Calculate the average, meadian, and standard deviation of the height of a labelled object
        def ___cloud_label_height(H_, I_labels_, m_to_km = 1000., verbose = False):
            # Find Labels
            labels_ = np.unique(I_labels_)
            index_  = np.nonzero(labels_)[0]
            labels_ = labels_[index_]
            # Variables Initialization
            h_mean_ = np.zeros(labels_.shape[0])
            h_medi_ = np.zeros(labels_.shape[0])
            h_std_  = np.zeros(labels_.shape[0])
            # Loop over labels
            for i, j in zip(labels_, range(labels_.shape[0])):
                # Calculate Statistics
                h_mean_[j] = np.mean(H_[I_labels_ == i])/m_to_km
                h_medi_[j] = np.median(H_[I_labels_ == i])/m_to_km
                h_std_[j]  = np.std(H_[I_labels_ == i])/m_to_km
                if verbose:
                    print('>> Label: {} Avg. Height: {} km Median Height: {} km Std. Height: {} km'.format(i, h_mean_[j], h_medi_[j], h_std_[j]))
            return h_mean_, h_medi_, h_std_
        # Calculate Average Height of the clouds in each Pixel
        h_ = ___cloud_label_height(H_, I_labels_, verbose = False)
        # Keep Lowest Cloud Layer in index 0
        if len(h_[0]) > 1:
            if h_[0][1] > h_[0][0]:
                # Invert Probabilities
                Q_ = W_.copy()
                W_[..., 0] = Q_[..., 1]
                W_[..., 1] = Q_[..., 0]
                # Invert Labels
                Z_ = I_labels_.copy()
                I_labels_[Z_ == 1] = 2
                I_labels_[Z_ == 2] = 1
                # Get ordered heights
                h_ = ___cloud_label_height(H_, I_labels_, verbose = False)
            W_[I_labels_ == 0, 0] = 0
            W_[I_labels_ == 0, 1] = 0
        else:
            W_[I_labels_ == 1, 0] = 1

        return W_, I_labels_, h_

    # Compute Pixels Temperature Distribution Probabilites
    I_labels_, W_, scores_ = __cloud_layer_prob(I_diffuse_, I_segm_2_, H_, index_, n_layers)
    # Label Pixels by temperatures labels
    I_labels_ = __label_clouds(I_labels_, I_segm_2_)
    # Calculate Average Height of the clouds in each Pixel
    W_, I_labels_, h_ = __order_cloud_height(I_labels_, H_, W_)
    return W_, I_labels_, h_, scores_

# Transform velocity vectors velocities according to the the average hight of a layer label
def _transform_velocity_vectors(I_norm_1_, I_norm_2_, I_labels_, Q_, dXYZ_, h_, opt_, n_layers, fps = 1./15.):
    #def __sigma(h, w_ = np.array([1.82859813, 0.01995626])):
    #    #return w_.T @ np.array((1., h))
    #    #return 2.3691
    #    return 2.2931530606046238
    sigma = 2.2931530606046238
    # Variables Initialization
    F_       = np.zeros((I_norm_2_.shape[0], I_norm_2_.shape[1], 2))
    F_prime_ = np.zeros((I_norm_2_.shape[0], I_norm_2_.shape[1], 2))
    # Case of No Clouds
    if n_layers == 0:
        return F_, F_prime_
    # Case of Clouds in one layer
    if n_layers == 1:
        F_ = _lucas_kanade(I_norm_1_, I_norm_2_, window_size = opt_[0], tau = opt_[1], sigma = opt_[3])
        # Force to 0 velocity vectors where cloud pixels where not segmented
        F_[I_labels_ == 0, :] = 0
        # Prespective Transformation Weighted Sum
        F_prime_[..., 0] = F_[..., 0] * dXYZ_[..., 0] * h_[0][0] * fps
        F_prime_[..., 1] = F_[..., 1] * dXYZ_[..., 1] * h_[0][0] * fps
        return F_, sigma*F_prime_
    # Case of Clouds in two layers
    if n_layers == 2:
        # Compute Velocity vectors Transformation Layer 2
        F_0_ = _weighted_lucas_kanade(I_norm_1_, I_norm_2_, Q_[..., 0], window_size = opt_[0], tau = opt_[1], sigma = opt_[3])
        F_1_ = _weighted_lucas_kanade(I_norm_1_, I_norm_2_, Q_[..., 1], window_size = opt_[0], tau = opt_[1], sigma = opt_[3])
        # Force to 0 velocity vectors where cloud pixels where not segmented
        F_0_[I_labels_ == 0, :] = 0
        F_1_[I_labels_ == 0, :] = 0
        # Weight Velocity Vectors
        F_0_[..., 0] = F_0_[..., 0] * Q_[..., 0]
        F_0_[..., 1] = F_0_[..., 1] * Q_[..., 0]
        F_1_[..., 0] = F_1_[..., 0] * Q_[..., 1]
        F_1_[..., 1] = F_1_[..., 1] * Q_[..., 1]
        # Prespective Transformation Weighted Sum
        F_prime_[..., 0] = (F_0_[..., 0] * dXYZ_[..., 0] * h_[0][0] * fps) + (F_1_[..., 0] * dXYZ_[..., 0] * h_[0][1] * fps)
        F_prime_[..., 1] = (F_0_[..., 1] * dXYZ_[..., 1] * h_[0][0] * fps) + (F_1_[..., 1] * dXYZ_[..., 1] * h_[0][1] * fps)
        return F_0_ + F_1_, sigma*F_prime_

# Compute the probability of a Velocity Vectors given a wind layer
def _velocity_vector_probability(U_, V_, U_p_, V_p_, F_, F_p_, I_, I_labels_, V_segm_, W_2_, n_layers,
                                 transform, velocity_inference_method, height_inference_method):
    # ICM 2-class inference with noise level
    def __Iterated_Conditional_Modes(X_, gamma = 1e-2, beta = - np.inf, n_init = 5, n_eval = 10):
        def __rand_init(X_, M):
            w_init_ = np.ones((M,))
            w_init_[np.random.randint(0, M - 1, size = M//2)] = 2.
            return w_init_
        # Fit Multivariate Normal Distribution to each class samples
        def __class_distribution(x_, w_, D, gamma):
            # Find each class elements
            idx_1_ = w_ == 1
            idx_2_ = w_ == 2
            # sample mean for each class
            mu_1_ = np.mean(x_[idx_1_, :], axis = 0)
            mu_2_ = np.mean(x_[idx_2_, :], axis = 0)
            # sample covariance for each class
            E_1_ = np.cov(x_[idx_1_, :].T) + np.eye(D)*gamma
            E_2_ = np.cov(x_[idx_2_, :].T) + np.eye(D)*gamma
            # Define Normal Distribution for each clasee
            return multivariate_normal(mu_1_, E_1_), multivariate_normal(mu_2_, E_2_)
        # Evaluate Likelihood with Uniform Noise
        def __likelihood(x_, _N_1, _N_2, beta, M):
            # Variables Initialization
            Z_ = np.zeros((M, 3))
            # Calculate Pixels-Class Energy
            Z_[..., 0] = _N_1.logpdf(x_)
            Z_[..., 1] = _N_2.logpdf(x_)
            Z_[..., 2] = np.ones((M,))*beta
            return Z_
        # Evaluate Energy and Classification
        def __energy(Z_):
            # Maximum Energy Classification
            W_ = np.argmax(Z_, axis = 1) + 1
            # Maximum Pixel's Energy
            U_ = np.max(Z_, axis = 1)
            return U_, W_
        # Constants Initialization
        M, D = X_.shape

        # Save Results
        class_, energy_ = [], []
        # Loop over No. of Inititialization
        for i in range(n_init):
            t1 = time()
            # Random Initialization
            w_init_ = __rand_init(X_, M)
            # Stopping criteria Initialization
            u_k    = - np.inf
            u_k_1_ = np.zeros((M,))
            w_k_1_ = w_init_.copy()
            # Class Distribution Inference
            _N_1, _N_2 = __class_distribution(X_, w_k_1_, D, gamma)
            # Loop over No. of evaluation
            for j in range(n_eval):
                # Evaluate likelihood function
                Z_ = __likelihood(X_, _N_1, _N_2, beta, M)
                # Current Evaluation Weights Initialization
                u_k_1_, w_k_1_ = __energy(Z_)
                # Current Evaluation Total Energy
                u_k_1 = u_k_1_.sum()
                # Stop if it is a minima
                if u_k >= u_k_1:
                    class_.append(w_)
                    energy_.append(u_k)
                    print('>>> ICM-MRF: No. Init.: {} Time :{} s Energy: {}'.format(i, time() - t1, u_k))
                    break
                # If not keep optimizing
                else:
                    # Update next iteration objective functions
                    w_  = w_k_1_.copy()
                    u_k = u_k_1.copy()
                    # Class Distribution Inference
                    _N_1, _N_2 = __class_distribution(X_, w_k_1_, D, gamma)
        # Select Maximum Energy Pixels State
        w_ = class_[np.argmax(energy_)]
        print('>>> ICM-MRF: Energy: {}'.format(np.max(energy_)))
        # Define Class Distribution
        _N_1, _N_2 = __class_distribution(X_, w_, D, gamma)
        # Evaluate Pixels Likelihood
        Z_ = __likelihood(X_, _N_1, _N_2, beta, M)
        return w_, [_N_1, _N_2]
    # Maximum Likelihood Prediction of a class
    def __predict(X_, _N_):
        # Evaluate Likelihood with Uniform Noise
        def __likelihood(x_, _N_1, _N_2, M):
            # Variables Initialization
            Z_ = np.zeros((M, 2))
            # Calculate Pixels-Class Energy
            Z_[..., 0] = _N_1.logpdf(x_)
            Z_[..., 1] = _N_2.logpdf(x_)
            return Z_
        # Evaluate Energy and Classification
        def __energy(Z_):
            # Maximum Energy Classification
            W_ = np.argmax(Z_, axis = 1)
            # Maximum Pixel's Energy
            U_ = np.max(Z_, axis = 1)
            return U_, W_
        M, D = X_.shape
        Z_ = __likelihood(X_, _N_[0], _N_[1], M)
        U_, W_ = __energy(Z_)
        return W_, U_.sum()

    # ICM Optimization for Markov Random Field Model of 1 Dimensional feature Space
    def __ICM_MRF(X_, W_, index_, beta, clique, n_init = 5, n_eval = 10):
        # Cliques
        C_0_ = [[0, 0]]
        C_1_ = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        C_2_ = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
        # Fit Multivariate Normal Distribution to each class samples
        def __class_distribution(X_, W_, index_):
            # Reshape Features
            x_ = X_[index_].flatten()
            w_ = W_[index_].flatten()
            # Find each class elements
            idx_1_ = w_ == 1
            idx_2_ = w_ == 2
            # Define Normal Distribution for each clasee
            return norm(np.mean(x_[idx_1_]), np.std(x_[idx_1_])), norm(np.mean(x_[idx_2_]), np.std(x_[idx_2_]))
        # Evaluate Likelihood with Uniform Noise
        def __likelihood(X_, _N_1, _N_2, M, N):
            x_ = X_.flatten()
            return _N_1.logpdf(x_).reshape(M, N), _N_2.logpdf(x_).reshape(M, N)
        # Energy Potential Function
        def __prior(W_, cliques_, beta, M, N):
            # Prior based on neigborhood class
            def ___neigborhood(w, W_, i, j, cliques_, beta, M, N):
                prior = 0
                # Loop over neigbors
                for clique_ in cliques_:
                    k = i + clique_[0]
                    m = j + clique_[1]
                    if k < 0 or m < 0 or k >= M or m >= N:
                        pass
                    else:
                        if w == W_[k, m]:
                            prior += beta
                        else:
                            prior -= beta
                return prior
            # Variable Initialization
            prior_0_ = np.zeros((M, N))
            prior_1_ = np.zeros((M, N))
            # Loop over Pixels in an Image
            for i in range(M):
                for j in range(N):
                    # Energy function Value and Prior Probability
                    prior_0_[i, j] = ___neigborhood(1, W_, i, j, cliques_, beta, M, N)
                    prior_1_[i, j] = ___neigborhood(2, W_, i, j, cliques_, beta, M, N)
            return prior_0_, prior_1_

        # Evaluate Energy and Classification
        def __energy(lik_, pri_, I_segm_, M, N):
            # Variables Initialization
            U_ = np.zeros((M, N, 2))
            Q_ = np.zeros((M, N, 2))
            # Calculate Pixels-Class Energy
            U_[..., 0] = lik_[0] + pri_[0]
            U_[..., 1] = lik_[1] + pri_[1]
            Q_[..., 0] = pri_[0]
            Q_[..., 1] = pri_[1]
            # Maximum Energy Classification
            W_ = np.argmax(U_, axis = 2) + 1
            W_[I_segm_] = 0.
            # Maximum Pixel's Energy
            U_ = np.max(U_, axis = 2)
            Q_ = np.max(Q_, axis = 2)
            return U_, Q_, W_
        # Select Cliques Set
        cliques_ = [C_0_, C_1_, C_1_ + C_2_][clique]
        # Find Good Velocity Vectors
        I_segm_ = W_ == 0.
        # Constants Initialization
        M, N = X_.shape
        # Save Results
        class_, energy_ = [], []
        # Loop over No. of Inititialization
        for i in range(n_init):
            # Stopping criteria Initialization
            u_k  = - np.inf
            W_k_ = W_.copy()
            U_k_1_ = np.zeros((M, N))
            W_k_1_ = W_.copy()
            # Class Distribution Inference
            _N_1, _N_2 = __class_distribution(X_, W_k_1_, index_)
            # Loop over No. of evaluation
            for j in range(n_eval):
                print('>>> ICM-MRF: No. Eval.: {} Energy: {}'.format(j, u_k))
                # Evaluate likelihood function
                lik_ = __likelihood(X_, _N_1, _N_2, M, N)
                # Evaluate prior function
                prior_ = __prior(W_k_1_, cliques_, beta, M, N)
                # Current Evaluation Weights Initialization
                U_k_1_, Q_k_1_, W_k_1_ = __energy(lik_, prior_, I_segm_, M, N)
                # Current Evaluation Total Energy
                u_k_1 = U_k_1_.sum()
                # Stop if it is a minima
                if u_k >= u_k_1:
                    print('>>> ICM-MRF: No. Init.: {} Energy: {}'.format(i, u_k))
                    energy_.append(u_k)
                    class_.append(W_k_)
                    break
                # If not keep optimizing
                else:
                    # Update next iteration objective functions
                    u_k  = u_k_1.copy()
                    W_k_ = W_k_1_.copy()
                    # Class Distribution Inference
                    _N_1, _N_2 = __class_distribution(X_, W_k_1_, index_)
        # Find State of Maximum Energy
        i = np.argmax(energy_)
        return class_[i], energy_[i]

    # Identify Which layer is lower and which one is higher
    def __find_layer(X_, W_, z_1_, z_2_):
        h_1 = np.mean(X_[W_ == 1])
        h_2 = np.mean(X_[W_ == 2])
        if h_1 > h_2:
            return z_1_, z_2_, W_
        else:
            idx_1_ = W_ == 1
            idx_2_ = W_ == 2
            W_[idx_1_] = 2
            W_[idx_2_] = 1
            return z_2_, z_1_, W_
    # Maximum Likelihood Inference
    def _ML_inference(X_, w_):
        m_ = (X_.T @ w_)/w_.sum()
        M_ = np.tile(m_, (X_.shape[0], 1))
        S_ = ((X_ - M_).T @ np.diag(w_) @ (X_ - M_)) / w_.sum()
        return multivariate_normal(m_, S_ + np.eye(m_.shape[0])*1e-5).pdf(Y_)
    # Evaluate VV probabilities
    def _eval_proba(X_, _N_):
        Z_hat_  = np.zeros((X_.shape[0], 2))
        Z_hat_[:, 0] = _N_[0].pdf(X_)
        Z_hat_[:, 1] = _N_[1].pdf(X_)
        return Z_hat_
    # Evaluate VV posterior probabilities
    def _eval_post(X_):
        Z_ = X_.copy()
        Z_[:, 0] = .5*(Z_[:, 0]/Z_[:, 0].sum())
        Z_[:, 1] = .5*(Z_[:, 1]/Z_[:, 1].sum())
        Z_ /= np.tile(np.sum(Z_, axis = 1), (2, 1)).T
        return Z_
    # Compute Pixels Probabilities to belong to a wind layer
    def __height_prob(L_hat_, W_2_):
        # Variables Initialization
        L_0_ = np.zeros((L_hat_.shape))
        L_1_ = np.zeros((L_hat_.shape))
        P_0_ = np.zeros((L_hat_.shape))
        P_1_ = np.zeros((L_hat_.shape))
        # Combination 1 Likelihood and Posterior
        L_0_[..., 0] = L_hat_[..., 0]*W_2_[..., 0]
        L_0_[..., 1] = L_hat_[..., 1]*W_2_[..., 1]
        M_0_         = L_0_[..., 0] + L_0_[..., 1]
        P_0_[..., 0] = np.nan_to_num(L_0_[..., 0]/M_0_)
        P_0_[..., 1] = np.nan_to_num(L_0_[..., 1]/M_0_)
        # Combination 2 Likelihood and Posterior
        L_1_[..., 0] = L_hat_[..., 1]*W_2_[..., 0]
        L_1_[..., 1] = L_hat_[..., 0]*W_2_[..., 1]
        M_1_         = L_1_[..., 0] + L_1_[..., 1]
        P_1_[..., 0] = np.nan_to_num(L_1_[..., 0]/M_1_)
        P_1_[..., 1] = np.nan_to_num(L_1_[..., 1]/M_1_)
        return L_0_, L_1_, P_0_, P_1_

    # Constants Definition
    N, M, D = F_.shape
    # Converting list of sample velecity vectors to matrix form
    Y_ = np.concatenate((np.hstack(U_p_)[:, np.newaxis], np.hstack(V_p_)[:, np.newaxis]), axis = 1)
    # Reshape Dataset
    F_tilde_ = F_.reshape(N*M, D)
    # Converting list of sample velecity vectors to matrix form
    X_ = np.concatenate((np.hstack(U_)[:, np.newaxis], np.hstack(V_)[:, np.newaxis]), axis = 1)
    Y_ = np.concatenate((np.hstack(U_p_)[:, np.newaxis], np.hstack(V_p_)[:, np.newaxis]), axis = 1)
    # Case of 1 Detected Layer
    if n_layers == 1:
        #print(X_)
        #print(X_.shape)
        m_ = np.mean(X_, axis = 0)
        S_ = np.cov(X_.T)
        #print(S_)
        z_0_ = multivariate_normal(m_, S_ + np.eye(m_.shape[0])*1e-5).pdf(X_)
        z_1_ = np.zeros((X_.shape[0],))
        if transform == 0:
            z_0_ = np.ones((X_.shape[0],))
            z_0_ = _ML_inference(Y_, z_0_)
        z_0_/= z_0_.sum()
    # Case of 2 Detected Layers
    if n_layers == 2:
        xx_, yy_ = np.meshgrid(np.linspace(X_[:, 0].min(), X_[:, 0].max(), 100), np.linspace(X_[:, 1].min(), X_[:, 1].max(), 100))

        P_ = []
        e_ = []
        for i in range(3):
            # Infer Distribution via Iterative Conditional Models with Noise level threshold
            W_, _N_ = __Iterated_Conditional_Modes(X_, gamma = 1e-4, beta = -1000, n_init = 1, n_eval = 1000)
            w_hat_, w = __predict(F_tilde_, _N_)
            # Infer Height Distribution from Velocity Vectors Labels Initialization
            W_hat_ = w_hat_.reshape(N, M) + 1
            Q_, q = __ICM_MRF(I_, W_hat_, V_segm_, beta = 0., clique = 0, n_init = 1, n_eval = 1000)
            P_.append([W_, Q_, W_hat_, _N_])
            e_.append(w + q)
        W_, Q_, W_hat_, _N_ = P_[np.argmax(e_)]

        Z_hat_  = _eval_proba(X_, _N_)
        Q_hat_  = _eval_post(Z_hat_)
        L_hat_  = _eval_post(_eval_proba(F_tilde_, _N_)).reshape(N, M, 2)
        # Set To 0 Pixels without Clouds
        W_hat_[I_labels_ == 0.] = 0.
        # Get Velocity Vectors Probabilities
        z_0_ = Z_hat_[:, 0]
        z_1_ = Z_hat_[:, 1]
        # If VV transformation is not applied
        if transform == 0:
            z_0_ = Q_hat_[:, 0]
            z_1_ = Q_hat_[:, 1]
            z_0_ = _ML_inference(Y_, z_0_)
            z_1_ = _ML_inference(Y_, z_1_)
        # Prabability Density Normalization
        z_0_/= z_0_.sum()
        z_1_/= z_1_.sum()
        # Organize the Probability according to height
        z_0_, z_1_, Q_ = __find_layer(I_, Q_, z_0_, z_1_)

    return z_0_, z_1_

# Clouds velocity field features extraction, and remove pixels with no clouds
def _cloud_features(F_):
    # Calculate Divergence and Vorticity
    M_ = _magnitude(F_[..., 0], F_[..., 1])
    D_ = _divergence(F_[..., 0], F_[..., 1])
    V_ = _vorticity(F_[..., 0], F_[..., 1])
    return [M_, D_, V_]

# Extract current time instant Features
def _feature_extraction(_tools, i, I_global_, I_direct_, I_diffuse_, I_scatter_, I_norm_2_, I_segm_2_,
                        x_0_2_, i_sun_max, T):

    # Unpack current Variables to extract the features
    I_, A_0_, dt_ = _tools.get_image_variables(i)

    # Unpack Lagged Variables
    X_, Y_, U_, V_, U_prime_, V_prime_ = _tools.wind_field_
    # Unpack Constants
    X, Y, x_, y_, XY_stdz_, xy_stdz_, _stdz_x, _stdz_y = _tools.get_constants(0)

    # Unpack Parameters
    p_segment, n_select, p_train, n_layers, lag, step_size = _tools.get_constants(1)

    index_ = _tools.get_index(x_0_2_)

    # Unpack Lag tracking Variables
    I_norm_1_, I_segm_1_, I_labels_1_, F_1_, x_0_1_ = _tools.lag_var_

    # Unpack current Weather Vraibles
    ws_ = _tools.get_weather_variables(i)

    irradiance_2_ = [i, dt_, [_tools.py_[i], _tools.gsi_[i], _tools.csi_[i]], [_tools.el_[i], _tools.az_[i]]]

    # Compute Velocity Vectors For Sky-Conditions Model
    #opt_ = [16., 0., 1.]
    F_lk_, M_lk_ = _LK_cloud_velocity_vector(I_norm_1_, I_norm_2_, opt_ = lk_opt_)

    # Compute the different Temperatures and Height After Processing the Solar Irradiance
    dT_, K_0_, H_0_, K_1_, H_1_, K_2_, H_2_ = _extract_radiometric_features(T, ws_,
                                                I_global_, I_direct_, I_diffuse_, i_sun_max)

    # Cloud Segmentation using Optimal Voting Scheme
    I_segm_2_, Z_segm_2_ = _image_segmentation_v0(models_, names_, I_norm_2_, M_lk_, I_scatter_,
                                                      dT_, K_0_, H_0_, K_1_, H_1_, K_2_, H_2_)
    #I_segm_2_ = np.ones(I_segm_2_.shape, dtype = int)
    #Z_segm_2_ = np.ones(Z_segm_2_.shape, dtype = float)

    # Rebust SVC-Persistent Classification for currenct frame Sky Condistions
    label, _tools = _atmospheric_condition_v1(_tools, i, K_0_, M_lk_, m_2_, verbose = True)


    weather_2_ = [label, _tools.temp_[i], _tools.dew_[i], _tools.pres_[i], _tools.ang_[i], _tools.mag_[i], _tools.hum_[i]]

    # Update Window Artifact Persistent Model
    _tools = _window_persistent_model(_tools, I_scatter_, label, n_samples = 250,
                                      n_burnt = 25, verbose = False)

    radiometry_2_ = [K_0_, K_1_, K_2_, H_0_, H_1_, H_2_, I_norm_2_, I_segm_2_, Z_segm_2_, x_0_2_, A_0_, T]


    #if I_segm_2_[index_].sum() > 5:
    if I_segm_2_[index_].sum() > 100000:

        _tools.n_layers = 1

        # Cloud Labels
        W_2_, I_labels_2_, h_2_, temp_scores_ = _wind_layer_inference(I_diffuse_, I_segm_2_, H_2_,
                                                                      index_, n_layers = _tools.n_layers)

        if I_labels_2_.sum() > 2:
            _tools.n_layers = 1

            # Tilt Camera Prespective Transformation
            XYZ_, dXYZ_, A_0_ = _perspective_transformation_v4(_tools.X_, _tools.Y_, _tools.N_x, _tools.N_y,
                                                               x_0_2_, A_0_, height = 1000., display = False)
            # Constant Intensity assumption
            I_norm_1_ *= I_norm_2_.sum()/I_norm_1_.sum()

            # Motion vectors approximation and transformation of velocity vectors velocities to meters per second
            F_2_, F_2_prime_ = _transform_velocity_vectors(I_norm_1_, I_norm_2_, I_labels_2_, W_2_, dXYZ_, h_2_,
                                                           opt_ = [window, tau, n_pyramids, sigma], n_layers = _tools.n_layers)

            if F_2_.sum() != 0:
                _tools.n_layers = 1

                features_2_  = _cloud_features(F_2_prime_)

                # Segmentation of the Velocity Vectors
                V_segm_2_ = _velocity_vectors_segmentation(_tools, I_norm_1_, I_norm_2_, I_labels_2_, x_0_2_, index_,
                                                           percentage = p_segment, method = edge_segm)

                if V_segm_2_.sum() > 2:
                    print(V_segm_2_.sum())
                    # Update Autoregressive Dataset
                    X_, Y_, U_, V_, U_prime_, V_prime_ = _get_autoregressive_dataset(X_, Y_, U_, V_, U_prime_, V_prime_,
                                                                                     [_tools.X_[V_segm_2_], _tools.Y_[V_segm_2_]],
                                                                                     [F_2_[V_segm_2_, 0], F_2_[V_segm_2_, 1]],
                                                                                     [F_2_prime_[V_segm_2_, 0], F_2_prime_[V_segm_2_, 1]],
                                                                                     lag = lag)
                    print(F_2_[V_segm_2_, 0], F_2_[V_segm_2_, 1])
                    print(F_2_prime_[V_segm_2_, 0], F_2_prime_[V_segm_2_, 1])
                    # Infer Velocity Vectors per layer Distributions
                    W_, Z_ = _velocity_vector_probability(U_, V_, U_prime_, V_prime_, F_2_, F_2_prime_, H_2_, I_labels_2_, V_segm_2_, W_2_,
                                                          n_layers = _tools.n_layers, transform = transform,
                                                          velocity_inference_method = velocity_inference_method,
                                                          height_inference_method = height_inference_method)
                    # Select Velocty Vectors to train and test the SVM
                    X_tr_, X_ts_, wind_flow_indicator_ = _cloud_velocity_vector_selection(X_, Y_, U_prime_, V_prime_, W_, Z_,
                                                                                          n_select = n_select, n_layers = _tools.n_layers,
                                                                                          percentage = p_train, method = sampling_method)
                    # Approximate Velocity Field on the entire frame
                    flow_2_ = _wind_velocity_field(_tools, X_tr_, X_ts_, XYZ_, dXYZ_, H_2_, h_2_, A_0_, x_0_2_, I_norm_2_, wind_flow_indicator_)


                    _tools.flow_.append([irradiance_2_, radiometry_2_, flow_2_, features_2_])

                    # Update list lag variables
                    _tools.lag_var_ = I_norm_2_, I_segm_2_, I_labels_2_, F_2_, x_0_2_

                    # Update Dataset Variables
                    _tools.wind_field_ = X_, Y_, U_, V_, U_prime_, V_prime_

    else:
        _tools.n_layers = 0
        _tools.flow_.append([irradiance_2_, weather_2_, radiometry_2_])

    return _tools

# Main Function do drive feature extraction and models updates
def _get_features(_tools, i, i_sun_max = 45057.):

    # Unpack current Variables to extract the features
    I_, A_0_, dt_ = _tools.get_image_variables(i)

    # Unpack current Weather Vraibles
    ws_ = _tools.get_weather_variables(i)

    # Unpack Lag tracking Variables
    I_norm_1_, I_segm_1_, I_labels_1_, F_1_, x_0_1_ = _tools.lag_var_

    # Detect sun and find euclidean pair of coordinates
    x_0_2_ = _sun_coordinates_persistent(_tools, I_, x_0_1_, tau = 33000.)

    # Image-processing to obtain raw, normalized, and cloud-segmented image, plus optical-depth geometric transformation
    _tools, I_scatter_, T = _infrared_images_processing(_tools, I_, x_0_2_, m_1_, i_sun_max)

    # Remove windon scattering effects from global radiation
    I_global_, I_direct_, I_diffuse_ = _remove_window_artifacts(_tools, I_, I_scatter_,
                                                                tau = 1100, verbose = False)
    # 8 bits Normalization of the Image for features extraction
    I_norm_2_ = _normalize_infrared_image(I_diffuse_, i_max = 9700.)

    I_diffuse_ += T

    # Radiometry Camera Functionality to obtain clouds height
    dT_, K_0_, H_0_, K_1_, H_1_, K_2_, H_2_ = _extract_radiometric_features(T, ws_, I_global_, I_direct_,
                                                                            I_diffuse_, i_sun_max)
    # Cloud Segmentation using best model
    I_segm_2_ = _image_segmentation_v1(models_, names_, dT_, H_2_)
    #I_segm_2_ = np.ones(I_segm_2_.shape, dtype = int)

    # Make sure that the first frame has been processed before extracting features...
    if _tools.FE:
        _tools = _feature_extraction(_tools, i, I_global_, I_direct_, I_diffuse_, I_scatter_,
                                     I_norm_2_, I_segm_2_, x_0_2_, i_sun_max, T)
    else:
        index_ = _tools.get_index(x_0_2_)

        if I_segm_2_[index_].sum() > 5:
            # Cloud Labels
            W_2_, I_labels_2_, h_2_, temp_scores_ = _wind_layer_inference(I_diffuse_, I_segm_2_, H_0_,
                                                                          index_ = _tools.get_index(x_0_2_),
                                                                          n_layers = _tools.n_layers)
        else:
            I_labels_2_ = np.zeros(I_segm_2_.shape, dtype = int)
        # Save on a list lag tracking variables
        _tools.lag_var_ = I_norm_2_, I_segm_2_, I_labels_2_, None, x_0_2_

        # Feature Extraction for Next Iteration
        _tools.FE = True

    return _tools

# Processing the recordins of the original data-processing file to return cloud features samples in each frame
def _process_file(_tools, file_name, range_ = [1, -1], el = 20):
    # Sequential processing of the images on an original data-processing file
    for i in range(_tools.N)[range_[0]:range_[1]]:
        if _tools.el_[i] > el:
            # Extract Features from current Variables
            _tools = _get_features(_tools, i)
    return _tools

def _extract_dataset(config, p_segment, n_select, p_train, lag, n_layers, file_name):
    # Initilize Day Class
    _tools = _frame(file_name, m_5_name, p_segment = p_segment, lag = lag, n_select = n_select,
                 p_train = p_train, n_layers = n_layers)
    # This function has to extract the predictors necessary for the ARMA function to generate the dataset.
    return _process_file(_tools, file_name, el = 31)

#10-31-95-6_10100
# Parameters
p_segment = 95
n_select  = 200
p_train   = 75
lag       = 6
params    = r'{}-{}-{}-{}'.format(p_segment, n_select, p_train, lag)
print(params)
# Configuration
sampling_method           = 1
velocity_inference_method = 0
edge_segm                 = 1
transform                 = 0
height_inference_method   = 0
config                    = r'{}{}{}{}{}'.format(sampling_method, velocity_inference_method, edge_segm,
                                                 transform, height_inference_method)
print(config)
# Lucas-Kanade
window     = 16.
tau        = 0.
n_pyramids = 1.
sigma      = 1.

# Feature Extraction Algorithm
i      = [4, 15, 16, 22, 26, 27, 28, 30, 38, 39, 51, 61, 62, 67, 68, 70, 81, 86, 89, 92, 96, 98, 100, 102,
          103, 114, 115, 123, 124, 131, 133, 140, 146, 151, 154, 157, 159, 167, 177, 211, 214, 215, 217,
          220, 222, 224, 226, 228, 229, 234, 235, 237][int(sys.argv[1])]

_tools = _extract_dataset(config, p_segment, n_select, p_train, lag, n_layers = 1, file_name = file_names_[i])
#_save_window_persistent_model(_tools, name_ = r'{}{}.pkl'.format(m_6_name, file_names_[i][i + 1][-14:-4]), n_samples = 250)

# Save Features Extracted
filename = r'{}/{}'.format(save_path, file_names_[i][-14:])
with open(filename, 'wb') as file:
    pickle.dump(_tools.flow_, file, protocol = pickle.HIGHEST_PROTOCOL)

# 217 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_12_16.pkl
# 220 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_12_19.pkl
# 222 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_12_21.pkl
# 224 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_12_23.pkl
# 226 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_12_25.pkl
# 228 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_12_27.pkl
#
# 229 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2019_01_08.pkl
# 234 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2019_01_13.pkl
# 235 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2019_01_14.pkl
# 237 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2019_01_16.pkl

# 103 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_05_14.pkl
# 114 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_05_26.pkl
# 115 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_06_06.pkl
# 123 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_06_14.pkl
# 124 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_06_15.pkl
# 131 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_07_02.pkl
#
# 133 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_07_08.pkl
# 140 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_07_15.pkl
# 146 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_07_24.pkl
# 151 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_08_24.pkl
# 154 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_08_27.pkl
# 157 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_08_30.pkl
#
# 159 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_09_02.pkl
# 167 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_09_16.pkl
# 177 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_10_11.pkl
# 211 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_12_09.pkl
# 214 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_12_12.pkl
# 215 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_12_13.pkl

# 4 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2017_12_11.pkl
# 15 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2017_12_23.pkl
# 16 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2017_12_25.pkl
# 22 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2017_12_31.pkl
# 26 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_01_04.pkl
# 27 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_01_05.pkl
#
# 28 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_01_06.pkl
# 30 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_01_08.pkl
# 38 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_01_16.pkl
# 39 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_01_17.pkl
# 51 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_01_30.pkl
# 61 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_02_13.pkl
#
# 62 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_02_14.pkl
# 67 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_02_19.pkl
# 68 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_02_20.pkl
# 70 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_02_22.pkl
# 81 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_03_07.pkl
# 86 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_04_05.pkl
#
# 89 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_04_08.pkl
# 92 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_04_11.pkl
# 96 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_04_16.pkl
# 98 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_04_19.pkl
# 100 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_04_21.pkl
# 102 /Users/Guille/Desktop/cloud_feature_extraction/data/pickle_data_v4/2018_04_23.pkl
#
