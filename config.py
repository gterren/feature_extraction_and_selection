import pickle, glob, sys, os

import numpy as np

# Import All custom functions
from datetime import datetime
from utils import *
from wind_velocity_field_utils import _wind_velocity_field_coordinates

# Define Object with all vairables and constants for a day feature extration
class _frame:
    def __init__(self, file_name, model_name, p_segment, n_select, p_train, n_layers, lag):

        # Prediction Interval
        self.g_ = [2, 4, 8, 16]

        # Variables defintion of feature Extraction algorithm
        self.p_segment = p_segment
        self.n_select  = n_select
        self.p_train   = p_train
        self.n_layers  = n_layers
        self.field_lag = lag
        self.field_step_size = 1

        # Constants defintion of feature Extraction algorithm
        self.file_name = file_name
        self.N_x = 80
        self.N_y = 60
        self.dim = 3

        # Define Euclidian coordiantes system for each pixel on an image frame
        self.X_, self.Y_ = _euclidian_coordiantes(self.N_x, self.N_y)
        # Define the wind velocity sef of coordinates in matrix, and vector form,
        # either as integer as standardized real numbers. And, full frame or reduce dimensions
        self.x_, self.y_, self.XY_stdz_, self.xy_stdz_, self._stdz_x, self._stdz_y \
        = _wind_velocity_field_coordinates(self.X_, self.Y_, self.field_step_size)

        # Variables initialization...
        self.Xi_ = np.empty((2, len(self.g_), self.dim, 0))  # Covariate dataset variable
        self.I_  = np.empty((1, 0))                          # Predictor dataset variable
        self.M_  = np.empty((1, 0))                          # Predictor dataset variable
        self.P_  = np.empty((1, 0))                          # Predictor dataset variable
        self.Z_  = np.empty((2, 0))                          # Covariate auxiliar dataset variable
        self.T_  = np.empty((1, 0), dtype = 'datetime64[s]') # Time label dataset variable
        self.xi_ = None

        # lag variables lists initialization
        self.position_   = [np.array((40, 30))[:, np.newaxis]]
        self.labels_     = []
        self.lag_var_    = [None, None, None, None, np.array((40, 30))]
        self.flow_       = []
        self.wind_field_ = [[], [], [], [], [], []]
        # Do Features Extraction
        self.FE  = False
        self.FV_ = []
        self.MA_ = []
        # Load Window Persistent Model
        window_name = r'{}{}'.format(model_name, file_name[-14:])
        m_ = _load_file(window_name)[0]
        self.W_     = m_[0]
        self.W_lag_ = m_[1]

        # Load pre-Processed GHI recordings from the pyranometer, and images from the IR camera
        X_ = _load_file(file_name)[0]
        self.py_, self.gsi_, self.csi_                                     = X_[0]
        self.el_, self.az_                                                 = X_[1]
        self.temp_, self.dew_, self.pres_, self.ang_, self.mag_, self.hum_ = X_[2]
        self.t_, self.img_                                                 = X_[3]
        self.N                                                             = len(self.t_)

        # Remove possible outliers because of clouds appearing in the frames
        self.idx_frame_x_ = (self.X_ < 5) | (self.X_ >= 75)
        self.idx_frame_y_ = (self.Y_ < 5) | (self.Y_ >= 55)

    def get_index(self, x_0_):
        idx_cirumsolar_ = _cart_to_polar(self.X_ - x_0_[0], self.Y_ - x_0_[1])[0] > 5.
        return np.invert(self.idx_frame_x_ | self.idx_frame_y_) & idx_cirumsolar_

    # Return 3 Sets of consntant
    def get_constants(self, c):
        if c == 0: return self.X_, self.Y_, self.x_, self.y_, self.XY_stdz_, self.xy_stdz_, self._stdz_x, self._stdz_y
        if c == 1: return self.p_segment, self.n_select, self.p_train, self.n_layers, self.field_lag, self.field_step_size
        if c == 2: return self.g_, self.N_y, self.N_x, self.dim

    # Return list with all data
    def get_data(self):
        return [self.Xi_, self.I_, self.W_, self.P_, self.Z_, self.T_]

    # And concatenate current feature extraction data to form the entire dataset
    def stack_data(self, flow):
        if not flow:
            self.xi_= np.zeros((2, len(self.g_), self.dim, 1))
        self.Xi_ = np.concatenate((self.Xi_, self.xi_), axis = 3)
        self.I_  = np.concatenate((self.I_, self.i), axis = 1)
        self.M_  = np.concatenate((self.M_, self.m), axis = 1)
        self.P_  = np.concatenate((self.P_, self.p), axis = 1)
        self.Z_  = np.concatenate((self.Z_, self.z), axis = 1)
        self.T_  = np.concatenate((self.T_, self.t), axis = 1)

    # The weather station variables in an array
    def get_weather_variables(self, i):
        return np.asarray((self.temp_[i], self.dew_[i], self.pres_[i], self.ang_[i], self.mag_[i], self.hum_[i]))

    # Unpack the data that is on the original data processing file...
    def get_image_variables(self, i, verbose = True):
        self.j = i
        # ith-sample UNIX captured-time
        dt_ = datetime.fromtimestamp(self.t_[i])
        # Get ith-sample variables
        self.i = self.csi_[i][np.newaxis, np.newaxis] # CSI
        self.m = self.gsi_[i][np.newaxis, np.newaxis] # GHI_theorical
        self.p = self.py_[i][np.newaxis, np.newaxis]  # GHI_pyra
        self.t = np.datetime64(dt_)[np.newaxis, np.newaxis] # Time label for current frame
        self.z = np.array((self.el_[i], self.az_[i]))[:, np.newaxis] # Sun angles on the horizont
        # Covariates Atmospheric Model
        # Model v1-1
        self.x_0_ = np.array((self.temp_[i], self.dew_[i], self.el_[i]))[np.newaxis, :]
        self.x_1_ = np.array((dt_.timetuple().tm_yday, self.el_[i], self.az_[i]))[np.newaxis, :]
        # Infrared image
        I_ = self.img_[i]
        # Trajectory of the Sun in horizon
        A_sun_ = np.array((self.el_[i:i + self.g_[-1] + 1], self.az_[i:i + self.g_[-1] + 1]))
        if verbose: print( '>>> Date: {} No. frame: {} - {} Elvation: {} Azimuth: {} CSI: {}'.format(dt_, i, self.N, self.el_[i], self.az_[i], self.csi_[i]) )
        # Return All Variables that are necessary
        return I_, A_sun_, dt_
