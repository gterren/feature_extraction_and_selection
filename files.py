import pickle, glob, sys, os
import numpy as np

# Import All custom functions
from utils import *

# Software directories
code_path   = r'/users/terren'
method_path = r'{}/cloud_feature_extraction'.format(code_path)
model_path  = r'{}/models'.format(method_path)
# data directories
files_path = r'/users/terren/wheeler-scratch'
load_path  = r'{}/data_v4'.format(files_path)
save_path  = r'{}/data_feature_extraction_v4'.format(files_path)
# Models files names
m_1_name = r'atmospheric_parameters_model_v6-1.pkl'
m_2_name = r'atmospheric_condition_model_v8-T6_0.pkl'
m_5_name = r'persistence_v7/' # Robust-persistent classification
#m_5_name = r'persistence_v8/' # persistent classification
m_6_name = r'persistence_v9/' # Robust-persistent classification with last model
# Atmospheric models paramters v1
m_1_name = '{}/{}'.format(model_path, m_1_name)
m_1_ = _load_file(m_1_name)[0]
# Sky Conditions model
m_2_name = '{}/{}'.format(model_path, m_2_name)
m_2_ = _load_file(m_2_name)[0]
# Window Artifact model
m_5_name = '{}/{}'.format(model_path, m_5_name)
m_6_name = '{}/{}'.format(model_path, m_6_name)
# Load data files
name = r'{}/*'.format(load_path)
file_names_ = sorted(glob.glob(name))
fb_opt_ = [0.26066805, 9.37731822, 2.35361781, 78.92822339, 1.32486497, 0.63680272]
hs_opt_ = [3.34485977, 244.26498405]
lk_opt_ = [8.74308106, 5.21506878e-3, 1.01125549]
nx_opt_ = [18.5478764, 17.21399269]
sx_opt_ = [23.71361892, 11.05462314]
# Cloud Segmentation Optimal Voting Scheme set of Models
names_, models_ = _load_segm_models(r'/users/terren/cloud_feature_extraction/models/segmentation_models_v0')
