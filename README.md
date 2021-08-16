# Feature Extraction and Selection

See these paper for more information about feature extraction: https://arxiv.org/pdf/2101.08694.pdf.

The class file that contains the information about the processing frame is config.py, the file with all paths to models and data is files.py. The files extra.py and utils.py contains the functions necessaries to drive the feature extraction and selection. The remaning files contains the functions introduced in other repositories that needed for the implementation of the feature extraction and selection algorithm. 


## Feature Extraction

The file run_feature_extraction.py performs the feature extraction using the processing methods in the resitories: https://github.com/gterren/cloud_segmentation, https://github.com/gterren/signal_and_image_processing, https://github.com/gterren/multiple_cloud_dection_and_segmentation, https://github.com/gterren/multiple_velocity_fields_visualization, https://github.com/gterren/girasol_machine, and https://github.com/gterren/geospatial_perspective_reprojection.

## Feature Selection

After the features are extracted and the wind velocity field is estimated the files are save in a directory. These files are loaded and processed again to select the pixels that will more likely intercept the Sun. The itersecting probabilies are used to weighted the stastistics computed for each features in file run_feature_selection.py.

## Dataset

A sample dataset is publicaly available in DRYAD repository: https://datadryad.org/stash/dataset/doi%253A10.5061%252Fdryad.zcrjdfn9m
