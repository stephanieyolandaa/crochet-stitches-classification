# crochet-stitches-classification
This repository is for the classification of crochet stitches using pre-trained models.

Crochet Stitches Classification.ipynb - Table of contents
  1 Prepare dataset
    1.1 Import images and labels
    1.2 Display imported images with labels
  2 Pre-process data
    2.1 Convert to NumPy arrays
    2.2 Convert to grayscale
    2.3 Pre-process data (label vectorization and data normalization)
    2.4 Split dataset
  3 Data augmentation
    3.1 Online data augmentation
    3.2 Offline data augmentation
  4 Define required functions
    4.1 Initialize variables (hyperparameters)
    4.2 Function for choosing dataset
    4.3 Functions for documentation purposes
    4.4 Functions for building and training models
    4.5 Functions for model evaluation
    4.6 Functions for the complete process
  5 Get best dataset option
    5.1 Explore different datasets on models
    5.2 Analysis
  6 Baseline model
  7 Pre-trained models
    7.1 Xception model
    7.2 VGG-16 model
    7.3 ResNet-50 model
    7.4 Analysis
  8 Hyperparameter tuning
    8.1 Implement grid search
    8.2 Results
    8.3 Best model obtained from grid search
  9 Final model
    9.1 Prepare train set
    9.2 Implement final model
  10 Model deployment
