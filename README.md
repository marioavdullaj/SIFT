# Fast matching of SIFT features
Authors: Alessio Mazzetto, Mario Avdullaj

## Introduction
Based on the "Fast Matching of Binary Features" paper, we developed a C++ program which performs approximate matching of SIFT features using as dataset 10'000 gray-scale 64x64 pixel images from the Notredame dataset (Winder/Brown).

## Details of the implementation
The code has been developed using the following framework:
- the dataset is stored in ./data/notredame. This consists in 406 patches of 256 images each of details of the Notredame building.
- the graphs obtained in the experimentations are stored in ./graph_results. These are grouped by two different methods used to select the centers in the hierarchical clustering trees: FLANN_CENTERS_RANDOM (random selection) and FLANN_CENTERS_GONZALES (farthest-first traversal).
- the source code can be found in ./src. We provide the implementation of two C++ programs: ./src/feature_compute and ./src/main.
- a Python script can be found at ./plot_graphs.py . The script receives in input the results of the k-NN searches, and it plots specific graphs which compare the linear search over approximate methods for the k-NN problem.


The feature_compute program takes in input the number of patches and the number of images per patch set. Consequently, it uses the OpenCV library to load the images from the Notredame dataset and crop them in a 64x64 format, extracting each image from its patch. The SIFT features are computed for each extracted image and the final results are stored in a YAML format file (features_data.yaml).
The main program takes this file as input to build the experimental dataset, and performs both the linear k-NN search and the approximate k-NN search which uses hierarchical clustering trees, using 10 random query images (from the Notredame dataset). To build the hierarchical clustering trees, we used the FLANN Library. Since the experiments require a considerable amount of time, the results are stored in a YAML format file (results.yaml).

## Compile and execution of the project
The code can be executed by compiling all the project with the provided bash script create.sh.
To compile, type 'sh create.sh' in the root directory.
The compile requires the OpenCV library. Once the project is built, the executables can be launched directly
from the root folder.
