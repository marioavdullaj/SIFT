# Fast matching of SIFT features
Authors: Alessio Mazzetto, Mario Avdullaj

## Introduction
Based on the Fast Matching of Binary Features paper, we developed a C++ program which computes the SIFT
features of 10.000 gray-scale 64x64 pixel notredame images. These are then used to perform feature matching strategies using a query of images over the entire dataset. Finally, the results of the experiment are store in YAML format files, which are taken in input by a Python script. This script parses the results and plots the graphs using the matplotlib library.

## Details of the implementation
The code has been developed using the following framework:
- the dataset is store in ./data/notredame. This consists in 406 patches of 256 images of some notredame architectures.
- the graph results have been store in ./graph_results. These are grouped by two different clustering methods, using  FLANN_CENTERS_RANDOM and FLANN_CENTERS_GONZALES as parameters for building the hierarchical clustering trees.
- the source code can be found in ./src. We provided the implementation of two C++ programs: ./src/feature_compute and ./src/main.
- the Python script can be found at ./plot_graphs.py, which receives in input the results of the k-NN searches, and
  by the user input parameters it plots specific graphs of the speed up of the fast approximate search algorithm over the linear search, as function of the percentage of the desired precision.


The feature_compute program takes in input the number of patches and the number of images per patch set by the user. Consequently, it uses the OpenCV library to interface to the images, and crop them in a 64x64 format, extracting each image from its patch. The SIFT features are then computed and the final results are store in a YAML format file (features_data.yaml).
The main program then takes this file as input, crops the images of the dataset, and performs both the linear k-NN search and the fast approximate using the hierarchical clustering trees for a query of images, using the FLANN library provided by OpenCV. Since the experiment requires a considerable amount of time, the results are then store in a YAML format file (results.yaml).

## Compile and execution of the project
The code can be executed by compiling all the project with the provided bash script create.sh.
To compile, type sh create.sh.
The compilation requires the OpenCV library. Once the project is built, the executables can be launched directly
from the root folder.
