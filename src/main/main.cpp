#include <iostream>
#include <sstream>
#include <numeric>
#include <string>
#include <regex>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/ccalib.hpp>
#include <map>
#include <opencv2/stitching.hpp>
#include "include/FeatureMatch.h"
#include <chrono>

using namespace std;
using namespace cv;

vector< vector< std::pair<std::vector<cv::KeyPoint>, cv::Mat> > > load(string);
vector< vector<Mat> > load_patches(string, int, int);
double precision_computation(Mat*, Mat*, int);

// Main function
int main(int argc, char** argv) {
  string filename("features_data");
  vector< vector<Mat> > patches_cropped = load_patches("data/notredame",64,64);
  cout << "Loading the features.." << endl;
  vector< vector< std::pair<std::vector<cv::KeyPoint>, cv::Mat> > > features = load(filename);


  FeatureMatch fm(patches_cropped, features);
  int number_of_query = 5;
  fm.query_train_split(number_of_query);

  // Number of nearest neighbors
  int knn = 10;

  // Linear KNN (Also used to compute ground truth)
  std::pair< std::vector<cv::Mat>, std::vector<cv::Mat> > linear_ind_dist = fm.linear_knn(knn);
  std::vector<cv::Mat> truth_indicies = linear_ind_dist.first;
  std::vector<cv::Mat> truth_distance = linear_ind_dist.second;


  // Hierarchical k-means clustering
  std::vector<int> branching{2,4,8,16,32};
  std::vector<int> leaf_size{16,64,128,256};
  std::vector<int> L_max{20, 30, 40, 50, 60, 70, 80};

  std::pair< std::vector<cv::Mat>, std::vector<cv::Mat> > hier_ind_dist = fm.hierarchical_knn_vs_linear(truth_indicies, branching, leaf_size, L_max, knn);

  int index_of_query = 0;
  fm.imageMatching(&truth_indicies[index_of_query], &truth_distance[index_of_query], index_of_query);
  return 0;
}


vector< vector<Mat> > load_patches(string dir, int w_size, int h_size) {
  vector<Mat> patches;
  vector< vector<Mat> > patches_cropped;
  vector< String > fn;
  stringstream dirname; dirname << dir;
  glob(dirname.str(), fn, false);
  for (size_t j=0; j<fn.size(); j++) {
    string s = (string) fn[j];
    cout << s;
    if( regex_match(s, regex("(.*)(patches)(.*)")) ) {
      cout << " ----> Patch loaded!";
      patches.push_back(imread(fn[j]));
    }
    cout << endl;
  }


  cout << "Cropping the patches" << endl;
  for(int i = 0; i < patches.size(); i++) {
    // We are now analyzing the i-th patch image
    vector<Mat> patch_images;
    for(int j = 0; j < patches[i].cols / w_size; j++) {
      for(int k = 0; k < patches[i].rows / h_size; k++) {
        Rect ROI(j*w_size,k*h_size, w_size, h_size);
        patch_images.push_back(patches[i](ROI));
      }
    }
    patches_cropped.push_back(patch_images);
  }
  return patches_cropped;
}

vector< vector< std::pair<std::vector<cv::KeyPoint>, cv::Mat> > > load(string filename) {
  int num_patches;
  cv::FileStorage store(filename, cv::FileStorage::READ);
  cv::FileNode n1 = store["num_patches"];
  cv::read(n1,num_patches,-1);

  if(num_patches == -1) { cout << "Error reading" << endl; }

  vector< vector< std::pair<std::vector<cv::KeyPoint>, cv::Mat> > > read_features;
  for(int p = 0; p < num_patches; ++p) {
    int size_patch;
    stringstream ss; ss << "size_patch_" << p;
    cv::FileNode n = store[ss.str()];
    cv::read(n, size_patch, -1);

    vector< std::pair<std::vector<cv::KeyPoint>, cv::Mat> > ff;
    for(int i = 0; i < size_patch; i++) {
      std::vector<cv::KeyPoint> kp;
      cv::Mat dsc;
      stringstream ss,ss1;
      ss << "keypoints" << p << "-" << i;
      ss1 << "descriptors" << p << "-" << i;
      cv::FileNode n1 = store[ss.str()];
      cv::FileNode n2 = store[ss1.str()];
      cv::read(n1, kp);
      cv::read(n2, dsc);

      std::pair<std::vector<cv::KeyPoint>, cv::Mat> temp;
      temp = make_pair(kp, dsc);
      ff.push_back(temp);
    }
    read_features.push_back(ff);
  }
  store.release();

  return read_features;
}
