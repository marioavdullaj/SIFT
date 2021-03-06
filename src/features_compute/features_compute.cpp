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
#include <opencv2/stitching.hpp>
#include "../include/ObjectDetection.h"

using namespace std;
using namespace cv;

void show(Mat m, string name, double res) {
  resize(m, m, cv::Size(), res, res);
  namedWindow(name, CV_WINDOW_AUTOSIZE); imshow(name, m);
  return;
}

vector< vector< std::pair<std::vector<cv::KeyPoint>, cv::Mat> > > load(string);
void save(vector< vector< std::pair<std::vector<cv::KeyPoint>, cv::Mat> > >, string);

int num_patches, images_per_patch;

// Main function
int main(int argc, char** argv) {
  vector<Mat> patches;
  ObjectDetection od;
  string filename("features_data.yaml");

  vector< String > fn;
  stringstream dirname; dirname << "data/notredame";
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

  int w_size = 64;
  int h_size = 64;

  vector< vector<Mat> > patches_cropped;
  vector< vector< std::pair<std::vector<cv::KeyPoint>, cv::Mat> > > features, ff;

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

  cout << "Found " << patches.size() << " patches: how many to select for SIFT computation?: ";
  cin >> num_patches;
  cout << "How many images per patch? (each patch has " << patches_cropped[0].size() << " images): ";
  cin >> images_per_patch;

  cout << "Computing the features for each image..." << endl;
  for(int i = 0; i < num_patches; i++) {
    cout << ((double) i / num_patches) * 100 << " %" << endl;
    vector< std::pair<std::vector<cv::KeyPoint>, cv::Mat> > ft;
    for(int j = 0; j < images_per_patch; j++) {
      Mat im = patches_cropped[i][j];
      ft.push_back( od.compute_features(im) );
    }
    features.push_back(ft);
  }

  cout << "Saving features into file..." << endl;
  save(features, filename);


  return 0;
}

void save(vector< vector< std::pair<std::vector<cv::KeyPoint>, cv::Mat> > > features, string filename) {
  cv::FileStorage store(filename, cv::FileStorage::WRITE);
  cv::write(store, "num_patches", (int) features.size());
  for(int p = 0; p < features.size(); ++p) {
    stringstream patch_size; patch_size << "size_patch_" << p;
    cv::write(store, patch_size.str(), (int) (features[p].size()));
    for(int i = 0; i < features[p].size(); i++) {
      stringstream ss, ss1;
      ss << "keypoints" << p << "-" << i;
      ss1 << "descriptors" << p << "-" << i;
      cv::write(store,ss.str(),features[p][i].first);
      cv::write(store,ss1.str(),features[p][i].second);
    }
  }
  store.release();

  return;
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
