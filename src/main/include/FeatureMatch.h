#ifndef SIFT__FEATURE_MATCH__H
#define SIFT__FEATURE_MATCH__H

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
#include <chrono>

class FeatureMatch {
  int tot_images;
  int number_of_query;
  std::vector< std::vector<cv::Mat> > patches;
  std::vector< std::vector< std::pair<std::vector<cv::KeyPoint>, cv::Mat> > > features;
  cv::Mat all_features;
  std::map< int, std::tuple<int,int,int> > row_to_descriptor;
  std::map< int, std::tuple<int,int> > query_to_image;
  std::vector< std::pair<std::vector<cv::KeyPoint>, cv::Mat> > query_images;
  int linear_duration;

  public:
    FeatureMatch(
      std::vector< std::vector<cv::Mat> >,
      std::vector< std::vector< std::pair<std::vector<cv::KeyPoint>, cv::Mat> > >
    );
    void query_train_split(int);
    std::vector<cv::Mat> get_desc_query();
    std::tuple< std::vector<cv::Mat>, std::vector<cv::Mat>, double > linear_knn(int);
    std::tuple< std::vector<cv::Mat>, std::vector<cv::Mat>,  std::vector< std::vector< std::vector< std::vector< std::tuple< int, int, double > > > > > > hierarchical_knn_vs_linear(std::vector<cv::Mat>, std::vector<int>, std::vector<int>, std::vector<int>, int);
    void imageMatching(cv::Mat*, cv::Mat*, int);

  private:
    cv::Mat descriptor_from_query(std::pair<std::vector<cv::KeyPoint>, cv::Mat>);
    double precision_computation(std::vector<cv::Mat>, std::vector<cv::Mat>, int);

};

#endif
