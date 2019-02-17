#ifndef SIFT__OBJECT_DETECTION__H
#define SIFT__OBJECT_DETECTION__H

#include <memory>
#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ccalib.hpp>
#include <opencv2/stitching.hpp>

class ObjectDetection {
  cv::Mat object, scene;
  std::vector<cv::KeyPoint> keypoints_object, keypoints_scene;
  cv::Mat descriptors_object, descriptors_scene;
  public:
    void load(cv::Mat);
    cv::Mat find_object(cv::Mat);
    cv::Mat get_object();

    std::pair<std::vector<cv::KeyPoint>, cv::Mat> compute_features(cv::Mat);

  private:
    float min_distance(std::vector<cv::DMatch>);
    std::vector<cv::DMatch> refine_match(std::vector<cv::DMatch>, double);
    std::vector<cv::DMatch> find_matches(double);
    cv::Mat find_homography_matrix();
};

#endif
