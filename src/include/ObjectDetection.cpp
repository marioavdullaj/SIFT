#include <memory>
#include <iostream>
#include <sstream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "ObjectDetection.h"

// Functions
void ObjectDetection::load(cv::Mat obj) {
  object = obj.clone();
}

cv::Mat ObjectDetection::get_object() { return object; }

float ObjectDetection::min_distance(std::vector<cv::DMatch> m) {
  float min = m[0].distance;
  for(int i = 1; i < m.size(); ++i)
    if(m[i].distance < min)
      min = m[i].distance;
  return min;
}

std::vector<cv::DMatch> ObjectDetection::refine_match(std::vector<cv::DMatch> matches, double ratio) {
  float min_d = ObjectDetection::min_distance(matches);
  std::vector<cv::DMatch> temp;
  for(int j = 0; j < matches.size(); j++)
    if(matches[j].distance <= min_d*ratio)
      temp.push_back(matches[j]);
  return temp;
}

std::vector<cv::DMatch> ObjectDetection::find_matches(double ratio) {
  std::vector<cv::DMatch> raw_matches,refined_matches;
  cv::Ptr< cv::xfeatures2d::SIFT > sif = cv::xfeatures2d::SIFT::create();
  cv::xfeatures2d::SIFT *s = sif.get();
  cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2, false);

  s->detect(object, keypoints_object);
  s->detect(scene, keypoints_scene);
  s->compute(object, keypoints_object, descriptors_object);
  s->compute(scene, keypoints_scene, descriptors_scene);

  matcher->train();
  matcher->match(descriptors_object,descriptors_scene,raw_matches,cv::noArray());
  refined_matches = ObjectDetection::refine_match(raw_matches, ratio);

  return refined_matches;
}

cv::Mat ObjectDetection::find_homography_matrix() {
  std::vector<cv::Point2f> obj_kpoints, scene_kpoints;
  std::vector<cv::DMatch> matches = ObjectDetection::find_matches(2);
  for(int j = 0; j < matches.size(); ++j) {
    obj_kpoints.push_back(keypoints_object[matches[j].queryIdx].pt);
    scene_kpoints.push_back(keypoints_scene[matches[j].trainIdx].pt);
  }
  if(!obj_kpoints.size() || !scene_kpoints.size())
    return cv::Mat();
  else
    return cv::findHomography(obj_kpoints, scene_kpoints, CV_RANSAC);
}

cv::Mat ObjectDetection::find_object(cv::Mat scn) {
  scene = scn;
  std::vector<cv::Point2f> obj_corners(4);
  obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint(object.cols, 0);
  obj_corners[2] = cvPoint(object.cols, object.rows); obj_corners[3] = cvPoint(0, object.rows);
  std::vector<cv::Point2f> scene_corners(4);
  cv::Mat H = ObjectDetection::find_homography_matrix();

  if( H.empty() )
    return scene;

  cv::perspectiveTransform( obj_corners, scene_corners, H);
  //Draw lines between the corners
  cv::Mat img_matches = scene.clone();
  cv::line( img_matches, scene_corners[0] , scene_corners[1] , cv::Scalar(0, 255, 0), 4 );
  cv::line( img_matches, scene_corners[1] , scene_corners[2] , cv::Scalar(0, 255, 0), 4 );
  cv::line( img_matches, scene_corners[2] , scene_corners[3] , cv::Scalar(0, 255, 0), 4 );
  cv::line( img_matches, scene_corners[3] , scene_corners[0] , cv::Scalar(0, 255, 0), 4 );

  return img_matches;
}
