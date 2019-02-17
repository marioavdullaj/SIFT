#include <iostream>
#include <sstream>
#include <numeric>
#include <string>
#include <regex>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/ccalib.hpp>
#include <opencv2/stitching.hpp>
#include "include/ObjectDetection.h"

using namespace std;
using namespace cv;

void show(Mat m, string name, double res) {
  resize(m, m, cv::Size(), res, res);
  namedWindow(name, CV_WINDOW_AUTOSIZE); imshow(name, m);
  return;
}

// Main function
int main(int argc, char** argv) {

  vector<Mat> obj_images;
  vector<Mat> scene_images;

  scene_images.push_back(imread("data/dataset1/scene1.png"));
  obj_images.push_back(imread("data/dataset1/obj1.png"));

  vector<ObjectDetection> objects;
  for(int j = 0; j < obj_images.size(); ++j) {
    ObjectDetection od;
    od.load(obj_images[j]);
    objects.push_back(od);
  }

  for(int j = 0; j < objects.size(); j++) {
    for(int k = 0; k < scene_images.size(); k++) {
      Mat scene_result = objects[j].find_object(scene_images[k]);

      // Resizing the scene object in order visualize the scene image next to the object one
      double res = ((double) objects[j].get_object().rows) / ((double) scene_result.rows);
      resize(scene_result, scene_result, cv::Size(), res,res);
      hconcat(objects[j].get_object(), scene_result,scene_result);
      if(scene_result.cols > 1200) resize(scene_result,scene_result,cv::Size(), 0.7,0.7);
      
      show(scene_result, "Result", 1);
      waitKey(0);
      cv::destroyAllWindows();
    }
  }

  return 0;
}
