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
  vector<Mat> patches;
  /*
  scene_images.push_back(imread("data/dataset1/scene1.png"));
  obj_images.push_back(imread("data/dataset1/obj1.png"));
  */

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

  // Lets just take the first patch for the moment
  Mat m = patches_cropped[0][0];
  ObjectDetection od;
  std::pair<std::vector<cv::KeyPoint>, cv::Mat> res;
  res = od.compute_features(m);
  cout << (res.first).size() << endl;

  return 0;
}

/*
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
*/
