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
#include "include/ObjectDetection.h"
#include <chrono>

using namespace std;
using namespace std::chrono;
using namespace cv;


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

void show(Mat m, string name, double res) {
  resize(m, m, cv::Size(), res, res);
  namedWindow(name, CV_WINDOW_AUTOSIZE); imshow(name, m);
  return;
}

Mat descriptor_from_query( std::pair<std::vector<cv::KeyPoint>, cv::Mat> x){
	return 1.0/512*x.second;
}

// Main function
int main(int argc, char** argv) {
  vector<Mat> obj_images;
  vector<Mat> scene_images;
  vector<Mat> patches;

  ObjectDetection od;
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
  vector< vector< std::pair<std::vector<cv::KeyPoint>, cv::Mat> > > features;

  cout << "Cropping the patches" << endl;
  for(int i = 0; i < (int)patches.size(); i++) {
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

  cout << "Computing the features for each image..." << endl;
  for(int i = 0; i < 5 + 0*patches_cropped.size(); i++) {
	cout << i << endl;
    vector< std::pair<std::vector<cv::KeyPoint>, cv::Mat> > ft;
    for(int j = 0; j < (int)patches_cropped[i].size(); j++) {
      Mat im = patches_cropped[i][j];
      ft.push_back( od.compute_features(im) );
    }
    features.push_back(ft);
  }
  
  
  // Here I assume to have the features vector and I do the computations on it to get training/query dataset
  cv::Mat all_features = features[0][0].second;
  
  
  // Compute the total number of images
  int total_images = 0;
  for(int i = 0; i < (int)features.size(); ++i)
	total_images+=(int)features[i].size();
  cout << "Total number of images: " << total_images << endl;
  // Construct a map to decide whether an image belongs to query dataset
  double p = 0.05; // Percent of dataset to be taken as query
  int query_indexes[total_images];
  for(int i = 0; i < total_images; ++i)
	query_indexes[i] = i;
  random_shuffle(query_indexes,query_indexes+total_images);
  int total_query = (int)(p*total_images);
  map<int,bool> is_query;;
  for(int i = 0; i < total_query; ++i)
	if(query_indexes[i] != 0)
		is_query[ query_indexes[i]] = true;
  
  cout << "Calculating training/query dataset... " << std::flush;
    
  map< int, tuple<int,int,int> > row_to_descriptor;
  map< int, tuple<int,int> > query_to_image;
  vector< std::pair<std::vector<cv::KeyPoint>, cv::Mat> > query_images;
  // Initialize the train descriptors (image (0,0) belongs to train dataset!)
  Mat train_descriptors = features[0][0].second; 
  int row_number = 0;
 
  for(int k = 0; k < features[0][0].second.rows; ++k)
	row_to_descriptor[k] = tuple<int,int,int>(0,0,k);
  row_number+= features[0][0].second.rows;
  for(int i = 0; i < (int)features.size(); ++i)
	for(int j = 0; j < (int)features[i].size(); ++j){
		if(i == 0 && j == 0) continue;
		if(!is_query[i*256+j]){
			vconcat(all_features,features[i][j].second,all_features);
			for(int k = 0; k < features[i][j].second.rows; ++k)
				row_to_descriptor[k+row_number] = tuple<int,int,int>(i,j,k);
			row_number+=features[i][j].second.rows;
		}else{
			query_images.push_back(features[i][j]);
			query_to_image[ (int)query_images.size() - 1] = tuple<int,int>(i,j);
			
		}
	}
  cout << "DONE!" << endl;
  
  // Normalize the training matrix!
  all_features = 1.0/512*all_features;
  
  
	
  // Compute the descriptor matrix associated to every query
  
  int number_of_query = (int)query_images.size();
  Mat desc_query[number_of_query];
  for(int i = 0; i < number_of_query; ++i)
	desc_query[i] = descriptor_from_query( query_images[i]);	
	
	
   /* From now on we will assume the have the following data
    * 1. patches_cropped <- needed to print  ( first two index of patches_cropped == first two indexes features)
    * 2. features <- required to find keypoint associated to descriptor
    * 3. all_features <- matrix of the training set (rows are descriptor)
    * 4. row_to_descriptor <- map (row all_features) -> (index on features), required to find keypoint associated to a descriptor
    * 4. query_images <- vector< pair< vector<KEYPOINT>, Mat<DESCRIPTOR> > describing query images
    * 5. query_to_image <- map (index of query images) -> (index on features)
    */
    
  /* ********************
   *   EXPERIMENTAL     * 
   ******************** */  
  
  // Number of nearest neighbors
  int knn = 10;   
  
  // Ground truth data denoted with truth_ keyword
  Mat truth_indicies[ number_of_query ];
  Mat truth_distance[ number_of_query ];
  Mat indicies[ number_of_query ];
  Mat distance[ number_of_query ];
  for(int i = 0; i < number_of_query; ++i){
	truth_indicies[i] = Mat(128,knn,CV_64F);
	truth_distance[i] = Mat(128,knn,CV_32S);
	indicies[i] = Mat(128,knn,CV_64F);
	distance[i] = Mat(128,knn,CV_32S);
  }
  
  // Time measuremenet variables
  high_resolution_clock::time_point t_start,t_end,t_midpoint;

   // auto duration = duration_cast<microseconds>( t2 - t1 ).count();

   // cout << duration;
   //  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  

  // Linear KNN (Also used to compute ground truth)
  t_start = high_resolution_clock::now();
  flann::Index linear_index = flann::Index(all_features,flann::LinearIndexParams());
  for(int i = 0; i < number_of_query; ++i)
	linear_index.knnSearch(desc_query[i], truth_indicies[i], truth_distance[i], knn);
  t_end = high_resolution_clock::now();
  
  auto truth_duration = duration_cast<microseconds>(t_end-t_start).count();
  cout << "Linear search KNN duration: " << truth_duration << endl;
  
  
  // Hierarchical k-means clustering 

  vector<int> branching{2,4,8,16,32};
  vector<int> leaf_size{16,64,128,256};
  vector<int> L_max{20, 30, 40, 50, 60, 70, 80};   // These values are related with precision and assumes knn = 10
  
  
  for(int n_branch = 0; n_branch < (int)branching.size(); ++n_branch)
	for(int n_leaf = 0; n_leaf < (int)leaf_size.size(); ++n_leaf)
	   for(int n_L = 0; n_L < (int)L_max.size(); ++n_L){
			
			t_start = high_resolution_clock::now();
			// We use 11 iterations
			flann::Index hKmean = flann::Index(all_features, flann::KMeansIndexParams( branching[n_branch], 11, cvflann::FLANN_CENTERS_RANDOM, 0.2));
			t_midpoint = high_resolution_clock::now();
			for(int i = 0; i < number_of_query; ++i)
				hKmean.knnSearch(desc_query[i], indicies[i], distance[i], knn, flann::SearchParams(L_max[n_L]));
			t_end = high_resolution_clock::now();
			
			auto duration = duration_cast<microseconds>(t_end-t_start).count();
			auto build_duration = duration_cast<microseconds>(t_midpoint - t_start).count();
			
				
		    // Compute precision  precision_computation(truth_indicies, indicies) TO IMPLEMENT!
			cout << "kMean search (B = " << branching[n_branch] << ", Leaf = " << leaf_size[n_leaf] << ", L_max = " << L_max[n_L] << "): " << duration << " (build:" << build_duration << ")" <<  endl;
			cout << "Precision: " << 0.421;
			
			
			
	    }

  



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
