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
void save_results(double, std::vector<int>, std::vector<int>, std::vector<int>, std::vector< std::vector< std::vector< std::vector< std::tuple< int, int, double > > > > >, string);
// Main function
int main(int argc, char** argv) {

  std::srand(std::time(nullptr));

  string filename("features_data.yaml");
  vector< vector<Mat> > patches_cropped = load_patches("data/notredame",64,64);
  cout << "Loading the features.." << endl;
  vector< vector< std::pair<std::vector<cv::KeyPoint>, cv::Mat> > > features = load(filename);

  int n_features = 0;
  for (int i = 0; i < features.size(); i++)
    for (int j = 0; j < features[i].size(); j++)
      n_features += (features[i][j].first).size();
  std::cout << "Number of features: " << n_features << std::endl;

  FeatureMatch fm(patches_cropped, features);
  int number_of_query = 10;
  fm.query_train_split(number_of_query);

  // Number of nearest neighbors
  int knn = 8;

  // Linear KNN (Also used to compute ground truth)
  std::tuple< std::vector<cv::Mat>, std::vector<cv::Mat>, double > linear_ind_dist = fm.linear_knn(knn);
  std::vector<cv::Mat> truth_indicies = std::get<0>(linear_ind_dist);
  std::vector<cv::Mat> truth_distance = std::get<1>(linear_ind_dist);
	double duration_linear = std::get<2>(linear_ind_dist);


  // Hierarchical k-means clustering
/*
  // Experiment set up
  std::vector<int> branching{2,4,8,16,32,64};
  std::vector<int> leaf_size{16,150,500};
  std::vector<int> trees{1,2,3,4,8,16};
*/
  std::vector<int> branching{32};
  std::vector<int> leaf_size{150};
  std::vector<int> trees{8};

 	std::tuple< std::vector<cv::Mat>,
							std::vector<cv::Mat>,
							std::vector< std::vector< std::vector< std::vector< std::tuple< int, int, double > > > > >
						> hier_ind_dist = fm.hierarchical_knn_vs_linear(truth_indicies, branching, leaf_size, trees, knn);

/*
	The performance data structure is a 3-dimensional data structure of pair values
	which contains for each branching value, leaf_size value and trees value the
	performance of the Hierarchical kmeans compared to the linear knn one.
	Specifically, the values of the pairs are relative to the duration time and the precision.
*/
	std::vector< std::vector< std::vector< std::vector< std::tuple< int, int, double > > > > >
		performance = std::get<2>(hier_ind_dist);

/*
	These are the indicies and the distance of the last branching, leaf_size and trees parameters.
	The compute them all do some cycles in the main giving single parameters to the method.
*/
  std::vector<cv::Mat> indicies = std::get<0>(hier_ind_dist);
  std::vector<cv::Mat> distance = std::get<1>(hier_ind_dist);

  std::cout << "Feature matching with linear kNN search" << std::endl;
  for(int i = 0; i < number_of_query; ++i)
		fm.imageMatching(&truth_indicies[i], &truth_distance[i], i);

  std::cout << "Feature matching with fast approximate search" << std::endl;
  for(int i = 0; i < number_of_query; ++i)
    fm.imageMatching(&indicies[i], &distance[i], i);


	save_results(duration_linear, branching, leaf_size, trees, performance, "results.yaml");
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

void save_results(double duration_linear,
									std::vector<int> branching,
									std::vector<int> leaf_size,
									std::vector<int> trees,
									std::vector< std::vector< std::vector< std::vector< std::tuple< int, int, double > > > > > performance,
									string filename)
{
  cv::FileStorage store(filename, cv::FileStorage::WRITE);
  cv::write(store, "duration_linear", (double) duration_linear);

	stringstream s; s << branching[0];
	for(int i = 1; i < branching.size(); i++)
		s << "," << branching[i];
	cv::write(store, "branching", (string) s.str());

	s.str(""); s << leaf_size[0];
	for(int i = 1; i < leaf_size.size(); i++)
		s << "," << leaf_size[i];
	cv::write(store, "leaf_size", (string) s.str());

	s.str(""); s << trees[0];
	for(int i = 1; i < trees.size(); i++)
		s << "," << trees[i];
	cv::write(store, "trees", (string) s.str());

  for(int b = 0; b < branching.size(); b++) {
    for(int l = 0; l < leaf_size.size(); l++) {
			for(int t = 0; t < trees.size(); t++) {

				stringstream ss1, ss2; ss1 << std::get<0>(performance[b][l][t][0]);
				for(int i = 1; i < performance[b][l][t].size(); i++)
					ss1 << "," << std::get<0>(performance[b][l][t][i]);
				ss2 << "L_max" << branching[b] << "-" << leaf_size[l] << "-" << trees[t];
				cv::write(store, ss2.str(), (string) ss1.str());

				for(int lm = 0; lm < performance[b][l][t].size(); lm++) {
		      stringstream ss;
					ss << "duration" << branching[b] << "-" << leaf_size[l] << "-" << trees[t] << "-" << std::get<0>(performance[b][l][t][lm]);
					cv::write(store, ss.str(), (int) std::get<1>(performance[b][l][t][lm]) );
					ss.str("");

					ss << "precision" << branching[b] << "-" << leaf_size[l] << "-" << trees[t] << "-" << std::get<0>(performance[b][l][t][lm]);
					cv::write(store, ss.str(), (double) std::get<2>(performance[b][l][t][lm]) );
					ss.str("");
				}
			}
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
