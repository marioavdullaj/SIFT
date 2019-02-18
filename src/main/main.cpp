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

using namespace std;
using namespace cv;

void show(Mat m, string name, double res) {
  resize(m, m, cv::Size(), res, res);
  namedWindow(name, CV_WINDOW_AUTOSIZE); imshow(name, m);
  return;
}

Mat descriptor_from_query( std::pair<std::vector<cv::KeyPoint>, cv::Mat> x){
	return 1.0/512*x.second;
}

vector< vector< std::pair<std::vector<cv::KeyPoint>, cv::Mat> > > load(string);
vector< vector<Mat> > load_patches(string, int, int);

// Main function
int main(int argc, char** argv) {
  string filename("features_data");
  vector< vector<Mat> > patches_cropped = load_patches("data/notredame",64,64);
  cout << "Loading the features.." << endl;
  vector< vector< std::pair<std::vector<cv::KeyPoint>, cv::Mat> > > features = load(filename);

  // Here I assume to have the features vector and I do the computations on it to get training/query dataset
  cv::Mat all_features = features[0][0].second;

  cout << "continuing..." << endl;
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


  cout << "Total number of query images: " <<  query_images.size() << endl;


  cout << "Trying to linear search " << endl;

  //Do some magic here :D
  flann::Index linear_index = flann::Index(all_features,flann::LinearIndexParams());

  cout << "Done!" << endl;
  int knn = 20;
  Mat desc_query = descriptor_from_query( query_images[3]);
  cout << desc_query << endl;
  Mat indicies(desc_query.rows, knn, CV_64F);
  Mat distance(desc_query.rows, knn, CV_64F);

  linear_index.knnSearch(desc_query, indicies,distance,knn);
  cout << desc_query.rows << " " << endl;
  cout << distance.rows << " " << distance.cols << endl;


  for(int i = 0; i < desc_query.rows; ++i){
	cout << "ROW " << i << ": ";
	for(int j = 0; j < distance.cols; ++j)
		cout << distance.at<float>(i,j) << " ";
	cout << endl;
	}

	for(int i = 0; i < desc_query.rows; ++i){
		cout << "ROW " << i << ": ";
		for(int j = 0; j < distance.cols; ++j)
			cout << get<0>(row_to_descriptor[ indicies.at<int>(i,j)]) << " "  << get<1>(row_to_descriptor[ indicies.at<int>(i,j)])  << " " << get<2>(row_to_descriptor[ indicies.at<int>(i,j)]) << ") ";
		cout << endl;
	}

  int where = -1;
  float dist = 1232142;
  int rMATCH;
  int cMATCH;
  map< pair<int,int> ,int> image_matches;
  for(int j = 0; j < distance.cols; ++j){

	for(int i =0 ; i < distance.rows; ++i){
		int r = get<0>(row_to_descriptor[ indicies.at<int>(i,j)]);
		int c = get<1>(row_to_descriptor[ indicies.at<int>(i,j)]);
		image_matches[ pair<int,int>(r,c)]++;
		if(image_matches[pair<int,int>(r,c)] >= 3){
			rMATCH = r;
			cMATCH = c;
			cout << " FOUND STRONG MATCH " << r << " " << c << endl;
			i = 1232412;
			j = 3214211;
		}
	}
  }


  int ii, jj;
  ii = get<0>(query_to_image[1]); jj = get<0>(query_to_image[1]);
  show(patches_cropped[ii][jj],"Query",1);
  waitKey(0);
  cout << dist << endl;
  show(patches_cropped[rMATCH][cMATCH],"Best match",1);
  waitKey(0);
  cv::destroyAllWindows();

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
