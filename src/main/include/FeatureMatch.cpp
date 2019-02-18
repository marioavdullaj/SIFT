#include <memory>
#include <iostream>
#include <sstream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "FeatureMatch.h"

FeatureMatch::FeatureMatch(
  std::vector< std::vector<cv::Mat> > p,
  std::vector< std::vector< std::pair<std::vector<cv::KeyPoint>, cv::Mat> > > f
) {
  features = f;
  patches = p;

  int total_images = 0;
  for(int i = 0; i < (int)features.size(); ++i) total_images+=(int)features[i].size();
  std::cout << "Total number of images: " << total_images << std::endl;

  tot_images = total_images;
}

cv::Mat FeatureMatch::descriptor_from_query( std::pair<std::vector<cv::KeyPoint>, cv::Mat> x){
	return 1.0/512*x.second;
}

void FeatureMatch::query_train_split(int query_size) {
  int query_indexes[tot_images];
  for(int i = 0; i < tot_images; ++i) {
    query_indexes[i] = i;
  }
  std::random_shuffle(query_indexes,query_indexes+tot_images);
  std::map<int,bool> is_query;
  for(int i = 0; i < query_size; ++i)
  		is_query[ query_indexes[i]] = true;

  std::cout << "Calculating training/query dataset... " << std::flush;

  // Initialize the train descriptors (image (0,0) belongs to train dataset!)

  int row_number = 0;
  for(int i = 0; i < (int)features.size(); ++i) {
    for(int j = 0; j < (int)features[i].size(); ++j) {
      if(!is_query[i*256+j]) {
        for(int k = 0; k < features[i][j].second.rows; ++k){
          all_features.push_back( features[i][j].second.row(k));
          row_to_descriptor[k+row_number] = std::tuple<int,int,int>(i,j,k);
        }
        row_number+=features[i][j].second.rows;
      } else {
        query_images.push_back(features[i][j]);
        query_to_image[ (int)query_images.size() - 1] = std::tuple<int,int>(i,j);
      }
    }
  }
  std::cout << "DONE!" << std::endl;

  // Normalize the training matrix!
  number_of_query = (int) query_images.size();
  all_features = 1.0/512*all_features;


  return;
}

std::vector<cv::Mat> FeatureMatch::get_desc_query() {
  std::vector<cv::Mat> desc_query;
  for(int i = 0; i < number_of_query; ++i)
   desc_query.push_back(FeatureMatch::descriptor_from_query( query_images[i] ));

  return desc_query;
}

std::pair< std::vector<cv::Mat>, std::vector<cv::Mat> > FeatureMatch::linear_knn(int knn) {
  std::vector<cv::Mat> indicies, distance;
  std::chrono::high_resolution_clock::time_point t_start,t_end,t_midpoint;

  for(int i = 0; i < number_of_query; ++i){
  	indicies.push_back(cv::Mat(query_images[i].first.size(),knn,CV_64F));
  	distance.push_back(cv::Mat(query_images[i].first.size(),knn,CV_32S));
  }

  std::vector<cv::Mat> desc_query = FeatureMatch::get_desc_query();
  cv::flann::Index linear_index = cv::flann::Index(all_features,cv::flann::LinearIndexParams());

  t_start = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < number_of_query; ++i)
	   linear_index.knnSearch(desc_query[i], indicies[i], distance[i], knn);

  t_end = std::chrono::high_resolution_clock::now();
  auto truth_duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count();
  std::cout << "Linear search KNN duration: " << truth_duration << std::endl;

  std::pair< std::vector<cv::Mat>, std::vector<cv::Mat> > ret;
  return ret = std::make_pair(indicies, distance);
}

std::pair< std::vector<cv::Mat>, std::vector<cv::Mat> > FeatureMatch::hierarchical_knn_vs_linear(
  std::vector<cv::Mat> truth_indicies,
  std::vector<int> branching,
  std::vector<int> leaf_size,
  std::vector<int> L_max,
  int knn)
{
  std::vector<cv::Mat> indicies, distance;
  std::chrono::high_resolution_clock::time_point t_start,t_end,t_midpoint;

  for(int i = 0; i < number_of_query; ++i){
  	indicies.push_back(cv::Mat(query_images[i].first.size(),knn,CV_64F));
  	distance.push_back(cv::Mat(query_images[i].first.size(),knn,CV_32S));
  }

  std::vector<cv::Mat> desc_query = FeatureMatch::get_desc_query();

  for(int n_branch = 0; n_branch < (int)branching.size(); ++n_branch) {
  	for(int n_leaf = 0; n_leaf < (int)leaf_size.size(); ++n_leaf) {
  	   for(int n_L = 0; n_L < (int)L_max.size(); ++n_L) {
    			t_start = std::chrono::high_resolution_clock::now();
    		
    			cv::flann::Index hKmean = cv::flann::Index(all_features, cv::flann::HierarchicalClusteringIndexParams( branching[n_branch], cvflann::FLANN_CENTERS_RANDOM, 1, leaf_size[n_leaf]));
    			t_midpoint = std::chrono::high_resolution_clock::now();
    			for(int i = 0; i < number_of_query; ++i)
    				hKmean.knnSearch(desc_query[i], indicies[i], distance[i], knn, cv::flann::SearchParams(L_max[n_L]));
    			t_end = std::chrono::high_resolution_clock::now();

    			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_midpoint).count();
    			auto build_duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_midpoint - t_start).count();

          std::cout << "kMean search (B = " << branching[n_branch] << ", Leaf = " << leaf_size[n_leaf] << ", L_max = " << L_max[n_L] << "): " << duration << " (build:" << build_duration << ")" <<  std::endl;
          double precision = precision_computation(truth_indicies, indicies, number_of_query);
          std::cout << "PRECISION: " << precision << std::endl;
       }
    }
  }

  std::pair< std::vector<cv::Mat>, std::vector<cv::Mat> > ret;
  return ret = std::make_pair(indicies, distance);
}

double FeatureMatch::precision_computation(std::vector<cv::Mat> truth_indicies, std::vector<cv::Mat> indicies, int queries) {
	int total_features = 0;
	for(int i = 0; i < queries; ++i)
		total_features+=indicies[i].rows;
	//cout << "TOTAL FEATURES: " << total_features << endl;
	int knn = truth_indicies[0].cols;
	std::vector<int> intersect(2*knn);
	std::vector<int> truth_feat(knn);
	std::vector<int> feat(knn);
	std::vector<int>::iterator it;
	double precision = 0;
	for(int i = 0; i < queries; ++i)
		for(int j = 0; j < indicies[i].rows; ++j){
			intersect.resize(2*knn);
			truth_indicies[i].row(j).copyTo(truth_feat);
			indicies[i].row(j).copyTo(feat);
			sort(truth_feat.begin(),truth_feat.begin()+knn);
			sort(feat.begin(),feat.begin()+knn);

			it=std::set_intersection (truth_feat.begin(), truth_feat.begin()+knn, feat.begin(), feat.begin()+knn, intersect.begin());
			intersect.resize(it-intersect.begin());
			precision+= 1.0*intersect.size()/knn;
			//cout << i << " " << j << " : " << intersect.size() << endl;

		}
	return precision = precision/total_features;
}

void FeatureMatch::imageMatching(cv::Mat* indicies, cv::Mat* dist, int index_of_query) {
  std::pair<int,int> idx_query( std::get<0>(query_to_image[index_of_query]), std::get<1>(query_to_image[index_of_query]));
  std::map< int, std::tuple<int,int,int> >* row_to_idx = &(this->row_to_descriptor);
  std::vector< std::vector< std::pair<std::vector<cv::KeyPoint>, cv::Mat> > >* features = &(this->features);
  std::vector< std::vector<cv::Mat> >* images = &(this->patches);

	int q_features = indicies->rows;
	int knn = indicies->cols;

	// Compute minimum distance on first column
	double min_dist = 123456; // The maximum should be sqrt( 1/4 * 128) anyway
	for(int i = 0; i < q_features; ++i)
		min_dist = std::min( dist->at<double>(i,0), min_dist);

	std::map< std::pair<int,int>, std::vector< std::pair<int,int> > > keypoint_association;
	std::map< std::pair<int,int>, int> n_association;
	int max_val = 0;
	for(int j = 0; j < knn; ++j){
		for(int i = 0; i < q_features; ++i){
			// Good matching definition (Lowe test is not suitable as there are multiple images of the same object)
			if( dist->at<double>(i,j) < std::max(2*min_dist,0.02)){
				std::tuple<int,int,int> tup = row_to_idx->at( indicies->at<int>(i,j));
				std::pair<int,int> idx_image( std::get<0>(tup), std::get<1>(tup));
				int n_descriptor = std::get<2>(tup);
				keypoint_association[ idx_image].push_back(std::make_pair( i, n_descriptor));
				n_association[idx_image]++;
				max_val = std::max( n_association[idx_image], max_val);
			}
			
			if(max_val >= 5) break;
		}
	}

	using pair_type = decltype(n_association)::value_type;
	auto pr = std::max_element
	(
		std::begin(n_association), std::end(n_association),
		[] (const pair_type & p1, const pair_type & p2) {
			return p1.second < p2.second;
		}
	);

	if(pr->second < 3 ){
		std::cout << "What is a match?! " << pr->second << std::endl;
		return;
	} else
		std::cout << "Found " << pr->second << " possible matching with image " << pr->first.first << " " << pr->first.second << " ( query: " << idx_query.first << " " << idx_query.second << ")" <<  std::endl;

	std::vector< cv::DMatch > good_matches;
	std::pair<int,int> keyy( pr->first.first, pr->first.second);
	for(int i = 0; i < (int)keypoint_association[keyy].size(); ++i){
		int qd = keypoint_association[keyy][i].first;
		int id = keypoint_association[keyy][i].second;
		good_matches.push_back( cv::DMatch( qd,id,0));
	}

	cv::Mat img_matches;
	cv::drawMatches( images->at(idx_query.first)[idx_query.second],
				 features->at(idx_query.first)[idx_query.second].first,
				 images->at(pr->first.first)[pr->first.second],
				 features->at(pr->first.first)[pr->first.second].first,
                 good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                 std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
  //-- Show detected match
  cv::imshow( "Good Matches", img_matches );
  cv::waitKey(0);
  return;
}
