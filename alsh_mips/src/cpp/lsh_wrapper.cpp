#include "lsh_wrapper.h"
#include <iostream>
#include <algorithm>

using namespace std;

LshWrapper::LshWrapper() {
}

LshWrapper::~LshWrapper() {
}

int LshWrapper::init(LshType type, int d, double r , int k, int L) {
  release_memory();

  type_ = type;
  d_ = d;
  r_ = r;
  k_ = k;
  L_ = L;

  create_hash();

  for (int i = 0; i < L; ++i) {
    std::multimap<std::string, long> *m = new std::multimap<std::string, long>();
    hash_map_.push_back(m);
  }

  return 0;
}

int LshWrapper::create_hash() {
  for (int i = 0; i < L_; ++i) {
    std::vector<Hash *> h_vec;
    for (int j = 0; j < k_; ++j) {
      Hash *h = NULL;
      if (type_ == L2_LSH) {
        h = new L2Lsh(r_, d_);
        // h->init(r_, d_);
      } else if (type_ == CONSINE_LSH) {
        h = new ConsineLsh(r_, d_);
        // h->init(r_, d_);
      } else {
        cout<<"error LSH type !"<<endl;
      }

      h_vec.push_back(h);
    }

    hashes_func_.push_back(h_vec);
  }
  return 0;
}
void LshWrapper::release_memory() {
  for (int i = 0; i < hashes_func_.size(); ++i) {
    for (int j = 0; j < hashes_func_[i].size(); ++j) {
      delete hashes_func_[i][j];
      hashes_func_[i][j] = NULL;
    }

    hashes_func_[i].clear();
  }

  hashes_func_.clear();
  for (int i = 0; i < hash_map_.size(); ++i) {
    delete hash_map_[i];
     hash_map_[i] = NULL;
  }

  hash_map_.clear();
}

int LshWrapper::resize(int L) {
  return 0;
}

std::string LshWrapper::hash(const std::vector<Hash *> &hash_vec, const VectorXd &data) {
  std::vector<int> hash_val;
  int vec_size = hash_vec.size();
  for (int i = 0; i < vec_size; ++i) {
    hash_val.push_back(hash_vec[i]->hash(data));
  }

  if (type_ == L2_LSH) {
    return hash_combine(hash_val);
  } else if (type_ == CONSINE_LSH) {
    return cosine_hash_combine(hash_val);
  } else {
    cout<<"error LSH type !"<<endl;
    return "";
  }
}

int LshWrapper::index(const MatrixXd &datas) {
  this->datas = datas;
  std::vector<std::vector<Hash *> >::const_iterator hash_iter;
  std::vector<std::multimap<std::string, long> * >::const_iterator vec_iter;
  for (hash_iter = hashes_func_.begin(), vec_iter = hash_map_.begin();
      hash_iter != hashes_func_.end() && vec_iter != hash_map_.end();
      vec_iter++, hash_iter++) {
    int row_size = datas.rows();
    for (int i = 0; i < row_size; ++i) {
      std::multimap<std::string, long> *cur_map = *vec_iter;
      cur_map->insert(std::pair<std::string, long>(this->hash(*hash_iter, datas.row(i)), i));
    }
  }

  tot_touched_ = 0.0;
  num_queries_ = 0;
  return 0;
}

VectorXd LshWrapper::query(const VectorXd &q, metric_func metric, int max_result) {
  VectorXd ret_v(max_result);

  #if 0
  triple_l = 3 * self.L
  if max_results > triple_l:
      max_results = triple_l
  elif
  #endif

  if (0 == max_result) {
    max_result = 1;
  }

  // find the max_results closest indexed datas to q according to the supplied metric
  multimap<double,long> candidates;  //double:distance    long:index
  vector<long> candidate_index;
  std::vector<std::vector<Hash *> >::const_iterator hash_iter;
  std::vector<std::multimap<std::string, long> * >::const_iterator vec_iter;
  typedef multimap<string, long>::iterator multi_iter;
  pair<multi_iter, multi_iter> iter;
  string footprint_q;
  double distance;

  // cout<<"max_result = "<<max_result<<endl;
  // cout<<"hashes_func_size = "<<hashes_func_.size()<<endl;
  for (hash_iter = hashes_func_.begin(), vec_iter = hash_map_.begin();
      hash_iter != hashes_func_.end() && vec_iter != hash_map_.end();
      vec_iter++, hash_iter++) {
    footprint_q = this->hash(*hash_iter, q);
    // cout<<"footprint_q= "<<footprint_q<<endl;
    std::multimap<std::string, long> *cur_map = *vec_iter;
    // for(std::multimap<std::string, long>::iterator it=(*cur_map).begin();it!=(*cur_map).end();it++)
    //  cout<<it->first<<" "<<it->second<<endl;
    iter = cur_map->equal_range(footprint_q);
    for(multi_iter k = iter.first; k != iter.second; k++) {
      distance = metric(q, datas.row(k->second));
      if(find(candidate_index.begin(),candidate_index.end(),k->second) == candidate_index.end()) {
        candidates.insert(pair<double,long>(distance,k->second));
        candidate_index.push_back(k->second);
      }
    }
  }

  // for(multimap<double,long>::iterator it = candidates.begin();it!= candidates.end();it++) {
  //  cout<<"distance "<<it->first<<" index "<<it->second<<endl;
  // }

  // update stats
  tot_touched_ += candidates.size();
  num_queries_ += 1;
  // cout<<"candidates.size "<<candidates.size()<<endl;
  multimap<double,long>::iterator iter_sort=candidates.begin();
  for(int i=0;i<max_result&&iter_sort!=candidates.end();i++,iter_sort++) {
    ret_v(i) = iter_sort->second;
  }

  return ret_v;
}

double LshWrapper::get_avg_touched() {
  return double(tot_touched_ / num_queries_);
}
