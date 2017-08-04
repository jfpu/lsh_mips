#include "lsh.h"

#include <random>
#include <iostream>
#include <math.h>
#include <sstream>
#include <cmath>

using namespace std;

std::string hash_combine(const std::vector<int> &hashes) {
  int size = hashes.size();
  std::stringstream ss;

  for (int i = 0; i < size; ++i) {
    ss << hashes.at(i) << ",";
  }

  std::string str;
  ss >> str;
  return  str;
}

std::string cosine_hash_combine(const std::vector<int> &hashes) {
  // return sum(2**i if h > 0 else 0 for i, h in enumerate(hashes))

  long s = 0;
  for (int i = 0; i < hashes.size(); ++i) {
    if(hashes[i]>0)
      s += pow(2,i);
  }

  string str = to_string(s);
  return  str;
}


double dot(const VectorXd &u, const VectorXd &v) {
  return u.adjoint()*v;
}

VectorXd g_ext_norm(const VectorXd &vec, const int m) {
  double l2norm_square = dot(vec, vec);
  VectorXd res(m);
  for(int i=0;i<m;i++) {
    res(i) = pow(l2norm_square,i+1);
  }

  return res;
}

VectorXd g_ext_half(const int m) {
  VectorXd res(m);
  for(int i=0;i<m;i++)  {
    res(i) = 0.5;
  }

  return res;
}

VectorXd g_ext_norm_cosine(const VectorXd &vec, const int m) {
  double l2norm_square = dot(vec, vec);
  VectorXd res(m);

  for(int i=0;i<m;i++) {
    res(i) = 0.5 - pow(l2norm_square, i+1);
  }

  return res;
}

VectorXd g_ext_norm_simple(const VectorXd &vec, const int m) {
  double l2norm_square = dot(vec, vec);
  VectorXd res(m);

  for(int i=0;i<m;i++) {
    res(i) = sqrt(1 - l2norm_square);
  }

  return res;
}

VectorXd g_ext_zero(const int m) {
  VectorXd res(m);
  for(int i=0;i<m;i++) {
    res(i) = 0;
  }

  return res;
}

MatrixXd g_index_extend(const MatrixXd &datas, const int m) {   //TODO why 2*m
  MatrixXd res(datas.rows(),datas.cols()+2*m);

  for(int i=0;i<datas.rows();i++) {
    res.row(i).head(datas.cols()) = datas.row(i);
    res.row(i).segment(datas.cols(),m) = g_ext_norm(datas.row(i), m);
    res.row(i).tail(m) = g_ext_half(m);
  }

  return res;
}

MatrixXd g_query_extend(const MatrixXd &queries, const int m) {
  MatrixXd res(queries.rows(),queries.cols()+2*m);

  for(int i=0;i<queries.rows();i++) {
    res.row(i).head(queries.cols()) = queries.row(i);
    res.row(i).segment(queries.cols(),m) = g_ext_half(m);
    res.row(i).tail(m) = g_ext_norm(queries.row(i), m);
  }

  return res;
}

MatrixXd g_index_cosine_extend(const MatrixXd &datas, const int m) {
  MatrixXd res(datas.rows(),datas.cols()+2*m);

  for(int i=0;i<datas.rows();i++) {
    res.row(i).head(datas.cols()) = datas.row(i);
    res.row(i).segment(datas.cols(),m) = g_ext_norm_cosine(datas.row(i), m);
    res.row(i).tail(m) = g_ext_zero(m);
  }

  return res;
}

MatrixXd g_query_cosine_extend(const MatrixXd &queries, const int m) {
  MatrixXd res(queries.rows(),queries.cols()+2*m);

  for(int i=0;i<queries.rows();i++) {
    res.row(i).head(queries.cols()) = queries.row(i);
    res.row(i).segment(queries.cols(),m) = g_ext_zero(m);
    res.row(i).tail(m) = g_ext_norm_cosine(queries.row(i), m);
  }

  return res;
}

MatrixXd g_index_simple_extend(const MatrixXd &datas, const int m) {
  assert (1 == m);
  MatrixXd res(datas.rows(),datas.cols()+2);

  for(int i=0;i<datas.rows();i++) {
    res.row(i).head(datas.cols()) = datas.row(i);
    res.row(i).segment(datas.cols(),m) = g_ext_norm_simple(datas.row(i), m);
    res.row(i).tail(m) = g_ext_zero(m);
  }

  return res;
}

MatrixXd g_query_simple_extend(const MatrixXd &queries, const int m) {
  assert (1 == m);
  MatrixXd res(queries.rows(),queries.cols()+2);

  for(int i=0;i<queries.rows();i++) {
    res.row(i).head(queries.cols()) = queries.row(i);
    res.row(i).segment(queries.cols(),m) = g_ext_zero(m);
    res.row(i).tail(m) = g_ext_norm_simple(queries.row(i), m);
  }

  return res;
}

double g_max_norm(const MatrixXd &datas) {
  VectorXd data_row(datas.rows());

  for(int i=0;i<datas.rows();i++) {
    data_row(i) = datas.row(i).norm();
  }

  return data_row.maxCoeff();
}

// datas transformation. S(xi) = (U / M) * xi
MatrixXd g_transformation(const MatrixXd &datas, double &ratio, double &max_norm) {
  // U < 1  ||xi||2 <= U <= 1. recommend for 0.83
  const double U = 0.83;

  //U = 0.75
  max_norm = g_max_norm(datas);
  ratio = double(U / max_norm);
  MatrixXd res(datas.rows(), datas.cols());

  for(int i=0;i<datas.rows();i++) {
    res.row(i) = datas.row(i) * ratio;
  }

  return res;
}

// normalization for each query
MatrixXd g_normalization(const MatrixXd &queries) {
  const double U = 0.83;

  //U = 0.75
  MatrixXd norm_queries(queries.rows(), queries.cols());
  double ratio;

  for(int i=0;i<queries.rows();i++) {
    ratio = double(U / queries.row(i).norm());
    norm_queries.row(i) = queries.row(i) * ratio;
  }

  return norm_queries;
}

double L2Lsh_distance(const VectorXd &u, const VectorXd &v) {
  return sqrt((u - v).squaredNorm());
}

double ConsineLsh_distance(const VectorXd &u, const VectorXd &v) {
  return 1 - (dot(u, v) / sqrt(dot(u, u) * dot(v, v)));
}

L2Lsh::L2Lsh(double r, int d) {
  srand((unsigned int) time(0));
  r_ = r;
  d_ = d;
  VectorXd v = VectorXd::Random(1);
  b_ = v(0) * r_ / 2 ;        // 0 < b < r    TODO

  std::random_device rd;
  std::mt19937 e2(rd());

  //gauss gen
  std::normal_distribution<> distribution(0, 1.0);
  auto gauss = [&] () {return distribution(e2);};
  data_ = VectorXd::NullaryExpr(d, gauss);
}

L2Lsh::~L2Lsh() {
}

int L2Lsh::init(double r, int d) {
  return 0;
}

int L2Lsh::hash(const VectorXd &vec) {
  return (int)((dot(vec, data_) + b_) / r_);
}

double L2Lsh::distance(const VectorXd &u, const VectorXd &v) {
  return sqrt((u - v).squaredNorm());
}

ConsineLsh::ConsineLsh(double r, int d) {
  srand((unsigned int) time(0));
  d_ = d;

  // gauss gen
  std::random_device rd;
  std::mt19937 e2(rd());
  std::normal_distribution<> distribution(0, 1);
  auto gauss = [&] () {return distribution(e2);};
  data_ = VectorXd::NullaryExpr(d, gauss);
}

ConsineLsh::~ConsineLsh() {
}

int ConsineLsh::init(double r, int d) {
  return 0;
}

int ConsineLsh::hash(const VectorXd &vec) {
  double dot_val = dot(vec, data_);

  if (dot_val > 0) {
      return 1;
  }

  return 0;
}

double ConsineLsh::distance(const VectorXd &u, const VectorXd &v) {
  return 1 - (dot(u, v) / sqrt(dot(u, u) * dot(v, v)));
}


