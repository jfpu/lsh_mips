#ifndef LSH_MIPS_SRC_CPP_LSH_H
#define LSH_MIPS_SRC_CPP_LSH_H

#include <vector>

#include <Eigen/Dense>
#include <Eigen/Core>

using Eigen::RowVectorXi;
using Eigen::MatrixXd;
using Eigen::VectorXd;

std::string hash_combine(const std::vector<int> &hashes);
std::string cosine_hash_combine(const std::vector<int> &hashes);
double dot(const VectorXd &u, const VectorXd &v);
VectorXd g_ext_norm(const VectorXd &vec, const int m);
VectorXd g_ext_half(const int m);
VectorXd g_ext_norm_cosine(const VectorXd &vec, const int m);
VectorXd g_ext_norm_simple(const VectorXd &vec, const int m);
VectorXd g_ext_zero(const int m);
MatrixXd g_index_extend(const MatrixXd &datas, const int m);
MatrixXd g_query_extend(const MatrixXd &queries, const int m);
MatrixXd g_index_cosine_extend(const MatrixXd &datas, const int m);
MatrixXd g_query_cosine_extend(const MatrixXd &queries, const int m);
MatrixXd g_index_simple_extend(const MatrixXd &datas, const int m);
MatrixXd g_query_simple_extend(const MatrixXd &queries, const int m);
double g_max_norm(const MatrixXd &datas);
MatrixXd g_transformation(const MatrixXd &datas, double &ratio, double &max_norm);
MatrixXd g_normalization(const MatrixXd &queries);
double L2Lsh_distance(const VectorXd &u, const VectorXd &v);
double ConsineLsh_distance(const VectorXd &u, const VectorXd &v);

class Hash {
 public:
  Hash(){}
  virtual ~Hash(){}

  virtual int init(double r, int d) = 0;
  virtual int hash(const VectorXd &vec) = 0;
  virtual double distance(const VectorXd &u, const VectorXd &v) = 0;
};

class L2Lsh : public Hash {
 public:
  L2Lsh(double r, int d);
  ~L2Lsh();
  int init(double r, int d);
  int hash(const VectorXd &vec);
  double distance(const VectorXd &u, const VectorXd &v);

 private:
  // r: fixed size
  // d: data length
  // RandomData: random data vector
  VectorXd data_;
  double r_;
  double b_;
  int d_;
};

class ConsineLsh : public Hash {
 public:
  ConsineLsh(double r, int d);
  ~ConsineLsh();

  int init(double r, int d);
  int hash(const VectorXd &vec);
  double distance(const VectorXd &u, const VectorXd &v);
  // double combine(const VectorXd &hashes);

 private:
  VectorXd data_;
  int d_;
};

#endif
