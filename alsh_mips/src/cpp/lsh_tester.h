#ifndef LSH_MIPS_SRC_CPP_LSH_TESTER_H
#define LSH_MIPS_SRC_CPP_LSH_TESTER_H

#include <vector>
#include <string>
#include <lsh.h>
#include <lsh_wrapper.h>

#include <Eigen/Dense>
#include <Eigen/Core>

using Eigen::RowVectorXi;
using Eigen::MatrixXd;
using Eigen::VectorXd;

using namespace std;

class LshTester {
 public:
  LshTester();
  virtual ~LshTester();

  virtual int init(MatrixXd datas, MatrixXd queries, double rand_range = 1.0, int num_neighbours = 1, int m = 3);
  virtual VectorXd linear(const VectorXd &q, metric_func metric, int max_results);
  // type: hash class type
  // k_vec: k vector for hash wrapper. default as [2]
  // l_vec: L vector for hash wrapper. default as [2]
  virtual void run(string type, vector<int> k_vec={2}, vector<int> l_vec={2});

// private:
  // datas: datas for build hash index
  // queries: query datas
  // rand_range: random range for norm
  // num_neighbours: query top num_neighbours
  int d;
  MatrixXd datas;
  MatrixXd queries;
  double rand_range;
  int num_neighbours;
  int q_num;
};

class L2AlshTester : public LshTester {
 public:
  L2AlshTester();
  ~L2AlshTester();

  int init(MatrixXd datas, MatrixXd queries, double rand_range, int num_neighbours, int m);
  VectorXd linear(const VectorXd &q, metric_func metric, int max_results);
  void run(string type, vector<int> k_vec, vector<int> l_vec);

 private:
  // m: ALSH extend metrix length. default 3
  int m;
  MatrixXd origin_datas;
  MatrixXd origin_queries;
  MatrixXd norm_datas;
  MatrixXd norm_queries;
  MatrixXd ext_datas;
  MatrixXd ext_queries;
};

class CosineAlshTester : public LshTester {
 public:
  CosineAlshTester();
  ~CosineAlshTester();

  int init(MatrixXd datas, MatrixXd queries, double rand_range, int num_neighbours, int m);
  VectorXd linear(const VectorXd &q, metric_func metric, int max_results);
  void run(string type, vector<int> k_vec, vector<int> l_vec);

 private:
  // m: ALSH extend metrix length. default 3
  int m;
  MatrixXd origin_datas;
  MatrixXd origin_queries;
  MatrixXd norm_datas;
  MatrixXd norm_queries;
  MatrixXd ext_datas;
  MatrixXd ext_queries;
};

class SimpleAlshTester : public LshTester {
 public:
  SimpleAlshTester();
  ~SimpleAlshTester();

  int init(MatrixXd datas, MatrixXd queries, double rand_range, int num_neighbours, int m);
  VectorXd linear(const VectorXd &q, metric_func metric, int max_results);
  void run(string type, vector<int> k_vec, vector<int> l_vec);

 private:
  // m: ALSH extend metrix length. default 3
  int m;
  MatrixXd origin_datas;
  MatrixXd origin_queries;
  MatrixXd norm_datas;
  MatrixXd norm_queries;
  MatrixXd ext_datas;
  MatrixXd ext_queries;
};

// LSH Test factory
class LshTesterFactory {
 public:
  LshTesterFactory(){}
  ~LshTesterFactory(){}

  LshTester *createTester(string type, bool mips, MatrixXd datas, MatrixXd queries, int rand_num, int num_neighbours, int m = 3);
};

#endif
