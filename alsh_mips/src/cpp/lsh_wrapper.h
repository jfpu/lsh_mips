#ifndef LSH_MIPS_SRC_CPP_LSH_WRAPER_H
#define LSH_MIPS_SRC_CPP_LSH_WRAPER_H

#include <vector>
#include <map>
#include <string>
#include "lsh.h"

using namespace std;

enum LshType {
  L2_LSH = 0,
  CONSINE_LSH
};

typedef double (*hash_func)(const VectorXd &vec);
typedef double (*metric_func)(const VectorXd &u, const VectorXd &v);

#if 0
typedef struct _buckets { //footprint是多个hash函数的hash值合并结果，indexs是该合并hash值对应的节点索引
  string footprint;
  vector<long> indexs;
} buckets;
#endif

class LshWrapper {
 public:
  LshWrapper();
  ~LshWrapper();

  int init(LshType tpye, int d, double r, int k, int L);
  void release_memory();
  int resize(int L);
  int create_hash();
  std::string hash(const std::vector<Hash *> &hash_vec, const VectorXd &data);
  int index(const MatrixXd &datas);
  VectorXd query(const VectorXd &q, metric_func megtric, int max_result = 1);
  double get_avg_touched();

 private:
  LshType type_;
  int d_;
  double r_;
  int k_;
  int L_;
  std::vector<std::vector<Hash *> > hashes_func_;
  std::vector<std::multimap<string, long> *> hash_map_;

  double tot_touched_;
  int num_queries_;
  MatrixXd datas;
};

#endif
