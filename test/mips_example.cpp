#include "falconn/lsh_nn_table.h"

#include <cstdio>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include <cassert>

using std::cout;
using std::ends;
using std::endl;

using std::make_pair;
using std::unique_ptr;
using std::vector;

using falconn::compute_number_of_hash_functions;
using falconn::construct_table;
using falconn::DenseVector;
using falconn::DistanceFunction;
using falconn::LSHConstructionParameters;
using falconn::LSHFamily;
using falconn::LSHNearestNeighborTable;
using falconn::SparseVector;
using falconn::StorageHashTable;

using falconn::DenseMips;
using falconn::DenseMipsParaProber;
using falconn::AccuracySpeed;

void mips_example_dense(const LSHConstructionParameters& params,
    falconn::NORM_TYPE mtype = falconn::NORM_TYPE::SIMPLE) {
  typedef DenseVector<float> Point;

  Point p1(params.dimension);
  p1[0] = 3;
  p1[1] = 5;
  p1[2] = 2;
  p1[3] = 4;

  Point p2(params.dimension);
  p2[0] = 6;
  p2[1] = 0;
  p2[2] = 1;
  p2[3] = 1;

  Point p3(params.dimension);
  p3[0] = 2;
  p3[1] = 1;
  p3[2] = 5;
  p3[3] = 4;

  Point p4(params.dimension);
  p4[0] = 1;
  p4[1] = 3;
  p4[2] = 2;
  p4[3] = 5;

  vector<Point> in_points;
  in_points.push_back(p1);
  in_points.push_back(p2);
  in_points.push_back(p3);
  in_points.push_back(p4);

  DenseMips<float> dm(params, in_points, mtype);
  // dm.set_debug_file();
  dm.norm_action();

  Point q1(params.dimension);
  q1[0] = 1;
  q1[1] = 2;
  q1[2] = 3;
  q1[3] = 4;

  Point q2(params.dimension);
  q2[0] = 1;
  q2[1] = 3;
  q2[2] = 4;
  q2[3] = 2;

  vector<Point> queries;
  queries.push_back(q1);
  queries.push_back(q2);

  dm.set_queries(queries);
  int32_t q1_index = dm.find_mips_point(q1);
  assert (2 == q1_index);

  int32_t q2_index = dm.find_mips_point(q2);
  assert (0 == q2_index);

  printf("mips_example_dense: dim(%d), mips_type(%d)\n", params.dimension, (int)mtype);
}

void do_hyperplane_mips_dense(falconn::NORM_TYPE mtype = falconn::NORM_TYPE::SIMPLE) {
  const int dim = 4;

  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::Hyperplane;
  params.distance_function = DistanceFunction::NegativeInnerProduct;
  params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
  params.k = 2;
  params.l = 8;
  params.num_setup_threads = 0;

  mips_example_dense(params, mtype);
}


void do_hyperplane_mips_validate(falconn::NORM_TYPE mtype = falconn::NORM_TYPE::SIMPLE) {
  // typedef DenseVector<float> Point;
  const int dim = 4;

  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::Hyperplane;
  params.distance_function = DistanceFunction::NegativeInnerProduct;
  params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
  params.k = 2;
  params.l = 8;
  params.num_setup_threads = 0;

  DenseMipsParaProber<float> dmt(dim, 100);
  // dmt.set_print_file();
  bool bret = dmt.validate_mips(params, mtype);
  assert (true == bret);

  printf("validate_mips: dim(%d), mips_type(%d)\n", params.dimension, (int)mtype);
}

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void gen_datas(vector<Point>* datas, const uint32_t dim,
    const uint32_t data_num, const uint32_t min = 0, const uint32_t max = 100) {
  datas->clear();
  datas->assign(data_num, Point(dim));

  random_device rd;
  mt19937_64 gen(rd());
  uniform_int_distribution<> u(min, max - 1);
  for (auto i = 0; i < data_num; ++i) {
    for (auto j = 0; j < dim; ++j) {
      const int ind = u(gen);
      assert ((min <= ind) && (max > ind));
      (*datas)[i][j] = ind;
    }
  }
}

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void gen_queries(vector<Point>* queries, const uint32_t dim,
    const uint32_t query_num, const uint32_t min = 0, const uint32_t max = 100) {
  queries->clear();
  queries->assign(query_num, Point(dim));

  random_device rd;
  mt19937_64 gen(rd());
  uniform_int_distribution<> u(min, max - 1);
  for (auto i = 0; i < query_num; ++i) {
    for (auto j = 0; j < dim; ++j) {
      const int ind = u(gen);
      assert ((min <= ind) && (max > ind));
      (*queries)[i][j] = ind;
    }
  }
}

void do_hyperplane_mips_para_optimum(falconn::NORM_TYPE mtype = falconn::NORM_TYPE::SIMPLE) {
  typedef DenseVector<float> Point;
  const int dim = 4;

  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::Hyperplane;
  params.distance_function = DistanceFunction::NegativeInnerProduct;
  params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
  params.k = 0;
  params.l = 0;
  params.num_setup_threads = 0;

  vector<Point>* datas = new vector<Point>();
  gen_datas<float>(datas, dim, 1000);
  assert (1000 == datas.size());

  vector<Point>* queries = new vector<Point>();
  gen_queries<float>(queries, dim, 100);
  assert (100 == queries.size());

  DenseMipsParaProber<float> dmt(dim, datas, queries);
  dmt.set_print_file();
  // dmt.print_para();

  AccuracySpeed as;
  bool bret = dmt.evaluate_optimum_paras(as, params, mtype, 10, 0.96);
  assert (true == bret);

  printf("evaluate_optimum_paras: dim(%d), mips_type(%d)\n", params.dimension, (int)mtype);
  cout << "best paras:" << endl;
  cout << "\t k: " << as._k << endl;
  cout << "\t L: " << as._L << endl;
  cout << "\t avg_time: " << as._time << endl;
  cout << "\t accuracy: " << as._accuracy << endl;
}

int main() {
  do_hyperplane_mips_dense();
  do_hyperplane_mips_dense(falconn::NORM_TYPE::SIGN);

  do_hyperplane_mips_validate();
  do_hyperplane_mips_validate(falconn::NORM_TYPE::SIGN);

  do_hyperplane_mips_para_optimum();
  do_hyperplane_mips_para_optimum(falconn::NORM_TYPE::SIGN);

  return 0;
}

