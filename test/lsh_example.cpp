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
using falconn::AccuracySpeed;

using falconn::DenseLshParaProber;

void basic_example_dense(const LSHConstructionParameters& params) {
  typedef DenseVector<float> Point;

  Point p1(params.dimension);
  p1[0] = 1.0;
  p1[1] = 0.0;
  p1[2] = 0.0;
  p1[3] = 0.0;
  p1[4] = 0.0;
  p1[5] = 0.0;

  Point p2(params.dimension);
  p2[0] = 0.6;
  p2[1] = 0.8;
  p2[2] = 0.0;
  p2[3] = 0.0;
  p2[4] = 0.0;
  p2[5] = 0.0;

  Point p3(params.dimension);
  p3[0] = 0.0;
  p3[1] = 0.0;
  p3[2] = 1.0;
  p3[3] = 0.0;
  p3[4] = 0.0;
  p3[5] = 0.0;

  vector<Point> points;
  points.push_back(p1);
  points.push_back(p2);
  points.push_back(p3);

  unique_ptr<LSHNearestNeighborTable<Point> > table(
      std::move(construct_table<Point>(points, params)));

  int32_t res1 = table->find_nearest_neighbor(p1);
  assert (0 == res1);
  int32_t res2 = table->find_nearest_neighbor(p2);
  assert(1 == res2);
  int32_t res3 = table->find_nearest_neighbor(p3);
  assert(2 == res3);

  Point p4(params.dimension);
  p4[0] = 0.0;
  p4[1] = 1.0;
  p4[2] = 0.0;
  p4[3] = 0.0;
  p4[4] = 0.0;
  p4[5] = 0.0;

  int32_t res4 = table->find_nearest_neighbor(p4);
  assert(1 == res4);

  printf("basic_example_dense: dim(%d), type(%d)\n", params.dimension, (int)params.lsh_family);
}

void basic_example_sparse(const LSHConstructionParameters& params) {
  typedef SparseVector<float> Point;

  Point p1;
  p1.push_back(make_pair(24, 1.0));
  Point p2;
  p2.push_back(make_pair(7, 0.8));
  p2.push_back(make_pair(24, 0.6));
  Point p3;
  p3.push_back(make_pair(50, 1.0));

  vector<Point> points;
  points.push_back(p1);
  points.push_back(p2);
  points.push_back(p3);

  unique_ptr<LSHNearestNeighborTable<Point> > table(
      std::move(construct_table<Point>(points, params)));

  int32_t res1 = table->find_nearest_neighbor(p1);
  assert (0 == res1);
  int32_t res2 = table->find_nearest_neighbor(p2);
  assert(1 == res2);
  int32_t res3 = table->find_nearest_neighbor(p3);
  assert(2 == res3);

  Point p4;
  p4.push_back(make_pair(7, 1.0));
  int32_t res4 = table->find_nearest_neighbor(p4);
  assert(1 == res4);

  printf("basic_example_sparse: dim(%d), type(%d)\n", params.dimension, (int)params.lsh_family);
}

void do_hyperplane_dense() {
  const int dim = 6;

  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::Hyperplane;
  params.distance_function = DistanceFunction::NegativeInnerProduct;
  params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
  params.k = 2;
  params.l = 4;
  params.num_setup_threads = 0;

  basic_example_dense(params);
}

void do_hyperplane_sparse() {
  const int dim = 100;

  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::Hyperplane;
  params.distance_function = DistanceFunction::NegativeInnerProduct;
  params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
  params.k = 2;
  params.l = 4;
  params.num_setup_threads = 0;

  basic_example_sparse(params);
}

void do_crosspolytope_dense() {
  const int dim = 6;

  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::CrossPolytope;
  params.distance_function = DistanceFunction::NegativeInnerProduct;
  params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
  params.k = 2;
  params.l = 4;
  params.feature_hashing_dimension = 8;
  params.last_cp_dimension = 8;
  params.num_rotations = 3;
  params.num_setup_threads = 0;

  basic_example_dense(params);
}

void do_crosspolytope_sparse() {
  const int dim = 100;

  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::CrossPolytope;
  params.distance_function = DistanceFunction::NegativeInnerProduct;
  params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
  params.k = 2;
  params.l = 4;
  params.feature_hashing_dimension = 8;
  params.last_cp_dimension = 8;
  params.num_rotations = 3;
  params.num_setup_threads = 0;

  basic_example_sparse(params);
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

void do_hyperplane_lsh_validate() {
  typedef DenseVector<float> Point;

  const int dim = 6;

  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::Hyperplane;
  params.distance_function = DistanceFunction::NegativeInnerProduct;
  params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
  params.k = 2;
  params.l = 8;
  params.num_setup_threads = 0;

  DenseLshParaProber<float> dlt(dim, 100);
  dlt.set_print_file();
  bool bret = dlt.validate_lsh(params);
  assert (true == bret);

  printf("validate_mips: dim(%d)\n", params.dimension);
}

void do_hyperplane_dense_para_optimum() {
  typedef DenseVector<float> Point;

  const int dim = 6;

  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::Hyperplane;
  params.distance_function = DistanceFunction::NegativeInnerProduct;
  params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
  params.k = 2;
  params.l = 4;
  params.num_setup_threads = 0;

  vector<Point>* datas = new vector<Point>();
  gen_datas<float>(datas, dim, 1000);
  assert (1000 == datas.size());

  vector<Point>* queries = new vector<Point>();
  gen_queries<float>(queries, dim, 100);
  assert (100 == queries.size());

  DenseLshParaProber<float> dlt(dim, datas, queries);
  dlt.set_print_file();
  // dlt.print_para();

  AccuracySpeed as;
  bool bret = dlt.evaluate_optimum_paras(as, params, 10, 0.96);
  assert (true == bret);

  printf("evaluate_optimum_paras: dim(%d)\n", params.dimension);
  cout << "best paras:" << endl;
  cout << "\t k: " << as._k << endl;
  cout << "\t L: " << as._L << endl;
  cout << "\t avg_time: " << as._time << endl;
  cout << "\t accuracy: " << as._accuracy << endl;
}

int main() {
  do_hyperplane_dense();
  do_hyperplane_sparse();

  do_crosspolytope_dense();
  do_crosspolytope_sparse();

  do_hyperplane_lsh_validate();
  do_hyperplane_dense_para_optimum();

  return 0;
}

