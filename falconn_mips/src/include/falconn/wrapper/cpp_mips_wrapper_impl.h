#ifndef __CPP_MIPS_WRAPPER_IMPL_H__
#define __CPP_MIPS_WRAPPER_IMPL_H__

#include <memory>
#include <utility>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>

#include <algorithm>
#include <chrono>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdio>

#include <cassert>
#include <stdexcept>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>

#include <type_traits>

#include "../core/bit_packed_flat_hash_table.h"
#include "../core/composite_hash_table.h"
#include "../core/cosine_distance.h"
#include "../core/data_storage.h"
#include "../core/euclidean_distance.h"
#include "../core/flat_hash_table.h"
#include "../core/hyperplane_hash.h"
#include "../core/lsh_table.h"
#include "../core/nn_query.h"
#include "../core/polytope_hash.h"
#include "../core/probing_hash_table.h"
#include "../core/stl_hash_table.h"

#include "prober_data.h"

using std::make_pair;
using std::pair;
using std::unique_ptr;
using std::vector;
using std::string;
using std::cin;
using std::cout;
using std::ends;
using std::endl;
using std::exception;
using std::runtime_error;
using std::ceil;
using std::log;

using std::mt19937_64;
using std::runtime_error;
using std::uniform_int_distribution;
using std::random_device;

using falconn::DenseVector;
using falconn::SparseVector;

using falconn::compute_number_of_hash_functions;
using falconn::construct_table;
using falconn::DistanceFunction;
using falconn::LSHConstructionParameters;
using falconn::LSHFamily;
using falconn::LSHNearestNeighborTable;
using falconn::get_default_parameters;
using falconn::SparseVector;
using falconn::StorageHashTable;

namespace falconn {
namespace wrapper {

typedef enum _NORM_TYPE {
  INVALID = 0,
  SIMPLE,
  SIGN,
} NORM_TYPE;

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void self_normalization(vector<Point>& in_points, vector<Point>& out_points,
    bool multiple_factor, const double U);

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void max_normalization(vector<Point>& in_points, vector<Point>& out_points,
  bool multiple_factor, const double U);

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void extend_half(const size_t m, vector<CoordinateType>& ext_half);

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void extend_norm_sign(const size_t m, Point& point, vector<CoordinateType>& ext_ns);

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
double norm_simple(Point& point);

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void simple_extend_for_index(vector<Point>& in_points, vector<Point>& out_points);

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void simple_extend_for_single_query(Point& in_point, Point& out_point);

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void simple_extend_for_query(vector<Point>& in_points, vector<Point>& out_points);

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void sign_extend_for_index(const size_t m, vector<Point>& in_points, vector<Point>& out_points);

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void sign_extend_for_single_query(const size_t m, Point& in_point, Point& out_point);

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void sign_extend_for_query(const size_t m, vector<Point>& in_points, vector<Point>& out_points);



template <typename CoordinateType>
class DenseMipsWrapper {
 public:
  typedef CoordinateType value_type;
  typedef DenseVector<value_type> Point;

  DenseMipsWrapper(const LSHConstructionParameters& paras,
      vector<Point>& points, NORM_TYPE norm_type, size_t m, double U)
      : _ready_to_alsh(false), _m(m), _U(U), _norm_type(NORM_TYPE::INVALID),
      _paras(paras), _origin_points(points), _batch_queries(NULL) {
    _norm_type = norm_type;

     assert (_paras.lsh_family == LSHFamily::Hyperplane);

    if (NORM_TYPE::SIMPLE == _norm_type) {
      _m = 1;
    }
  }

  virtual ~DenseMipsWrapper() {
    _m = 0;

    _norm_points.clear();
    _ext_points.clear();

    _batch_queries = NULL;
  }

  void set_queries(vector<Point>& queries) {
    _batch_queries = &queries;

    __normlize_batch_queries();
  }

  int32_t find_mips_point(const Point& query) {
    if (!_ready_to_alsh) {
      __normlize_points();
    }

    Point real_query;
    __normlize_single_query(query, real_query);

    return _table->find_nearest_neighbor(real_query);
  }

  void find_k_nearest_mips_points(const Point& query, uint32_t k,
      std::vector<int32_t>* result) {
    if (!_ready_to_alsh) {
      __normlize_points();
    }

    Point real_query;
    __normlize_single_query(query, real_query);

  }

  void find_near_mips_points(const Point& q,
      typename PointTypeTraits<Point>::ScalarType threshold,
      std::vector<int32_t>* result) {
    if (!_ready_to_alsh) {
      __normlize_points();
    }

    Point real_query;
    __normlize_single_query(q, real_query);
  }

  void set_debug_file(string& file) {
    _debug_file = file;
  }

  void norm_action() {
    __normlize_points();
  }

  double get_avg_time() const {
    auto statistics = _table->get_query_statistics();
    return statistics.average_total_query_time;
  }

 private:
  void __set_ready() {
     _table = std::move(construct_table<Point>(_ext_points, _paras));
    _ready_to_alsh = true;
  }

  void __normlize_points() {
    max_normalization<CoordinateType>(_origin_points, _norm_points, true, _U);

    if (NORM_TYPE::SIMPLE == _norm_type) {
      simple_extend_for_index<CoordinateType>(_norm_points, _ext_points);
    } else {
      sign_extend_for_index<CoordinateType>(_m, _norm_points, _ext_points);
    }

    debug_index_print();

    _norm_points.clear();

    // ready to work
    __set_ready();
  }

 void __normlize_batch_queries() {
   self_normalization<CoordinateType>(*_batch_queries, _norm_batch_queries, true, _U);

   if (NORM_TYPE::SIMPLE == _norm_type) {
     simple_extend_for_query<CoordinateType>(_norm_batch_queries, _ext_batch_queries);
   } else {
     sign_extend_for_query<CoordinateType>(_m, _norm_batch_queries, _ext_batch_queries);
   }

   debug_batch_queries_print();

   _norm_batch_queries.clear();
 }

  void __normlize_single_query(const Point& query, Point& real_query) {
    Point norm_q(query);
    norm_q.normalize();
    norm_q *= _U;

    if (NORM_TYPE::SIMPLE == _norm_type) {
      simple_extend_for_single_query<CoordinateType>(norm_q, real_query);
    } else {
      sign_extend_for_single_query<CoordinateType>(_m, norm_q, real_query);
    }

    debug_single_query_print(query, norm_q, real_query);
  }

  void debug_index_print() {
    if (_debug_file.empty()) {
      return;
    }

    FILE* fp = fopen(_debug_file.c_str(), "a");
    if (NULL == fp) {
      string throw_info = string("can't open the file: ") + _debug_file;
      throw runtime_error(throw_info);
      return;
    }

    debug_point(fp, _origin_points, "_origin_points");
    debug_point(fp, _norm_points, "_norm_points");
    debug_point(fp, _ext_points, "_ext_points");

    fclose(fp);
  }

  void debug_batch_queries_print() {
    if (_debug_file.empty()) {
      return;
    }

    FILE* fp = fopen(_debug_file.c_str(), "a");
    if (NULL == fp) {
      string throw_info = string("can't open the file: ") + _debug_file;
      throw runtime_error(throw_info);
      return;
    }

    debug_point(fp, *_batch_queries, "_batch_queries");
    debug_point(fp, _norm_batch_queries, "_norm_batch_queries");
    debug_point(fp, _ext_batch_queries, "_ext_batch_queries");

    fclose(fp);
  }

  void debug_single_query_print(const Point& q, const Point& norm_q, const Point& real_q) {
    if (_debug_file.empty()) {
      return;
    }

    FILE* fp = fopen(_debug_file.c_str(), "a");
    if (NULL == fp) {
      string throw_info = string("can't open the file: ") + _debug_file;
      throw runtime_error(throw_info);
      return;
    }

    debug_point(fp, q, "q");
    debug_point(fp, norm_q, "norm_q");
    debug_point(fp, real_q, "real_q");
  }

  void debug_point(FILE* fp, vector<Point>& points, const string& explain) {
    assert (NULL != fp);
    assert (0 < points.size());

    fprintf(fp, "\n%s:\n", explain.c_str());

    size_t rows = points[0].rows();
    for (auto& p : points) {
      for (auto i = 0; i < rows; ++i) {
        fprintf(fp, "%lf, ", p[i]);
      }
      fprintf(fp, "\n");
    }
    fprintf(fp, "============================================================\n");
  }

   void debug_point(FILE* fp, const Point& point, const string& explain) {
    assert (NULL != fp);

    fprintf(fp, "\n%s:\n", explain.c_str());

    for (auto i = 0; i < point.rows(); ++i) {
      fprintf(fp, "%lf, ", point[i]);
    }
    fprintf(fp, "\n");
    fprintf(fp, "============================================================\n");
  }

 private:
  bool _ready_to_alsh;
  size_t _m;
  double _U;
  NORM_TYPE _norm_type;
  const LSHConstructionParameters& _paras;

  vector<Point>& _origin_points;
  vector<Point> _norm_points;
  vector<Point> _ext_points;

  vector<Point>* _batch_queries;
  vector<Point> _norm_batch_queries;
  vector<Point> _ext_batch_queries;

  string _debug_file;

  std::unique_ptr<LSHNearestNeighborTable<Point> > _table = nullptr;

 private:
  DenseMipsWrapper(const DenseMipsWrapper&) = delete;
  DenseMipsWrapper& operator=(const DenseMipsWrapper&) = delete;
};

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
falconn::wrapper::DenseMipsWrapper<CoordinateType>*
create_dense_wrapper(const LSHConstructionParameters& paras,
  vector<Point>& points, NORM_TYPE norm_type, size_t m, double U) {
  return new falconn::wrapper::DenseMipsWrapper<CoordinateType>(paras, points, norm_type, m, U);
}

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void self_normalization(vector<Point>& in_points, vector<Point>& out_points,
    bool multiple_factor, const double U) {
  if (in_points.empty()) {
    return;
  }

  out_points.clear();
  out_points.reserve(in_points.size());

  size_t i = 0;
  for (auto& p : in_points) {
    out_points.push_back(p);
    out_points[i++].normalize();
  }
  assert (in_points.size() == i);

  do {
    if (!multiple_factor) {
      break;
    }

    for (auto& p : out_points) {
      p *= U;
    }
  } while (false);

  assert (in_points.size() == out_points.size());
  assert (in_points[0].rows() == out_points[0].rows());
  assert (in_points[0].cols() == out_points[0].cols());
}

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void max_normalization(vector<Point>& in_points, vector<Point>& out_points,
  bool multiple_factor, const double U) {
  if (in_points.empty()) {
    return;
  }

  out_points.clear();
  out_points.reserve(in_points.size());

  CoordinateType max_norm = 0;

  for (auto& p : in_points) {
    out_points.push_back(p);

    CoordinateType norm = p.norm();
    if (max_norm < norm) {
      max_norm = norm;
    }
  }

  do {
    for (auto& p : out_points) {
      p /= max_norm;
    }

    if (!multiple_factor) {
      break;
    }

    for (auto& p : out_points) {
      p *= U;
    }
  } while (false);

  assert (in_points.size() == out_points.size());
  assert (in_points[0].rows() == out_points[0].rows());
  assert (in_points[0].cols() == out_points[0].cols());
}

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void extend_half(const size_t m, vector<CoordinateType>& ext_half) {
  ext_half.clear();
  ext_half.assign(m, 0.5);
}

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void extend_norm_sign(const size_t m, Point& point, vector<CoordinateType>& ext_ns) {
    CoordinateType l2norm_square = point.squaredNorm();
  assert (1 >= l2norm_square);
  ext_ns.clear();
  ext_ns.reserve(m);
  for (size_t i = 1; i <= m; ++i) {
    ext_ns.push_back(0.5 - pow(l2norm_square, i));
  }
  assert (ext_ns.size() == m);
}

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
double norm_simple(Point& point) {
  CoordinateType l2norm_square = point.squaredNorm();
  assert (1 >= l2norm_square);
  return sqrt(1 - l2norm_square);
}

// Simple-ALSH: build data exend: [x] => [x;    sqrt(1 - ||x||**2);    0]
template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void simple_extend_for_index(vector<Point>& in_points, vector<Point>& out_points) {
  if (in_points.empty()) {
    return;
  }

  out_points.clear();

  size_t new_rows = in_points[0].rows() + 2;
  size_t new_cols = in_points[0].cols();
  out_points.assign(in_points.size(), Point(new_rows, new_cols));

  size_t i = 0;
  for (auto& p : in_points) {
    auto& op = out_points[i++];
    double ns = norm_simple<CoordinateType>(p);
    op << p, ns, 0;
  }

  assert (in_points.size() == out_points.size());
  assert ((in_points[0].rows() + 2) == out_points[0].rows());
  assert (in_points[0].cols() == out_points[0].cols());
}

// Simple-ALSH: query extend: [x] => [x;    0;    sqrt(1 - ||x||**2)]
template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void simple_extend_for_single_query(Point& in_point, Point& out_point) {
  size_t new_rows = in_point.rows() + 2;
  size_t new_cols = in_point.cols();
  out_point.resize(new_rows, new_cols);

  out_point << in_point, 0, norm_simple<CoordinateType>(in_point);

  assert ((in_point.rows() + 2) == out_point.rows());
  assert (in_point.cols() == out_point.cols());
}

// Simple-ALSH: query extend: [x] => [x;    0;    sqrt(1 - ||x||**2)]
template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void simple_extend_for_query(vector<Point>& in_points, vector<Point>& out_points) {
  if (in_points.empty()) {
    return;
  }

  out_points.clear();
  out_points.reserve(in_points.size());

  size_t i = 0;
  for (auto& p : in_points) {
    out_points.push_back(p);
    simple_extend_for_single_query<CoordinateType>(p, out_points[i++]);
  }

  assert (in_points.size() == i);
  assert (in_points.size() == out_points.size());
}

// Sign-ALSH: build data exend: [x] => [x;    1/2 - ||x||**2; 1/2 - ||x||**4; ...; 1/2 - ||x||**(2*m);    0; ...; 0(m)]
template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void sign_extend_for_index(const size_t m, vector<Point>& in_points, vector<Point>& out_points) {
  if (in_points.empty()) {
    return;
  }

  out_points.clear();

  size_t new_rows = in_points[0].rows() + 2 * m;
  size_t new_cols = in_points[0].cols();
  out_points.assign(in_points.size(), Point(new_rows, new_cols));

  size_t i = 0;
  for (auto& p : in_points) {
    auto& op = out_points[i++];

    vector<CoordinateType> ext_ns;
    extend_norm_sign<CoordinateType>(m, p, ext_ns);
    assert (m = ext_ns.size());

    Point p_ens(m);
    for (size_t i = 0; i < ext_ns.size(); ++i) {
      p_ens[i] = ext_ns[i];
    }

    Point p_zero(m);
    for (size_t i = 0; i < (size_t)m; ++i) {
      p_zero[i] = 0;
    }

    op << p, p_ens, p_zero;
  }

  assert (in_points.size() == out_points.size());
  assert ((in_points[0].rows() + 2 * m) == out_points[0].rows());
  assert (in_points[0].cols() == out_points[0].cols());
}

// Sign-ALSH: query extend: [x] => [x;    0; ...; 0(m);    1/2 - ||x||**2; 1/2 - ||x||**4; ...; 1/2 - ||x||**(2*m)]
template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void sign_extend_for_single_query(const size_t m, Point& in_point, Point& out_point) {
  vector<CoordinateType> ext_ns;
  extend_norm_sign<CoordinateType>(m, in_point, ext_ns);
  assert (m = ext_ns.size());

  size_t new_rows = in_point.rows() + 2 * m;
  size_t new_cols = in_point.cols();
  out_point.resize(new_rows, new_cols);

  Point p_ens(m);
  for (size_t i = 0; i < ext_ns.size(); ++i) {
    p_ens[i] = ext_ns[i];
  }

  Point p_zero(m);
  for (size_t i = 0; i < (size_t)m; ++i) {
    p_zero[i] = 0;
  }

  out_point << in_point, p_zero, p_ens;

  assert ((in_point.rows() + 2 * m) == out_point.rows());
  assert (in_point.cols() == out_point.cols());
}

// Sign-ALSH: query extend: [x] => [x;    0; ...; 0(m);    1/2 - ||x||**2; 1/2 - ||x||**4; ...; 1/2 - ||x||**(2*m)]
template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void sign_extend_for_query(const size_t m, vector<Point>& in_points, vector<Point>& out_points) {
  if (in_points.empty()) {
    return;
  }

  out_points.clear();
  out_points.reserve(in_points.size());

  size_t i = 0;
  for (auto& p : in_points) {
   out_points.push_back(p);
   sign_extend_for_single_query<CoordinateType>(m, p, out_points[i++]);
  }

  assert (i == out_points.size());
  assert (in_points.size() == out_points.size());
}




}
}

namespace falconn {

typedef falconn::wrapper::NORM_TYPE NORM_TYPE;

template <typename CoordinateType>
class DenseMips {
 public:
  typedef CoordinateType value_type;
  typedef DenseVector<value_type> Point;

 DenseMips(const LSHConstructionParameters& paras, vector<Point>& points,
    NORM_TYPE norm_type = NORM_TYPE::SIMPLE, size_t m = 3, double U = 0.83) {
    _wrapper.reset(std::move(falconn::wrapper::create_dense_wrapper<value_type>(paras, points, norm_type, m, U)));
  }

  void set_queries(vector<Point>& queries) {
    _wrapper->set_queries(queries);
  }

 int32_t find_mips_point(const Point& q) {
    return _wrapper->find_mips_point(q);
  }

  void find_k_nearest_mips_points(const Point& query, uint32_t k,
      std::vector<int32_t>* result) {
    _wrapper->find_k_nearest_mips_points(query, k, result);
  }

  void find_near_mips_points(const Point& q,
      typename PointTypeTraits<Point>::ScalarType threshold,
      std::vector<int32_t>* result) {
   _wrapper->find_near_mips_points(q, threshold, result);
  }

  void set_debug_file(const char* file = NULL) {
    string sf;
    if (NULL == file) {
      sf = "mips_log.txt";
    } else {
      sf = file;
    }

    _wrapper->set_debug_file(sf);
  }

  void norm_action() {
    _wrapper->norm_action();
  }

  double get_avg_time() const {
    return _wrapper->get_avg_time();
  }

 private:
  std::unique_ptr<falconn::wrapper::DenseMipsWrapper<value_type> > _wrapper = nullptr;

  private:
   DenseMips(const DenseMips&) = delete;
   DenseMips& operator=(const DenseMips&) = delete;
};

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void self_normalization(vector<Point>& in_points, vector<Point>& out_points,
    bool multiple_factor = false, const double U = 0.83) {
  falconn::wrapper::self_normalization<CoordinateType, Point>(in_points, out_points, multiple_factor, U);
}

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void max_normalization(vector<Point>& in_points, vector<Point>& out_points,
    bool multiple_factor = false, const double U = 0.83) {
  falconn::wrapper::max_normalization<CoordinateType, Point>(in_points, out_points, multiple_factor, U);
}

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void extend_half(const size_t m, vector<CoordinateType>& ext_half) {
  falconn::wrapper::extend_half<CoordinateType, Point>(m, ext_half);
}

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void extend_norm_sign(const size_t m, Point& point, vector<CoordinateType>& ext_ns) {
  falconn::wrapper::extend_norm_sign<CoordinateType, Point>(m, point, ext_ns);
}

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
double norm_simple(Point& point) {
  return falconn::wrapper::norm_simple<CoordinateType, Point>(point);
}

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void simple_extend_for_index(vector<Point>& in_points, vector<Point>& out_points) {
  return falconn::wrapper::simple_extend_for_index<CoordinateType, Point>(in_points, out_points);
}

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void simple_extend_for_single_query(Point& in_point, Point& out_point) {
  falconn::wrapper::simple_extend_for_single_query<CoordinateType, Point>(in_point, out_point);
}

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void simple_extend_for_query(vector<Point>& in_points, vector<Point>& out_points) {
  falconn::wrapper::simple_extend_for_query<CoordinateType, Point>(in_points, out_points);
}

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void sign_extend_for_index(const size_t m, vector<Point>& in_points, vector<Point>& out_points) {
  falconn::wrapper::sign_extend_for_index<CoordinateType, Point>(m, in_points, out_points);
}

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void sign_extend_for_single_query(const size_t m, Point& in_point, Point& out_point) {
  falconn::wrapper::sign_extend_for_single_query<CoordinateType, Point>(m, in_point, out_point);
}

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void sign_extend_for_query(const size_t m, vector<Point>& in_points, vector<Point>& out_points) {
  falconn::wrapper::sign_extend_for_query<CoordinateType, Point>(m, in_points, out_points);
}

// MIPS Parameter Prober for Dense Datas

// TODO: Pls. do not check the return value for any member function of DenseMipsParaProber by using assert
// TODO: Because it might can not call the member without any compiling error

template <typename CoordinateType>
class DenseMipsParaProber {
 public:
  typedef CoordinateType value_type;
  typedef DenseVector<value_type> Point;

 public:
  DenseMipsParaProber(const uint32_t dim, const uint32_t data_num = 1000,
      const uint32_t query_num = 100, const uint32_t min = 0, const uint32_t max = 100)
      : _dim(dim), _data_num(data_num), _query_num(query_num), _min(min), _max(max) {
    assert ((0 < _dim) && (10000 > _dim));
    assert (0 < _data_num);
    if ((0 == _query_num) || (query_num >= data_num)) {
      _query_num = _data_num / 10;
    }

    if (1 > _query_num) {
      _query_num = 1;
    }
    assert (_query_num < _data_num);

    __gen_datas();
    __gen_queries();
    __gen_answers();
  }

  DenseMipsParaProber(const uint32_t dim,
      vector<Point>* const datas, vector<Point>* const queries)
      : _dim(dim), _data_num(0), _query_num(0) {
    _datas.reset(std::move(datas));
    _data_num = _datas->size();

    _queries.reset(std::move(queries));
    _query_num = _queries->size();

    assert ((0 < _dim) && (10000 > _dim));
    assert (0 < _data_num);
    assert (0 < _query_num);

    __gen_answers();
  }

  ~DenseMipsParaProber() {}

  void set_print_file(const char* file = "dense_mips_tester.txt") {
    _print_file = file;
  }

  // TODO: Pls. do not check the return value for validate_mips function of DenseMipsParaProber by using assert
  // TODO: Because it might can not call the member without any compiling error
  bool validate_mips(LSHConstructionParameters& params,
      falconn::NORM_TYPE type = falconn::NORM_TYPE::SIMPLE) {
    _print();

    DenseMips<float> dm(params, *_datas, type);
    // dm.set_debug_file();
    dm.norm_action();
    // dm.set_queries(_queries);

    size_t r_cnt = 0;
    size_t index = 0;
    for (auto& q : *_queries) {
      int q_index = dm.find_mips_point(q);
      int ans = _answers[index][0].first;
      if (q_index != ans) {
        cout << "validate fail: " << index << ", ret(" << q_index << "), ans(" << ans << ")" << endl;
        return false;
      }

      ++r_cnt;
      ++index;
    }

    if (_queries->size() == r_cnt) {
      return true;
    }

    return true;
  }

  double evaluate_mips_accuracy(LSHConstructionParameters& params,
      falconn::NORM_TYPE type = falconn::NORM_TYPE::SIMPLE) {
    _print();

    DenseMips<float> dm(params, *_datas, type);
    // dm.set_debug_file();
    dm.norm_action();
    // dm.set_queries(_queries);

    return __evaluate_accuracy(dm);
  }

  // TODO: Pls. do not check the return value for evaluate_optimum_paras function of DenseMipsParaProber by using assert
  // TODO: Because it might can not call the member without any compiling error
  bool evaluate_optimum_paras(AccuracySpeed& as, LSHConstructionParameters& params,
      falconn::NORM_TYPE type = falconn::NORM_TYPE::SIMPLE, const uint32_t multiple = 10,
      const double min_accuracy = 0.9) {
    assert (_datas->size() == _data_num);
    assert (_queries->size() == _query_num);

    uint32_t i_k = 0;
    if (300 >_data_num) {
      i_k = below100IndexSet;
    } else if (3000 > _data_num) {
      i_k = below1KIndexSet;
    } else {
      i_k = beyond10KIndexSet;
    }
    uint32_t i_L = i_k;

    #if 0
    cout << "data_num: " << _data_num << endl;
    cout << "query_num: " << _query_num << endl;
    cout << "\t i_k: " << i_k << endl;
    cout << "\t i_L: " << i_L << endl;
    #endif

    if (!__evaluate_good_as(i_k, i_L, params, type, min_accuracy)) {
      return false;
    }

    // select k & L probe range
    assert (LSetLength() > i_L);
    assert (kSetLength() > (i_k + 1));
    if (LSetLength() == (i_L + 1)) {
      _predict_as.push_back(AccuracySpeed(kSet[i_k], (1.5 * LSet[i_L])));
    } else {
      uint32_t min_k = kSet[i_k];
      uint32_t max_k = 0;
      uint32_t min_L = LSet[i_L];
      uint32_t max_L = 0;
      if (10 >= multiple) {
        max_k = kSet[i_k + 1];
        max_L = LSet[i_L + 1];
      } else {
        // max_k will <= 20
        max_k = (kSetLength() > (i_k + 2)) ? kSet[i_k + 2] : (kSet[i_k] * 1.15);
        max_L = (LSetLength() > (i_L + 2)) ? LSet[i_L + 2] : (LSet[i_L] * 1.5);
      }

      __get_radnom_as(min_k, max_k, min_L, max_L);
    }
    assert (evaluate_num * 3 > _predict_as.size());

    __evaluate_random_as(params, type, min_accuracy);

    assert (0 < _qualified_as.size());
    for (auto& qas : _qualified_as) {
      assert ((0.0 < qas._time) && (0.0 < qas._accuracy));
      (void)qas;
    }

    sort(_qualified_as.begin(), _qualified_as.end(), as_time_comp);
    sort(_qualified_as.begin(), (_qualified_as.begin() + 3), as_accuracy_comp);

    as = _qualified_as[0];

    return true;
  }

  void print_para() {
    if (_print_file.empty()) {
      return;
    }

    FILE* fp = fopen(_print_file.c_str(), "a");
    if (NULL == fp) {
      string throw_info = string("can't open the file: ") + _print_file;
      throw runtime_error(throw_info);
      return;
    }

    fprintf(fp, "\n DenseMipsParaProber:\n");
    fprintf(fp, "\t dim: %d, dnum: %d, qnum: %d, range(%d, %d)\n",
        _dim, _data_num, _query_num, _min, _max);

    fclose(fp);
  }

 private:
  void __evaluate_as(AccuracySpeed& as, LSHConstructionParameters& params, falconn::NORM_TYPE type) {
    LSHConstructionParameters t_para;
    memcpy(&t_para, &params, sizeof(LSHConstructionParameters));
    t_para.k = as._k;
    t_para.l = as._L;

    double time = 0.0;
    double accuracy = 0.0;
    __get_mips_paras(time, accuracy, t_para, type);
    as._time = time;
    as._accuracy = accuracy;
  }

  bool __evaluate_good_as(uint32_t& i_k, uint32_t& i_L,
      LSHConstructionParameters& params, falconn::NORM_TYPE type,
      const double min_accuracy){
    assert ((kSetLength() > i_k) && (LSetLength() > i_L));

    AccuracySpeed as_predict(kSet[i_k], LSet[i_L]);
    if (__evaluate_single_as(as_predict, params, type, min_accuracy)) {
      return true;
    }

    // un-qualified right move L and back move k for k_back_step
    assert (i_L >= i_k);
    if (i_L > i_k) {
      ++i_k;
    }
    ++i_L;

    // beyond the border. evaluate fail
    if ((kSetLength() == i_k) || (LSetLength() == i_L)) {
      return false;
    }

    return __evaluate_good_as(i_k, i_L, params, type, min_accuracy);
  }

  bool __evaluate_random_as(LSHConstructionParameters& params, falconn::NORM_TYPE type,
      const double min_accuracy) {
    for (auto& as_predict : _predict_as) {
      __evaluate_single_as(as_predict, params, type, min_accuracy);
    }

    return true;
  }

  bool __evaluate_single_as(AccuracySpeed& as_predict, LSHConstructionParameters& params,
      falconn::NORM_TYPE type, const double min_accuracy) {
    __evaluate_as(as_predict, params, type);

    if (min_accuracy <= as_predict._accuracy) {
      _qualified_as.push_back(as_predict);
      return true;
    }

    return false;
  }

  void __get_radnom_as(uint32_t min_k, uint32_t max_k, uint32_t min_L , uint32_t max_L ) {
    if (min_k > max_k) {
      min_k ^= max_k;
      max_k ^= min_k;
      min_k ^= max_k;
    }

    if (min_L > max_L ) {
      min_L ^= max_L;
      max_L ^= min_L;
      min_L ^= max_L;
    }

    random_device rd;
    mt19937_64 gen(rd());
    uniform_int_distribution<> rand_k(min_k + 1, max_k);
    uniform_int_distribution<> rand_L(min_L + 1, max_L );

    vector<uint32_t> ks(evaluate_num);
    vector<uint32_t> Ls(evaluate_num);
    for (auto i = 0; i < evaluate_num; ++i) {
      ks[i] = rand_k(gen);
      Ls[i] = rand_L(gen);
    }

    sort(ks.begin(), ks.end());
    sort(Ls.begin(), Ls.end());

    // push five predict as: min /|/|/ max
    _predict_as.push_back(AccuracySpeed(min_k, Ls[0]));
    _predict_as.push_back(AccuracySpeed(ks[0], Ls[0]));
    _predict_as.push_back(AccuracySpeed(ks[0], Ls[1]));
    _predict_as.push_back(AccuracySpeed(ks[1], Ls[1]));
    _predict_as.push_back(AccuracySpeed(ks[1], max_L));
  }

  void __gen_datas() {
    _datas.reset(std::move(new vector<Point>()));

    _datas->clear();
    _datas->assign(_data_num, Point(_dim));

    random_device rd;
    mt19937_64 gen(rd());
    uniform_int_distribution<> u(_min, _max - 1);
    for (auto i = 0; i < _data_num; ++i) {
      for (auto j = 0; j < _dim; ++j) {
        const int ind = u(gen);
        assert ((_min <= ind) && (_max > ind));
        (*_datas)[i][j] = ind;
      }
    }
  }

  void __gen_queries() {
    _queries.reset(std::move(new vector<Point>()));

    _queries->clear();
    _queries->assign(_query_num, Point(_dim));

    random_device rd;
    mt19937_64 gen(rd());
    uniform_int_distribution<> u(_min, _max - 1);
    for (auto i = 0; i < _query_num; ++i) {
      for (auto j = 0; j < _dim; ++j) {
        const int ind = u(gen);
        assert ((_min <= ind) && (_max > ind));
        (*_queries)[i][j] = ind;
      }
    }
  }

  void __gen_answers() {
    assert (0 < _queries->size());
    _answers.clear();
    _answers.assign(_queries->size(), vector<pair<int, float> >());

    for (auto& answer : _answers) {
      answer.assign(_datas->size(), pair<int, float>(0, 0.0));
    }

    int outer_counter = 0;
    for (const auto& query : *_queries) {
      float best = -10.0;
      int inner_counter = 0;
      for (const auto& datapoint : *_datas) {
        // score for vector innert product sum: dot
        float score = query.dot(datapoint);
          _answers[outer_counter][inner_counter].first = inner_counter;
          _answers[outer_counter][inner_counter].second = score;
        if (score > best) {
          best = score;
        }
        ++inner_counter;
      }
      sort(_answers[outer_counter].begin(), _answers[outer_counter].end(), int_comp_greater);
      assert (_answers[outer_counter][0].second == best);
      ++outer_counter;
    }
  }

  double __evaluate_accuracy(DenseMips<value_type>& dm) {
    int outer_counter = 0;
    int num_matches = 0;
    int32_t candidates;
    for (const auto& query : *_queries) {
      candidates = dm.find_mips_point(query);
      const int ans = _answers[outer_counter][0].first;
      if (ans == candidates) {
        ++num_matches;
      }
      ++outer_counter;
    }
    return (num_matches + 0.0) / (_queries->size() + 0.0);
  }

  void __get_mips_paras(double& time, double& accuracy, LSHConstructionParameters& params,
      falconn::NORM_TYPE type = falconn::NORM_TYPE::SIMPLE) {
    DenseMips<float> dm(params, *_datas, type);
    dm.norm_action();

    accuracy = __evaluate_accuracy(dm);
    time = dm.get_avg_time();
  }

  void _print_point(FILE* fp, vector<Point>& points, const string& explain) {
    assert (NULL != fp);
    assert (0 < points.size());

    fprintf(fp, "\n%s:\n", explain.c_str());

    size_t seq = 0;
    size_t rows = points[0].rows();
    for (auto& p : points) {
      fprintf(fp, "[%zu]:\t", seq++);
      for (auto i = 0; i < rows; ++i) {
        fprintf(fp, "%lf, ", p[i]);
      }
      fprintf(fp, "\n");
    }
    fprintf(fp, "============================================================\n");
  }

  void _print_point(FILE* fp, vector<vector<pair<int, float> > >& answers, const string& explain) {
    assert (NULL != fp);
    assert (0 < answers.size());

    fprintf(fp, "\n%s:\n", explain.c_str());

    for (auto& p : answers) {
      for (auto& a : p) {
        fprintf(fp, "%d(%f), ", a.first, a.second);
      }
      fprintf(fp, "\n");
    }
    fprintf(fp, "============================================================\n");
  }

  void _print() {
    if (_print_file.empty()) {
      return;
    }

    FILE* fp = fopen(_print_file.c_str(), "a");
    if (NULL == fp) {
      string throw_info = string("can't open the file: ") + _print_file;
      throw runtime_error(throw_info);
      return;
    }

    fprintf(fp, "\n DenseMipsParaProber:\n");
    fprintf(fp, "\t dim: %d, dnum: %d, qnum: %d, range(%d, %d)\n",
        _dim, _data_num, _query_num, _min, _max);

    _print_point(fp, *_datas, "_datas");
    _print_point(fp, *_queries, "_queries");
    _print_point(fp, _answers, "_answers");

    fclose(fp);
  }

 private:
  uint32_t _dim;
  uint32_t _data_num;
  uint32_t _query_num;
  uint32_t _min;
  uint32_t _max;

  string _print_file;

  std::unique_ptr<vector<Point> > _datas;
  std::unique_ptr<vector<Point> > _queries;
  vector<vector<pair<int, float> > > _answers;

  vector<AccuracySpeed> _qualified_as;
  vector<AccuracySpeed> _predict_as;

 private:
  DenseMipsParaProber(const DenseMipsParaProber&) = delete;
  DenseMipsParaProber& operator=(const DenseMipsParaProber&) = delete;
};

}

#endif
