#include "falconn/lsh_nn_table.h"

#include <memory>
#include <utility>
#include <vector>
#include <iostream>

#include <algorithm>
#include <chrono>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdio>

#include "gtest/gtest.h"

using std::cin;
using std::cout;
using std::cerr;
using std::ends;
using std::endl;
using std::exception;
using std::max;
using std::mt19937_64;
using std::runtime_error;
using std::uniform_int_distribution;
using std::random_device;

using std::string;
using std::vector;
using std::pair;
using std::make_pair;
using std::unique_ptr;

using std::sort;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

using falconn::compute_number_of_hash_functions;
using falconn::construct_table;
using falconn::DenseVector;
using falconn::DistanceFunction;
using falconn::LSHConstructionParameters;
using falconn::LSHFamily;
using falconn::LSHNearestNeighborTable;
using falconn::get_default_parameters;
using falconn::SparseVector;
using falconn::StorageHashTable;

using falconn::DenseMips;
using falconn::DenseMipsParaProber;
using falconn::AccuracySpeed;
using falconn::self_normalization;
using falconn::max_normalization;
using falconn::extend_half;
using falconn::extend_norm_sign;
using falconn::norm_simple;
using falconn::simple_extend_for_index;
using falconn::simple_extend_for_single_query;
using falconn::simple_extend_for_query;
using falconn::sign_extend_for_index;
using falconn::sign_extend_for_single_query;
using falconn::sign_extend_for_query;

static bool print = false;

template <typename CoordinateType>
void print_points(vector<CoordinateType>& points, const string& explain) {
  cout << explain.c_str() << ": \t" << endl;
  for (auto& p : points) {
    cout << p << ", ";
  }
  cout << endl;
}

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
void print_points(vector<Point>& points, int dim, const string& explain) {
  cout << explain.c_str() << ": \t" << endl;
  for (auto& p : points) {
    for (int i = 0; i < dim; ++i) {
      cout << p[i] << ", ";
    }
    cout << endl;
  }
}

template <typename Point>
void print_points(Point& points, const string& explain) {
  cout << explain.c_str() << ": \t" << endl;
  for (auto i = 0; i < points.rows(); ++i) {
    cout << points[i] << ", ";
  }
  cout << endl;
}

TEST(AlshForMispTest, k_L_length) {
  EXPECT_EQ(6, falconn::kSetLength());
  EXPECT_EQ(6, falconn::LSetLength());
}

TEST(AlshForMispTest, self_normalization_test_1) {
  typedef DenseVector<float> Point;
  int dim = 4;
  const float multiple = 3.0;

  Point p1(dim);
  p1[0] = 0.8;
  p1[1] = 0.0;
  p1[2] = 0.6;
  p1[3] = 0.0;

  Point p2(dim);
  p2[0] = 0.0;
  p2[1] = 0.8;
  p2[2] = 0.0;
  p2[3] = 0.6;

  vector<Point> valid_points;
  valid_points.push_back(p1);
  valid_points.push_back(p2);

  p1 *= multiple;
  p2 *= multiple;

  vector<Point> in_points;
  in_points.push_back(p1);
  in_points.push_back(p2);

  vector<Point> out_points;
  falconn::self_normalization<float>(in_points, out_points);

  if (print) {
    print_points<float>(out_points, dim, "outs");
    print_points<float>(in_points, dim, "ins");
    print_points<float>(valid_points, dim, "valids");
  }

  float eps = 0.00001;
  EXPECT_EQ(out_points.size(), in_points.size());
  EXPECT_EQ(out_points.size(), valid_points.size());
  EXPECT_EQ(out_points[0].rows(), valid_points[0].rows());
  EXPECT_EQ(out_points[0].cols(), valid_points[0].cols());

  size_t i = 0;
  for (auto& p : out_points) {
    auto& vp = valid_points[i];
    auto& ip = in_points[i];

    for (int j = 0; j < dim; ++j) {
      EXPECT_NEAR(p[j], vp[j], eps);
      EXPECT_NEAR((p[j] * multiple), ip[j], eps);
    }

    ++i;
  }
}

TEST(AlshForMispTest, self_normalization_test_2) {
  typedef DenseVector<float> Point;
  int dim = 4;
  const float multiple = 3.0;

  double U = 0.83;

  Point p1(dim);
  p1[0] = 0.8;
  p1[1] = 0.0;
  p1[2] = 0.6;
  p1[3] = 0.0;

  Point p2(dim);
  p2[0] = 0.0;
  p2[1] = 0.8;
  p2[2] = 0.0;
  p2[3] = 0.6;

  vector<Point> valid_points;
  valid_points.push_back(p1);
  valid_points.push_back(p2);

  p1 *= multiple;
  p2 *= multiple;

  vector<Point> in_points;
  in_points.push_back(p1);
  in_points.push_back(p2);

  vector<Point> out_points;
  falconn::self_normalization<float>(in_points, out_points, true, U);

  if (print) {
    print_points<float>(out_points, dim, "outs");
    print_points<float>(in_points, dim, "ins");
    print_points<float>(valid_points, dim, "valids");
  }

  float eps = 0.00001;
  EXPECT_EQ(out_points.size(), in_points.size());
  EXPECT_EQ(out_points.size(), valid_points.size());
  EXPECT_EQ(out_points[0].rows(), valid_points[0].rows());
  EXPECT_EQ(out_points[0].cols(), valid_points[0].cols());
  size_t i = 0;
  for (auto& p : out_points) {
    auto& vp = valid_points[i];
    auto& ip = in_points[i];

    for (int j = 0; j < dim; ++j) {
      EXPECT_NEAR(p[j], (U * vp[j]), eps);
      EXPECT_NEAR((p[j] * multiple), (U * ip[j]), eps);
    }

    ++i;
  }
}

TEST(AlshForMispTest, max_normalization_test_1) {
  typedef DenseVector<float> Point;
  int dim = 4;

  Point p1(dim);
  p1[0] = 0.8;
  p1[1] = 0.0;
  p1[2] = 0.6;
  p1[3] = 0.0;

  Point p2(dim);
  p2[0] = 2.4;
  p2[1] = 0.0;
  p2[2] = 1.8;
  p2[3] = 0.0;

  vector<Point> in_points;
  in_points.push_back(p1);
  in_points.push_back(p2);

  float max_norm = 0.0;
  for (auto& p : in_points) {
    float norm = p.norm();
    if (max_norm < norm) {
      max_norm = norm;
    }
  }

  vector<Point> valid_points;
  valid_points.push_back(p1 / max_norm);
  valid_points.push_back(p2 / max_norm);

  vector<Point> out_points;
  falconn::max_normalization<float>(in_points, out_points);

  if (print) {
    print_points<float>(out_points, dim, "outs");
    print_points<float>(in_points, dim, "ins");
    print_points<float>(valid_points, dim, "valids");
  }

  float eps = 0.00001;
  EXPECT_EQ(out_points.size(), in_points.size());
  EXPECT_EQ(out_points.size(), valid_points.size());
  EXPECT_EQ(out_points[0].rows(), valid_points[0].rows());
  EXPECT_EQ(out_points[0].cols(), valid_points[0].cols());
  size_t i = 0;
  for (auto& p : out_points) {
    auto& vp = valid_points[i];
    auto& ip = in_points[i];

    for (int j = 0; j < dim; ++j) {
      EXPECT_NEAR(p[j], vp[j], eps);
      EXPECT_NEAR((max_norm * p[j]), ip[j], eps);
    }

    ++i;
  }
}

TEST(AlshForMispTest, max_normalization_test_2) {
  typedef DenseVector<float> Point;
  int dim = 4;

  double U = 0.83;

  Point p1(dim);
  p1[0] = 0.8;
  p1[1] = 0.0;
  p1[2] = 0.6;
  p1[3] = 0.0;

  Point p2(dim);
  p2[0] = 2.4;
  p2[1] = 0.0;
  p2[2] = 1.8;
  p2[3] = 0.0;

  vector<Point> in_points;
  in_points.push_back(p1);
  in_points.push_back(p2);

  float max_norm = 0.0;
  for (auto& p : in_points) {
    float norm = p.norm();
    if (max_norm < norm) {
      max_norm = norm;
    }
  }

  vector<Point> valid_points;
  valid_points.push_back(p1 / max_norm);
  valid_points.push_back(p2 / max_norm);

  vector<Point> out_points;
  falconn::max_normalization<float>(in_points, out_points, true, U);

  if (print) {
    print_points<float>(out_points, dim, "outs");
    print_points<float>(in_points, dim, "ins");
    print_points<float>(valid_points, dim, "valids");
  }

  float eps = 0.00001;
  EXPECT_EQ(out_points.size(), in_points.size());
  EXPECT_EQ(out_points.size(), valid_points.size());
  EXPECT_EQ(out_points[0].rows(), valid_points[0].rows());
  EXPECT_EQ(out_points[0].cols(), valid_points[0].cols());
  size_t i = 0;
  for (auto& p : out_points) {
    auto& vp = valid_points[i];
    auto& ip = in_points[i];

    for (int j = 0; j < dim; ++j) {
      EXPECT_NEAR(p[j], (U * vp[j]), eps);
      EXPECT_NEAR((max_norm * p[j]), (U * ip[j]), eps);
    }

    ++i;
  }
}

TEST(AlshForMispTest, extend_half_test) {
  int dim = 4;

  vector<float> out_points;
  falconn::extend_half<float>(dim, out_points);

  if (print) {
    print_points<float>(out_points, "outs");
  }

  float eps = 0.00001;
  EXPECT_EQ(dim, out_points.size());

  for (auto& p : out_points) {
    EXPECT_NEAR(0.5, p, eps);
  }
}

TEST(AlshForMispTest, extend_norm_sign_test) {
  typedef DenseVector<float> Point;
  int dim = 4;

  Point p1(dim);
  p1[0] = 0.7;
  p1[1] = 0.3;
  p1[2] = 0.6;
  p1[3] = 0.4;

  auto squared_norm = pow(p1.norm(), 2);
  Point p2(dim);
  p2[0] = squared_norm;
  p2[1] = pow(squared_norm, 2);
  p2[2] = pow(squared_norm, 3);
  p2[3] = pow(squared_norm, 4);

  vector<float> out_points;
  falconn::extend_norm_sign<float>(dim, p1, out_points);

  if (print) {
    print_points<float>(out_points, "outs");
  }

  float eps = 0.00001;
  EXPECT_EQ(dim, out_points.size());

  size_t i = 0;
  for (auto& p : out_points) {
    EXPECT_NEAR((p + p2[i++]), 0.5, eps);
  }
}

TEST(AlshForMispTest, norm_simple_test) {
  typedef DenseVector<float> Point;
  int dim = 4;

  Point p1(dim);
  p1[0] = 0.2;
  p1[1] = 0.3;
  p1[2] = 0.5;
  p1[3] = 0.4;

  double squared_norm = pow(p1.norm(), 2);
  double norm_sp = sqrt(1 - squared_norm);

  double ns = falconn::norm_simple<float>(p1);

  if (print) {
    cout << "norm_simple: " << norm_sp << " , " << ns << endl;
  }

  float eps = 0.00001;
  EXPECT_NEAR(ns, norm_sp, eps);
}

TEST(AlshForMispTest, simple_extend_for_index_test) {
  typedef DenseVector<float> Point;
  int dim = 4;
  int out_dim = 6;

  Point p1(dim);
  p1[0] = 0.7;
  p1[1] = 0.3;
  p1[2] = 0.3;
  p1[3] = 0.4;

  Point p2(dim);
  p2[0] = 0.3;
  p2[1] = 0.5;
  p2[2] = 0.2;
  p2[3] = 0.6;

  vector<Point> in_points;
  in_points.push_back(p1);
  in_points.push_back(p2);

  vector<Point> valid_points;
  for (auto& p : in_points) {
    double squared_norm = pow(p.norm(), 2);
    assert (1 > squared_norm);
    double ns = sqrt(1 - squared_norm);

    Point ptmp(dim);

    size_t new_rows = ptmp.rows() + 2;
    size_t new_cols = ptmp.cols();
    ptmp.resize(new_rows, new_cols);

    ptmp << p, ns, 0.0;
    valid_points.push_back(ptmp);
  }

  vector<Point> out_points;
  falconn::simple_extend_for_index<float>(in_points, out_points);

  if (print) {
    EXPECT_EQ(out_dim, out_points[0].rows());
    EXPECT_EQ(dim, in_points[0].rows());
    EXPECT_EQ(out_dim, valid_points[0].rows());

    print_points<float>(out_points, out_dim, "outs");
    print_points<float>(in_points, dim, "ins");
    print_points<float>(valid_points, out_dim, "valids");
  }

  float eps = 0.00001;
  EXPECT_EQ(out_points.size(), in_points.size());
  EXPECT_EQ(out_points.size(), valid_points.size());
  EXPECT_EQ(out_points[0].rows(), (in_points[0].rows() + 2));
  EXPECT_EQ(out_points[0].cols(), in_points[0].cols());
  EXPECT_EQ(out_points[0].rows(), valid_points[0].rows());
  EXPECT_EQ(out_points[0].cols(), valid_points[0].cols());

  size_t i = 0;
  for (auto& p : out_points) {
    auto& vp = valid_points[i];
    auto& ip = in_points[i];

    for (int j = 0; j < p.rows(); ++j) {
      EXPECT_NEAR(p[j], vp[j], eps);
      if (j < ip.rows()) {
        EXPECT_NEAR(p[j], ip[j], eps);
      }
    }

    ++i;
  }
}

TEST(AlshForMispTest, simple_extend_for_single_query_test) {
  typedef DenseVector<float> Point;
  int dim = 4;
  int out_dim = 6;

  Point p1(dim);
  p1[0] = 0.8;
  p1[1] = 0.2;
  p1[2] = 0.3;
  p1[3] = 0.1;

  double squared_norm = pow(p1.norm(), 2);
  assert (1 > squared_norm);
  double ns = sqrt(1 - squared_norm);

  Point valid_p(dim);
  size_t new_rows = p1.rows() + 2;
  size_t new_cols = p1.cols();
  valid_p.resize(new_rows, new_cols);

  valid_p << p1, 0.0, ns;

  Point out_p;
  falconn::simple_extend_for_single_query<float>(p1, out_p);

  if (print) {
    EXPECT_EQ(out_dim, out_p.rows());
    EXPECT_EQ(dim, p1.rows());
    EXPECT_EQ(out_dim, valid_p.rows());

    print_points<Point>(out_p, "outs");
    print_points<Point>(p1, "ins");
    print_points<Point>(valid_p, "valids");

    #if 0
    cout << "outs: " << out_p << endl;
    cout << "ins: " << p1 << endl;
    cout << "valids: " << valid_p << endl;
    #endif
  }

  float eps = 0.00001;
  for (auto i = 0; i < out_p.rows(); ++i) {
    EXPECT_NEAR(out_p[i], valid_p[i], eps);

    if (p1.rows() > i ) {
      EXPECT_NEAR(out_p[i], p1[i], eps);
    }
  }
}

TEST(AlshForMispTest, simple_extend_for_query_test) {
  typedef DenseVector<float> Point;
  int dim = 4;
  int out_dim = 6;

  Point p1(dim);
  p1[0] = 0.9;
  p1[1] = 0.2;
  p1[2] = 0.1;
  p1[3] = 0.1;

  Point p2(dim);
  p2[0] = 0.5;
  p2[1] = 0.3;
  p2[2] = 0.2;
  p2[3] = 0.5;

  vector<Point> in_points;
  in_points.push_back(p1);
  in_points.push_back(p2);

  vector<Point> valid_points;
  for (auto& p : in_points) {
    double squared_norm = pow(p.norm(), 2);
    assert (1 > squared_norm);
    double ns = sqrt(1 - squared_norm);

    Point ptmp(dim);

    size_t new_rows = ptmp.rows() + 2;
    size_t new_cols = ptmp.cols();
    ptmp.resize(new_rows, new_cols);

    ptmp << p, 0.0, ns;
    valid_points.push_back(ptmp);
  }

  vector<Point> out_points;
  falconn::simple_extend_for_query<float>(in_points, out_points);

  if (print) {
    EXPECT_EQ(out_dim, out_points[0].rows());
    EXPECT_EQ(dim, in_points[0].rows());
    EXPECT_EQ(out_dim, valid_points[0].rows());

    print_points<float>(out_points, out_dim, "outs");
    print_points<float>(in_points, dim, "ins");
    print_points<float>(valid_points, out_dim, "valids");
  }

  float eps = 0.00001;
  EXPECT_EQ(out_points.size(), in_points.size());
  EXPECT_EQ(out_points.size(), valid_points.size());
  EXPECT_EQ(out_points[0].rows(), (in_points[0].rows() + 2));
  EXPECT_EQ(out_points[0].cols(), in_points[0].cols());
  EXPECT_EQ(out_points[0].rows(), valid_points[0].rows());
  EXPECT_EQ(out_points[0].cols(), valid_points[0].cols());

  size_t i = 0;
  for (auto& p : out_points) {
    auto& vp = valid_points[i];
    auto& ip = in_points[i];

    for (int j = 0; j < p.rows(); ++j) {
      EXPECT_NEAR(p[j], vp[j], eps);
      if (j < ip.rows()) {
        EXPECT_NEAR(p[j], ip[j], eps);
      }
    }

    ++i;
  }
}

TEST(AlshForMispTest, sign_extend_for_index) {
  typedef DenseVector<float> Point;
  int dim = 4;
  int m = 3;
  int out_dim = dim + 2 * m;

  Point p1(dim);
  p1[0] = 0.7;
  p1[1] = 0.3;
  p1[2] = 0.3;
  p1[3] = 0.4;

  Point p2(dim);
  p2[0] = 0.3;
  p2[1] = 0.5;
  p2[2] = 0.2;
  p2[3] = 0.6;

  vector<Point> in_points;
  in_points.push_back(p1);
  in_points.push_back(p2);

  vector<Point> valid_points;
  for (auto& p : in_points) {
    vector<float> ext_ns;
    extend_norm_sign<float>(m, p, ext_ns);
    assert (m = ext_ns.size());

    #if 0
    print_points<float>(ext_ns, "ext_ns");
    #endif

    Point ptmp(dim);

    size_t new_rows = p.rows() + 2 * m;
    size_t new_cols = p.cols();
    ptmp.resize(new_rows, new_cols);

    Point p_ens(m);
    for (size_t i = 0; i < ext_ns.size(); ++i) {
      p_ens[i] = ext_ns[i];
    }

    Point p_zero(m);
    for (size_t i = 0; i < (size_t)m; ++i) {
      p_zero[i] = 0;
    }

    ptmp << p, p_ens, p_zero;

    #if 0
    print_points<Point>(p, "p");
    print_points<Point>(ptmp, "ptmp");
    #endif

    valid_points.push_back(ptmp);
  }

  vector<Point> out_points;
  falconn::sign_extend_for_index<float>(m, in_points, out_points);

  if (print) {
    EXPECT_EQ(out_dim, out_points[0].rows());
    EXPECT_EQ(dim, in_points[0].rows());
    EXPECT_EQ(out_dim, valid_points[0].rows());

    print_points<float>(out_points, out_dim, "outs");
    print_points<float>(in_points, dim, "ins");
    print_points<float>(valid_points, out_dim, "valids");
  }

  float eps = 0.00001;
  EXPECT_EQ(out_points.size(), in_points.size());
  EXPECT_EQ(out_points.size(), valid_points.size());
  EXPECT_EQ(out_points[0].rows(), (in_points[0].rows() + 2 * m));
  EXPECT_EQ(out_points[0].cols(), in_points[0].cols());
  EXPECT_EQ(out_points[0].rows(), valid_points[0].rows());
  EXPECT_EQ(out_points[0].cols(), valid_points[0].cols());

  size_t i = 0;
  for (auto& p : out_points) {
    auto& vp = valid_points[i];
    auto& ip = in_points[i];

    for (int j = 0; j < p.rows(); ++j) {
      EXPECT_NEAR(p[j], vp[j], eps);
      if (j < ip.rows()) {
        EXPECT_NEAR(p[j], ip[j], eps);
      }
    }

    ++i;
  }
}

TEST(AlshForMispTest, sign_extend_for_single_query_test) {
  typedef DenseVector<float> Point;
  int dim = 4;
  int m = 3;
  int out_dim = dim + 2 * m;

  Point p1(dim);
  p1[0] = 0.8;
  p1[1] = 0.2;
  p1[2] = 0.3;
  p1[3] = 0.1;

  vector<float> ext_ns;
  extend_norm_sign<float>(m, p1, ext_ns);
  assert (m = ext_ns.size());

  Point valid_p(dim);
  size_t new_rows = p1.rows() + 2 * m;
  size_t new_cols = p1.cols();
  valid_p.resize(new_rows, new_cols);

  Point p_ens(m);
  for (size_t i = 0; i < ext_ns.size(); ++i) {
    p_ens[i] = ext_ns[i];
  }
  
  Point p_zero(m);
  for (size_t i = 0; i < (size_t)m; ++i) {
    p_zero[i] = 0;
  }
  
  valid_p << p1, p_zero, p_ens;

  Point out_p;
  falconn::sign_extend_for_single_query<float>(m, p1, out_p);

  if (print) {
    EXPECT_EQ(out_dim, out_p.rows());
    EXPECT_EQ(dim, p1.rows());
    EXPECT_EQ(out_dim, valid_p.rows());

    print_points<Point>(out_p, "outs");
    print_points<Point>(p1, "ins");
    print_points<Point>(valid_p, "valids");
  }

  float eps = 0.00001;
  for (auto i = 0; i < out_p.rows(); ++i) {
    EXPECT_NEAR(out_p[i], valid_p[i], eps);

    if (p1.rows() > i ) {
      EXPECT_NEAR(out_p[i], p1[i], eps);
    }
  }
}

TEST(AlshForMispTest, sign_extend_for_query_test) {
  typedef DenseVector<float> Point;
  int dim = 4;
  int m = 3;
  int out_dim = dim + 2 * m;

  Point p1(dim);
  p1[0] = 0.6;
  p1[1] = 0.5;
  p1[2] = 0.2;
  p1[3] = 0.4;

  Point p2(dim);
  p2[0] = 0.1;
  p2[1] = 0.7;
  p2[2] = 0.0;
  p2[3] = 0.2;

  vector<Point> in_points;
  in_points.push_back(p1);
  in_points.push_back(p2);

  vector<Point> valid_points;
  for (auto& p : in_points) {
    vector<float> ext_ns;
    extend_norm_sign<float>(m, p, ext_ns);
    assert (m = ext_ns.size());

    #if 0
    print_points<float>(ext_ns, "ext_ns");
    #endif

    Point ptmp(dim);

    size_t new_rows = p.rows() + 2 * m;
    size_t new_cols = p.cols();
    ptmp.resize(new_rows, new_cols);

    Point p_ens(m);
    for (size_t i = 0; i < ext_ns.size(); ++i) {
      p_ens[i] = ext_ns[i];
    }

    Point p_zero(m);
    for (size_t i = 0; i < (size_t)m; ++i) {
      p_zero[i] = 0;
    }

    ptmp << p, p_zero, p_ens;

    #if 0
    print_points<Point>(p, "p");
    print_points<Point>(ptmp, "ptmp");
    #endif

    valid_points.push_back(ptmp);
  }

  vector<Point> out_points;
  falconn::sign_extend_for_query<float>(m, in_points, out_points);

  if (print) {
    EXPECT_EQ(out_dim, out_points[0].rows());
    EXPECT_EQ(dim, in_points[0].rows());
    EXPECT_EQ(out_dim, valid_points[0].rows());

    print_points<float>(out_points, out_dim, "outs");
    print_points<float>(in_points, dim, "ins");
    print_points<float>(valid_points, out_dim, "valids");
  }

  float eps = 0.00001;
  EXPECT_EQ(out_points.size(), in_points.size());
  EXPECT_EQ(out_points.size(), valid_points.size());
  EXPECT_EQ(out_points[0].rows(), (in_points[0].rows() + 2 * m));
  EXPECT_EQ(out_points[0].cols(), in_points[0].cols());
  EXPECT_EQ(out_points[0].rows(), valid_points[0].rows());
  EXPECT_EQ(out_points[0].cols(), valid_points[0].cols());

  size_t i = 0;
  for (auto& p : out_points) {
    auto& vp = valid_points[i];
    auto& ip = in_points[i];

    for (int j = 0; j < p.rows(); ++j) {
      EXPECT_NEAR(p[j], vp[j], eps);
      if (j < ip.rows()) {
        EXPECT_NEAR(p[j], ip[j], eps);
      }
    }

    ++i;
  }
}

#if 0
// Simple-ALSH
#endif

TEST(AlshForMispTest, DenseMips_Data_Norm_Simple_Test_Norm) {
  typedef DenseVector<float> Point;
  int dim = 4;

  Point p1(dim);
  p1[0] = 5;
  p1[1] = 5;
  p1[2] = 2;
  p1[3] = 4;

  Point p2(dim);
  p2[0] = 1;
  p2[1] = 7;
  p2[2] = 0;
  p2[3] = 2;

  Point p3(dim);
  p3[0] = 5;
  p3[1] = 6;
  p3[2] = 2;
  p3[3] = 4;

  Point p4(dim);
  p4[0] = 4;
  p4[1] = 6;
  p4[2] = 2;
  p4[3] = 4;

  vector<Point> in_points;
  in_points.push_back(p1);
  in_points.push_back(p2);
  in_points.push_back(p3);
  in_points.push_back(p4);

  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::Hyperplane;
  params.distance_function = DistanceFunction::NegativeInnerProduct;
  params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
  params.k = 2;
  params.l = 8;
  params.num_setup_threads = 0;

  DenseMips<float> dm(params, in_points);
  // dm.set_debug_file();
  dm.norm_action();

  Point q1(dim);
  q1[0] = 4;
  q1[1] = 5;
  q1[2] = 2;
  q1[3] = 4;

  Point q2(dim);
  q2[0] = 1;
  q2[1] = 7;
  q2[2] = 1;
  q2[3] = 2;

  vector<Point> queries;
  queries.push_back(q1);
  queries.push_back(q2);

  dm.set_queries(queries);

}

TEST(AlshForMispTest, DenseMips_Data_Norm_Simple_Test) {
  typedef DenseVector<float> Point;
  int dim = 4;

  Point p1(dim);
  p1[0] = 5;
  p1[1] = 5;
  p1[2] = 2;
  p1[3] = 4;

  Point p2(dim);
  p2[0] = 1;
  p2[1] = 7;
  p2[2] = 0;
  p2[3] = 2;

  Point p3(dim);
  p3[0] = 5;
  p3[1] = 6;
  p3[2] = 2;
  p3[3] = 4;

  Point p4(dim);
  p4[0] = 4;
  p4[1] = 6;
  p4[2] = 2;
  p4[3] = 4;

  vector<Point> in_points;
  in_points.push_back(p1);
  in_points.push_back(p2);
  in_points.push_back(p3);
  in_points.push_back(p4);

  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::Hyperplane;
  params.distance_function = DistanceFunction::NegativeInnerProduct;
  params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
  params.k = 2;
  params.l = 8;
  params.num_setup_threads = 0;

  DenseMips<float> dm(params, in_points);
  // dm.set_debug_file();
  dm.norm_action();

  Point q1(dim);
  q1[0] = 4;
  q1[1] = 5;
  q1[2] = 2;
  q1[3] = 4;

  Point q2(dim);
  q2[0] = 1;
  q2[1] = 7;
  q2[2] = 1;
  q2[3] = 2;

  vector<Point> queries;
  queries.push_back(q1);
  queries.push_back(q2);

  dm.set_queries(queries);
  int32_t q1_index = dm.find_mips_point(q1);
  EXPECT_TRUE(2 == q1_index);
  if (print) {
    cout << "index for q1: " << q1_index << endl;
  }

  int32_t q2_index = dm.find_mips_point(q2);
  EXPECT_TRUE(2 == q2_index);
  if (print) {
    cout << "index for q2: " << q2_index << endl;
  }
}

#if 0
// Sign-ALSH
#endif

TEST(AlshForMispTest, DenseMips_Data_Norm_Sign_Test_Norm) {
  typedef DenseVector<float> Point;
  int dim = 4;

  Point p1(dim);
  p1[0] = 5;
  p1[1] = 5;
  p1[2] = 2;
  p1[3] = 4;

  Point p2(dim);
  p2[0] = 1;
  p2[1] = 7;
  p2[2] = 0;
  p2[3] = 2;

  Point p3(dim);
  p3[0] = 5;
  p3[1] = 6;
  p3[2] = 2;
  p3[3] = 4;

  Point p4(dim);
  p4[0] = 4;
  p4[1] = 6;
  p4[2] = 2;
  p4[3] = 4;

  vector<Point> in_points;
  in_points.push_back(p1);
  in_points.push_back(p2);
  in_points.push_back(p3);
  in_points.push_back(p4);

  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::Hyperplane;
  params.distance_function = DistanceFunction::NegativeInnerProduct;
  params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
  params.k = 2;
  params.l = 8;
  params.num_setup_threads = 0;

  DenseMips<float> dm(params, in_points, falconn::NORM_TYPE::SIGN);
  // dm.set_debug_file();
  dm.norm_action();

  Point q1(dim);
  q1[0] = 4;
  q1[1] = 5;
  q1[2] = 2;
  q1[3] = 4;

  Point q2(dim);
  q2[0] = 1;
  q2[1] = 7;
  q2[2] = 1;
  q2[3] = 2;

  vector<Point> queries;
  queries.push_back(q1);
  queries.push_back(q2);

  dm.set_queries(queries);
}

TEST(AlshForMispTest, DenseMips_Data_Norm_Sign_Test) {
  typedef DenseVector<float> Point;
  int dim = 4;

  Point p1(dim);
  p1[0] = 3;
  p1[1] = 5;
  p1[2] = 2;
  p1[3] = 4;

  Point p2(dim);
  p2[0] = 6;
  p2[1] = 0;
  p2[2] = 1;
  p2[3] = 1;

  Point p3(dim);
  p3[0] = 2;
  p3[1] = 1;
  p3[2] = 5;
  p3[3] = 4;

  Point p4(dim);
  p4[0] = 1;
  p4[1] = 3;
  p4[2] = 2;
  p4[3] = 5;

  vector<Point> in_points;
  in_points.push_back(p1);
  in_points.push_back(p2);
  in_points.push_back(p3);
  in_points.push_back(p4);

  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::Hyperplane;
  params.distance_function = DistanceFunction::NegativeInnerProduct;
  params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
  params.k = 2;
  params.l = 8;
  params.num_setup_threads = 0;

  DenseMips<float> dm(params, in_points, falconn::NORM_TYPE::SIGN);
  // dm.set_debug_file();
  dm.norm_action();

  Point q1(dim);
  q1[0] = 1;
  q1[1] = 2;
  q1[2] = 3;
  q1[3] = 4;

  Point q2(dim);
  q2[0] = 1;
  q2[1] = 3;
  q2[2] = 4;
  q2[3] = 2;

  vector<Point> queries;
  queries.push_back(q1);
  queries.push_back(q2);

  dm.set_queries(queries);
  int32_t q1_index = dm.find_mips_point(q1);
  EXPECT_TRUE(2 == q1_index);
  if (print) {
    cout << "index for q1: " << q1_index << endl;
  }

  int32_t q2_index = dm.find_mips_point(q2);
  EXPECT_TRUE(0 == q2_index);
  if (print) {
    cout << "index for q2: " << q2_index << endl;
  }
}

#if 0
// DenseMipsTester
#endif

TEST(AlshForMispTest, DenseMipsParaProberSimple) {
  // typedef DenseVector<float> Point;
  int dim = 4;

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
  EXPECT_TRUE(dmt.validate_mips(params));
}

TEST(AlshForMispTest, DenseMipsTesterSign) {
  // typedef DenseVector<float> Point;
  int dim = 4;

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
  bool bret = dmt.validate_mips(params, falconn::NORM_TYPE::SIGN);
  assert (true == bret);
}


TEST(AlshForMispTest, DenseMipsParaProberSimpleAccuracy) {
  // typedef DenseVector<float> Point;
  int dim = 4;

  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::Hyperplane;
  params.distance_function = DistanceFunction::NegativeInnerProduct;
  params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
  params.k = 2;
  params.l = 8;
  params.num_setup_threads = 0;

  DenseMipsParaProber<float> dmt(dim, 1000);
  // dmt.set_print_file();
  double accuracy = dmt.evaluate_mips_accuracy(params);

  if (print) {
    cout << "Accuracy: " << accuracy << endl;
  }
}

TEST(AlshForMispTest, DenseMipsParaProberSimpleAccuracyEvaluate) {
  // typedef DenseVector<float> Point;
  int dim = 4;

  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::Hyperplane;
  params.distance_function = DistanceFunction::NegativeInnerProduct;
  params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
  params.k = 0;
  params.l = 0;
  params.num_setup_threads = 0;

  DenseMipsParaProber<float> dmt(dim, 10000, 200);
  dmt.set_print_file();
  // dmt.print_para();

  AccuracySpeed as;
  bool bret = dmt.evaluate_optimum_paras(as, params, falconn::NORM_TYPE::SIMPLE, 10, 0.96);
  assert (true == bret);

  if (true) {
    cout << "best paras:" << endl;
    cout << "\t k: " << as._k << endl;
    cout << "\t L: " << as._L << endl;
    cout << "\t avg_time: " << as._time << endl;
    cout << "\t accuracy: " << as._accuracy << endl;
  }

}

#if 0
TEST(AlshForMispTest, DenseMipsParaProberSign_recursive) {
  // typedef DenseVector<float> Point;
  int dim = 4;

  uint32_t count = 0;
  while(true) {
    LSHConstructionParameters params;
    params.dimension = dim;
#if 0
    params.lsh_family = LSHFamily::CrossPolytope;
    params.last_cp_dimension = dim;
    params.num_rotations = 3;
#endif
    params.lsh_family = LSHFamily::Hyperplane;
    params.distance_function = DistanceFunction::NegativeInnerProduct;
    params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
    params.k = 2;
    params.l = 8;
    params.num_setup_threads = 0;

    DenseMipsParaProber<float> dmt(dim, 100);
    // dmt.set_print_file();
    dmt.validate_mips(params, falconn::NORM_TYPE::SIGN);

    cout << "count: " << count++ << endl;
  }
}
#endif

