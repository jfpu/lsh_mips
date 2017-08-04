#ifndef __CPP_WRAPPER_IMPL_H__
#define __CPP_WRAPPER_IMPL_H__

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

namespace falconn {
namespace wrapper {

template <typename PointType>
struct PointTypeTraitsInternal {};

// TODO: get rid of these type trait classes once CosineDistance and the LSH
// classes are specialized on PointType (if we want to specialize on point
// type).
template <typename CoordinateType>
class PointTypeTraitsInternal<DenseVector<CoordinateType> > {
 public:
  typedef core::CosineDistanceDense<CoordinateType> CosineDistance;
  typedef core::EuclideanDistanceDense<CoordinateType> EuclideanDistance;
  template <typename HashType>
  using HPHash = core::HyperplaneHashDense<CoordinateType, HashType>;
  template <typename HashType>
  using CPHash = core::CrossPolytopeHashDense<CoordinateType, HashType>;

  template <typename HashType>
  static std::unique_ptr<CPHash<HashType> > construct_cp_hash(
      const LSHConstructionParameters& params) {
    std::unique_ptr<CPHash<HashType> > res(new CPHash<HashType>(
        params.dimension, params.k, params.l, params.num_rotations,
        params.last_cp_dimension, params.seed ^ 93384688));
    return std::move(res);
  }
};

template <typename CoordinateType, typename IndexType>
class PointTypeTraitsInternal<SparseVector<CoordinateType, IndexType> > {
 public:
  typedef core::CosineDistanceSparse<CoordinateType, IndexType> CosineDistance;
  typedef core::EuclideanDistanceSparse<CoordinateType, IndexType>
      EuclideanDistance;
  template <typename HashType>
  using HPHash =
      core::HyperplaneHashSparse<CoordinateType, HashType, IndexType>;
  template <typename HashType>
  using CPHash =
      core::CrossPolytopeHashSparse<CoordinateType, HashType, IndexType>;

  template <typename HashType>
  static std::unique_ptr<CPHash<HashType> > construct_cp_hash(
      const LSHConstructionParameters& params) {
    std::unique_ptr<CPHash<HashType> > res(new CPHash<HashType>(
        params.dimension, params.k, params.l, params.num_rotations,
        params.feature_hashing_dimension, params.last_cp_dimension,
        params.seed ^ 93384688));
    return std::move(res);
  }
};

template <typename PointSet>
class DataStorageAdapter {
 public:
  DataStorageAdapter() {
    static_assert(FalseStruct<PointSet>::value,
                  "Point set type not supported.");
  }

  template <typename PS>
  struct FalseStruct : std::false_type {};
};

template <typename PointType>
class DataStorageAdapter<std::vector<PointType> > {
 public:
  template <typename KeyType>
  using DataStorage = core::ArrayDataStorage<PointType, KeyType>;

  template <typename KeyType>
  static std::unique_ptr<DataStorage<KeyType> > construct_data_storage(
      const std::vector<PointType>& points) {
    std::unique_ptr<DataStorage<KeyType> > res(new DataStorage<KeyType>(points));
    return std::move(res);
  }
};

template <typename CoordinateType>
class DataStorageAdapter<PlainArrayPointSet<CoordinateType> > {
 public:
  template <typename KeyType>
  using DataStorage =
      core::PlainArrayDataStorage<DenseVector<CoordinateType>, KeyType>;

  template <typename KeyType>
  static std::unique_ptr<DataStorage<KeyType> > construct_data_storage(
      const PlainArrayPointSet<CoordinateType>& points) {
    std::unique_ptr<DataStorage<KeyType> > res(new DataStorage<KeyType>(
        points.data, points.num_points, points.dimension));
    return std::move(res);
  }
};

template <typename PointType>
struct ComputeNumberOfHashFunctions {
  static void compute(int_fast32_t, LSHConstructionParameters*) {
    static_assert(FalseStruct<PointType>::value, "Point type not supported.");
  }
  template <typename T>
  struct FalseStruct : std::false_type {};
};

template <typename CoordinateType>
struct ComputeNumberOfHashFunctions<DenseVector<CoordinateType> > {
  static void compute(int_fast32_t number_of_hash_bits,
                      LSHConstructionParameters* params) {
    if (params->lsh_family == LSHFamily::Hyperplane) {
      params->k = number_of_hash_bits;
    } else if (params->lsh_family == LSHFamily::CrossPolytope) {
      if (params->dimension <= 0) {
        throw LSHNNTableSetupError(
            "Vector dimension must be set to determine "
            "the number of dense cross polytope hash functions.");
      }
      int_fast32_t rotation_dim =
          core::find_next_power_of_two(params->dimension);
      core::cp_hash_helpers::compute_k_parameters_for_bits(
          rotation_dim, number_of_hash_bits, &(params->k),
          &(params->last_cp_dimension));
    } else {
      throw LSHNNTableSetupError(
          "Cannot set paramters for unknown hash "
          "family.");
    }
  }
};

template <typename CoordinateType, typename IndexType>
struct ComputeNumberOfHashFunctions<SparseVector<CoordinateType, IndexType> > {
  static void compute(int_fast32_t number_of_hash_bits,
                      LSHConstructionParameters* params) {
    if (params->lsh_family == LSHFamily::Hyperplane) {
      params->k = number_of_hash_bits;
    } else if (params->lsh_family == LSHFamily::CrossPolytope) {
      if (params->feature_hashing_dimension <= 0) {
        throw LSHNNTableSetupError(
            "Feature hashing dimension must be set to "
            "determine  the number of sparse cross polytope hash functions.");
      }
      // TODO: add check here for power-of-two feature hashing dimension
      // (or allow non-power-of-two feature hashing dimension in the CP hash)
      int_fast32_t rotation_dim =
          core::find_next_power_of_two(params->feature_hashing_dimension);
      core::cp_hash_helpers::compute_k_parameters_for_bits(
          rotation_dim, number_of_hash_bits, &(params->k),
          &(params->last_cp_dimension));
    } else {
      throw LSHNNTableSetupError(
          "Cannot set paramters for unknown hash "
          "family.");
    }
  }
};

template <typename PointType>
struct ComputeNumberOfHashBits {
  static int_fast32_t compute(const LSHConstructionParameters&) {
    static_assert(FalseStruct<PointType>::value, "Point type not supported.");
    return 0;
  }
  template <typename T>
  struct FalseStruct : std::false_type {};
};

template <typename CoordinateType>
struct ComputeNumberOfHashBits<DenseVector<CoordinateType> > {
  static int_fast32_t compute(const LSHConstructionParameters& params) {
    if (params.k <= 0) {
      throw LSHNNTableSetupError(
          "Number of hash functions k must be at least "
          "1 to determine the number of hash bits.");
    }
    if (params.lsh_family == LSHFamily::Hyperplane) {
      return params.k;
    } else if (params.lsh_family == LSHFamily::CrossPolytope) {
      if (params.dimension <= 0) {
        throw LSHNNTableSetupError(
            "Vector dimension must be set to determine "
            "the number of dense cross polytope hash bits.");
      }
      if (params.last_cp_dimension <= 0) {
        throw LSHNNTableSetupError(
            "Last cross-polytope dimension must be set "
            "to determine the number of dense cross polytope hash bits.");
      }
      return core::cp_hash_helpers::compute_number_of_hash_bits(
          params.dimension, params.last_cp_dimension, params.k);
    } else {
      throw LSHNNTableSetupError(
          "Cannot compute number of hash bits for "
          "unknown hash family.");
    }
  }
};

template <typename CoordinateType, typename IndexType>
struct ComputeNumberOfHashBits<SparseVector<CoordinateType, IndexType> > {
  static int_fast32_t compute(const LSHConstructionParameters& params) {
    if (params.k <= 0) {
      throw LSHNNTableSetupError(
          "Number of hash functions k must be at least "
          "1 to determine the number of hash bits.");
    }
    if (params.lsh_family == LSHFamily::Hyperplane) {
      return params.k;
    } else if (params.lsh_family == LSHFamily::CrossPolytope) {
      if (params.feature_hashing_dimension <= 0) {
        throw LSHNNTableSetupError(
            "Feature hashing dimension must be set to "
            "determine the number of dense cross polytope hash bits.");
      }
      if (params.last_cp_dimension <= 0) {
        throw LSHNNTableSetupError(
            "Last cross-polytope dimension must be set "
            "to determine the number of dense cross polytope hash bits.");
      }
      return core::cp_hash_helpers::compute_number_of_hash_bits(
          params.feature_hashing_dimension, params.last_cp_dimension, params.k);
    } else {
      throw LSHNNTableSetupError(
          "Cannot compute number of hash bits for "
          "unknown hash family.");
    }
  }
};

template <typename PointType>
struct GetDefaultParameters {
  static LSHConstructionParameters get(int_fast64_t, int_fast32_t,
                                       DistanceFunction, bool) {
    static_assert(FalseStruct<PointType>::value, "Point type not supported.");
    LSHConstructionParameters tmp;
    return tmp;
  }
  template <typename T>
  struct FalseStruct : std::false_type {};
};

template <typename CoordinateType>
struct GetDefaultParameters<DenseVector<CoordinateType> > {
  static LSHConstructionParameters get(int_fast64_t dataset_size,
                                       int_fast32_t dimension,
                                       DistanceFunction distance_function,
                                       bool is_sufficiently_dense) {
    LSHConstructionParameters result;
    result.dimension = dimension;
    result.distance_function = distance_function;
    result.lsh_family = LSHFamily::CrossPolytope;

    result.num_rotations = 2;
    if (is_sufficiently_dense) {
      result.num_rotations = 1;
    }

    result.l = 10;
    result.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
    result.num_setup_threads = 0;

    int_fast32_t number_of_hash_bits = 1;
    while ((1 << (number_of_hash_bits + 2)) <= dataset_size) {
      ++number_of_hash_bits;
    }
    compute_number_of_hash_functions<DenseVector<CoordinateType> >(
        number_of_hash_bits, &result);

    return result;
  }
};

template <typename CoordinateType>
struct GetDefaultParameters<SparseVector<CoordinateType> > {
  static LSHConstructionParameters get(int_fast64_t dataset_size,
                                       int_fast32_t dimension,
                                       DistanceFunction distance_function,
                                       bool) {
    LSHConstructionParameters result;
    result.dimension = dimension;
    result.distance_function = distance_function;
    result.lsh_family = LSHFamily::CrossPolytope;
    result.feature_hashing_dimension = 1024;
    result.num_rotations = 2;

    result.l = 10;
    result.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
    result.num_setup_threads = 0;

    int_fast32_t number_of_hash_bits = 1;
    while ((1 << (number_of_hash_bits + 2)) <= dataset_size) {
      ++number_of_hash_bits;
    }
    compute_number_of_hash_functions<SparseVector<CoordinateType> >(
        number_of_hash_bits, &result);

    return result;
  }
};

template <typename PointType, typename KeyType, typename DistanceType,
          typename DistanceFunction, typename LSHTable, typename LSHFunction,
          typename HashTableFactory, typename CompositeHashTable,
          typename NNQuery, typename DataStorage>
class LSHNNTableWrapper : public LSHNearestNeighborTable<PointType, KeyType> {
 public:
  LSHNNTableWrapper(std::unique_ptr<LSHFunction> lsh,
                    std::unique_ptr<LSHTable> lsh_table,
                    std::unique_ptr<HashTableFactory> hash_table_factory,
                    std::unique_ptr<CompositeHashTable> composite_hash_table,
                    std::unique_ptr<typename LSHTable::Query> query,
                    std::unique_ptr<NNQuery> nn_query,
                    std::unique_ptr<DataStorage> data_storage)
      : lsh_(std::move(lsh)),
        lsh_table_(std::move(lsh_table)),
        hash_table_factory_(std::move(hash_table_factory)),
        composite_hash_table_(std::move(composite_hash_table)),
        query_(std::move(query)),
        nn_query_(std::move(nn_query)),
        data_storage_(std::move(data_storage)) {
    num_probes_ = lsh_->get_l();
  }

  void set_num_probes(int_fast64_t num_probes) {
    if (num_probes <= 0) {
      throw LSHNearestNeighborTableError(
          "Number of probes must be at least 1.");
    }
    num_probes_ = num_probes;
  }

  int_fast64_t get_num_probes() { return num_probes_; }

  void set_max_num_candidates(int_fast64_t max_num_candidates) {
    max_num_candidates_ = max_num_candidates;
  }

  int_fast64_t get_max_num_candidates() { return max_num_candidates_; }

  KeyType find_nearest_neighbor(const PointType& q) {
    return nn_query_->find_nearest_neighbor(q, q, num_probes_,
                                            max_num_candidates_);
  }

  void find_k_nearest_neighbors(const PointType& q, int_fast64_t k,
                                std::vector<KeyType>* result) {
    nn_query_->find_k_nearest_neighbors(q, q, k, num_probes_,
                                        max_num_candidates_, result);
  }

  void find_near_neighbors(const PointType& q, DistanceType threshold,
                           std::vector<KeyType>* result) {
    nn_query_->find_near_neighbors(q, q, threshold, num_probes_,
                                   max_num_candidates_, result);
  }

  void get_candidates_with_duplicates(const PointType& q,
                                      std::vector<KeyType>* result) {
    query_->get_candidates_with_duplicates(q, num_probes_, max_num_candidates_,
                                           result);
  }

  void get_unique_candidates(const PointType& q, std::vector<KeyType>* result) {
    query_->get_unique_candidates(q, num_probes_, max_num_candidates_, result);
  }

  void reset_query_statistics() { nn_query_->reset_query_statistics(); }

  QueryStatistics get_query_statistics() {
    return nn_query_->get_query_statistics();
  }

  ~LSHNNTableWrapper() {}

 protected:
  std::unique_ptr<LSHFunction> lsh_;
  std::unique_ptr<LSHTable> lsh_table_;
  std::unique_ptr<HashTableFactory> hash_table_factory_;
  std::unique_ptr<CompositeHashTable> composite_hash_table_;
  std::unique_ptr<typename LSHTable::Query> query_;
  std::unique_ptr<NNQuery> nn_query_;
  std::unique_ptr<DataStorage> data_storage_;

  int_fast64_t num_probes_;
  int_fast64_t max_num_candidates_ = this->kNoMaxNumCandidates;
};

template <typename PointType, typename KeyType, typename PointSet>
class StaticTableFactory {
 public:
  typedef typename PointTypeTraits<PointType>::ScalarType ScalarType;

  typedef typename DataStorageAdapter<PointSet>::template DataStorage<KeyType>
      DataStorageType;

  StaticTableFactory(const PointSet& points,
                     const LSHConstructionParameters& params)
      : points_(points), params_(params) {}

  std::unique_ptr<LSHNearestNeighborTable<PointType, KeyType> > setup() {
    if (params_.dimension < 1) {
      throw LSHNNTableSetupError(
          "Point dimension must be at least 1. Maybe "
          "you forgot to set the point dimension in the parameter struct?");
    }
    if (params_.k < 1) {
      throw LSHNNTableSetupError(
          "The number of hash functions k must be at "
          "least 1. Maybe you forgot to set k in the parameter struct?");
    }
    if (params_.l < 1) {
      throw LSHNNTableSetupError(
          "The number of hash tables l must be at "
          "least 1. Maybe you forgot to set l in the parameter struct?");
    }
    if (params_.num_setup_threads < 0) {
      throw LSHNNTableSetupError(
          "The number of setup threads cannot be "
          "negative. Maybe you forgot to set num_setup_threads in the "
          "parameter struct? A value of 0 indicates that FALCONN should use "
          "the maximum number of available hardware threads.");
    }

    data_storage_ = std::move(
        DataStorageAdapter<PointSet>::template construct_data_storage<KeyType>(
            points_));

    ComputeNumberOfHashBits<PointType> helper;
    num_bits_ = helper.compute(params_);

    n_ = data_storage_->size();

    setup0();

    return std::move(table_);
  }

 private:
  void setup0() {
    if (num_bits_ <= 32) {
      typedef uint32_t HashType;
      HashType tmp;
      setup1(std::make_tuple(tmp));
    } else if (num_bits_ <= 64) {
      typedef uint64_t HashType;
      HashType tmp;
      setup1(std::make_tuple(tmp));
    } else {
      throw LSHNNTableSetupError(
          "More than 64 hash bits are currently not "
          "supported.");
    }
  }

  template <typename V>
  void setup1(V vals) {
    typedef typename std::tuple_element<kHashTypeIndex, V>::type HashType;

    if (params_.lsh_family == LSHFamily::Hyperplane) {
      typedef typename wrapper::PointTypeTraitsInternal<
          PointType>::template HPHash<HashType>
          LSH;
      std::unique_ptr<LSH> lsh(new LSH(params_.dimension, params_.k, params_.l,
                                       params_.seed ^ 93384688));
      setup2(std::tuple_cat(vals, std::make_tuple(std::move(lsh))));
    } else if (params_.lsh_family == LSHFamily::CrossPolytope) {
      if (params_.num_rotations < 0) {
        throw LSHNNTableSetupError(
            "The number of pseudo-random rotations for "
            "the cross polytope hash must be non-negative. Maybe you forgot to "
            "set num_rotations in the parameter struct?");
      }
      if (params_.last_cp_dimension <= 0) {
        throw LSHNNTableSetupError(
            "The last cross polytope dimension for "
            "the cross polytope hash must be at least 1. Maybe you forgot to "
            "set last_cp_dimension in the parameter struct?");
      }

      // TODO: for sparse vectors, also check feature_hashing_dimension here (it
      // is checked in the CP hash class, but the error message is less
      // verbose).

      typedef typename wrapper::PointTypeTraitsInternal<
          PointType>::template CPHash<HashType>
          LSH;
      std::unique_ptr<LSH> lsh(
          std::move(wrapper::PointTypeTraitsInternal<
                    PointType>::template construct_cp_hash<HashType>(params_)));
      setup2(std::tuple_cat(vals, std::make_tuple(std::move(lsh))));
    } else {
      throw LSHNNTableSetupError(
          "Unknown hash family. Maybe you forgot to set "
          "the hash family in the parameter struct?");
    }
  }

  template <typename V>
  void setup2(V vals) {
    if (params_.distance_function == DistanceFunction::NegativeInnerProduct) {
      typedef
          typename wrapper::PointTypeTraitsInternal<PointType>::CosineDistance
              DistanceFunc;
      DistanceFunc tmp;
      setup3(std::tuple_cat(std::move(vals), std::make_tuple(tmp)));
    } else if (params_.distance_function ==
               DistanceFunction::EuclideanSquared) {
      typedef typename wrapper::PointTypeTraitsInternal<
          PointType>::EuclideanDistance DistanceFunc;
      DistanceFunc tmp;
      setup3(std::tuple_cat(std::move(vals), std::make_tuple(tmp)));
    } else {
      throw LSHNNTableSetupError(
          "Unknown distance function. Maybe you forgot "
          "to set the hash family in the parameter struct?");
    }
  }

  template <typename V>
  void setup3(V vals) {
    typedef typename std::tuple_element<kHashTypeIndex, V>::type HashType;

    if (params_.storage_hash_table == StorageHashTable::FlatHashTable) {
      typedef core::FlatHashTable<HashType> HashTable;
      std::unique_ptr<typename HashTable::Factory> factory(
          new typename HashTable::Factory(1 << num_bits_));

      typedef core::StaticCompositeHashTable<HashType, KeyType, HashTable>
          CompositeTable;
      std::unique_ptr<CompositeTable> composite_table(
          new CompositeTable(params_.l, factory.get()));
      setup4(std::tuple_cat(std::move(vals),
                            std::make_tuple(std::move(factory)),
                            std::make_tuple(std::move(composite_table))));
    } else if (params_.storage_hash_table ==
               StorageHashTable::BitPackedFlatHashTable) {
      typedef core::BitPackedFlatHashTable<HashType> HashTable;
      std::unique_ptr<typename HashTable::Factory> factory(
          new typename HashTable::Factory(1 << num_bits_, n_));

      typedef core::StaticCompositeHashTable<HashType, KeyType, HashTable>
          CompositeTable;
      std::unique_ptr<CompositeTable> composite_table(
          new CompositeTable(params_.l, factory.get()));
      setup4(std::tuple_cat(std::move(vals),
                            std::make_tuple(std::move(factory)),
                            std::make_tuple(std::move(composite_table))));
    } else if (params_.storage_hash_table == StorageHashTable::STLHashTable) {
      typedef core::STLHashTable<HashType> HashTable;
      std::unique_ptr<typename HashTable::Factory> factory(
          new typename HashTable::Factory());

      typedef core::StaticCompositeHashTable<HashType, KeyType, HashTable>
          CompositeTable;
      std::unique_ptr<CompositeTable> composite_table(
          new CompositeTable(params_.l, factory.get()));
      setup4(std::tuple_cat(std::move(vals),
                            std::make_tuple(std::move(factory)),
                            std::make_tuple(std::move(composite_table))));
    } else if (params_.storage_hash_table ==
               StorageHashTable::LinearProbingHashTable) {
      typedef core::StaticLinearProbingHashTable<HashType, KeyType> HashTable;
      std::unique_ptr<typename HashTable::Factory> factory(
          new typename HashTable::Factory(2 * n_));

      typedef core::StaticCompositeHashTable<HashType, KeyType, HashTable>
          CompositeTable;
      std::unique_ptr<CompositeTable> composite_table(
          new CompositeTable(params_.l, factory.get()));
      setup4(std::tuple_cat(std::move(vals),
                            std::make_tuple(std::move(factory)),
                            std::make_tuple(std::move(composite_table))));
    } else {
      throw LSHNNTableSetupError(
          "Unknown storage hash table type. Maybe you "
          "forgot to set the hash table type in the parameter struct?");
    }
  }

  template <typename V>
  void setup4(V vals) {
    setup_final(std::move(vals));
  }

  template <typename V>
  void setup_final(V vals) {
    typedef typename std::tuple_element<kHashTypeIndex, V>::type HashType;

    typedef
        typename std::tuple_element<kLSHFamilyIndex, V>::type LSHPointerType;
    typedef typename LSHPointerType::element_type LSHType;

    typedef typename std::tuple_element<kDistanceFunctionIndex, V>::type
        DistanceFunctionType;

    typedef typename std::tuple_element<kHashTableFactoryIndex, V>::type
        HashTableFactoryPointerType;
    typedef
        typename HashTableFactoryPointerType::element_type HashTableFactoryType;

    typedef typename std::tuple_element<kCompositeHashTableIndex, V>::type
        CompositeHashTablePointerType;
    typedef typename CompositeHashTablePointerType::element_type
        CompositeHashTableType;

    std::unique_ptr<LSHType>& lsh = std::get<kLSHFamilyIndex>(vals);
    std::unique_ptr<HashTableFactoryType>& factory =
        std::get<kHashTableFactoryIndex>(vals);
    std::unique_ptr<CompositeHashTableType>& composite_table =
        std::get<kCompositeHashTableIndex>(vals);

    typedef core::StaticLSHTable<PointType, KeyType, LSHType, HashType,
                                 CompositeHashTableType, DataStorageType>
        LSHTableType;
    std::unique_ptr<LSHTableType> lsh_table(
        new LSHTableType(lsh.get(), composite_table.get(), *data_storage_,
                         params_.num_setup_threads));

    std::unique_ptr<typename LSHTableType::Query> query(
        new typename LSHTableType::Query(*lsh_table));

    typedef core::NearestNeighborQuery<typename LSHTableType::Query, PointType,
                                       KeyType, PointType, ScalarType,
                                       DistanceFunctionType, DataStorageType>
        NNQueryType;
    std::unique_ptr<NNQueryType> nn_query(
        new NNQueryType(query.get(), *data_storage_));

    table_.reset(
        new LSHNNTableWrapper<PointType, KeyType, ScalarType,
                              DistanceFunctionType, LSHTableType, LSHType,
                              HashTableFactoryType, CompositeHashTableType,
                              NNQueryType, DataStorageType>(
            std::move(lsh), std::move(lsh_table), std::move(factory),
            std::move(composite_table), std::move(query), std::move(nn_query),
            std::move(data_storage_)));
  }

  const static int_fast32_t kHashTypeIndex = 0;
  const static int_fast32_t kLSHFamilyIndex = 1;
  const static int_fast32_t kDistanceFunctionIndex = 2;
  const static int_fast32_t kHashTableFactoryIndex = 3;
  const static int_fast32_t kCompositeHashTableIndex = 4;

  const PointSet& points_;
  const LSHConstructionParameters& params_;
  std::unique_ptr<DataStorageType> data_storage_;
  int_fast32_t num_bits_;
  int_fast64_t n_;
  std::unique_ptr<LSHNearestNeighborTable<PointType, KeyType> > table_ = nullptr;
};

}  // namespace wrapper
}  // namespace falconn

namespace falconn {

template <typename PointType>
void compute_number_of_hash_functions(int_fast32_t number_of_hash_bits,
                                      LSHConstructionParameters* params) {
  wrapper::ComputeNumberOfHashFunctions<PointType>::compute(number_of_hash_bits,
                                                            params);
}

template <typename PointType>
LSHConstructionParameters get_default_parameters(
    int_fast64_t dataset_size, int_fast32_t dimension,
    DistanceFunction distance_function, bool is_sufficiently_dense) {
  return wrapper::GetDefaultParameters<PointType>::get(
      dataset_size, dimension, distance_function, is_sufficiently_dense);
}

template <typename PointType, typename KeyType, typename PointSet>
std::unique_ptr<LSHNearestNeighborTable<PointType, KeyType> > construct_table(
    const PointSet& points, const LSHConstructionParameters& params) {
  wrapper::StaticTableFactory<PointType, KeyType, PointSet> factory(points,
                                                                    params);
  return std::move(factory.setup());
}

// LSH Parameter Prober for Dense Datas

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
static double CosineDistance(const Point& u, const Point& v) {
  // return 1 - pow(u.dot(v) / (u.squaredNorm() * v.squaredNorm()), 0.5);

  falconn::core::CosineDistanceDense<CoordinateType> distance_function;
  return distance_function(u, v);
}

template <typename CoordinateType, typename Point = DenseVector<CoordinateType> >
static double EuclideanDistanceDense(const Point& u, const Point& v) {
  falconn::core::EuclideanDistanceDense<CoordinateType> distance_function;
  return distance_function(u, v);
}

// TODO: Pls. do not check the return value for any member function of DenseLshParaProber by using assert
// TODO: Because it might can not call the member without any compiling error

template <typename CoordinateType>
class DenseLshParaProber {
 public:
  typedef CoordinateType value_type;
  typedef DenseVector<value_type> Point;

 public:
  DenseLshParaProber(const uint32_t dim, const uint32_t data_num = 1000,
      const uint32_t query_num = 100, DistanceFunction df = DistanceFunction::NegativeInnerProduct,
      const uint32_t min = 0, const uint32_t max = 100)
      : _dim(dim), _data_num(data_num), _query_num(query_num),
      _min(min), _max(max), _df(df) {
    assert (DistanceFunction::Unknown != _df);
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

  DenseLshParaProber(const uint32_t dim,
      vector<Point>* const datas, vector<Point>* const queries,
      DistanceFunction df = DistanceFunction::NegativeInnerProduct)
      : _dim(dim), _data_num(0), _query_num(0), _df(df) {
    assert (DistanceFunction::Unknown != _df);
    _datas.reset(std::move(datas));
    _data_num = _datas->size();

    _queries.reset(std::move(queries));
    _query_num = _queries->size();

    assert ((0 < _dim) && (10000 > _dim));
    assert (0 < _data_num);
    assert (0 < _query_num);

    __gen_answers();
  }

  ~DenseLshParaProber() {}

  void set_print_file(const char* file = "dense_mips_tester.txt") {
    _print_file = file;
  }

  // TODO: Pls. do not check the return value for validate_lsh function of DenseLshParaProber by using assert
  // TODO: Because it might can not call the member without any compiling error
  bool validate_lsh(LSHConstructionParameters& params) {
    _print();

    unique_ptr<LSHNearestNeighborTable<Point> > table(
        std::move(construct_table<Point>(*_datas, params)));

    size_t r_cnt = 0;
    size_t index = 0;
    for (auto& q : *_queries) {
      int q_index = table->find_nearest_neighbor(q);
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

  double evaluate_mips_accuracy(LSHConstructionParameters& params) {
    _print();

    unique_ptr<LSHNearestNeighborTable<Point> > table(
        std::move(construct_table<Point>(*_datas, params)));


    return __evaluate_accuracy(table);
  }

  // TODO: Pls. do not check the return value for evaluate_optimum_paras function of DenseLshParaProber by using assert
  // TODO: Because it might can not call the member without any compiling error
  bool evaluate_optimum_paras(AccuracySpeed& as, LSHConstructionParameters& params,
      const uint32_t multiple = 10, const double min_accuracy = 0.9) {
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

    if (!__evaluate_good_as(i_k, i_L, params, min_accuracy)) {
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

    __evaluate_random_as(params, min_accuracy);

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

    fprintf(fp, "\n DenseLshParaProber:\n");
    fprintf(fp, "\t dim: %d, dnum: %d, qnum: %d, range(%d, %d)\n",
        _dim, _data_num, _query_num, _min, _max);

    fclose(fp);
  }

 private:
  void __evaluate_as(AccuracySpeed& as, LSHConstructionParameters& params) {
    LSHConstructionParameters t_para;
    memcpy(&t_para, &params, sizeof(LSHConstructionParameters));
    t_para.k = as._k;
    t_para.l = as._L;

    double time = 0.0;
    double accuracy = 0.0;
    __get_mips_paras(time, accuracy, t_para);
    as._time = time;
    as._accuracy = accuracy;
  }

  bool __evaluate_good_as(uint32_t& i_k, uint32_t& i_L,
      LSHConstructionParameters& params, const double min_accuracy){
    assert ((kSetLength() > i_k) && (LSetLength()() > i_L));

    AccuracySpeed as_predict(kSet[i_k], LSet[i_L]);
    if (__evaluate_single_as(as_predict, params, min_accuracy)) {
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

    return __evaluate_good_as(i_k, i_L, params, min_accuracy);
  }

  bool __evaluate_random_as(LSHConstructionParameters& params, const double min_accuracy) {
    for (auto& as_predict : _predict_as) {
      __evaluate_single_as(as_predict, params, min_accuracy);
    }

    return true;
  }

  bool __evaluate_single_as(AccuracySpeed& as_predict,
      LSHConstructionParameters& params, const double min_accuracy) {
    __evaluate_as(as_predict, params);

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
        float score = 0.0;
        if (DistanceFunction::NegativeInnerProduct == _df) {
          score = CosineDistance<value_type>(query, datapoint);
        } else {
          score = EuclideanDistanceDense<value_type>(query, datapoint);
        }

        _answers[outer_counter][inner_counter].first = inner_counter;
        _answers[outer_counter][inner_counter].second = score;

        if (score > best) {
          best = score;
        }
        ++inner_counter;
      }
      sort(_answers[outer_counter].begin(), _answers[outer_counter].end(), int_comp_little);
      assert (_answers[outer_counter][0].second == best);
      ++outer_counter;
    }
  }

  double __evaluate_accuracy(unique_ptr<LSHNearestNeighborTable<Point> >& table) {
    int outer_counter = 0;
    int num_matches = 0;
    int32_t candidates;
    for (const auto& query : *_queries) {
      candidates = table->find_nearest_neighbor(query);
      const int ans = _answers[outer_counter][0].first;
      if (ans == candidates) {
        ++num_matches;
      }
      ++outer_counter;
    }
    return (num_matches + 0.0) / (_queries->size() + 0.0);
  }

  void __get_mips_paras(double& time, double& accuracy, LSHConstructionParameters& params) {

    unique_ptr<LSHNearestNeighborTable<Point> > table(
        std::move(construct_table<Point>(*_datas, params)));

    accuracy = __evaluate_accuracy(table);

    auto statistics = table->get_query_statistics();
    time = statistics.average_total_query_time;
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

    fprintf(fp, "\n DenseLshParaProber:\n");
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

  DistanceFunction _df = DistanceFunction::NegativeInnerProduct;
  string _print_file;

  std::unique_ptr<vector<Point> > _datas;
  std::unique_ptr<vector<Point> > _queries;
  vector<vector<pair<int, float> > > _answers;

  vector<AccuracySpeed> _qualified_as;
  vector<AccuracySpeed> _predict_as;

 private:
  DenseLshParaProber(const DenseLshParaProber&) = delete;
  DenseLshParaProber& operator=(const DenseLshParaProber&) = delete;
};

}  // namespace falconn

#endif
