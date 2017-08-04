#ifndef _PROBER_DATA_H_
#define _PROBER_DATA_H_

#include <utility>
#include <cmath>

using std::make_pair;
using std::pair;
using std::ceil;
using std::log;

namespace falconn {

const uint32_t kSet[] = {2, 6, 12, 14, 16, 18};
const uint32_t LSet[] = {8, 16, 32, 48, 96, 200};
const uint32_t below100IndexSet = 0;  // index for data_num < 100
const uint32_t below1KIndexSet = 1;   // index for data_num < 1K
const uint32_t beyond10KIndexSet = 2;  // index for data_num > 10K
const uint32_t evaluate_num = 2;      // select evaluate k & L number

static uint32_t kSetLength() {
  return sizeof(kSet) / sizeof(kSet[0]);
}

static uint32_t LSetLength() {
  return sizeof(LSet) / sizeof(LSet[0]);
}

// delta: the probability that nearest neighbor is reported GE than: 1 - delta. defalut 0.9
// p1: default 0.9
// markdown: $ L = \lceil \frac{log(\delta)}{log(1 - p_1^k)} \rceil  $
static uint32_t get_min_L(const uint32_t k, const double p1 = 0.9, const double delta = 0.1) {
  return ceil(log(delta) / log(1 - pow(p1, k)));
}

typedef struct _AccuracySpeed {
  uint32_t _k;
  uint32_t _L;
  double _time;
  double _accuracy;

  _AccuracySpeed() : _k(0), _L(0), _time(0.0), _accuracy(0.0) {}

  _AccuracySpeed(uint32_t k, uint32_t L) : _k(k), _L(L), _time(0.0), _accuracy(0.0) {
    assert (1 <=  _k);
    assert (_k <= _L);

    const uint32_t min_L = get_min_L(_k);
    if (_L < min_L) {
      _L = min_L;
    }
  }

  _AccuracySpeed& operator=(const _AccuracySpeed& rhs) {
    if (this != &rhs) {
      _time = rhs._time;
      _accuracy = rhs._accuracy;
      _k = rhs._k;
      _L = rhs._L;
    }

    return *this;
  }

  bool operator<(const _AccuracySpeed& rhs) const {
    if (_k <rhs._k) {
      return true;
    }

    if (_L < rhs._L) {
      return true;
    }

    return false;
  }
} AccuracySpeed;

bool as_time_comp(const AccuracySpeed& a, const AccuracySpeed& b) {
  return a._time < b._time;
}

bool as_accuracy_comp(const AccuracySpeed& a, const AccuracySpeed& b) {
  return a._accuracy > b._accuracy;
}

bool int_comp_greater(const pair<int, float>& a, const pair<int, float>& b) {
  return a.second > b.second;
}

bool int_comp_little(const pair<int, float>& a, const pair<int, float>& b) {
  return a.second < b.second;
}

}

#endif
