#include "lsh_tester.h"

#include <iostream>

using namespace std;

LshTester::LshTester() {
}

LshTester::~LshTester() {
}

int LshTester::init(MatrixXd datas, MatrixXd queries, double rand_range , int num_neighbours,int m ) {
  int kdata = datas.row(0).size();
  int qdata = queries.row(0).size();
  assert(kdata == qdata);

  this->d = kdata;
  this->datas = datas;
  this->queries = queries;
  this->rand_range = rand_range;
  this->num_neighbours = num_neighbours;
  this->q_num = queries.rows();
  return 1;
}

VectorXd LshTester::linear(const VectorXd &q, metric_func metric, int max_results) {
  multimap<double,long> candidates;
  double distance;

  for(int i=0;i<datas.rows();i++) {
    distance = (*metric)(q, datas.row(i));
    candidates.insert(pair<double,long>(distance, i));
  }

  VectorXd res(max_results);
  multimap<double,long>::iterator iter=candidates.begin();
  for(int j=0;j<max_results;j++,iter++) {
    res(j) = iter->second;
  }

  return res;
}

void LshTester::run(string type, vector<int> k_vec, vector<int> l_vec) {
  // set distance func object
  metric_func metric;
  LshType ltype;

  if("l2" == type) {
    metric = L2Lsh_distance;
    ltype = L2_LSH;
  }
  else if("cosine" == type) {
    metric = ConsineLsh_distance;
    ltype = CONSINE_LSH;
  }
  else {
    std::cout<<"error distance type !";
    return ;
  }

  MatrixXd exact_hits(queries.rows(), num_neighbours);
  for(int i=0;i<queries.rows();i++) {
    exact_hits.row(i) = linear(queries.row(i), metric, num_neighbours);
  }

  cout<<"=============================="<<endl;
  cout<<type<<" TEST: "<<endl;
  cout<<"L\tk\tacc\ttouch"<<endl;

  // concatenating more hash functions increases selectivity
  for(int i= 0;i<k_vec.size();i++) {
    int k = k_vec[i];
    LshWrapper lsh;

    // lsh.init(ltype, d, rand_range, k, 0);
    // using more hash tables increases recall
    for(int j=0;j<l_vec.size();j++) {
      int L = l_vec[j];
      lsh.init(ltype, d, rand_range, k, L);
      //lsh.resize(L);
      lsh.index(datas);

      int correct = 0;
      for(int k=0;k<queries.size();k++) {
        VectorXd q = queries.row(k);
        VectorXd hits = exact_hits.row(k);
        VectorXd lsh_hits = lsh.query(q, metric, num_neighbours);

        if(lsh_hits == hits) {
          correct += 1;
        }
      }
      cout<< L << "\t" << k << "\t" << float(correct)/q_num << "\t" << float(lsh.get_avg_touched())/datas.rows() <<endl;
    }
  }
}

L2AlshTester::L2AlshTester() {
}

L2AlshTester::~L2AlshTester() {
}

int L2AlshTester::init(MatrixXd datas, MatrixXd queries, double rand_range, int num_neighbours, int m ) {
  int kdata = datas.row(0).size();
  int qdata = queries.row(0).size();
  assert(kdata == qdata);

  this->m = m;
  //this->half_extend = g_ext_half(m);
  //storage original datas & queries. used for validation
  this->origin_datas = datas;
  this->origin_queries = queries;
  this->q_num = origin_queries.rows();
  // datas & queries transformation
  double dratio,dmax_norm;
  this->norm_datas = g_transformation(origin_datas, dratio, dmax_norm);
  this->norm_queries = g_normalization(origin_queries);
  assert (kdata == norm_datas.row(0).size());
  assert (qdata == norm_queries.row(0).size());
  assert (origin_datas.rows() == norm_datas.rows());
  assert (origin_queries.rows() == norm_queries.rows());

  // expand k dimension into k+2m dimension
  this->ext_datas = g_index_extend(norm_datas, m);
  this->ext_queries = g_query_extend(norm_queries, m);
  int new_len = kdata + 2 * m;
  assert (new_len == ext_datas.row(0).size());
  assert (new_len == ext_queries.row(0).size());
  assert (origin_datas.rows() == ext_datas.rows());
  assert (origin_queries.rows() == ext_queries.rows());

  LshTester::init(ext_datas, ext_queries, rand_range, num_neighbours);
  return 1;
}

VectorXd L2AlshTester::linear(const VectorXd &q, metric_func metric, int max_results) {
  // brute force search by linear scan
  // print 'MipsLshTester linear:
  multimap<double,long> candidates;
  double mips;

  for(int i=0;i<origin_datas.rows();i++) {
    mips = dot(q, origin_datas.row(i));
    candidates.insert(pair<double,long>(mips, i));
  }

  VectorXd res(max_results);
  multimap<double,long>::iterator iter=candidates.end();
  --iter;

  for(int j=0;j<max_results;j++,iter--) {
    res(j) = iter->second;
  }

  return res;
}

void L2AlshTester::run(string type, vector<int> k_vec, vector<int> l_vec) {
  assert("l2" == type);
  metric_func validate_metric = dot, compute_metric = L2Lsh_distance;

  MatrixXd exact_hits(origin_queries.rows(), num_neighbours);
  for(int i=0;i<origin_queries.rows();i++) {
      exact_hits.row(i) = linear(origin_queries.row(i), validate_metric, num_neighbours);
  }

  cout<<"=============================="<<endl;
  cout<<"L2AlshTester "<<type<<" TEST: "<<endl;
  cout<<"L\tk\tacc\ttouch"<<endl;

  // concatenating more hash functions increases selectivity
  for(int i= 0;i<k_vec.size();i++) {
    int k = k_vec[i];
    LshWrapper lsh;
    // lsh.init(L2_LSH, d, rand_range, k, 0);

    // using more hash tables increases recall
    for(int j=0;j<l_vec.size();j++) {
      int L = l_vec[j];
      lsh.init(L2_LSH, d, rand_range, k, L);
      // lsh.resize(L);
      lsh.index(ext_datas);

      int correct = 0;
      for(int k=0;k<ext_queries.rows();k++) {
        VectorXd q = ext_queries.row(k);
        VectorXd hits = exact_hits.row(k);
        VectorXd lsh_hits = lsh.query(q, compute_metric, num_neighbours);

        if(lsh_hits == hits) {
            correct += 1;
        }
      }

      cout<< L << "\t" << k << "\t" << float(correct)/q_num << "\t" << float(lsh.get_avg_touched())/datas.rows() <<endl;
    }
  }
}

CosineAlshTester::CosineAlshTester() {
}

CosineAlshTester::~CosineAlshTester() {
}

int CosineAlshTester::init(MatrixXd datas, MatrixXd queries, double rand_range, int num_neighbours, int m) {
  int kdata = datas.row(0).size();
  int qdata = queries.row(0).size();
  assert(kdata == qdata);

  this->m = m;
  //this->half_extend = g_ext_half(m);
  //storage original datas & queries. used for validation
  this->origin_datas = datas;
  this->origin_queries = queries;
  this->q_num = origin_queries.rows();
  // datas & queries transformation
  double dratio,dmax_norm;
  this->norm_datas = g_transformation(origin_datas, dratio, dmax_norm);
  this->norm_queries = g_normalization(origin_queries);
  assert (kdata == norm_datas.row(0).size());
  assert (qdata == norm_queries.row(0).size());
  assert (origin_datas.rows() == norm_datas.rows());
  assert (origin_queries.rows() == norm_queries.rows());

  // expand k dimension into k+2m dimension
  this->ext_datas = g_index_cosine_extend(norm_datas, m);
  this->ext_queries = g_query_cosine_extend(norm_queries, m);
  int new_len = kdata + 2 * m;
  assert (new_len == ext_datas.row(0).size());
  assert (new_len == ext_queries.row(0).size());
  assert (origin_datas.rows() == ext_datas.rows());
  assert (origin_queries.rows() == ext_queries.rows());

  LshTester::init(ext_datas, ext_queries, rand_range, num_neighbours);
  return 1;
}

VectorXd CosineAlshTester::linear(const VectorXd &q, metric_func metric, int max_results) {
  multimap<double,long> candidates;
  double mips;

  for(int i=0;i<origin_datas.rows();i++) {
    mips = dot(q, origin_datas.row(i));
    candidates.insert(pair<double,long>(mips, i));
  }

  VectorXd res(max_results);
  multimap<double,long>::iterator iter=candidates.end();
  --iter;

  for(int j=0;j<max_results;j++,iter--) {
    res(j) = iter->second;
  }

  return res;
}

void CosineAlshTester::run(string type, vector<int> k_vec, vector<int> l_vec) {
  assert("cosine" == type);
  metric_func validate_metric = dot, compute_metric = L2Lsh_distance;

  MatrixXd exact_hits(origin_queries.rows(), num_neighbours);
  for(int i=0;i<origin_queries.rows();i++) {
    exact_hits.row(i) = linear(origin_queries.row(i), validate_metric, num_neighbours);
  }

  cout<<"=============================="<<endl;
  cout<<"CosineAlshTester "<<type<<" TEST: "<<endl;
  cout<<"L\tk\tacc\ttouch"<<endl;

  // concatenating more hash functions increases selectivity
  for(int i= 0;i<k_vec.size();i++) {
    int k = k_vec[i];
    LshWrapper lsh;
    // lsh.init(CONSINE_LSH, d, rand_range, k, 0);

    // using more hash tables increases recall
    for(int j=0;j<l_vec.size();j++) {
      int L = l_vec[j];
      lsh.init(CONSINE_LSH, d, rand_range, k, L);
      //lsh.resize(L);
      lsh.index(ext_datas);

      int correct = 0;
      for(int k=0;k<ext_queries.rows();k++) {
        VectorXd q = ext_queries.row(k);
        VectorXd hits = exact_hits.row(k);
        VectorXd lsh_hits = lsh.query(q, compute_metric, num_neighbours);
        // cout<<"lsh_hits"<<endl<<lsh_hits<<endl;
        // cout<<"hits "<<hits<<endl;
        if(lsh_hits == hits) {
            correct += 1;
        }
      }

      cout<< L << "\t" << k << "\t" << float(correct)/q_num << "\t" << float(lsh.get_avg_touched())/datas.rows() <<endl;
    }
  }
}

SimpleAlshTester::SimpleAlshTester() {
}

SimpleAlshTester::~SimpleAlshTester() {
}

int SimpleAlshTester::init(MatrixXd datas, MatrixXd queries, double rand_range, int num_neighbours, int m ) {
  int kdata = datas.row(0).size();
  int qdata = queries.row(0).size();
  assert(kdata == qdata);

  this->m = m = 1;
  //this->half_extend = g_ext_half(m);
  //storage original datas & queries. used for validation
  this->origin_datas = datas;
  this->origin_queries = queries;
  this->q_num = origin_queries.rows();
  // datas & queries transformation

  double dratio,dmax_norm;
  this->norm_datas = g_transformation(origin_datas, dratio, dmax_norm);
  this->norm_queries = g_normalization(origin_queries);
  assert (kdata == norm_datas.row(0).size());
  assert (qdata == norm_queries.row(0).size());
  assert (origin_datas.rows() == norm_datas.rows());
  assert (origin_queries.rows() == norm_queries.rows());

  // expand k dimension into k+2m dimension
  this->ext_datas = g_index_simple_extend(norm_datas, m);
  this->ext_queries = g_query_simple_extend(norm_queries, m);
  int new_len = kdata + 2 * m;
  assert (new_len == ext_datas.row(0).size());
  assert (new_len == ext_queries.row(0).size());
  assert (origin_datas.rows() == ext_datas.rows());
  assert (origin_queries.rows() == ext_queries.rows());

  LshTester::init(ext_datas, ext_queries, rand_range, num_neighbours);
  return 1;
}

VectorXd SimpleAlshTester::linear(const VectorXd &q, metric_func metric, int max_results) {
  multimap<double,long> candidates;

  double mips;
  for(int i=0;i<origin_datas.rows();i++) {
    mips = dot(q, origin_datas.row(i));
    candidates.insert(pair<double,long>(mips, i));
    // cout<<"mips "<<mips<<" index "<<i<<endl;
  }

  VectorXd res(max_results);
  multimap<double,long>::iterator iter=candidates.end();
  --iter;

  for(int j=0;j<max_results;j++,iter--) {
    res(j) = iter->second;
  }

  return res;
}

void SimpleAlshTester::run(string type, vector<int> k_vec, vector<int> l_vec) {
  assert("simple" == type);
  metric_func validate_metric = dot, compute_metric = L2Lsh_distance;

  MatrixXd exact_hits(origin_queries.rows(), num_neighbours);

  for(int i=0;i<origin_queries.rows();i++) {
    exact_hits.row(i) = linear(origin_queries.row(i), validate_metric, num_neighbours);
  }

  cout<<"=============================="<<endl;
  cout<<"SimpleAlshTester "<<type<<" TEST: "<<endl;
  cout<<"L\tk\tacc\ttouch"<<endl;

  // concatenating more hash functions increases selectivity
  for(int i= 0;i<k_vec.size();i++) {
    int k = k_vec[i];
    LshWrapper lsh;
    //lsh.init(CONSINE_LSH, d, rand_range, k, 0);

    // using more hash tables increases recall
    for(int j=0;j<l_vec.size();j++) {
      int L = l_vec[j];
      lsh.init(CONSINE_LSH, d, rand_range, k, L);
      //lsh.resize(L);
      lsh.index(ext_datas);

      int correct = 0;
      for(int k=0;k<ext_queries.rows();k++) {
        VectorXd q = ext_queries.row(k);
        VectorXd hits = exact_hits.row(k);
        clock_t start, end;
        start = clock();
        VectorXd lsh_hits = lsh.query(q, compute_metric, num_neighbours);
        end = clock();
        cout<<((double)(end - start) / CLOCKS_PER_SEC)<<endl;
        // cout<<"lsh_hits"<<endl<<lsh_hits<<endl;
        // cout<<"hits "<<hits<<endl;
        if(lsh_hits == hits) {
          correct += 1;
        }
      }

      cout<< L << "\t" << k << "\t" << float(correct)/q_num << "\t" << float(lsh.get_avg_touched())/datas.rows() <<endl;
    }
  }
}

LshTester *LshTesterFactory::createTester(string type, bool mips, MatrixXd datas, MatrixXd queries, int rand_num, int num_neighbours, int m) {
  LshTester *tester = NULL;

  if(false == mips) {
    tester = new LshTester();
    tester->init(datas, queries, rand_num, num_neighbours, m);
  }
  else if("l2" == type) {
    tester = new L2AlshTester();
    tester->init(datas, queries, rand_num, num_neighbours, m);
  }
  else if("cosine" == type) {
    tester = new CosineAlshTester();
    tester->init(datas, queries, rand_num, num_neighbours, m);
  }
  else if("simple" == type) {
    tester = new SimpleAlshTester();
    tester->init(datas, queries, rand_num, num_neighbours, 1);
  }
  else {
    cout<<"error LshTesterFactory type !"<<endl;
  }

  return tester;
}
