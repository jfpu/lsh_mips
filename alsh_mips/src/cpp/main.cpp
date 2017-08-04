#include "lsh_tester.h"
#include <iostream>

void lsh_test(MatrixXd datas, MatrixXd queries, int rand_num, int num_neighbours, bool mips = false) {
  #if 0
  type = 'l2'
  tester = LshTesterFactory.createTester(type, mips, datas, queries, rand_num, num_neighbours)
  args = {
              'type':      type,
              'k_vec':     [1, 2, 4, 8],
              #'k_vec':    [2],
              'l_vec':     [2, 4, 8, 16, 32]
              #'l_vec':    [3]
          }
  tester.run(**args)

  type = 'cosine'
  tester = LshTesterFactory.createTester(type, mips, datas, queries, rand_num, num_neighbours)

  args = {
              'type':      type,
              'k_vec':    [1, 2, 4, 8],
              #'k_vec':     [2],
              'l_vec':    [2, 4, 8, 16, 32]
              #'l_vec':     [3]
          }
  tester.run(**args)
  #endif

  LshTesterFactory factory;
  // 'k_vec': [2], 'l_vec': [3]
  vector<int> k_vec = {1, 2, 4, 8}, l_vec = {2, 4, 8, 16, 32};

  #if 0
  string type = "l2";           // l2   cosine  simple
  LshTester *tester = factory.createTester(type, mips, datas, queries, rand_num, num_neighbours);
  tester->run(type, k_vec, l_vec);

  string type2 = "cosine";
  LshTester *tester2 = factory.createTester(type2, mips, datas, queries, rand_num, num_neighbours);
  tester2->run(type2, k_vec, l_vec);
  #endif

  string type3 = "simple";
  LshTester *tester3 = factory.createTester(type3, mips, datas, queries, rand_num, num_neighbours);
  tester3->run(type3, k_vec, l_vec);
}

void lshtest();
int main(int argc, char ** argv) {
  #if 0
  srand((unsigned int) time(0));
  ConsineLsh lsh;
  int d = 10;
  VectorXd vec = VectorXd::Random(d);
  std::cout << vec << std::endl << std::endl;
  lsh.init(0, d);
  std::cout << "hash: " << lsh.hash(vec) << std::endl;

  L2Lsh l2lsh;
  float r = 0.8;
  l2lsh.init(r, d);
  std::cout << "l2lsh: " << l2lsh.hash(vec) << std::endl;
  #endif

  // lshtest();
  // create a test dataset of vectors of non-negative integers
  int num_neighbours = 1;
  double radius = 0.3;
  int r_range = 10 * radius;

  int d = 2000;
  int xmin = 0;
  int xmax = 10;
  int num_datas = 10000;
  int num_queries = num_datas / 10;

  MatrixXd datas = MatrixXd::Random(num_datas, d);
  datas = datas * 5 + MatrixXd::Constant(num_datas, d, 5);

  MatrixXd queries = datas.topRows(num_queries);
  MatrixXd wave = MatrixXd::Random(num_queries,d) * radius;
  queries = queries + wave;

  #if 0
  cout<<"datas :"<<endl;
  cout<<datas<<endl;
  cout<<"queries :"<<endl;
  cout<<queries<<endl;

  queries = []
  for point in datas[:num_queries]:
      queries.append([x + random.uniform(-radius, radius) for x in point])

  lsh_test(datas, queries, r_range, num_neighbours)
  #endif

  // MIPS
  lsh_test(datas, queries, r_range, num_neighbours, true);

  return 0;
}

void lshtest() {
  MatrixXd datas(3, 3);
  int num = 1;
  for(int i=0;i<3;i++) {
    for(int j=0;j<3;j++) {
      datas(i,j) = num++;
    }
  }

  VectorXd u(3), v(3);
  v << 0.5, 0.3;  u << 1, 4, 3;
  vector<int> hashes = {1,-2,3};
  int m = 3;

  //cout<<"hash_combine"<< hash_combine(hashes)<<endl;  //ok
  cout<<"cosine_hash_combine"<< cosine_hash_combine(hashes)<<endl;  //ok
  //cout<<"dot"<< dot( u, v)<<endl;  //ok
  //cout<<"g_ext_norm"<< g_ext_norm(v,  m)<<endl;  //ok
  //cout<<"g_ext_half"<< g_ext_half( m)<<endl;  //ok
  //cout<<"g_ext_norm_cosine"<<  g_ext_norm_cosine(v,  m)<<endl;  //ok
  cout<<"g_ext_norm_simple"<<  g_ext_norm_simple(v, 1)<<endl; //ok
  //cout<<"g_ext_zero"<<  g_ext_zero( m)<<endl;  //ok
  //cout<<"g_index_extend"<<  g_index_extend(datas,  m)<<endl;  //ok
  //cout<<"g_query_extend"<<  g_query_extend(datas,  m)<<endl;  //ok
  //cout<<"g_index_cosine_extend"<<  g_index_cosine_extend(datas,  m)<<endl;  //ok
  //cout<<"g_query_cosine_extend"<<  g_query_cosine_extend(datas,  m)<<endl;  //ok
  cout<<"g_index_simple_extend"<<  g_index_simple_extend(datas,  1)<<endl;  //ok
  cout<<"g_query_simple_extend"<<  g_query_simple_extend(datas,  1)<<endl;  //ok
  //cout<<"g_max_norm"<<  g_max_norm(datas)<<endl;  //ok
  //cout<<"g_transformation"<<  g_transformation(datas, ratio, max_norm)<<endl;  //ok
  cout<<"g_normalization"<<  g_normalization(datas)<<endl; //ok
  //cout<<"L2Lsh_distance"<<  L2Lsh_distance(u, v)<<endl; //ok
  cout<<"ConsineLsh_distance"<<  ConsineLsh_distance(u, v)<<endl;  //ok
}
