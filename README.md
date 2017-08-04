# 位置敏感哈希LSH应用用计算向量最大内积和

---

> **Locality Sensitive Hashing:** 位置敏感哈希

---

## 1.LSH简介：

位置敏感哈希可以快速计算向量之间相似度（aNN问题）。与一般用于替代场景的线性扫描机树形相似性搜索算法比较而言，局部敏感哈希具备亚线性的计算时耗；
尤其在高维度向量计算中，位置敏感哈希具备性能上的优点；与此相反，应用位置敏感哈希是有一定程度的精度损失，
在一定程度上可以根据候选数据来选取哈希参数，有利于减少损失。

---

## 2.LSH实现MIPS理论基础:

- LSH:
[Locality-Sensitive Hashing Scheme Based on p-Stable Distributions](http://www.cs.princeton.edu/courses/archive/spr05/cos598E/bib/p253-datar.pdf)

- L2-ALSH:
[Asymmetric LSH (ALSH) for Sublinear Time Maximum Inner Product Search (MIPS)](https://papers.nips.cc/paper/5329-asymmetric-lsh-alsh-for-sublinear-time-maximum-inner-product-search-mips.pdf)

- Sign-ALSH(Cosine):
[Improved Asymmetric Locality Sensitive Hashing (ALSH) for Maximum Inner Product Search (MIPS)](http://auai.org/uai2015/proceedings/papers/96.pdf)

- Simple-ALSH(Cosine):
[On Symmetric and Asymmetric LSHs for Inner Product Search](https://arxiv.org/abs/1410.5518)

---

## 3.代码介绍：

### 3.1 alsh_mips

自研版本LSH及ALSH，开发了python和c++版本。包含了三种MIPS算法：
- L2-ALSH： 基于欧式距离的MIPS算法
- Sign-ALSH：基于余弦距离的MIPS算法1
- Simple-ALSH： 基于余弦距离的MIPS算法2

### 3.2 falconn源码分析

falconn源码是模板实现，封装较严格；以下仅以Hyperplane方法查询最近邻调用关系供参考：

HyperPlane stack:

```
falconn::wrapper::LSHNNTableWrapper::find_nearest_neighbor
falconn::core::NearestNeighborQuery::find_nearest_neighbor
falconn::core::StaticLSHTable::Query::get_unique_candidates
falconn::core::StaticLSHTable::Query::get_unique_candidates_internal
falconn::core::HashObjectQuery::get_probes_by_table
falconn::core::HyperplaneHashDense::get_multiplied_vector_all_tables
```

### 3.3 falconn_mips代码修改

基于开源LSH库 [FALCONN](https://github.com/falconn-lib/falconn/wiki) 实现MIPS方法。

**新增代码：**
- src/include/falconn/wrapper/cpp_mips_wrapper_impl.h
- falconn_mips/src/test/cpp_mips_wrapper_test.cc

**修改代码：**

- Makefile
- src/include/falconn/lsh_nn_table.h

### 3.4 falconn_mips功能及参数选优算法

新增源码内容：
- 1.实现密集向量MIPS功能：

开发基于余弦距离的Simple-ALSH数据处理功能及Sign-ALSH的数据处理功能。

- 2.实现密集向量MIPS参数选优功能：

k: 2, 6, 12, 14, 16, 18
L: 8, 16, 32, 48, 96, 200

根据LSH特性及性能，分别将k & L设置候选集。

**探测最优参数算法：**

A: 找寻第一个满足精度的k & L (初始位置由数据量决定)

```
计算k & L是否满足精度；如果满足，则标记k & L并返回成功；否则:
步进k & L:
如果k == L; ++L；否则(k < L)，++k, ++L;
最后，校验k & L是否到达边界；如果是，则返回失败
```

B: 如果A不成功，则代表无法找到满足的参数，返回识别；否则进入C

C: 在k & L下一步进范围内随机选取2组值排序：

```
-> L  L_1  L_2  next_L ->
-> k  k_1  k_2  next_k ->
```

构成5组候选集合：

```
candidate1: [k, L_1]
candidate1: [k_1, L_1]
candidate1: [k_1, L_2]
candidate1: [k_2, L_2]
candidate1: [k_2, next_L]
```

```
candidate1: [k, L]
```

计算5组候选集是否满足最低精度要求，并与原始参数一起进行排序，最终选取速度最快前三名中精度最高的参数返回。

---

## 4.注意事项：

### 4.1 DenseMipsParaProber接口返回值判断

请不要使用assert语句判断DenseMipsParaProber成员函数的返回值，尤其是对以下成员函数：

```
DenseMipsParaProber::validate_mips
DenseMipsParaProber::evaluate_optimum_paras
```

```使用assert时，已经出现过调用代码根本不会进入调用函数逻辑；即使将调用函数定义注释后，也不会出现任何编译错误!!!```

当将assert判断去除后，能正常执行函数内部逻辑

如：

```
DenseMipsParaProber<float> dmt(dim, 1000);
dmt.set_print_file();
// dmt.print_para();

AccuracySpeed as;
assert (true == dmt.evaluate_optimum_paras(as, params, mtype, 10, 0.96));
```

```
best paras:
         k: 0
         L: 0
         avg_time: 0
         accuracy: 0
```

使用assert判断evaluate_optimum_paras返回值时，发现该函数根本没有进入

**对调用代码做如下修改后，发现程序执行正常，参数返回正常**

```
AccuracySpeed as;
bool bret = dmt.evaluate_optimum_paras(as, params, mtype, 10, 0.96);
assert (true == bret);

```

返回信息正常

```
best paras:
         k: 14
         L: 41
         avg_time: 0.00103649
         accuracy: 0.97
```






