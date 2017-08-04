# -*- coding: utf-8 -*-

"""
# Locality Sensitive Hashing for MIPS

## Reference:

- LSH:
[Locality-Sensitive Hashing Scheme Based on p-Stable Distributions](http://www.cs.princeton.edu/courses/archive/spr05/cos598E/bib/p253-datar.pdf)

- L2-ALSH:
[Asymmetric LSH (ALSH) for Sublinear Time Maximum Inner Product Search (MIPS)](https://papers.nips.cc/paper/5329-asymmetric-lsh-alsh-for-sublinear-time-maximum-inner-product-search-mips.pdf)

- Sign-ALSH(Cosine):
[Improved Asymmetric Locality Sensitive Hashing (ALSH) for Maximum Inner Product Search (MIPS)](http://auai.org/uai2015/proceedings/papers/96.pdf)

- Simple-ALSH(Cosine):
[On Symmetric and Asymmetric LSHs for Inner Product Search](https://arxiv.org/abs/1410.5518)

"""

import math
import random
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from operator import itemgetter
from global_mgr import gol

import numpy as np

# matrix inner product
def dot(u, v):
    # return sum(ux * vx for ux, vx in zip(u,v))
    return np.dot(u, v)

def g_ext_norm(vec, m):
    l2norm_square = dot(vec, vec)
    return [l2norm_square**(i+1) for i in xrange(m)]

def g_ext_half(m):
    return [0.5 for i in xrange(m)]

def g_ext_norm_cosine(vec, m):
    l2norm_square = dot(vec, vec)
    return [0.5 - l2norm_square**(i+1) for i in xrange(m)]

def g_ext_norm_simple(vec, m):
    l2norm_square = dot(vec, vec)
    return [np.sqrt(1 - l2norm_square) for i in xrange(m)]

def g_ext_zero(m):
    return [0 for i in xrange(m)]

# [x] => [x;    ||x||**2; ||x||**4; ...; ||x||**(2*m);    1/2; ...; 1/2(m)]
def g_index_extend(datas, m):
    return [(dv + g_ext_norm(dv, m) + g_ext_half(m)) for dv in datas]

# [x] => [x;    1/2; ...; 1/2(m);    ||x||**2; ||x||**4; ...; ||x||**(2*m)]
def g_query_extend(queries, m):
    return [(qv + g_ext_half(m) + g_ext_norm(qv, m)) for qv in queries]

# [x] => [x;    1/2 - ||x||**2; 1/2 - ||x||**4; ...; 1/2 - ||x||**(2*m);    0; ...; 0(m)]
def g_index_cosine_extend(datas, m):
    return [(dv + g_ext_norm_cosine(dv, m) + g_ext_zero(m)) for dv in datas]

# [x] => [x;    0; ...; 0(m);    1/2 - ||x||**2; 1/2 - ||x||**4; ...; 1/2 - ||x||**(2*m)]
def g_query_cosine_extend(queries, m):
    return [(qv + g_ext_zero(m) + g_ext_norm_cosine(qv, m)) for qv in queries]

# [x] => [x;    sqrt(1 - ||x||**2);    0]
def g_index_simple_extend(datas, m):
    assert (1 == m)
    return [(dv + g_ext_norm_simple(dv, m) + g_ext_zero(m)) for dv in datas]

# [x] => [x;    0;    sqrt(1 - ||x||**2)]
def g_query_simple_extend(queries, m):
    assert (1 == m)
    return [(qv + g_ext_zero(m) + g_ext_norm_simple(qv, m)) for qv in queries]

# get max norm for two-dimension list
def g_max_norm(datas):
    norm_list = [math.sqrt(dot(dd, dd)) for dd in datas]
    return max(norm_list)

# datas transformation. S(xi) = (U / M) * xi
def g_transformation(datas):
    # U < 1  ||xi||2 <= U <= 1. recommend for 0.83
    U = 0.83
    #U = 0.75
    max_norm = g_max_norm(datas)
    ratio = float(U / max_norm)
    return ratio, max_norm, [[ratio * dx for dx in dd] for dd in datas]

# normalization for each query
def g_normalization(queries):
    U = 0.83
    #U = 0.75
    norm_queries = []
    for qv in queries:
        norm = math.sqrt(dot(qv, qv))
        ratio = float(U / norm)
        norm_queries.append([ratio * qx for qx in qv])
    return norm_queries

class Hash:
    'hash base'

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def hash(self, vec):
        print "Hash.hash()"
        pass

    @staticmethod
    def distance(u, v):
        print "Hash.distance()"
        pass

    @staticmethod
    def combine(hashes):
        # print "Hash.combine()"
        return str(hashes)

class L2Lsh(Hash):
    'L2 LSH'

    # r: fixed size
    # d: data length
    # RandomData: random data vector
    def __init__(self, r, d):
        self.r = r
        self.d = d
        self.b = random.uniform(0, self.r)      # 0 < b < r
        self.Data = [random.gauss(0, 1) for i in xrange(self.d)]

        if True == gol.get_value('DEBUG'):
            print 'L2Lsh:\tr: ' + str(self.r) + '\td: ' + str(self.d) + '\tb: ' + str(self.b)
            print 'L2Lsh Data: '
            print self.Data

    def hash(self, vec):
        # use str() as a naive way of forming a single value
        return int((dot(vec, self.Data) + self.b) / self.r)

    # Euclidean Distance
    @staticmethod
    def distance(u, v):
        # print "L2Lsh.distance()"
        """
        u_na = np.array(u)
        v_na = np.array(v)
        return np.linalg.norm(u_na - v_na)
        """
        return sum((ux - vx)**2 for ux, vx in zip(u, v))**0.5

class CosineLsh(Hash):
    'Cosine LSH'

    def __init__(self, d):
        self.d = d
        self.Data = [random.gauss(0, 1) for i in xrange(self.d)]

        if True == gol.get_value('DEBUG'):
            print 'CosineLsh:\td: ' + str(self.d)
            print 'CosineLsh Data: '
            print self.Data

    def hash(self, vec):
        return int(0 < dot(vec, self.Data))

    # Cosine Distance
    @staticmethod
    def distance(u, v):
        # print "Cosine.distance()"
        return 1 - (dot(u, v) / (dot(u, u) * dot(v, v))**0.5)

    @staticmethod
    def combine(hashes):
        # print "Cosine.combine()"
        return sum(2**i if h > 0 else 0 for i, h in enumerate(hashes))
