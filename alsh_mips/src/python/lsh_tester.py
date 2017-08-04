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

import random
import numpy as np
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from operator import itemgetter

from lsh import *
from lsh_wrapper import *

class LshTester():
    'LSH Test parameters'

    # datas: datas for build hash index
    # queries: query datas
    # rand_range: random range for norm
    # num_neighbours: query top num_neighbours
    def __init__(self, datas, queries, rand_range = 1.0, num_neighbours = 1):
        kdata = len(datas[0])
        qdata = len(queries[0])
        assert kdata == qdata

        self.d = kdata
        self.datas = datas
        self.queries = queries
        self.rand_range = rand_range
        self.num_neighbours = num_neighbours

        self.q_num = len(self.queries)

        if True == gol.get_value('DEBUG'):
            print 'LshTester:\td: ' + str(self.d) + '\trand_range: ' + str(self.rand_range) \
            + '\tnum_beighbours: ' + str(self.num_neighbours) + '\tq_num: ' + str(self.q_num)

            print 'LshTester datas:\n'
            for i, vd in enumerate(self.datas):
                print str(i) + '\t->\t' + str(vd)
            print 'LshTester queries:\n'
            for i, vq in enumerate(self.queries):
                print str(i) + '\t->\t' + str(vq)

    def linear(self, q, metric, max_results):
        """ brute force search by linear scan """
        candidates = [(ix, metric(q, p)) for ix, p in enumerate(self.datas)]
        x =  sorted(candidates, key=itemgetter(1))[:max_results]

        if True == gol.get_value('DEBUG'):
            print 'LshTester Sorted Validation:'
            for i, vx in enumerate(x):
                print str(i) + '\t->\t' + str(vx)

        return x

    # type: hash class type
    # k_vec: k vector for hash wrapper. default as [2]
    # l_vec: L vector for hash wrapper. default as [2]
    def run(self, type, k_vec = [2], l_vec = [2]):
        # set distance func object
        try:
            if 'l2' == type:
                metric = L2Lsh.distance
            elif 'cosine' == type:
                metric = CosineLsh.distance
            else:
                raise ValueError
        except ValueError:
            print 'LshTester: type error: ' + str(type)
            return

        exact_hits = [[ix for ix, dist in self.linear(q, metric, self.num_neighbours)] for q in self.queries]

        print '=============================='
        print type + ' TEST:'
        print 'L\tk\tacc\ttouch'

        # concatenating more hash functions increases selectivity
        for k in k_vec:
            lsh = LshWrapper(type, self.d, self.rand_range, k, 0)

            # using more hash tables increases recall
            for L in l_vec:
                lsh.resize(L)
                lsh.index(self.datas)

                correct = 0
                for q, hits in zip(self.queries, exact_hits):
                    lsh_hits = [ix for ix, dist in lsh.query(q, metric, self.num_neighbours)]
                    if lsh_hits == hits:
                        correct += 1

                    if True == gol.get_value('DEBUG'):
                        print 'Queried Sorted Validation:'
                        for j, vd in enumerate(lsh_hits):
                            print str(q) + '\t->\t' + str(vd)

                # print 'correct: ' + str(correct)
                print "{0}\t{1}\t{2}\t{3}".format(L, k, float(correct) / self.q_num, float(lsh.get_avg_touched()) / len(self.datas))
                # print "{0}\t{1}\t{2}\t".format(L, k, float(correct) / self.q_num)

class L2AlshTester(LshTester):
    'L2-ALSH for MIPS Test parameters'

    # datas: datas for build hash index
    # queries: query datas
    # rand_range: random range for norm
    # num_neighbours: query top num_neighbours
    # m: ALSH extend metrix length. default 3
    def __init__(self, datas, queries, rand_range = 1.0, num_neighbours = 1, m = 3):
        kdata = len(datas[0])
        qdata = len(queries[0])
        assert (kdata == qdata)

        self.m = m
        self.half_extend = g_ext_half(self.m)

        # storage original datas & queries. used for validation
        self.origin_datas = datas
        self.origin_queries = queries

        self.q_num = len(self.origin_queries)

        # datas & queries transformation
        dratio, dmax_norm, self.norm_datas = g_transformation(self.origin_datas)
        self.norm_queries = g_normalization(self.origin_queries)
        assert (kdata == len(self.norm_datas[0]))
        assert (qdata == len(self.norm_queries[0]))
        assert (len(self.origin_datas) == len(self.norm_datas))
        assert (len(self.origin_queries) == len(self.norm_queries))

        # expand k dimension into k+2m dimension
        self.ext_datas = g_index_extend(self.norm_datas, self.m)
        self.ext_queries = g_query_extend(self.norm_queries, self.m)
        new_len = kdata + 2 * m
        assert (new_len == len(self.ext_datas[0]))
        assert (new_len == len(self.ext_queries[0]))
        assert (len(self.origin_datas) == len(self.ext_datas))
        assert (len(self.origin_queries) == len(self.ext_queries))

        if True == gol.get_value('DEBUG'):
            print 'L2AlshTester:\td: ' + str(kdata) + '\trand_range: ' + str(rand_range) \
            + '\tnum_beighbours: ' + str(num_neighbours) + '\tq_num: ' + str(self.q_num)

            print '\tdatas_ratio: ' + str(dratio) + '\tdmax_norm: ' + str(dmax_norm) + '\n'
            print 'L2AlshTester origin_datas:\n'
            for i, vd in enumerate(self.origin_datas):
                print str(i) + '\t->\t' + str(vd)
            print 'L2AlshTester origin_queries:\n'
            for i, vq in enumerate(self.origin_queries):
                print str(i) + '\t->\t' + str(vq)

            print 'L2AlshTester norm_datas:\n'
            for i, vd in enumerate(self.norm_datas):
                print str(i) + '\t->\t' + str(vd)
            print 'L2AlshTester norm_queries:\n'
            for i, vq in enumerate(self.norm_queries):
                print str(i) + '\t->\t' + str(vq)

        LshTester.__init__(self, self.ext_datas, self.ext_queries, rand_range, num_neighbours)

    # MIPS
    def linear(self, q, metric, max_results):
        """ brute force search by linear scan """
        # print 'MipsLshTester linear:'
        candidates = [(ix, dot(q, p)) for ix, p in enumerate(self.origin_datas)]
        x = sorted(candidates, key=itemgetter(1), reverse=True)[:max_results]

        if True == gol.get_value('DEBUG'):
            print 'L2AlshTester Sorted Validation:'
            for i, vx in enumerate(x):
                print str(i) + '\t->\t' + str(vx)

        return x

    # type: hash class type
    # k_vec: k vector for hash wrapper. default as [2]
    # l_vec: L vector for hash wrapper. default as [2]
    def run(self, type, k_vec = [2], l_vec = [2]):
        if True == gol.get_value('DEBUG'):
            print 'L2AlshTester Run:'

        try:
            if 'l2' == type:
                pass
            else:
                raise ValueError
        except ValueError:
            print 'L2AlshTester: type error: ' + str(type)
            return

        validate_metric = dot
        compute_metric = L2Lsh.distance

        exact_hits = [[ix for ix, dist in self.linear(q, validate_metric, self.num_neighbours)] for q in self.origin_queries]

        print '=============================='
        print 'L2AlshTester ' + type + ' TEST:'
        print 'L\tk\tacc\ttouch'

        # concatenating more hash functions increases selectivity
        for k in k_vec:
            lsh = LshWrapper(type, self.d, self.rand_range, k, 0)

            # using more hash tables increases recall
            for L in l_vec:
                lsh.resize(L)
                lsh.index(self.ext_datas)

                correct = 0
                for q, hits in zip(self.ext_queries, exact_hits):
                    lsh_hits = [ix for ix, dist in lsh.query(q, compute_metric, self.num_neighbours)]
                    if lsh_hits == hits:
                        correct += 1

                    if True == gol.get_value('DEBUG'):
                        print 'Queried Sorted Validation:'
                        for j, vd in enumerate(lsh_hits):
                            print str(q) + '\t->\t' + str(vd)

                # print 'correct: ' + str(correct)
                print "{0}\t{1}\t{2}\t{3}".format(L, k, float(correct) / self.q_num, float(lsh.get_avg_touched()) / len(self.datas))
                # print "{0}\t{1}\t{2}\t".format(L, k, float(correct) / self.q_num)

class CosineAlshTester(LshTester):
    'Cosine-ALSH for MIPS Test parameters'

    # datas: datas for build hash index
    # queries: query datas
    # rand_range: random range for norm
    # num_neighbours: query top num_neighbours
    # m: ALSH extend metrix length. default 3
    def __init__(self, datas, queries, rand_range = 1.0, num_neighbours = 1, m = 3):
        kdata = len(datas[0])
        qdata = len(queries[0])
        assert (kdata == qdata)

        self.m = m
        self.half_extend = g_ext_half(self.m)

        # storage original datas & queries. used for validation
        self.origin_datas = datas
        self.origin_queries = queries

        self.q_num = len(self.origin_queries)

        # datas & queries transformation
        dratio, dmax_norm, self.norm_datas = g_transformation(self.origin_datas)
        self.norm_queries = g_normalization(self.origin_queries)
        assert (kdata == len(self.norm_datas[0]))
        assert (qdata == len(self.norm_queries[0]))
        assert (len(self.origin_datas) == len(self.norm_datas))
        assert (len(self.origin_queries) == len(self.norm_queries))

        # expand k dimension into k+2m dimension
        self.ext_datas = g_index_cosine_extend(self.norm_datas, self.m)
        self.ext_queries = g_query_cosine_extend(self.norm_queries, self.m)
        new_len = kdata + 2 * m
        assert (new_len == len(self.ext_datas[0]))
        assert (new_len == len(self.ext_queries[0]))
        assert (len(self.origin_datas) == len(self.ext_datas))
        assert (len(self.origin_queries) == len(self.ext_queries))

        if True == gol.get_value('DEBUG'):
            print 'CosineAlshTester:\td: ' + str(kdata) + '\trand_range: ' + str(rand_range) \
            + '\tnum_beighbours: ' + str(num_neighbours) + '\tq_num: ' + str(self.q_num)

            print '\tdatas_ratio: ' + str(dratio) + '\tdmax_norm: ' + str(dmax_norm) + '\n'
            print 'CosineAlshTester origin_datas:\n'
            for i, vd in enumerate(self.origin_datas):
                print str(i) + '\t->\t' + str(vd)
            print 'CosineAlshTester origin_queries:\n'
            for i, vq in enumerate(self.origin_queries):
                print str(i) + '\t->\t' + str(vq)

            print 'CosineAlshTester norm_datas:\n'
            for i, vd in enumerate(self.norm_datas):
                print str(i) + '\t->\t' + str(vd)
            print 'CosineAlshTester norm_queries:\n'
            for i, vq in enumerate(self.norm_queries):
                print str(i) + '\t->\t' + str(vq)

        LshTester.__init__(self, self.ext_datas, self.ext_queries, rand_range, num_neighbours)

    # MIPS
    def linear(self, q, metric, max_results):
        """ brute force search by linear scan """
        # print 'MipsLshTester linear:'
        candidates = [(ix, dot(q, p)) for ix, p in enumerate(self.origin_datas)]
        x = sorted(candidates, key=itemgetter(1), reverse=True)[:max_results]

        if True == gol.get_value('DEBUG'):
            print 'CosineAlshTester Sorted Validation:'
            for i, vx in enumerate(x):
                print str(i) + '\t->\t' + str(vx)

        return x

    # type: hash class type
    # k_vec: k vector for hash wrapper. default as [2]
    # l_vec: L vector for hash wrapper. default as [2]
    def run(self, type, k_vec = [2], l_vec = [2]):
        if True == gol.get_value('DEBUG'):
            print 'CosineAlshTester Run:'

        try:
            if 'cosine' == type:
                pass
            else:
                raise ValueError
        except ValueError:
            print 'CosineAlshTester: type error: ' + str(type)
            return

        validate_metric = dot
        compute_metric = L2Lsh.distance

        exact_hits = [[ix for ix, dist in self.linear(q, validate_metric, self.num_neighbours)] for q in self.origin_queries]

        print '=============================='
        print 'CosineAlshTester ' + type + ' TEST:'
        print 'L\tk\tacc\ttouch'

        # concatenating more hash functions increases selectivity
        for k in k_vec:
            lsh = LshWrapper(type, self.d, self.rand_range, k, 0)

            # using more hash tables increases recall
            for L in l_vec:
                lsh.resize(L)
                lsh.index(self.ext_datas)

                correct = 0
                for q, hits in zip(self.ext_queries, exact_hits):
                    lsh_hits = [ix for ix, dist in lsh.query(q, compute_metric, self.num_neighbours)]
                    if lsh_hits == hits:
                        correct += 1

                    if True == gol.get_value('DEBUG'):
                        print 'Queried Sorted Validation:'
                        for j, vd in enumerate(lsh_hits):
                            print str(q) + '\t->\t' + str(vd)

                # print 'correct: ' + str(correct)
                print "{0}\t{1}\t{2}\t{3}".format(L, k, float(correct) / self.q_num, float(lsh.get_avg_touched()) / len(self.datas))
                # print "{0}\t{1}\t{2}\t".format(L, k, float(correct) / self.q_num)

class SimpleAlshTester(LshTester):
    'Simple-ALSH for MIPS Test parameters'

    # datas: datas for build hash index
    # queries: query datas
    # rand_range: random range for norm
    # num_neighbours: query top num_neighbours
    def __init__(self, datas, queries, rand_range = 1.0, num_neighbours = 1):
        kdata = len(datas[0])
        qdata = len(queries[0])
        assert (kdata == qdata)

        # m: ALSH extend metrix length. default 1
        self.m = 1
        self.half_extend = g_ext_half(self.m)

        # storage original datas & queries. used for validation
        self.origin_datas = datas
        self.origin_queries = queries

        self.q_num = len(self.origin_queries)

        # datas & queries transformation
        dratio, dmax_norm, self.norm_datas = g_transformation(self.origin_datas)
        self.norm_queries = g_normalization(self.origin_queries)
        assert (kdata == len(self.norm_datas[0]))
        assert (qdata == len(self.norm_queries[0]))
        assert (len(self.origin_datas) == len(self.norm_datas))
        assert (len(self.origin_queries) == len(self.norm_queries))

        # expand k dimension into k+2m dimension
        self.ext_datas = g_index_simple_extend(self.norm_datas, self.m)
        self.ext_queries = g_query_simple_extend(self.norm_queries, self.m)
        new_len = kdata + 2 * self.m
        assert (new_len == len(self.ext_datas[0]))
        assert (new_len == len(self.ext_queries[0]))
        assert (len(self.origin_datas) == len(self.ext_datas))
        assert (len(self.origin_queries) == len(self.ext_queries))

        if True == gol.get_value('DEBUG'):
            print 'SimpleAlshTester:\td: ' + str(kdata) + '\trand_range: ' + str(rand_range) \
            + '\tnum_beighbours: ' + str(num_neighbours) + '\tq_num: ' + str(self.q_num)

            print '\tdatas_ratio: ' + str(dratio) + '\tdmax_norm: ' + str(dmax_norm) + '\n'
            print 'SimpleAlshTester origin_datas:\n'
            for i, vd in enumerate(self.origin_datas):
                print str(i) + '\t->\t' + str(vd)
            print 'SimpleAlshTester origin_queries:\n'
            for i, vq in enumerate(self.origin_queries):
                print str(i) + '\t->\t' + str(vq)

            print 'SimpleAlshTester norm_datas:\n'
            for i, vd in enumerate(self.norm_datas):
                print str(i) + '\t->\t' + str(vd)
            print 'SimpleAlshTester norm_queries:\n'
            for i, vq in enumerate(self.norm_queries):
                print str(i) + '\t->\t' + str(vq)

        LshTester.__init__(self, self.ext_datas, self.ext_queries, rand_range, num_neighbours)

    # MIPS
    def linear(self, q, metric, max_results):
        """ brute force search by linear scan """
        # print 'MipsLshTester linear:'
        candidates = [(ix, dot(q, p)) for ix, p in enumerate(self.origin_datas)]
        x = sorted(candidates, key=itemgetter(1), reverse=True)[:max_results]

        if True == gol.get_value('DEBUG'):
            print 'SimpleAlshTester Sorted Validation:'
            for i, vx in enumerate(x):
                print str(i) + '\t->\t' + str(vx)

        return x

    # type: hash class type
    # k_vec: k vector for hash wrapper. default as [2]
    # l_vec: L vector for hash wrapper. default as [2]
    def run(self, type, k_vec = [2], l_vec = [2]):
        if True == gol.get_value('DEBUG'):
            print 'SimpleAlshTester Run:'

        try:
            if 'simple' == type:
                pass
            else:
                raise ValueError
        except ValueError:
            print 'SimpleAlshTester: type error: ' + str(type)
            return

        validate_metric = dot
        compute_metric = L2Lsh.distance

        exact_hits = [[ix for ix, dist in self.linear(q, validate_metric, self.num_neighbours)] for q in self.origin_queries]

        print '=============================='
        print 'SimpleAlshTester ' + type + ' TEST:'
        print 'L\tk\tacc\ttouch'

        # concatenating more hash functions increases selectivity
        for k in k_vec:
            lsh = LshWrapper('cosine', self.d, self.rand_range, k, 0)

            # using more hash tables increases recall
            for L in l_vec:
                lsh.resize(L)
                lsh.index(self.ext_datas)

                correct = 0
                for q, hits in zip(self.ext_queries, exact_hits):
                    lsh_hits = [ix for ix, dist in lsh.query(q, compute_metric, self.num_neighbours)]
                    if lsh_hits == hits:
                        correct += 1

                    if True == gol.get_value('DEBUG'):
                        print 'Queried Sorted Validation:'
                        for j, vd in enumerate(lsh_hits):
                            print str(q) + '\t->\t' + str(vd)

                # print 'correct: ' + str(correct)
                print "{0}\t{1}\t{2}\t{3}".format(L, k, float(correct) / self.q_num, float(lsh.get_avg_touched()) / len(self.datas))
                # print "{0}\t{1}\t{2}\t".format(L, k, float(correct) / self.q_num)

class LshTesterFactory():
    'LSH Test factory'

    @staticmethod
    # type: l2 & cosine
    # mips: True for ALSH
    def createTester(type, mips, datas, queries, rand_num, num_neighbours):
        try:
            if False == mips:
                return LshTester(datas, queries, rand_num, num_neighbours)

            if 'l2' == type:
                return L2AlshTester(datas, queries, rand_num, num_neighbours)
            elif 'cosine' == type:
                return CosineAlshTester(datas, queries, rand_num, num_neighbours)
            elif 'simple' == type:
                return SimpleAlshTester(datas, queries, rand_num, num_neighbours)
            else:
                raise ValueError
        except ValueError:
            print "LshTesterFactory type error: " + type
            return
