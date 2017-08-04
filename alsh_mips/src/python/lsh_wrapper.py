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
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from operator import itemgetter
from global_mgr import gol

from lsh import *

class LshWrapper:
    'LSH Wrapper'

    # lsh_type: lsh hash func type: in 'l2' 'cosine'
    # r: random float data
    # d: data vector size
    # k: number of hash func for each hashtable. default 2
    # L: number of hash tables for each hash type: default 2
    def __init__(self, lsh_type, d, r = 1.0, k = 2, L = 2):
        self.type = lsh_type
        self.d = d
        self.r = r
        self.k = k
        self.L = 0
        self.hash_tables = []
        self.resize(L)

        if True == gol.get_value('DEBUG'):
            print 'LshWrapper init:\ttype: ' + str(self.type) + '\td: ' + str(self.d) + '\tr: ' + str(self.r) + '\tk: ' + str(self.k) + '\tL: ' + str(self.L)

    def __get_hash_class__(self):
        try:
            if 'l2' == self.type:
                return L2Lsh
            elif 'cosine' == self.type:
                return CosineLsh
            else:
                raise ValueError
        except ValueError:
            print "LshWrapper type error: " + self.type
            return

    def __creat_ht__(self):
        try:
            if 'l2' == self.type:
                return L2Lsh(self.r, self.d)
            elif 'cosine' == self.type:
                return CosineLsh(self.d)
            else:
                raise ValueError
        except ValueError:
            print "LshWrapper type error: " + self.type
            return

    def resize(self, L):
        if True == gol.get_value('DEBUG'):
            print 'LshWrapper resize:\tnew L: ' + str(L) + '\tL: ' + str(self.L) + '\tk: ' + str(self.k)

        # shrink the number of hash tables to be used
        if L < self.L:
            self.hash_tables = self.hash_tables[:L]
        else:
            # initialise a new hash table for each hash function
            hash_funcs = [[self.__creat_ht__() for h in xrange(self.k)] for l in xrange(self.L, L)]
            self.hash_tables.extend([(g, defaultdict(lambda:[])) for g in hash_funcs])

            if True == gol.get_value('DEBUG'):
                print 'resize wrapper hashtable: '
                for ht, ct in self.hash_tables:
                    print 'ht: ' + str(type(ht)) + '\tct: ' + str(type(ct))
                    print ht
                    print ct
        self.L = L

    def hash(self, ht, data):
        #  used for combine
        return self.__get_hash_class__().combine([h.hash(data) for h in ht])

    def index(self, datas):
        # index the supplied datas
        self.datas = datas
        for ht, ct in self.hash_tables:
            for ix, p in enumerate(self.datas):
                ct[self.hash(ht, p)].append(ix)
        # reset stats
        self.tot_touched = 0
        self.num_queries = 0

        if True == gol.get_value('DEBUG'):
            print 'index wrapper hashtable: '
            for ht, ct in self.hash_tables:
                print 'ht: ' + str(type(ht)) + '\tct: ' + str(type(ct))
                print ht
                print ct

    def query(self, q, metric, max_results = 1):
        """
        triple_l = 3 * self.L
        if max_results > triple_l:
            max_results = triple_l
        elif
        """
        if 0 == max_results:
            max_results = 1

        # find the max_results closest indexed datas to q according to the supplied metric
        candidates = set()
        for ht, ct in self.hash_tables:
            matches = ct.get(self.hash(ht, q), [])
            candidates.update(matches)

        # update stats
        self.tot_touched += len(candidates)
        self.num_queries += 1

        # rerank candidates
        candidates = [(ix, metric(q, self.datas[ix])) for ix in candidates]
        candidates.sort(key=itemgetter(1))
        return candidates[:max_results]

    def get_avg_touched(self):
        # mean number of candidates inspected per query
        return self.tot_touched / self.num_queries

    def distance(self, u, v):
        return __get_hash_class__.distance(u, v)
