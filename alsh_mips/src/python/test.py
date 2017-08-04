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

from lsh import *
from lsh_wrapper import *
from lsh_tester import *
from global_mgr import gol

def lsh_test(datas, queries, rand_num, num_neighbours, mips = False):

    """

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

    """

    type = 'simple'
    tester = LshTesterFactory.createTester(type, mips, datas, queries, rand_num, num_neighbours)

    args = {
                'type':      type,
                'k_vec':    [1, 2, 4, 8],
                #'k_vec':     [2],
                'l_vec':    [2, 4, 8, 16, 32]
                #'l_vec':     [3]
            }
    tester.run(**args)

if __name__ == "__main__":
    gol._init()
    #gol.set_value('DEBUG', True)

    # create a test dataset of vectors of non-negative integers
    num_neighbours = 1
    radius = 0.3
    r_range = 10 * radius

    d = 100
    xmin = 0
    xmax = 10
    num_datas = 1000
    num_queries = num_datas / 10
    datas = [[random.randint(xmin, xmax) for i in xrange(d)] for j in xrange(num_datas)]

    #"""
    queries = []
    for point in datas[:num_queries]:
        queries.append([x + random.uniform(-radius, radius) for x in point])
    #"""

    #lsh_test(datas, queries, r_range, num_neighbours)

    # MIPS
    lsh_test(datas, queries, r_range, num_neighbours, True)

