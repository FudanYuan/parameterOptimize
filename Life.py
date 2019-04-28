# -*- encoding: utf-8 -*-

SCORE_NONE = 0


# 个体类
class Life(object):
    def __init__(self, gene=None):
        self._gene = gene
        self._fitness = SCORE_NONE
