# coding=utf-8
import numpy as np
from abc import ABCMeta, abstractmethod

class Fitness(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    # 适应度计算
    @abstractmethod
    def fitnessFunc(self, decodedGene):
        return 0
        pass