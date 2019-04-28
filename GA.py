# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import fsolve
import random
import time
import os
import matplotlib.pyplot as plt
import logging

from Life import Life
from Fitness import Fitness

random.seed(10)

logging.basicConfig(level=logging.INFO,  # 控制台打印的日志级别
                    filename='./logs/out_%s.log' % time.strftime("%Y%m%d%H%M", time.localtime(time.time())),
                    filemode='w',  # 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志, a是追加模式，默认如果不写的话，就是追加模式
                    format='[line:%(lineno)d] - %(levelname)s: %(message)s'
                    # 日志格式 %(asctime)s - %(pathname)s[line:%(lineno)d] -
                    )


# 遗传算法类
class GA(object):
    def __init__(self, populationSize=100, binaryEncode=True, geneLength=None, boundaryList=None, delta=0.01,
                 fitnessClass=None, crossoverRate=0.7, crossoverOneRate=0.5, crossoverTwoRate=0.5,
                 mutationRate=0.05, mutationLocationRate=0.5, mutationRotateRate=0.5, fitnessThreshold=0.9,
                 similarityThreshold=10, CF=10, punish=0.01, showBestFitness=True, showTotalFitness=False,
                 earlyStopRoundThreshold=100):
        self._populationSize = populationSize  # 种群数量
        self._binaryEncode = binaryEncode  # 基因编码方式
        self._geneLength = geneLength  # 基因长度
        self._boundaryList = boundaryList  # 变量区间列表
        self._delta = delta  # 变量精度
        self._encodeLengths = None  # 基因编码长度
        self._fitnessClass = fitnessClass  # 适配函数
        self._crossoverRate = crossoverRate  # 交叉概率
        self._crossoverOneRate = crossoverOneRate  # 单点交叉概率
        self._crossoverTwoRate = crossoverTwoRate  # 双点交叉概率
        self._mutationRate = mutationRate  # 突变概率
        self._mutationLocationRate = mutationLocationRate  # 位置变异概率
        self._mutationRotateRate = mutationRotateRate  # 旋转变异概率
        self._bestLife = None  # 一代中最好的个体
        self._fitnessThreshold = fitnessThreshold  # 适应度阈值，超过该阈值提前结束
        self._generation = 1  # 初始化，为第一代
        self._crossoverCount = 0  # 初始化，交叉次数是0
        self._mutationCount = 0  # 初始化，变异次数是0
        self._totalFitness = 0.0  # 适配值之和，用于选择时计算概率
        self._earlyStop = False  # 提前终止
        self._CF = CF  # 排挤因子
        self._punish = punish  # 惩罚因子
        self._similarityThresholdThreshold = similarityThreshold  # 相似度
        self._bestFitnessHistory = []  # 最佳适应度历史值
        self._totalFitnessHistory = []  # 总适应度历史值
        self._showBestFitness = showBestFitness  # 是否可视化最佳适应度
        self._showTotalFitness = showTotalFitness  # 是否可视化最佳适应度
        self._earlyStopRoundThreshold = earlyStopRoundThreshold  # 当适应度不再增加超过_earlyStopRoundThreshold时，提前终止
        self._round = 0  # 适应度不再增加的轮数
        self.initPopulation()  # 初始化种群

    # 初始化种群
    def initPopulation(self):
        logging.debug('init population')
        self._population = []
        if self._binaryEncode:  # 将变量进行二进制编码
            if self._boundaryList is None:
                raise ValueError("boundaryList must be configured!")
            # 获取编码长度列表
            self._encodeLengths = self.getEncodedLengths()
            # 基因长度为每个变量编码长度之和
            self._geneLength = np.sum(self._encodeLengths)
            # 随机化初始种群为0
            for i in range(self._populationSize):
                # 随机生成基因
                gene = np.random.randint(0, 2, self._geneLength)
                # 生成个体，并计算适应度
                life = Life(gene)
                life._fitness = self._fitnessClass.fitnessFunc(self.decodedOneGene(gene))
                # 把生成个体添加至种群集合里
                self._population.append(life)
        else:  # 编码方式为[0, 1, 2, ..., self._geneLength-1]
            if self._geneLength is None:
                raise ValueError("geneLength must be configured!")
            for i in range(self._populationSize):
                gene = np.array(range(self._geneLength))
                # 将基因乱序
                random.shuffle(gene)
                # 生成个体，并计算适应度
                life = Life(gene)
                life._fitness = self._fitnessClass.fitnessFunc(gene)
                # 把生成个体添加至种群集合里
                self._population.append(life)
        # 根据适应度值对种群的个体降序排列，并记录最佳个体和最佳适应度
        self._population = self.sortPopulation(self._population)
        self._bestLife = self._population[0]
        self._bestFitnessHistory.append(self._bestLife._fitness)
        # 计算总适应度
        self._totalFitness = self.calTotalFitness(self._population)
        self._totalFitnessHistory.append(self._totalFitness)

    # 根据解的精度确定基因(gene)的长度
    # 需要根据决策变量的上下边界来确定
    def getEncodedLengths(self):
        if self._boundaryList is None:
            raise ValueError("boundaryList must be configured!")
        # 每个变量的编码长度
        encodeLengths = []
        for i in self._boundaryList:
            lower = i[0]
            upper = i[1]
            # lamnda 代表匿名函数f(x)=0, 50代表搜索的初始解
            res = fsolve(lambda x: ((upper - lower) * 1 / self._delta) - 2 ** x - 1, 50)
            length = int(np.floor(res[0]))
            encodeLengths.append(length)
        return encodeLengths

    # 基因解码得到表现型的解
    def decodedGenes(self, population):
        if self._encodeLengths is None:
            raise ValueError("encodeLengths must be configured!")
        populationSize = len(population)
        variables = len(self._encodeLengths)
        decodeGenes = np.zeros((populationSize, variables))
        for k, life in enumerate(population):
            gene = life._gene.tolist()
            decodeGenes[k] = self.decodedOneGene(gene)
        return decodeGenes

    # 解码某一基因序列
    def decodedOneGene(self, encodeGene):
        decodeGene = np.zeros(len(self._encodeLengths))
        start = 0
        for index, length in enumerate(self._encodeLengths):
            # 将一个染色体进行拆分，得到染色体片段
            power = length - 1
            # 解码得到的10进制数字
            demical = 0
            for i in range(start, length + start):
                demical += encodeGene[i] * (2 ** power)
                power -= 1
            lower = self._boundaryList[index][0]
            upper = self._boundaryList[index][1]
            decodedValue = lower + demical * (upper - lower) / (2 ** length - 1)
            decodeGene[index] = decodedValue
            # 下一段染色体的编码
            start += length
        return decodeGene

    # 将种群按照适应度排序，默认倒序排列
    def sortPopulation(self, population, reverse=True):
        return sorted(population, key=lambda life: life._fitness, reverse=reverse)

    # 计算种群的总适应度
    def calTotalFitness(self, population):
        fitnessList = []
        for life in population:
            fitnessList.append(life._fitness)
        return np.sum(np.array(fitnessList))

    # 计算海明距离
    def hammingDist(self, s1, s2):
        assert len(s1) == len(s2)
        return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])

    # 小生境淘汰
    def smallHabitatElimination(self, population):
        count = 0
        popSize = len(population)
        for i in range(popSize - 1):
            for j in range(i + 1, popSize):
                gene1 = population[i]._gene
                gene2 = population[j]._gene
                # 计算海明距离
                distance = self.hammingDist(gene1, gene2)
                # 若距离小于阈值，则惩罚其中适应度值较差的个体。
                if distance < self._similarityThresholdThreshold:
                    count += 1
                    if population[i]._fitness < population[j]._fitness:
                        logging.debug("个体 #%s 基因组被惩罚：" % i)
                        logging.debug("被惩罚前分数: %f" % (population[i]._fitness))
                        population[i]._fitness = self._punish
                    else:
                        logging.debug("个体 #%s 基因组被惩罚：" % j)
                        logging.debug("被惩罚前分数: %f" % (population[j]._fitness))
                        population[j]._fitness = self._punish
        return population

    # 评估，计算每一个个体的适配值
    def evaluation(self, population):
        # 适配值之和，用于选择时计算概率
        totalFitness = 0.0
        bestLife = population[0]
        if self._binaryEncode:
            decodedGenes = self.decodedGenes(population)
            bestLife._fitness = self._fitnessClass.fitnessFunc(decodedGenes[0])
            for i in range(1, len(decodedGenes)):
                decodedGene = decodedGenes[i]
                life = self._population[i]
                life._fitness = self._fitnessClass.fitnessFunc(decodedGene)
                totalFitness += life._fitness
                # 如果新基因的适配值大于原先的best基因，就更新best基因
                if bestLife._fitness < life._fitness:
                    bestLife = life
        else:
            for life in population:
                life._fitness = self._fitnessClass.fitnessFunc(life._gene)
                totalFitness += life._fitness
                # 如果新基因的适配值大于原先的best基因，就更新best基因
                if bestLife < life._fitness:
                    bestLife = life
        return totalFitness, bestLife

    # 选择一个个体
    def selection(self):
        # 产生0到（适配值之和）之间的任何一个实数
        r = random.uniform(0, self._totalFitness)
        for life in self._population:
            r -= life._fitness
            if r <= 0:
                return life
        raise Exception("选择错误", self._totalFitness)

    # 交叉
    def crossover(self, parent1, parent2):
        # 交叉类型概率
        rate = random.random()
        newGene1 = []
        newGene2 = []
        variablesCount = len(self._boundaryList)  # 变量个数
        # 若该概率小于单点交叉概率，则进行单点交叉，否则，进行双点交叉
        if rate <= self._crossoverOneRate:  # 单点交叉
            startIndex = 0
            for i in range(variablesCount):
                index = random.randint(0, self._encodeLengths[i])
                endIndex = startIndex + self._encodeLengths[i]
                parentGene1 = parent1._gene[startIndex:endIndex]
                parentGene2 = parent2._gene[startIndex:endIndex]
                newGene1.extend(parentGene1[:index].tolist() + parentGene2[index:].tolist())
                newGene2.extend(parentGene2[:index].tolist() + parentGene1[index:].tolist())
                startIndex = self._encodeLengths[i]
        else:  # 双点交叉
            startIndex = 0
            for i in range(variablesCount):
                index1 = random.randint(0, self._encodeLengths[i] - 1)
                index2 = random.randint(index1, self._encodeLengths[i] - 1)
                endIndex = startIndex + self._encodeLengths[i]

                parentGene1 = parent1._gene[startIndex:endIndex]
                parentGene2 = parent2._gene[startIndex:endIndex]

                newGene1.extend(
                    parentGene1[:index1].tolist() + parentGene2[index1:index2].tolist() + parentGene1[index2:].tolist())
                newGene2.extend(
                    parentGene2[:index1].tolist() + parentGene1[index1:index2].tolist() + parentGene2[index2:].tolist())
                startIndex = self._encodeLengths[i]
        self._crossoverCount += 1
        return np.array(newGene1), np.array(newGene2)

    # 突变
    def mutation(self, gene):
        newGene = []
        variablesCount = len(self._boundaryList)  # 变量个数

        # 位置突变概率
        locationRate = random.random()
        if locationRate < self._mutationLocationRate:
            startIndex = 0
            for i in range(variablesCount):
                # 相当于取得0到self._geneLength - 1之间的一个数，包括0和self._geneLength - 1
                index1 = random.randint(0, self._encodeLengths[i] - 1)
                index2 = random.randint(0, self._encodeLengths[i] - 1)
                while index1 == index2:
                    index2 = random.randint(0, self._encodeLengths[i] - 1)
                    pass
                endIndex = startIndex + self._encodeLengths[i]
                genePart = gene[startIndex:endIndex]
                genePart[index1], genePart[index2] = genePart[index2], genePart[index1]
                newGene.extend(genePart)
                startIndex = self._encodeLengths[i]
            newGene = np.array(newGene)
            """
            # 整体突变
            index1 = random.randint(0, self._geneLength - 1)
            index2 = random.randint(0, self._geneLength - 1)
            newGene[index1], newGene[index2] = newGene[index2], newGene[index1]
            """
        else:  # 旋转突变
            startIndex = 0
            for i in range(variablesCount):
                # 相当于取得0到self._geneLength - 1之间的一个数，包括0和self._geneLength - 1
                index1 = random.randint(0, self._encodeLengths[i] - 1)
                index2 = random.randint(0, self._encodeLengths[i] - 1)
                while index1 == index2:
                    index2 = random.randint(0, self._encodeLengths[i] - 1)
                    pass
                #  保证index1 < index2
                if index1 > index2:
                    tmp = index1
                    index1 = index2
                    index2 = tmp
                endIndex = startIndex + self._encodeLengths[i]
                genePart = gene[startIndex:endIndex]
                genePart[index1:index2] = 1 - genePart[index1:index2]
                newGene.extend(genePart)
                startIndex = self._encodeLengths[i]
            newGene = np.array(newGene)
            """
            # 整体突变
            index1 = random.randint(0, self._geneLength - 1)
            index2 = random.randint(0, self._geneLength - 1)
            while index1 == index2:
                index2 = random.randint(0, self._geneLength - 1)
                pass
            #  保证index1 < index2
            if index1 > index2:
                tmp = index1
                index1 = index2
                index2 = tmp
            newGene[index1:index2] = 1 - newGene[index1:index2]
            """
        # 突变次数加1
        self._mutationCount += 1
        return newGene

    # 产生新后代
    def getNewChild(self):
        logging.debug('select 2 parent lives')
        parent1 = self.selection()
        parent2 = self.selection()
        while self.hammingDist(parent1._gene, parent2._gene) == 0:
            parent2 = self.selection()
            pass
        variableCount = len(self._boundaryList)
        encodeLength = int(self._geneLength / variableCount)
        # logging.debug('parent1: \n%s' % self.decodedOneGene(parent1._gene))
        # logging.debug('parent2: \n%s' % parent1._gene.reshape((variableCount, encodeLength)))
        # logging.debug('parent2: \n%s' % self.decodedOneGene(parent2._gene))
        # logging.debug('parent2: \n%s' % parent2._gene.reshape((variableCount, encodeLength)))
        logging.debug(
            'hamming distance between parent1 and parent2: %d' % self.hammingDist(parent1._gene, parent2._gene))

        # 按概率交叉
        rate = random.random()
        if rate < self._crossoverRate:
            logging.debug('crossover')
            # 交叉
            gene1, gene2 = self.crossover(parent1, parent2)
            logging.debug(
                'hamming distance between gene1 and gene2 after crossover: %d' % self.hammingDist(gene1, gene2))
        else:
            gene1, gene2 = parent1._gene, parent2._gene

        # 按概率突变
        rate = random.random()
        if rate < self._mutationRate:
            logging.debug('mutation')
            gene1, gene2 = self.mutation(gene1), self.mutation(gene2)
            logging.debug(
                'hamming distance between gene1 and gene2 after mutation: %d' % self.hammingDist(gene1, gene2))

        # 计算子代适应度
        life1 = Life(gene1)
        life2 = Life(gene2)
        life1._fitness = self._fitnessClass.fitnessFunc(self.decodedOneGene(gene1))
        life2._fitness = self._fitnessClass.fitnessFunc(self.decodedOneGene(gene2))
        return life1, life2

    # 产生下一代
    def getNewGeneration(self):
        # 记录上一代的最佳适应度得分
        bestFitnessCurrent = self._bestLife._fitness

        # 更新种群
        # 合并父代记忆的个体集合与子代个体组成新的种群,
        # 大小为self._populationSize + self._populationSize / CF
        newPopulation = self._population[:int(self._populationSize / self._CF)]

        # debug
        fitnessList = []
        for life in newPopulation:
            fitnessList.append(life._fitness)
        logging.debug('generation %d fitness list of origin populations before elimination' % self._generation)
        logging.debug(fitnessList)

        # 生成新子代
        i = 0
        while i < self._populationSize:
            child1, child2 = self.getNewChild()
            newPopulation.append(child1)
            newPopulation.append(child2)
            i += 2

        # debug
        fitnessList = []
        for life in newPopulation:
            fitnessList.append(life._fitness)
        logging.debug('generation %d fitness list of new populations' % self._generation)
        logging.debug(fitnessList)

        # 进行小生境淘汰，即对最新种群内的两两个体计算海明距离，进行相似比较，惩罚较差个体
        newPopulation = self.smallHabitatElimination(newPopulation)

        # 根据适应度值对最新种群的个体降序排列
        newPopulation = self.sortPopulation(newPopulation)

        # debug
        for life in newPopulation:
            fitnessList.append(life._fitness)
        logging.debug('generation %d fitness list of current populations after elimination' % self._generation)
        logging.debug(fitnessList[:10])
        logging.debug('\n')

        # 更新种群，大小为self._populationSize
        self._population = newPopulation[:self._populationSize]

        # 计算最佳个体，
        self._bestLife = self._population[0]
        self._bestFitnessHistory.append(self._bestLife._fitness)

        # 如果适应度没有提高，则self.round加1，否则置0
        if bestFitnessCurrent == self._bestLife._fitness:
            self._round += 1
        else:
            self._round = 0

        # 计算总适应度
        self._totalFitness = self.calTotalFitness(self._population)
        self._totalFitnessHistory.append(self._totalFitness)

        # 如果适应度值足够高，可提前结束
        if self._bestLife._fitness >= self._fitnessThreshold or self._round == self._earlyStopRoundThreshold:
            self._earlyStop = True

        # 迭代次数加1
        self._generation += 1

    # 可视化种群适应度
    def plotHistory(self):
        if not self._showBestFitness and not self._showTotalFitness:
            return
        plt.figure(figsize=(40, 10))
        path = './figures/%s/' % time.strftime("%Y%m%d", time.localtime(time.time()))
        if not os.path.exists(path):
            os.mkdir(path)
        if self._showBestFitness:
            plt.plot(self._bestFitnessHistory, label="best_fitness")
            plt.savefig(os.path.join(path, 'bestFitness_%s.png' % time.strftime("%H%M", time.localtime(time.time()))))
        if self._showTotalFitness:
            plt.plot(self._totalFitnessHistory, label="total_fitness")
            plt.savefig(os.path.join(path, 'totalFitness_%s.png' % time.strftime("%H%M", time.localtime(time.time()))))
        plt.legend()
        plt.show()

    # 参数设置
    def parameters(self):
        return 'populationSize=%d, binaryEncode=%s, geneLength=%d, boundaryList=%s,\n delta=%f,' \
               'fitnessClass=%s, crossoverRate=%f, crossoverOneRate=%f, crossoverTwoRate=%f,' \
               'mutationRate=%f, mutationLocationRate=%f, mutationRotateRate=%f, fitnessThreshold=%f,' \
               'similarityThreshold=%f, CF=%d, punish=%f, showBestFitness=%s, showTotalFitness=%s' \
               % (self._populationSize, self._binaryEncode, self._geneLength, self._boundaryList, self._delta,
                  self._fitnessClass, self._crossoverRate, self._crossoverOneRate, self._crossoverTwoRate,
                  self._mutationRate, self._mutationLocationRate, self._mutationRotateRate, self._fitnessThreshold,
                  self._similarityThresholdThreshold, self._CF, self._punish, self._showBestFitness,
                  self._showTotalFitness)


if __name__ == "__main__":
    # 适应函数度量
    fitnessClass = Fitness()
    ga = GA(crossoverRate=0.7, mutationRate=0.05, populationSize=100, boundaryList=[[0.5, 1.5]] * 2, delta=0.1,
            fitnessClass=fitnessClass)
    ga.plotHistory()
    for i in range(3):
        print(ga._population[0]._gene)
        gene = ga.mutation(ga._population[0]._gene)
        print(gene)
        print('\n')
