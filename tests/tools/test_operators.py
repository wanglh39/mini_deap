"""tests/tools/test_operators.py —— 算子单元测试（镜像 mini_deap/tools/operators.py）。

覆盖：initRepeat/initIterate/initCycle、selRandom/selBest/selTournament/selRoulette(+np)、
      cxOnePoint/cxTwoPoint/cxUniform/cxBlend、mutGaussian(+np)/mutFlipBit/mutShuffleIndexes/mutUniformInt。
      含纯 Python vs numpy 版分布对照。
"""

import random

import pytest

from mini_deap.base.fitness import Fitness
import mini_deap.base.creator as creator
from mini_deap.tools.operators import (
    initRepeat, initIterate, initCycle,
    selRandom, selBest, selTournament, selRoulette, selRoulette_np,
    cxOnePoint, cxTwoPoint, cxUniform, cxBlend,
    mutGaussian, mutGaussian_np, mutFlipBit, mutShuffleIndexes, mutUniformInt,
)

creator.create("FitMax", Fitness, weights=(1.0,))
creator.create("Ind", list, fitness=creator.FitMax)


def make_ind(values, fit):
    ind = creator.Ind(values)
    ind.fitness.values = (fit,)
    return ind


@pytest.fixture
def population():
    # fitness 1..5，全正（selRoulette 要求）
    return [make_ind([i], float(i + 1)) for i in range(5)]


class TestInit:
    def test_initRepeat(self):
        random.seed(42)
        result = initRepeat(list, lambda: random.randint(0, 9), 5)
        assert len(result) == 5
        assert all(0 <= x <= 9 for x in result)

    def test_initIterate(self):
        result = initIterate(list, lambda: [1, 2, 3])
        assert result == [1, 2, 3]

    def test_initCycle(self):
        result = initCycle(list, [lambda: 1, lambda: 2], n=3)
        assert result == [1, 2, 1, 2, 1, 2]


class TestSelection:
    def test_selRandom(self, population):
        random.seed(42)
        chosen = selRandom(population, 3)
        assert len(chosen) == 3
        assert all(ind in population for ind in chosen)

    def test_selBest(self, population):
        chosen = selBest(population, 2)
        assert chosen[0].fitness.values == (5.0,)
        assert chosen[1].fitness.values == (4.0,)

    def test_selTournament(self, population):
        random.seed(42)
        chosen = selTournament(population, 1000, tournsize=2)
        assert len(chosen) == 1000
        # 期望 max of 2 from {1..5} = 3.8，应明显高于随机期望 3.0
        avg_fit = sum(ind.fitness.values[0] for ind in chosen) / 1000
        assert avg_fit > 3.3

    def test_selTournament_size1_is_random(self, population):
        random.seed(42)
        chosen = selTournament(population, 5, tournsize=1)
        assert len(chosen) == 5

    def test_selRoulette(self, population):
        random.seed(42)
        chosen = selRoulette(population, 10)
        assert len(chosen) == 10

    def test_selRoulette_np(self, population):
        random.seed(42)
        chosen = selRoulette_np(population, 10)
        assert len(chosen) == 10
        assert all(ind in population for ind in chosen)

    def test_roulette_versions_similar_distribution(self, population):
        # 两版都应倾向高 fitness（大样本下均值接近期望 55/15≈3.67）
        random.seed(42)
        chosen_py = selRoulette(population, 2000)
        random.seed(42)
        chosen_np = selRoulette_np(population, 2000)
        avg_py = sum(ind.fitness.values[0] for ind in chosen_py) / 2000
        avg_np = sum(ind.fitness.values[0] for ind in chosen_np) / 2000
        assert 3.0 < avg_py < 4.5
        assert 3.0 < avg_np < 4.5


class TestCrossover:
    def test_cxOnePoint(self):
        ind1 = creator.Ind([1, 2, 3, 4])
        ind2 = creator.Ind([5, 6, 7, 8])
        random.seed(42)
        c1, c2 = cxOnePoint(ind1, ind2)
        assert len(c1) == 4 and len(c2) == 4
        assert sorted(c1 + c2) == [1, 2, 3, 4, 5, 6, 7, 8]   # 元素守恒

    def test_cxTwoPoint(self):
        ind1 = creator.Ind([1, 2, 3, 4, 5])
        ind2 = creator.Ind([6, 7, 8, 9, 10])
        random.seed(42)
        c1, c2 = cxTwoPoint(ind1, ind2)
        assert len(c1) == 5 and len(c2) == 5
        assert sorted(c1 + c2) == list(range(1, 11))

    def test_cxUniform(self):
        ind1 = creator.Ind([1, 2, 3, 4])
        ind2 = creator.Ind([5, 6, 7, 8])
        orig1, orig2 = list(ind1), list(ind2)
        random.seed(42)
        c1, c2 = cxUniform(ind1, ind2, indpb=0.5)
        assert len(c1) == 4 and len(c2) == 4
        for i in range(4):
            if c1[i] == orig1[i]:
                assert c2[i] == orig2[i]
            else:
                assert c1[i] == orig2[i] and c2[i] == orig1[i]

    def test_cxBlend(self):
        ind1 = creator.Ind([0.0, 0.0])
        ind2 = creator.Ind([10.0, 10.0])
        c1, c2 = cxBlend(ind1, ind2, alpha=0.5)
        assert c1 == [5.0, 5.0]   # 中点
        assert c2 == [5.0, 5.0]


class TestMutation:
    def test_mutGaussian(self):
        ind = creator.Ind([0.0, 0.0, 0.0, 0.0])
        random.seed(42)
        mut, = mutGaussian(ind, mu=0, sigma=1, indpb=1.0)
        assert any(x != 0.0 for x in mut)

    def test_mutGaussian_zero_indpb(self):
        ind = creator.Ind([1.0, 2.0])
        mut, = mutGaussian(ind, mu=0, sigma=1, indpb=0.0)
        assert mut == [1.0, 2.0]

    def test_mutGaussian_np(self):
        ind = creator.Ind([0.0, 0.0, 0.0, 0.0])
        mut, = mutGaussian_np(ind, mu=0, sigma=1, indpb=1.0)
        assert any(abs(x) > 1e-9 for x in mut)

    def test_mutGaussian_np_zero_indpb(self):
        ind = creator.Ind([1.0, 2.0])
        mut, = mutGaussian_np(ind, mu=0, sigma=1, indpb=0.0)
        assert mut == [1.0, 2.0]

    def test_mutFlipBit(self):
        ind = creator.Ind([0, 1, 0, 1])
        random.seed(42)
        mut, = mutFlipBit(ind, indpb=1.0)
        assert all(x in (0, 1) for x in mut)
        assert mut == [1, 0, 1, 0]   # 全翻转

    def test_mutFlipBit_zero(self):
        ind = creator.Ind([0, 1, 0, 1])
        mut, = mutFlipBit(ind, indpb=0.0)
        assert mut == [0, 1, 0, 1]

    def test_mutShuffleIndexes_preserves_permutation(self):
        ind = creator.Ind([0, 1, 2, 3, 4])
        random.seed(42)
        mut, = mutShuffleIndexes(ind, indpb=1.0)
        assert sorted(mut) == [0, 1, 2, 3, 4]   # 排列守恒

    def test_mutUniformInt(self):
        ind = creator.Ind([0, 0, 0])
        random.seed(42)
        mut, = mutUniformInt(ind, low=0, up=9, indpb=1.0)
        assert all(0 <= x <= 9 for x in mut)