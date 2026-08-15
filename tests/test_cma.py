"""tests/test_cma.py —— CMA-ES 单元测试。

覆盖：Strategy 初始化/generate/update、eaGenerateUpdate 端到端。
测试场景：Sphere 函数最小化。
"""

import random
import numpy

import pytest

from mini_deap.base.fitness import Fitness
from mini_deap.base.toolbox import Toolbox
import mini_deap.base.creator as creator
from mini_deap.cma import Strategy
from mini_deap.algorithms import eaGenerateUpdate
from mini_deap.tools.support import Statistics, HallOfFame


creator.create("FitMinCMA", Fitness, weights=(-1.0,))
creator.create("IndCMA", list, fitness=creator.FitMinCMA)


def sphere(ind):
    return (sum(x * x for x in ind),)


# ───────────────────────── Strategy ─────────────────────────

class TestStrategy:
    def test_init(self):
        s = Strategy(centroid=[0.0, 0.0, 0.0], sigma=1.0)
        assert s.dim == 3
        assert s.sigma == 1.0
        assert s.lambda_ > 0
        assert s.mu > 0

    def test_generate_count(self):
        s = Strategy(centroid=[0.0, 0.0], sigma=1.0, lambda_=10)
        pop = s.generate(creator.IndCMA)
        assert len(pop) == 10

    def test_generate_dim(self):
        s = Strategy(centroid=[0.0, 0.0, 0.0], sigma=1.0, lambda_=5)
        pop = s.generate(creator.IndCMA)
        assert all(len(ind) == 3 for ind in pop)

    def test_update_changes_centroid(self):
        """update 后 centroid 应移动（朝最优方向）。"""
        s = Strategy(centroid=[5.0, 5.0], sigma=1.0, lambda_=20)
        pop = s.generate(creator.IndCMA)
        for ind in pop:
            ind.fitness.values = sphere(ind)
        old_centroid = s.centroid.copy()
        s.update(pop)
        assert not numpy.allclose(old_centroid, s.centroid)

    def test_weights_normalized(self):
        s = Strategy(centroid=[0.0, 0.0], sigma=1.0)
        assert abs(sum(s.weights) - 1.0) < 1e-10


# ───────────────────────── 端到端 ─────────────────────────

class TestEaGenerateUpdate:
    def test_sphere_converges(self):
        """CMA-ES 优化 Sphere 应快速收敛到 0。"""
        random.seed(42)
        numpy.random.seed(42)

        dim = 5
        strategy = Strategy(centroid=[5.0] * dim, sigma=2.0)

        tb = Toolbox()
        tb.register("evaluate", sphere)
        tb.register("generate", strategy.generate, creator.IndCMA)
        tb.register("update", strategy.update)

        stats = Statistics(key=lambda ind: ind.fitness.values[0])
        stats.register("min", min)

        pop, log = eaGenerateUpdate(tb, 50, stats=stats, verbose=False)

        best = min(ind.fitness.values[0] for ind in pop)
        assert best < 1.0   # 50 代应收敛到 < 1

    def test_returns_pop_and_logbook(self):
        dim = 2
        strategy = Strategy(centroid=[1.0] * dim, sigma=1.0)
        tb = Toolbox()
        tb.register("evaluate", sphere)
        tb.register("generate", strategy.generate, creator.IndCMA)
        tb.register("update", strategy.update)
        result = eaGenerateUpdate(tb, 5, verbose=False)
        assert len(result) == 2
        pop, log = result
        assert len(log) == 5

    def test_halloffame_updated(self):
        dim = 3
        strategy = Strategy(centroid=[3.0] * dim, sigma=1.0)
        tb = Toolbox()
        tb.register("evaluate", sphere)
        tb.register("generate", strategy.generate, creator.IndCMA)
        tb.register("update", strategy.update)
        hof = HallOfFame(maxsize=1)
        eaGenerateUpdate(tb, 10, halloffame=hof, verbose=False)
        assert len(hof) == 1
        assert hof[0].fitness.values[0] < 10.0   # 应有改善