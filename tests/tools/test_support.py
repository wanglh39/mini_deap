"""tests/tools/test_support.py —— support 单元测试（镜像 mini_deap/tools/support.py）。

覆盖：Statistics register/compile/key、MultiStatistics、Logbook record/select/stream、
      HallOfFame update/排序/maxsize/去重/deepcopy隔离、ParetoFront 非支配保留/移除。
"""

import numpy as np
from operator import attrgetter, eq

from mini_deap.base.fitness import Fitness
import mini_deap.base.creator as creator
from mini_deap.tools.support import (
    Statistics, MultiStatistics, Logbook, HallOfFame, ParetoFront,
)

creator.create("FitMax", Fitness, weights=(1.0,))
creator.create("Ind", list, fitness=creator.FitMax)


def make_ind(values, fit):
    ind = creator.Ind(values)
    ind.fitness.values = (fit,)
    return ind


class TestStatistics:
    def test_register_and_compile(self):
        stats = Statistics()
        stats.register("mean", np.mean)
        stats.register("max", max)
        result = stats.compile([1, 2, 3, 4])
        assert result == {"mean": 2.5, "max": 4}

    def test_key(self):
        stats = Statistics(key=lambda x: x * 2)
        stats.register("sum", sum)
        result = stats.compile([1, 2, 3])
        assert result == {"sum": 12}   # (2+4+6)

    def test_key_fitness(self):
        stats = Statistics(key=attrgetter("fitness.values"))
        stats.register("avg", np.mean)
        pop = [make_ind([1], 3.0), make_ind([2], 5.0)]
        result = stats.compile(pop)
        assert result == {"avg": 4.0}


class TestMultiStatistics:
    def test_compile(self):
        fit_stats = Statistics(key=attrgetter("fitness.values"))
        size_stats = Statistics(key=len)
        mstats = MultiStatistics(fitness=fit_stats, size=size_stats)
        mstats.register("mean", np.mean)
        pop = [make_ind([1, 2], 3.0), make_ind([3, 4, 5], 5.0)]
        result = mstats.compile(pop)
        assert result == {"fitness": {"mean": 4.0}, "size": {"mean": 2.5}}


class TestLogbook:
    def test_record_and_select(self):
        log = Logbook()
        log.record(gen=0, avg=3.2, max=5.0)
        log.record(gen=1, avg=3.5, max=6.0)
        assert log.select("avg") == [3.2, 3.5]
        assert log.select("gen", "max") == ([0, 1], [5.0, 6.0])

    def test_stream_first_includes_header(self):
        log = Logbook()
        log.header = ["gen", "avg"]
        log.record(gen=0, avg=3.2)
        s1 = log.stream
        assert "gen" in s1 and "0" in s1

    def test_stream_second_no_header(self):
        log = Logbook()
        log.header = ["gen", "avg"]
        log.record(gen=0, avg=3.2)
        log.stream   # 消费第一次
        log.record(gen=1, avg=3.5)
        s2 = log.stream
        assert "1" in s2 and "gen" not in s2   # 第二次不输出表头


class TestHallOfFame:
    def test_update_keeps_best(self):
        hof = HallOfFame(maxsize=3)
        pop = [make_ind([i], float(i)) for i in range(5)]   # fitness 0..4
        hof.update(pop)
        assert len(hof) == 3
        assert hof[0].fitness.values == (4.0,)   # 最优在前
        assert hof[1].fitness.values == (3.0,)
        assert hof[2].fitness.values == (2.0,)

    def test_update_replaces_worst(self):
        hof = HallOfFame(maxsize=2)
        hof.update([make_ind([1], 1.0), make_ind([2], 2.0)])
        hof.update([make_ind([3], 3.0)])   # 3.0 > 1.0，替换
        assert hof[0].fitness.values == (3.0,)
        assert hof[1].fitness.values == (2.0,)

    def test_duplicate_not_inserted(self):
        hof = HallOfFame(maxsize=5, similar=eq)
        ind1 = make_ind([1, 2, 3], 5.0)
        ind2 = make_ind([1, 2, 3], 5.0)   # 相同内容
        hof.update([ind1, ind2])
        assert len(hof) == 1

    def test_deepcopy_isolation(self):
        hof = HallOfFame(maxsize=1)
        ind = make_ind([1, 2, 3], 5.0)
        hof.update([ind])
        ind[0] = 999   # 改原个体
        assert hof[0][0] == 1   # 名人堂里的不受影响


class TestParetoFront:
    def setup_method(self):
        creator.create("FitMulti", Fitness, weights=(1.0, 1.0))
        creator.create("IndMulti", list, fitness=creator.FitMulti)

    def make_multi(self, a, b):
        ind = creator.IndMulti([a, b])
        ind.fitness.values = (a, b)
        return ind

    def test_keeps_mutually_nondominated(self):
        pf = ParetoFront()
        pf.update([self.make_multi(1, 5), self.make_multi(2, 4), self.make_multi(5, 1)])
        # 三者互不支配（每个都有一维更优）
        assert len(pf) == 3

    def test_removes_dominated(self):
        pf = ParetoFront()
        pf.update([self.make_multi(1, 1)])
        pf.update([self.make_multi(2, 2)])   # (2,2) 支配 (1,1)，移除 (1,1)
        assert len(pf) == 1
        assert pf[0].fitness.values == (2.0, 2.0)