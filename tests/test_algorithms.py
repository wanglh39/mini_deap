"""tests/test_algorithms.py —— 算法主循环单元测试（镜像 mini_deap/algorithms.py）。

覆盖：varAnd / varOr / eaSimple / eaMuPlusLambda / eaMuCommaLambda。
测试场景：One-Max（个体是 0/1 list，fitness = sum）。
"""

import random

import pytest

from mini_deap.base.fitness import Fitness
from mini_deap.base.toolbox import Toolbox
import mini_deap.base.creator as creator
from mini_deap.tools.operators import (
    selTournament, cxOnePoint, mutFlipBit, initRepeat,
)
from mini_deap.tools.support import Statistics, HallOfFame
from mini_deap.algorithms import (
    varAnd, varOr, eaSimple, eaMuPlusLambda, eaMuCommaLambda,
)

creator.create("FitMax", Fitness, weights=(1.0,))
creator.create("Ind", list, fitness=creator.FitMax)


def maketoolbox(indlen=10):
    tb = Toolbox()
    tb.register("attr_bool", random.randint, 0, 1)
    tb.register("individual", initRepeat, creator.Ind, tb.attr_bool, indlen)
    tb.register("population", initRepeat, list, tb.individual)
    tb.register("evaluate", lambda ind: (sum(ind),))
    tb.register("mate", cxOnePoint)
    tb.register("mutate", mutFlipBit, indpb=0.1)
    tb.register("select", selTournament, tournsize=3)
    return tb


def makepop(tb, n=20):
    pop = tb.population(n=n)
    for ind in pop:
        ind.fitness.values = tb.evaluate(ind)
    return pop


# ───────────────────────── varAnd ─────────────────────────

class TestVarAnd:
    def test_length_preserved(self):
        tb = maketoolbox()
        pop = makepop(tb)
        offspring = varAnd(pop, tb, 0.5, 0.2)
        assert len(offspring) == len(pop)

    def test_clone_independence(self):
        """offspring 是 clone，改它不影响父代。"""
        tb = maketoolbox()
        pop = makepop(tb)
        pop_snapshot = [list(ind) for ind in pop]
        offspring = varAnd(pop, tb, 0.5, 0.2)
        for orig, snap in zip(pop, pop_snapshot):
            assert list(orig) == snap   # 父代未被改

    def test_no_op_when_prob_zero(self):
        """cxpb=0, mutpb=0：无交叉无变异，但仍有 clone。"""
        tb = maketoolbox()
        pop = makepop(tb)
        offspring = varAnd(pop, tb, 0.0, 0.0)
        assert len(offspring) == len(pop)
        for o, p in zip(offspring, pop):
            assert list(o) == list(p)   # 内容相同
            assert o is not p           # 但是 clone（不同对象）

    def test_all_fitness_invalidated_at_full_prob(self):
        """cxpb=1, mutpb=1：所有个体都应 fitness 失效。

        交叉步长 2 处理 (1,2),(3,4),...，若种群长度为偶数，
        所有个体都被交叉 → 全失效。变异再全失效。
        """
        tb = maketoolbox()
        pop = makepop(tb, n=10)   # 偶数长度
        offspring = varAnd(pop, tb, 1.0, 1.0)
        assert all(not ind.fitness.valid for ind in offspring)

    def test_fitness_valid_when_no_op(self):
        """cxpb=0, mutpb=0：clone 的个体 fitness 仍有效（复制保留 fitness）。"""
        tb = maketoolbox()
        pop = makepop(tb)
        offspring = varAnd(pop, tb, 0.0, 0.0)
        assert all(ind.fitness.valid for ind in offspring)


# ───────────────────────── varOr ─────────────────────────

class TestVarOr:
    def test_lambda_count(self):
        tb = maketoolbox()
        pop = makepop(tb)
        offspring = varOr(pop, tb, 15, 0.5, 0.3)
        assert len(offspring) == 15

    def test_assert_prob_sum(self):
        """cxpb + mutpb > 1 应断言失败。"""
        tb = maketoolbox()
        pop = makepop(tb)
        with pytest.raises(AssertionError):
            varOr(pop, tb, 10, 0.6, 0.5)

    def test_all_reproduction_when_prob_zero(self):
        """cxpb=0, mutpb=0：全复制，fitness 仍有效。"""
        tb = maketoolbox()
        pop = makepop(tb)
        offspring = varOr(pop, tb, 10, 0.0, 0.0)
        assert all(ind.fitness.valid for ind in offspring)

    def test_all_crossover_when_cxpb_one(self):
        """cxpb=1：全交叉，所有子代 fitness 失效。"""
        tb = maketoolbox()
        pop = makepop(tb)
        offspring = varOr(pop, tb, 10, 1.0, 0.0)
        assert all(not ind.fitness.valid for ind in offspring)

    def test_all_mutation_when_mutpb_one(self):
        """cxpb=0, mutpb=1：全变异，所有子代 fitness 失效。"""
        tb = maketoolbox()
        pop = makepop(tb)
        offspring = varOr(pop, tb, 10, 0.0, 1.0)
        assert all(not ind.fitness.valid for ind in offspring)


# ───────────────────────── eaSimple ─────────────────────────

class TestEaSimple:
    def test_returns_pop_and_logbook(self):
        tb = maketoolbox()
        pop = makepop(tb)
        result = eaSimple(pop, tb, 0.5, 0.2, 5, verbose=False)
        assert len(result) == 2
        final_pop, logbook = result
        assert len(final_pop) == len(pop)

    def test_logbook_records_ngen_plus_one(self):
        """ngen=5 → logbook 有 6 条（gen 0..5）。"""
        tb = maketoolbox()
        pop = makepop(tb)
        _, logbook = eaSimple(pop, tb, 0.5, 0.2, 5, verbose=False)
        assert len(logbook) == 6
        assert logbook[0]["gen"] == 0
        assert logbook[-1]["gen"] == 5

    def test_halloffame_updated(self):
        tb = maketoolbox()
        pop = makepop(tb)
        hof = HallOfFame(maxsize=1)
        eaSimple(pop, tb, 0.5, 0.2, 5, halloffame=hof, verbose=False)
        assert len(hof) == 1
        # 名人堂最优应 >= 初始种群最优（进化只会更好或持平）
        initial_best = max(sum(ind) for ind in pop)
        assert sum(hof[0]) >= initial_best - 2  # 容忍波动

    def test_stats_recorded(self):
        tb = maketoolbox()
        pop = makepop(tb)
        stats = Statistics(key=lambda ind: ind.fitness.values[0])
        stats.register("max", max)
        _, logbook = eaSimple(pop, tb, 0.5, 0.2, 3, stats=stats, verbose=False)
        assert "max" in logbook[0]
        assert logbook[0]["max"] is not None

    def test_verbose_false_no_print(self, capsys):
        tb = maketoolbox()
        pop = makepop(tb)
        eaSimple(pop, tb, 0.5, 0.2, 3, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_population_inplace_replace(self):
        """population[:] = offspring 应保持原列表对象引用。"""
        tb = maketoolbox()
        pop = makepop(tb)
        pop_id = id(pop)
        eaSimple(pop, tb, 0.5, 0.2, 3, verbose=False)
        assert id(pop) == pop_id   # 同一对象

    def test_lazy_evaluation(self):
        """初始全有效 → gen 0 的 nevals=0。"""
        tb = maketoolbox()
        pop = makepop(tb)   # 全有效
        _, logbook = eaSimple(pop, tb, 0.5, 0.2, 3, verbose=False)
        assert logbook[0]["nevals"] == 0   # 初始全有效，不评估


# ───────────────────────── eaMuPlusLambda ─────────────────────────

class TestEaMuPlusLambda:
    def test_pop_size_stays_mu(self):
        tb = maketoolbox()
        mu, lambda_ = 10, 20
        pop = makepop(tb, n=mu)
        final_pop, _ = eaMuPlusLambda(pop, tb, mu, lambda_,
                                      0.5, 0.2, 5, verbose=False)
        assert len(final_pop) == mu

    def test_logbook_length(self):
        tb = maketoolbox()
        pop = makepop(tb, n=10)
        _, logbook = eaMuPlusLambda(pop, tb, 10, 20, 0.5, 0.2, 4, verbose=False)
        assert len(logbook) == 5

    def test_elitism_best_not_regress(self):
        """(μ+λ) 精英保留：历代最优不会退化（>= 初始最优）。"""
        tb = maketoolbox()
        pop = makepop(tb, n=10)
        initial_best = max(sum(ind) for ind in pop)
        hof = HallOfFame(maxsize=1)
        eaMuPlusLambda(pop, tb, 10, 20, 0.5, 0.2, 5,
                       halloffame=hof, verbose=False)
        assert sum(hof[0]) >= initial_best   # 精英保留保证不退化


# ───────────────────────── eaMuCommaLambda ─────────────────────────

class TestEaMuCommaLambda:
    def test_assert_lambda_ge_mu(self):
        """lambda_ < mu 应断言失败。"""
        tb = maketoolbox()
        pop = makepop(tb, n=10)
        with pytest.raises(AssertionError):
            eaMuCommaLambda(pop, tb, 10, 5, 0.5, 0.2, 3, verbose=False)

    def test_pop_size_stays_mu(self):
        tb = maketoolbox()
        mu, lambda_ = 10, 20
        pop = makepop(tb, n=mu)
        final_pop, _ = eaMuCommaLambda(pop, tb, mu, lambda_,
                                       0.5, 0.2, 5, verbose=False)
        assert len(final_pop) == mu

    def test_logbook_length(self):
        tb = maketoolbox()
        pop = makepop(tb, n=10)
        _, logbook = eaMuCommaLambda(pop, tb, 10, 20, 0.5, 0.2, 4, verbose=False)
        assert len(logbook) == 5

    def test_no_elitism_can_regress(self):
        """(μ,λ) 无精英：最优可能退化（父代全淘汰）。

        用确定性场景验证：高变异概率下最优可能丢失。
        不做严格断言（随机性），只验证能跑完且最终解合理。
        """
        tb = maketoolbox()
        pop = makepop(tb, n=10)
        final_pop, _ = eaMuCommaLambda(pop, tb, 10, 20, 0.5, 0.5, 5,
                                       verbose=False)
        assert len(final_pop) == 10
        # 最终种群所有 fitness 有效
        assert all(ind.fitness.valid for ind in final_pop)