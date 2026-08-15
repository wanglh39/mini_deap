"""tests/tools/test_emo.py —— NSGA2 多目标算子单元测试。

覆盖：sortNondominated / assignCrowdingDist / selNSGA2 / selTournamentDCD。
测试场景：双目标最大化，手工构造已知支配关系的种群。
"""

import random

import pytest

from mini_deap.base.fitness import Fitness
import mini_deap.base.creator as creator
from mini_deap.tools.emo import (
    sortNondominated, assignCrowdingDist, selNSGA2, selTournamentDCD,
)

creator.create("FitMulti", Fitness, weights=(1.0, 1.0))
creator.create("IndMulti", list, fitness=creator.FitMulti)


def make_ind(f1, f2):
    ind = creator.IndMulti([f1, f2])
    ind.fitness.values = (f1, f2)
    return ind


# ───────────────────────── sortNondominated ─────────────────────────

class TestSortNondominated:
    def test_empty(self):
        assert sortNondominated([], 0) == []

    def test_single_individual(self):
        ind = make_ind(1.0, 2.0)
        fronts = sortNondominated([ind], 1)
        assert len(fronts) == 1
        assert fronts[0] == [ind]

    def test_first_front_is_nondominated(self):
        """第一前沿的个体互不支配。"""
        pop = [
            make_ind(1.0, 5.0),   # A
            make_ind(5.0, 1.0),   # B
            make_ind(2.0, 2.0),   # C：被 A? A=(1,5) 不支配 C=(2,2)（1<2 但 5>2）
            make_ind(0.5, 0.5),   # D：被 A/B/C 支配
        ]
        fronts = sortNondominated(pop, len(pop))
        # D 应不在第一前沿
        assert pop[3] not in fronts[0]
        # A/B/C 互不支配，应都在第一前沿
        assert pop[0] in fronts[0]
        assert pop[1] in fronts[0]
        assert pop[2] in fronts[0]

    def test_dominated_in_later_front(self):
        """被支配的个体在后续前沿。"""
        pop = [
            make_ind(3.0, 3.0),   # 支配下面所有
            make_ind(1.0, 1.0),   # 被上面支配
        ]
        fronts = sortNondominated(pop, len(pop))
        assert fronts[0] == [pop[0]]
        assert fronts[1] == [pop[1]]

    def test_k_limit(self):
        """k=1 只排到第一个前沿就够。"""
        pop = [make_ind(1.0, 1.0), make_ind(2.0, 2.0)]
        fronts = sortNondominated(pop, 1)
        assert len(fronts) == 1
        assert fronts[0] == [pop[1]]

    def test_first_front_only(self):
        pop = [make_ind(1.0, 1.0), make_ind(2.0, 2.0)]
        front = sortNondominated(pop, len(pop), first_front_only=True)
        assert front == [[pop[1]]]


# ───────────────────────── assignCrowdingDist ─────────────────────────

class TestAssignCrowdingDist:
    def test_empty(self):
        assignCrowdingDist([])   # 不应报错

    def test_boundary_is_inf(self):
        """每维的边界个体拥挤距离 = inf。"""
        pop = [make_ind(1.0, 5.0), make_ind(2.0, 3.0), make_ind(5.0, 1.0)]
        assignCrowdingDist(pop)
        # f1 维的 min/max 和 f2 维的 min/max
        # (1,5) 是 f1-min 和 f2-max → inf
        # (5,1) 是 f1-max 和 f2-min → inf
        # (2,3) 是中间 → 有限
        assert pop[0].fitness.crowding_dist == float("inf")
        assert pop[2].fitness.crowding_dist == float("inf")
        assert pop[1].fitness.crowding_dist < float("inf")

    def test_middle_is_finite(self):
        pop = [make_ind(1.0, 1.0), make_ind(2.0, 2.0), make_ind(3.0, 3.0)]
        assignCrowdingDist(pop)
        assert pop[1].fitness.crowding_dist < float("inf")

    def test_all_same_fitness(self):
        """所有个体 fitness 相同 → 不报错，距离为 0 或 inf。"""
        pop = [make_ind(1.0, 1.0), make_ind(1.0, 1.0)]
        assignCrowdingDist(pop)   # 不应报错


# ───────────────────────── selNSGA2 ─────────────────────────

class TestSelNSGA2:
    def test_select_k(self):
        pop = [make_ind(i, 10 - i) for i in range(5)]
        chosen = selNSGA2(pop, 3)
        assert len(chosen) == 3

    def test_pareto_front_preferred(self):
        """第一前沿的个体优先被选。"""
        front1 = [make_ind(3.0, 3.0)]           # 支配 front2
        front2 = [make_ind(1.0, 1.0)]
        pop = front1 + front2
        chosen = selNSGA2(pop, 1)
        assert chosen == front1

    def test_all_nondominated_selected_first(self):
        """互不支配的个体都应在第一前沿。"""
        pop = [make_ind(i, 10 - i) for i in range(5)]  # 互不支配
        chosen = selNSGA2(pop, 5)
        assert len(chosen) == 5

    def test_returns_references(self):
        """返回的是原个体的引用，不是拷贝。"""
        pop = [make_ind(1.0, 1.0), make_ind(2.0, 2.0)]
        chosen = selNSGA2(pop, 2)
        assert all(c in pop for c in chosen)


# ───────────────────────── selTournamentDCD ─────────────────────────

class TestSelTournamentDCD:
    def test_select_k(self):
        pop = [make_ind(i, 10 - i) for i in range(8)]
        for ind in pop:
            ind.fitness.crowding_dist = 1.0
        chosen = selTournamentDCD(pop, 4)
        assert len(chosen) == 4

    def test_k_too_large(self):
        pop = [make_ind(1.0, 1.0)]
        with pytest.raises(ValueError):
            selTournamentDCD(pop, 2)

    def test_k_not_divisible_by_4(self):
        """k == len 且 k % 4 != 0 应报错。"""
        pop = [make_ind(i, 10 - i) for i in range(6)]
        for ind in pop:
            ind.fitness.crowding_dist = 1.0
        with pytest.raises(ValueError):
            selTournamentDCD(pop, 6)

    def test_dominance_preferred(self):
        """支配者应胜出。"""
        pop = [make_ind(3.0, 3.0), make_ind(1.0, 1.0)] * 4   # len=8
        for ind in pop:
            ind.fitness.crowding_dist = 1.0
        chosen = selTournamentDCD(pop, 4)
        # 支配者 (3,3) 应更多被选
        assert any(sum(c) == 6.0 for c in chosen)