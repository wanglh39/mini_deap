"""tests/base/test_fitness.py —— Fitness 单元测试（镜像 mini_deap/base/fitness.py）。

覆盖：抽象基类保护、values property 三件套、惰性加权、
      最大化/最小化方向统一、多目标字典序与 Pareto 支配、deepcopy、hash、repr。
"""

import copy

import pytest

from mini_deap.base.fitness import Fitness


# ---- 测试用的具体适应度子类（模拟 creator.create 的产物）----
class FitnessMax(Fitness):
    weights = (1.0,)


class FitnessMin(Fitness):
    weights = (-1.0,)


class FitnessMulti(Fitness):
    weights = (1.0, 1.0)   # 双目标都最大化


class TestFitnessAbstraction:
    """抽象基类保护。"""

    def test_cannot_instantiate_base_fitness(self):
        # weights=None → 抛 TypeError
        with pytest.raises(TypeError, match="abstract attribute weights"):
            Fitness()

    def test_weights_must_be_sequence(self):
        class BadFitness(Fitness):
            weights = 42   # int 不是 Sequence

        with pytest.raises(TypeError, match="must be a sequence"):
            BadFitness()


class TestValuesProperty:
    """values property 的 get/set/del 与惰性加权。"""

    def test_set_values_stores_weighted(self):
        f = FitnessMax()
        f.values = (3.0,)
        # wvalues = values * weights = 3.0 * 1.0 = 3.0
        assert f.wvalues == (3.0,)

    def test_get_values_restores_original(self):
        f = FitnessMin()
        f.values = (5.0,)
        # wvalues = 5.0 * -1.0 = -5.0，get 时除回 → 5.0
        assert f.wvalues == (-5.0,)
        assert f.values == (5.0,)

    def test_init_with_values(self):
        f = FitnessMax((2.0,))
        assert f.values == (2.0,)
        assert f.valid

    def test_init_empty_is_invalid(self):
        f = FitnessMax()
        assert not f.valid
        assert f.values == ()

    def test_del_invalidates(self):
        f = FitnessMax((1.0,))
        assert f.valid
        del f.values
        assert not f.valid
        assert f.wvalues == ()

    def test_length_mismatch_raises(self):
        f = FitnessMax()
        with pytest.raises(AssertionError):
            f.values = (1.0, 2.0)   # weights 只有 1 维


class TestComparison:
    """比较运算符：基于 wvalues 字典序，永远越大越好。"""

    def test_maximization_larger_is_better(self):
        a = FitnessMax((3.0,))
        b = FitnessMax((2.0,))
        assert a > b
        assert b < a
        assert a >= b
        assert b <= a

    def test_minimization_smaller_is_better(self):
        # weights=(-1.0,)：原值越小 → wvalues 越大 → 越优
        a = FitnessMin((1.0,))   # wvalues = -1.0
        b = FitnessMin((5.0,))   # wvalues = -5.0
        assert a > b             # 1.0 比 5.0 更优（最小化）
        assert b < a

    def test_equality(self):
        a = FitnessMax((2.0,))
        b = FitnessMax((2.0,))
        assert a == b
        assert not (a != b)

    def test_multiobjective_lexicographic(self):
        # 多目标按 wvalues 字典序比较（非 Pareto，Pareto 用 dominates）
        a = FitnessMulti((1.0, 9.0))   # wvalues = (1.0, 9.0)
        b = FitnessMulti((1.0, 5.0))   # wvalues = (1.0, 5.0)
        assert a > b   # 第一维相等，第二维 9 > 5


class TestDominates:
    """Pareto 支配（多目标）。"""

    def test_dominates_strictly_better_in_one(self):
        a = FitnessMulti((1.0, 2.0))
        b = FitnessMulti((1.0, 1.0))
        # a 每维不劣，第二维严格更优 → a 支配 b
        assert a.dominates(b)
        assert not b.dominates(a)

    def test_non_dominating(self):
        a = FitnessMulti((1.0, 2.0))
        b = FitnessMulti((2.0, 1.0))
        # 互不支配
        assert not a.dominates(b)
        assert not b.dominates(a)

    def test_equal_does_not_dominate(self):
        a = FitnessMulti((1.0, 1.0))
        b = FitnessMulti((1.0, 1.0))
        assert not a.dominates(b)


class TestCopyAndHash:
    """deepcopy 与 hash。"""

    def test_deepcopy_preserves_values(self):
        f = FitnessMax((3.0,))
        g = copy.deepcopy(f)
        assert g.values == (3.0,)
        assert g is not f
        # 修改副本不影响原
        g.values = (5.0,)
        assert f.values == (3.0,)

    def test_deepcopy_invalid(self):
        f = FitnessMax()
        g = copy.deepcopy(f)
        assert not g.valid

    def test_hash_consistent_with_equality(self):
        a = FitnessMax((2.0,))
        b = FitnessMax((2.0,))
        assert hash(a) == hash(b)

    def test_repr_contains_class_and_value(self):
        f = FitnessMax((3.0,))
        r = repr(f)
        assert "FitnessMax" in r
        assert "3.0" in r

    def test_repr_invalid_shows_empty(self):
        f = FitnessMax()
        assert "()" in repr(f)