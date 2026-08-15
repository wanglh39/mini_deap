"""tests/base/test_creator.py —— creator 单元测试（镜像 mini_deap/base/creator.py）。

覆盖：create 基本用法、实例属性 vs 类属性、继承 list/set 行为、
      array.array 替换 + deepcopy 保留 fitness、numpy.ndarray 替换、
      deepcopy 隔离、pickle 动态类、重复 create 警告。
"""

import array
import copy
import pickle
import warnings

import pytest

from mini_deap.base.fitness import Fitness
import mini_deap.base.creator as creator


class TestCreateBasic:
    """create 基本用法：个体 = 容器 + fitness。"""

    def test_individual_is_list_with_fitness(self):
        creator.create("FitA", Fitness, weights=(1.0,))
        creator.create("IndA", list, fitness=creator.FitA)

        ind = creator.IndA([1, 2, 3])
        assert ind == [1, 2, 3]                       # list 行为
        assert isinstance(ind.fitness, creator.FitA)  # fitness 是实例属性
        assert not ind.fitness.valid                  # 空 fitness

    def test_instance_attribute_independent_per_instance(self):
        creator.create("FitB", Fitness, weights=(1.0,))
        creator.create("IndB", list, fitness=creator.FitB)

        ind1 = creator.IndB([1])
        ind2 = creator.IndB([2])
        # fitness 是实例属性：各实例独立
        assert ind1.fitness is not ind2.fitness
        ind1.fitness.values = (5.0,)
        assert ind2.fitness.valid is False            # 改 ind1 不影响 ind2

    def test_class_attribute_shared(self):
        creator.create("FitC", Fitness, weights=(1.0,))
        creator.create("IndC", list, fitness=creator.FitC, size=42)

        ind1 = creator.IndC([1])
        ind2 = creator.IndC([2])
        # size 是类属性（42 不是类型）→ 共享
        assert creator.IndC.size == 42
        assert ind1.size == 42 and ind2.size == 42

    def test_weights_is_class_attribute(self):
        creator.create("FitD", Fitness, weights=(1.0,))
        # weights 是 tuple（不是类型）→ 类属性
        assert creator.FitD.weights == (1.0,)
        f1 = creator.FitD()
        f2 = creator.FitD()
        assert f1.weights is creator.FitD.weights     # 共享同一对象


class TestInheritance:
    """create 的类继承 base 的行为。"""

    def test_inherits_list(self):
        creator.create("FitE", Fitness, weights=(1.0,))
        creator.create("IndE", list, fitness=creator.FitE)
        ind = creator.IndE()
        ind.extend([1, 2, 3])
        assert ind == [1, 2, 3]
        assert len(ind) == 3

    def test_inherits_set(self):
        creator.create("IndSet", set, marker="x")
        s = creator.IndSet([1, 2, 3])
        assert s == {1, 2, 3}
        s.add(4)
        assert 4 in s


class TestDeepCopy:
    """deepcopy 必须保留容器数据 + fitness + 类属性。"""

    def test_deepcopy_list_individual(self):
        creator.create("FitF", Fitness, weights=(1.0,))
        creator.create("IndF", list, fitness=creator.FitF)
        ind = creator.IndF([1, 2, 3])
        ind.fitness.values = (6.0,)

        clone = copy.deepcopy(ind)
        assert clone == [1, 2, 3]
        assert clone.fitness.values == (6.0,)
        assert clone is not ind
        assert clone.fitness is not ind.fitness


class TestArrayReplacer:
    """array.array 替换：修正 deepcopy 不拷 __dict__。"""

    def test_array_in_replacers(self):
        assert array.array in creator.class_replacers

    def test_create_array_individual(self):
        creator.create("FitG", Fitness, weights=(1.0,))
        creator.create("IndArrG", array.array, typecode="f", fitness=creator.FitG)
        ind = creator.IndArrG([1.0, 2.0, 3.0])
        assert list(ind) == [1.0, 2.0, 3.0]
        assert isinstance(ind.fitness, creator.FitG)

    def test_array_deepcopy_preserves_fitness(self):
        creator.create("FitH", Fitness, weights=(1.0,))
        creator.create("IndArrH", array.array, typecode="f", fitness=creator.FitH)
        ind = creator.IndArrH([1.0, 2.0])
        ind.fitness.values = (9.0,)

        clone = copy.deepcopy(ind)
        assert list(clone) == [1.0, 2.0]
        assert clone.fitness.values == (9.0,)
        assert clone.fitness is not ind.fitness


class TestNumpyReplacer:
    """numpy.ndarray 替换（numpy 可用时）。"""

    def test_numpy_in_replacers(self):
        numpy = pytest.importorskip("numpy")
        assert numpy.ndarray in creator.class_replacers

    def test_create_numpy_individual(self):
        numpy = pytest.importorskip("numpy")
        creator.create("FitI", Fitness, weights=(1.0,))
        creator.create("IndNpI", numpy.ndarray, fitness=creator.FitI)
        ind = creator.IndNpI([1.0, 2.0, 3.0])
        assert list(ind) == [1.0, 2.0, 3.0]
        assert isinstance(ind.fitness, creator.FitI)

    def test_numpy_deepcopy_preserves_fitness(self):
        numpy = pytest.importorskip("numpy")
        creator.create("FitJ", Fitness, weights=(1.0,))
        creator.create("IndNpJ", numpy.ndarray, fitness=creator.FitJ)
        ind = creator.IndNpJ([1.0, 2.0])
        ind.fitness.values = (7.0,)

        clone = copy.deepcopy(ind)
        assert list(clone) == [1.0, 2.0]
        assert clone.fitness.values == (7.0,)


class TestPickleAndWarning:
    """pickle 动态类 + 重复 create 警告。"""

    def test_pickle_individual(self):
        creator.create("FitK", Fitness, weights=(1.0,))
        creator.create("IndK", list, fitness=creator.FitK)
        ind = creator.IndK([1, 2, 3])
        ind.fitness.values = (5.0,)

        data = pickle.dumps(ind)
        ind2 = pickle.loads(data)
        assert ind2 == [1, 2, 3]
        assert ind2.fitness.values == (5.0,)

    def test_duplicate_create_warns(self):
        creator.create("FitL", Fitness, weights=(1.0,))
        with pytest.warns(RuntimeWarning, match="overwritten"):
            creator.create("FitL", Fitness, weights=(-1.0,))
        assert creator.FitL.weights == (-1.0,)