"""tests/test_gp.py —— 遗传规划单元测试。

覆盖：Primitive/Terminal、PrimitiveSet、PrimitiveTree、
      genFull/genGrow/genHalfAndHalf、compile、cxOnePoint、
      mutUniform/mutNodeReplacement/mutShrink。
"""

import random
import operator
import math

import pytest

from mini_deap.base.fitness import Fitness
from mini_deap.base.toolbox import Toolbox
import mini_deap.base.creator as creator
import mini_deap.gp as gp
from mini_deap.algorithms import eaSimple
from mini_deap.tools.operators import selTournament


def make_pset():
    pset = gp.PrimitiveSet("MAIN", 1)           # 1 个变量 ARG0
    pset.addPrimitive(operator.add, 2, name="add")
    pset.addPrimitive(operator.sub, 2, name="sub")
    pset.addPrimitive(operator.mul, 2, name="mul")
    pset.addTerminal(1.0, name="one")
    pset.addTerminal(2.0, name="two")
    return pset


creator.create("FitMinGP", Fitness, weights=(-1.0,))
creator.create("IndGP", gp.PrimitiveTree, fitness=creator.FitMinGP)


# ───────────────────────── 节点 ─────────────────────────

class TestPrimitive:
    def test_arity(self):
        p = gp.Primitive("add", [object, object], object)
        assert p.arity == 2

    def test_format(self):
        p = gp.Primitive("add", [object, object], object)
        assert p.format(1, 2) == "add(1, 2)"

    def test_eq(self):
        p1 = gp.Primitive("add", [object, object], object)
        p2 = gp.Primitive("add", [object, object], object)
        assert p1 == p2


class TestTerminal:
    def test_arity_zero(self):
        t = gp.Terminal(1.0, False, object)
        assert t.arity == 0

    def test_format(self):
        t = gp.Terminal(1.0, False, object)
        assert t.format() == "1.0"


# ───────────────────────── PrimitiveSet ─────────────────────────

class TestPrimitiveSet:
    def test_arguments(self):
        pset = gp.PrimitiveSet("MAIN", 2)
        assert pset.arguments == ["ARG0", "ARG1"]

    def test_add_primitive(self):
        pset = make_pset()
        assert len(pset.primitives[object]) == 3   # add, sub, mul

    def test_add_terminal(self):
        pset = make_pset()
        # ARG0 + one + two = 3 terminals
        assert len(pset.terminals[object]) == 3

    def test_context_has_funcs(self):
        pset = make_pset()
        assert "add" in pset.context
        assert "mul" in pset.context


# ───────────────────────── PrimitiveTree ─────────────────────────

class TestPrimitiveTree:
    def test_str(self):
        pset = make_pset()
        # 手构 [add, ARG0, 1.0] = add(ARG0, 1.0)
        tree = gp.PrimitiveTree([pset.primitives[object][0],   # add
                                 pset.terminals[object][0],    # ARG0
                                 pset.terminals[object][1]])   # 1.0
        assert str(tree) == "add(ARG0, 1.0)"

    def test_height(self):
        pset = make_pset()
        add = pset.primitives[object][0]
        arg0 = pset.terminals[object][0]   # ARG0
        one = pset.terminals[object][1]    # 1.0
        # [add, ARG0, 1.0] 高度 1
        tree = gp.PrimitiveTree([add, arg0, one])
        assert tree.height == 1
        # [add, add, ARG0, 1.0, 1.0] 高度 2
        tree2 = gp.PrimitiveTree([add, add, arg0, one, one])
        assert tree2.height == 2

    def test_searchSubtree(self):
        pset = make_pset()
        add = pset.primitives[object][0]
        arg0 = pset.terminals[object][0]
        one = pset.terminals[object][1]
        # [add, add, ARG0, 1.0, 1.0]
        #  index: 0   1    2    3    4
        tree = gp.PrimitiveTree([add, add, arg0, one, one])
        # 子树从 index=1 开始：[add, ARG0, 1.0] → slice(1, 4)
        s = tree.searchSubtree(1)
        assert s == slice(1, 4)
        # 子树从 index=0 开始：整棵树 → slice(0, 5)
        s0 = tree.searchSubtree(0)
        assert s0 == slice(0, 5)

    def test_deepcopy(self):
        import copy
        pset = make_pset()
        tree = gp.PrimitiveTree(gp.genFull(pset, 1, 2))
        tree2 = copy.deepcopy(tree)
        assert str(tree) == str(tree2)
        assert tree is not tree2


# ───────────────────────── 树生成 ─────────────────────────

class TestGeneration:
    def test_genFull(self):
        pset = make_pset()
        random.seed(42)
        expr = gp.genFull(pset, 1, 2)
        tree = gp.PrimitiveTree(expr)
        assert tree.height >= 1

    def test_genGrow(self):
        pset = make_pset()
        random.seed(42)
        expr = gp.genGrow(pset, 1, 2)
        tree = gp.PrimitiveTree(expr)
        assert tree.height >= 1

    def test_genHalfAndHalf(self):
        pset = make_pset()
        random.seed(42)
        expr = gp.genHalfAndHalf(pset, 1, 2)
        tree = gp.PrimitiveTree(expr)
        assert len(tree) >= 1


# ───────────────────────── compile ─────────────────────────

class TestCompile:
    def test_compile_and_eval(self):
        pset = make_pset()
        add = pset.primitives[object][0]
        arg0 = pset.terminals[object][0]
        one = pset.terminals[object][1]
        tree = gp.PrimitiveTree([add, arg0, one])   # add(ARG0, 1.0)
        func = gp.compile(tree, pset)
        assert func(2.0) == 3.0   # 2.0 + 1.0

    def test_compile_complex(self):
        pset = make_pset()
        # mul(add(ARG0, 1.0), 2.0) = (ARG0 + 1) * 2
        mul = [p for p in pset.primitives[object] if p.name == "mul"][0]
        add = [p for p in pset.primitives[object] if p.name == "add"][0]
        arg0 = pset.terminals[object][0]
        one = pset.terminals[object][1]
        two = pset.terminals[object][2]
        tree = gp.PrimitiveTree([mul, add, arg0, one, two])
        func = gp.compile(tree, pset)
        assert func(3.0) == 8.0   # (3+1)*2


# ───────────────────────── 交叉 ─────────────────────────

class TestCxOnePoint:
    def test_returns_two(self):
        pset = make_pset()
        random.seed(42)
        t1 = creator.IndGP(gp.genHalfAndHalf(pset, 1, 2))
        t2 = creator.IndGP(gp.genHalfAndHalf(pset, 1, 2))
        r1, r2 = gp.cxOnePoint(t1, t2)
        assert isinstance(r1, gp.PrimitiveTree)
        assert isinstance(r2, gp.PrimitiveTree)

    def test_single_node_no_cross(self):
        """单节点树不交叉。"""
        pset = make_pset()
        arg0 = pset.terminals[object][0]
        t1 = creator.IndGP([arg0])
        t2 = creator.IndGP([arg0])
        r1, r2 = gp.cxOnePoint(t1, t2)
        assert list(r1) == list(t1)
        assert list(r2) == list(t2)

    def test_cross_changes_tree(self):
        """交叉应改变树（大概率）。"""
        pset = make_pset()
        random.seed(42)
        t1 = creator.IndGP(gp.genFull(pset, 2, 3))
        t2 = creator.IndGP(gp.genFull(pset, 2, 3))
        s1, s2 = str(t1), str(t2)
        gp.cxOnePoint(t1, t2)
        # 大概率至少一个变了
        assert str(t1) != s1 or str(t2) != s2


# ───────────────────────── 变异 ─────────────────────────

class TestMutation:
    def test_mutUniform(self):
        pset = make_pset()
        random.seed(42)
        t = creator.IndGP(gp.genFull(pset, 2, 3))
        original = str(t)
        gp.mutUniform(t, lambda pset: gp.genGrow(pset, 0, 2), pset)
        # 变异后仍是合法树（能编译）
        func = gp.compile(t, pset)
        assert callable(func)

    def test_mutNodeReplacement(self):
        pset = make_pset()
        random.seed(42)
        t = creator.IndGP(gp.genFull(pset, 2, 3))
        gp.mutNodeReplacement(t, pset)
        func = gp.compile(t, pset)
        assert callable(func)

    def test_mutShrink(self):
        pset = make_pset()
        random.seed(42)
        t = creator.IndGP(gp.genFull(pset, 2, 3))
        original_len = len(t)
        gp.mutShrink(t)
        assert len(t) <= original_len   # 收缩后不更长


# ───────────────────────── 端到端：符号回归 ─────────────────────────

class TestSymbolicRegression:
    """用 GP 拟合 f(x) = x² + x + 1，验证全流程。"""

    def test_gp_fits_quadratic(self):
        random.seed(42)
        pset = gp.PrimitiveSet("MAIN", 1)
        pset.addPrimitive(operator.add, 2, name="add")
        pset.addPrimitive(operator.sub, 2, name="sub")
        pset.addPrimitive(operator.mul, 2, name="mul")
        for c in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            pset.addTerminal(c)

        tb = Toolbox()
        tb.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
        tb.register("individual", gp.PrimitiveTree, tb.expr)
        tb.register("mate", gp.cxOnePoint)
        tb.register("expr_mut", gp.genHalfAndHalf, pset=pset, min_=0, max_=2)
        tb.register("mutate", gp.mutUniform, expr=tb.expr_mut, pset=pset)
        tb.register("evaluate", self._eval_symbreg, pset=pset)
        tb.register("select", selTournament, tournsize=3)

        pop = [creator.IndGP(gp.genHalfAndHalf(pset, 1, 2)) for _ in range(50)]
        for ind in pop:
            ind.fitness.values = tb.evaluate(ind)

        pop, log = eaSimple(pop, tb, 0.5, 0.2, 20, verbose=False)
        # 验证能跑完且最优 fitness 合理（误差 < 50）
        best = min(ind.fitness.values[0] for ind in pop)
        assert best < 50.0

    @staticmethod
    def _eval_symbreg(ind, pset):
        """拟合 x² + x + 1，返回 MSE。"""
        func = gp.compile(ind, pset)
        xs = [x / 10.0 for x in range(-10, 11)]
        sq_errors = []
        for x in xs:
            try:
                pred = func(x)
                target = x ** 2 + x + 1
                sq_errors.append((pred - target) ** 2)
            except (OverflowError, ZeroDivisionError):
                sq_errors.append(1e6)
        return (sum(sq_errors) / len(sq_errors),)