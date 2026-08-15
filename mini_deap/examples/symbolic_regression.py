"""examples/symbolic_regression.py —— GP 符号回归：拟合 f(x) = x² + x + 1。

用遗传规划自动"发现"数学公式。个体是语法树，fitness = 预测值与目标值的 MSE。

GP 流程：
  1. 定义函数集（add/sub/mul）+ 终端集（变量 x + 常量）
  2. 随机生成树种群
  3. 编译每棵树为函数，在采样点上算 MSE 作为 fitness
  4. 用 eaSimple 进化（子树交叉 + 子树变异）

运行：python -m mini_deap.examples.symbolic_regression
"""

import random
import operator
import math

from mini_deap.base.fitness import Fitness
from mini_deap.base.toolbox import Toolbox
import mini_deap.base.creator as creator
import mini_deap.gp as gp
from mini_deap.tools.operators import selTournament
from mini_deap.tools.support import Statistics, HallOfFame
from mini_deap.algorithms import eaSimple


# 目标函数
def target(x):
    return x ** 2 + x + 1


def eval_symbreg(ind, pset):
    """编译树并在采样点上算 MSE。"""
    func = gp.compile(ind, pset)
    xs = [x / 10.0 for x in range(-10, 11)]
    sq_errors = []
    for x in xs:
        try:
            pred = func(x)
            if math.isfinite(pred):
                sq_errors.append((pred - target(x)) ** 2)
            else:
                sq_errors.append(1e6)
        except (OverflowError, ZeroDivisionError):
            sq_errors.append(1e6)
    return (sum(sq_errors) / len(sq_errors),)


def main(pop_size=300, ngen=80, seed=42):
    random.seed(seed)

    pset = gp.PrimitiveSet("MAIN", 1)           # 1 个变量 ARG0
    pset.addPrimitive(operator.add, 2, name="add")
    pset.addPrimitive(operator.sub, 2, name="sub")
    pset.addPrimitive(operator.mul, 2, name="mul")
    for c in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        pset.addTerminal(c)

    creator.create("FitMin", Fitness, weights=(-1.0,))
    creator.create("Ind", gp.PrimitiveTree, fitness=creator.FitMin)

    tb = Toolbox()
    tb.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    tb.register("individual", gp.PrimitiveTree, tb.expr)
    tb.register("mate", gp.cxOnePoint)
    tb.register("expr_mut", gp.genHalfAndHalf, pset=pset, min_=0, max_=2)
    tb.register("mutate", gp.mutUniform, expr=tb.expr_mut, pset=pset)
    tb.register("evaluate", eval_symbreg, pset=pset)
    tb.register("select", selTournament, tournsize=3)

    pop = [creator.Ind(gp.genHalfAndHalf(pset, 1, 2)) for _ in range(pop_size)]
    for ind in pop:
        ind.fitness.values = tb.evaluate(ind)

    hof = HallOfFame(maxsize=1)
    stats = Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda x: sum(x) / len(x))
    stats.register("min", min)

    print(f"=== GP 符号回归 (pop={pop_size}, ngen={ngen}) ===")
    print(f"目标: f(x) = x² + x + 1")
    pop, log = eaSimple(pop, tb, 0.5, 0.2, ngen,
                        stats=stats, halloffame=hof, verbose=True)

    best = hof[0]
    print(f"\n最优公式: {best}")
    print(f"MSE = {best.fitness.values[0]:.6f}")
    # 验证
    func = gp.compile(best, pset)
    for x in [-1.0, 0.0, 1.0, 2.0]:
        print(f"  f({x}) = {func(x):.4f}, 目标 = {target(x):.4f}")

    return best


if __name__ == "__main__":
    main()