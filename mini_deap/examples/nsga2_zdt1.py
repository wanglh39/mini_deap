"""examples/nsga2_zdt1.py —— NSGA2 求解 ZDT1 双目标问题。

ZDT1（Zitzler-Deb-Thiele 1）：
    f1(x) = x1
    g(x) = 1 + 9/(n-1) * sum(x[1:])
    f2(x) = g * (1 - sqrt(f1/g))

x ∈ [0,1]^30。Pareto 前沿：f2 = 1 - sqrt(f1), f1 ∈ [0,1]。
目标：最小化 (f1, f2)。

NSGA2 流程 = eaMuPlusLambda + selNSGA2：
  合并父子 → 非支配排序 → 按前沿秩+拥挤距离选 μ 个。
eaMuPlusLambda 的"父子合并选"正好是 NSGA2 的选择时机。

运行：python -m mini_deap.examples.nsga2_zdt1
"""

import random
import array
import math

from mini_deap.base.fitness import Fitness
from mini_deap.base.toolbox import Toolbox
import mini_deap.base.creator as creator
from mini_deap.tools.operators import initRepeat
from mini_deap.tools.support import Statistics, MultiStatistics
from mini_deap.tools.emo import selNSGA2, cxSimulatedBinaryBounded, mutPolynomialBounded
from mini_deap.algorithms import eaMuPlusLambda


def zdt1(ind):
    """ZDT1 评估：返回 (f1, f2)。"""
    n = len(ind)
    f1 = ind[0]
    g = 1.0 + 9.0 / (n - 1) * sum(ind[1:])
    h = 1.0 - math.sqrt(f1 / g) if g > 0 else 0.0
    f2 = g * h
    return (f1, f2)


def main(n=30, mu=100, lambda_=100, ngen=50, seed=42):
    random.seed(seed)

    # 双目标最小化
    creator.create("FitMulti", Fitness, weights=(-1.0, -1.0))
    creator.create("Ind", array.array, typecode="d", fitness=creator.FitMulti)

    tb = Toolbox()
    tb.register("attr_float", random.random)          # [0, 1]
    tb.register("individual", initRepeat, creator.Ind, tb.attr_float, n)
    tb.register("population", initRepeat, list, tb.individual)
    tb.register("evaluate", zdt1)
    # SBX 交叉 + 多项式变异（连续多目标标准算子）
    low, up = 0.0, 1.0
    tb.register("mate", cxSimulatedBinaryBounded,
                low=low, up=up, eta=20.0)
    tb.register("mutate", mutPolynomialBounded,
                low=low, up=up, eta=20.0, indpb=1.0 / n)
    tb.register("select", selNSGA2)

    pop = tb.population(n=mu)
    hof = creator.create  # 占位，NSGA2 用 ParetoFront 更好

    # 用 MultiStatistics 同时统计两个目标
    stats = MultiStatistics(
        f1=Statistics(key=lambda ind: ind.fitness.values[0]),
        f2=Statistics(key=lambda ind: ind.fitness.values[1]),
    )
    stats.register("min", min)
    stats.register("max", max)

    print(f"=== NSGA2-ZDT1 (n={n}, mu={mu}, lambda={lambda_}, ngen={ngen}) ===")
    pop, logbook = eaMuPlusLambda(pop, tb, mu, lambda_, 0.9, 0.1, ngen,
                                  stats=stats, verbose=True)

    # 输出 Pareto 前沿采样
    print("\nPareto 前沿采样（前 10 个）：")
    for ind in pop[:10]:
        print(f"  f1={ind.fitness.values[0]:.4f}, f2={ind.fitness.values[1]:.4f}")

    return pop


if __name__ == "__main__":
    main()