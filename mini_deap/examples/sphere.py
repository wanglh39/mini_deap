"""examples/sphere.py —— Sphere 函数最小化（连续优化）。

Sphere 函数：f(x) = sum(x_i^2)，x ∈ [-5.12, 5.12]^n。
全局最小 f(0,...,0) = 0。用 eaMuPlusLambda（(μ+λ) 进化策略）求解。

验证：连续个体（list of float）、mutGaussian 高斯变异、eaMuPlusLambda 精英保留。

运行：python -m mini_deap.examples.sphere
"""

import random
import array

from mini_deap.base.fitness import Fitness
from mini_deap.base.toolbox import Toolbox
import mini_deap.base.creator as creator
from mini_deap.tools.operators import initRepeat, selBest, cxBlend, mutGaussian
from mini_deap.tools.support import Statistics, HallOfFame
from mini_deap.algorithms import eaMuPlusLambda


def main(dim=10, mu=50, lambda_=100, ngen=100, seed=42):
    random.seed(seed)

    # 最小化：weights = (-1.0,)
    creator.create("FitMin", Fitness, weights=(-1.0,))
    # 用 array.array 存连续值（比 list 省内存，且测试 class_replacers）
    creator.create("Ind", array.array, typecode="d", fitness=creator.FitMin)

    tb = Toolbox()
    tb.register("attr_float", random.uniform, -5.12, 5.12)
    tb.register("individual", initRepeat, creator.Ind, tb.attr_float, dim)
    tb.register("population", initRepeat, list, tb.individual)
    # Sphere: f(x) = sum(x_i^2)，返回 tuple
    tb.register("evaluate", lambda ind: (sum(x * x for x in ind),))
    # cxBlend: alpha=0.5 混合交叉
    tb.register("mate", cxBlend, alpha=0.5)
    # mutGaussian: mu=0, sigma=0.2, 每维概率 1/dim
    tb.register("mutate", mutGaussian, mu=0, sigma=0.2, indpb=1.0 / dim)
    # selBest: (μ+λ) 用确定性选择也 OK（有 lambda_ 子代注入多样性）
    tb.register("select", selBest)

    pop = tb.population(n=mu)
    hof = HallOfFame(maxsize=1)
    stats = Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda x: sum(x) / len(x))
    stats.register("min", min)

    print(f"=== Sphere (dim={dim}, mu={mu}, lambda={lambda_}, ngen={ngen}) ===")
    pop, logbook = eaMuPlusLambda(pop, tb, mu, lambda_, 0.5, 0.2, ngen,
                                  stats=stats, halloffame=hof, verbose=True)

    best = hof[0]
    print(f"\n最优 f(x) = {best.fitness.values[0]:.6f} (目标 0.0)")
    return best.fitness.values[0]


if __name__ == "__main__":
    main()