"""examples/cma_es.py —— CMA-ES 优化 Sphere 和 Rosenbrock。

CMA-ES 是连续优化的"黄金标准"——自适应学习协方差矩阵，
无需调参，在非凸/非光滑/病态问题上表现优异。

对比：
  Sphere:   f(x) = Σx²，单峰，各向同性 → CMA-ES 快速收敛
  Rosenbrock: f(x) = Σ(100(x_{i+1}-x_i²)² + (1-x_i)²)，非凸，狭长谷 → 测试协方差学习

运行：python -m mini_deap.examples.cma_es
"""

import random
import numpy

from mini_deap.base.fitness import Fitness
from mini_deap.base.toolbox import Toolbox
import mini_deap.base.creator as creator
from mini_deap.cma import Strategy
from mini_deap.algorithms import eaGenerateUpdate
from mini_deap.tools.support import Statistics, HallOfFame


def sphere(ind):
    return (sum(x * x for x in ind),)


def rosenbrock(ind):
    return (sum(100 * (ind[i+1] - ind[i]**2)**2 + (1 - ind[i])**2
                for i in range(len(ind) - 1)),)


def run_cma(func, dim, centroid, sigma, ngen, name):
    creator.create("FitMin", Fitness, weights=(-1.0,))
    creator.create("Ind", list, fitness=creator.FitMin)

    strategy = Strategy(centroid=centroid, sigma=sigma)

    tb = Toolbox()
    tb.register("evaluate", func)
    tb.register("generate", strategy.generate, creator.Ind)
    tb.register("update", strategy.update)

    hof = HallOfFame(maxsize=1)
    stats = Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("min", min)
    stats.register("avg", lambda x: sum(x) / len(x))

    print(f"\n=== CMA-ES {name} (dim={dim}, ngen={ngen}) ===")
    pop, log = eaGenerateUpdate(tb, ngen, stats=stats, halloffame=hof,
                                verbose=True)

    best = hof[0]
    print(f"最优 f(x) = {best.fitness.values[0]:.8f}")
    print(f"均值 = {strategy.centroid}")
    return best.fitness.values[0]


def main(dim=10, ngen=100, seed=42):
    random.seed(seed)
    numpy.random.seed(seed)

    # Sphere
    f1 = run_cma(sphere, dim, [5.0] * dim, 2.0, ngen, "Sphere")

    # Rosenbrock
    random.seed(seed)
    numpy.random.seed(seed)
    f2 = run_cma(rosenbrock, dim, [0.0] * dim, 0.5, ngen, "Rosenbrock")

    print(f"\n=== 总结 ===")
    print(f"Sphere    最优 = {f1:.8f} (目标 0.0)")
    print(f"Rosenbrock 最优 = {f2:.8f} (目标 0.0)")
    return f1, f2


if __name__ == "__main__":
    main()