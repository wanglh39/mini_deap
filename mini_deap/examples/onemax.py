"""examples/onemax.py —— One-Max 问题：位串求和最大化。

最简单的进化算法验证：个体是 N 位 0/1 串，fitness = sum（1 的个数）。
最优解是全 1，fitness = N。用 eaSimple（简单 GA）求解。

运行：python -m mini_deap.examples.onemax
"""

import random

from mini_deap.base.fitness import Fitness
from mini_deap.base.toolbox import Toolbox
import mini_deap.base.creator as creator
from mini_deap.tools.operators import initRepeat, selTournament, cxOnePoint, mutFlipBit
from mini_deap.tools.support import Statistics, HallOfFame
from mini_deap.algorithms import eaSimple


def main(ind_len=50, pop_size=100, ngen=40, cxpb=0.5, mutpb=0.2, seed=42):
    random.seed(seed)

    creator.create("FitMax", Fitness, weights=(1.0,))
    creator.create("Ind", list, fitness=creator.FitMax)

    tb = Toolbox()
    tb.register("attr_bool", random.randint, 0, 1)
    tb.register("individual", initRepeat, creator.Ind, tb.attr_bool, ind_len)
    tb.register("population", initRepeat, list, tb.individual)
    tb.register("evaluate", lambda ind: (sum(ind),))
    tb.register("mate", cxOnePoint)
    tb.register("mutate", mutFlipBit, indpb=1.0 / ind_len)
    tb.register("select", selTournament, tournsize=3)

    pop = tb.population(n=pop_size)
    hof = HallOfFame(maxsize=1)
    stats = Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda x: sum(x) / len(x))
    stats.register("max", max)

    print(f"=== One-Max (len={ind_len}, pop={pop_size}, ngen={ngen}) ===")
    pop, logbook = eaSimple(pop, tb, cxpb, mutpb, ngen,
                            stats=stats, halloffame=hof, verbose=True)

    best = hof[0]
    print(f"\n最优 fitness = {best.fitness.values[0]:.0f} / {ind_len}")
    return best.fitness.values[0]


if __name__ == "__main__":
    main()