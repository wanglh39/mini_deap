"""examples/tsp.py —— 旅行商问题（排列编码 + 自定义算子）。

TSP：给定 n 个城市的坐标，找最短哈密顿回路。
个体是城市排列 [0,1,2,...,n-1] 的一个排列，fitness = 总回路长度（最小化）。

验证：自定义个体/交叉/变异、eaSimple 对排列编码的适用性。
展示：如何用 mini_deap 的框架跑自定义问题——只需注册 4 个算子即可。

运行：python -m mini_deap.examples.tsp
"""

import random
import math

from mini_deap.base.fitness import Fitness
from mini_deap.base.toolbox import Toolbox
import mini_deap.base.creator as creator
from mini_deap.tools.operators import initRepeat, selTournament
from mini_deap.tools.support import Statistics, HallOfFame
from mini_deap.algorithms import eaSimple


# ────────────── 自定义排列算子 ──────────────

def cxOrdered(ind1, ind2):
    """顺序交叉（OX）：保留 ind1 一段，其余从 ind2 按序填充。

    保证子代仍是合法排列（无重复）。in-place 修改 ind1, ind2。
    """
    size = len(ind1)
    a, b = sorted(random.sample(range(size), 2))
    # ind1[a:b] 保留，其余从 ind2[非 ind1[a:b]] 按序填
    hole = set(ind1[a:b])
    rest = [x for x in ind2 if x not in hole]
    ind1[:] = rest[:a] + ind1[a:b] + rest[a:]
    # 对称地处理 ind2
    hole2 = set(ind2[a:b])
    rest2 = [x for x in ind1 if x not in hole2]
    ind2[:] = rest2[:a] + ind2[a:b] + rest2[a:]
    return ind1, ind2


def mutSwap(ind, indpb=0.1):
    """交换变异：以 indpb 概率交换两个随机位置。in-place。"""
    if random.random() < indpb:
        i, j = random.sample(range(len(ind)), 2)
        ind[i], ind[j] = ind[j], ind[i]
    return ind,


# ────────────── TSP 评估 ──────────────

def make_distance(cities):
    """预计算距离矩阵（n×n），避免每次评估重算 sqrt。"""
    n = len(cities)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dx = cities[i][0] - cities[j][0]
            dy = cities[i][1] - cities[j][1]
            d = math.hypot(dx, dy)
            dist[i][j] = dist[j][i] = d
    return dist


def tour_length(ind, dist):
    """回路长度：ind[0]→ind[1]→...→ind[-1]→ind[0]。"""
    total = 0.0
    n = len(ind)
    for i in range(n):
        total += dist[ind[i]][ind[(i + 1) % n]]
    return (total,)


# ────────────── 主流程 ──────────────

def main(n_cities=30, pop_size=100, ngen=50, seed=42):
    random.seed(seed)

    # 随机生成城市坐标
    cities = [(random.uniform(0, 100), random.uniform(0, 100))
              for _ in range(n_cities)]
    dist = make_distance(cities)

    # 最小化距离：weights = (-1.0,)
    creator.create("FitMin", Fitness, weights=(-1.0,))
    creator.create("Ind", list, fitness=creator.FitMin)

    tb = Toolbox()
    # 个体是 [0,1,...,n-1] 的随机排列
    tb.register("individual", lambda: creator.Ind(random.sample(range(n_cities), n_cities)))
    tb.register("population", initRepeat, list, tb.individual)
    tb.register("evaluate", tour_length, dist=dist)
    tb.register("mate", cxOrdered)
    tb.register("mutate", mutSwap, indpb=0.2)
    tb.register("select", selTournament, tournsize=3)

    pop = tb.population(n=pop_size)
    hof = HallOfFame(maxsize=1)
    stats = Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda x: sum(x) / len(x))
    stats.register("min", min)

    print(f"=== TSP (cities={n_cities}, pop={pop_size}, ngen={ngen}) ===")
    pop, logbook = eaSimple(pop, tb, 0.7, 0.2, ngen,
                            stats=stats, halloffame=hof, verbose=True)

    best = hof[0]
    print(f"\n最短回路 = {best.fitness.values[0]:.2f}")
    print(f"路径 = {list(best)}")
    return best.fitness.values[0]


if __name__ == "__main__":
    main()