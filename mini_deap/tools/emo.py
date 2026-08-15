"""tools/emo.py —— 多目标进化算子（NSGA-II）。

NSGA-II (Non-dominated Sorting Genetic Algorithm II) 是最经典的多目标
进化算法。核心三件：

    sortNondominated   —— 快速非支配排序，把种群按 Pareto 支配关系分层
    assignCrowdingDist —— 拥挤距离，同层内衡量个体周围密度
    selNSGA2           —— NSGA2 选择：先按前沿秩，再按拥挤距离

多目标的核心难点：没有全局最优，而是一个 Pareto 前沿。选择要同时考虑
"接近前沿"（支配秩小）和"分布均匀"（拥挤距离大）。

参考：Deb, Pratap, Agarwal, Meyarivan, "A fast elitist non-dominated
sorting genetic algorithm for multi-objective optimization: NSGA-II", 2002.
"""

from collections import defaultdict
from itertools import chain
from operator import attrgetter
import random


# ───────────────────────── 非支配排序 ─────────────────────────

def sortNondominated(individuals, k, first_front_only=False):
    """快速非支配排序：把个体分入 successive Pareto 前沿。

    算法（Deb 2002, Fast Nondominated Sort）：
      1. 对每个个体 i，统计支配它的个体数 dominating_count[i]，
         及它支配的个体列表 dominated_list[i]。
      2. dominating_count == 0 的个体属于第一前沿（Pareto 最优）。
      3. 对第一前沿的每个个体，把它支配的个体的 dominating_count 减 1，
         减到 0 的进入下一前沿。重复直到排完 k 个或全部。

    复杂度 O(M·N²)，M 目标数，N 个体数。

    Args:
        individuals:       待排序个体列表
        k:                 最多排 k 个（提前终止省时）
        first_front_only:  只排第一前沿（求 Pareto 最优集）
    Returns:
        fronts: Pareto 前沿列表，fronts[0] 是非支配层（最优）
    """
    if k == 0:
        return []

    # 按 fitness 去重（相同 fitness 的个体共享支配关系）
    map_fit_ind = defaultdict(list)
    for ind in individuals:
        map_fit_ind[ind.fitness].append(ind)
    fits = list(map_fit_ind.keys())

    current_front = []
    next_front = []
    dominating_fits = defaultdict(int)   # fit 被多少个 fit 支配
    dominated_fits = defaultdict(list)    # fit 支配哪些 fit

    # 排第一前沿：两两比较 fitness
    for i, fit_i in enumerate(fits):
        for fit_j in fits[i + 1:]:
            if fit_i.dominates(fit_j):
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif fit_j.dominates(fit_i):
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)

    # 把第一前沿的 fitness 对应的个体放入 fronts[0]
    fronts = [[]]
    for fit in current_front:
        fronts[-1].extend(map_fit_ind[fit])
    pareto_sorted = len(fronts[-1])

    # 排后续前沿直到排完 k 个
    if not first_front_only:
        N = min(len(individuals), k)
        while pareto_sorted < N:
            fronts.append([])
            for fit_p in current_front:
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1
                    if dominating_fits[fit_d] == 0:
                        next_front.append(fit_d)
                        pareto_sorted += len(map_fit_ind[fit_d])
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []

    return fronts


# ───────────────────────── 拥挤距离 ─────────────────────────

def assignCrowdingDist(individuals):
    """计算并分配拥挤距离到 ind.fitness.crowding_dist。

    拥挤距离：个体在目标空间中两侧邻居的矩形包围盒周长。
    边界个体（每维的 min/max）距离 = inf，保证不被淘汰。
    距离越大 → 周围越稀疏 → 选择优先（维持前沿分布均匀）。

    算法：
      对每个目标维 i：
        按 f_i 排序个体
        边界两个赋 inf
        中间的 += (f_{i+1} - f_{i-1}) / (f_max - f_min)  归一化
    """
    if len(individuals) == 0:
        return

    distances = [0.0] * len(individuals)
    crowd = [(ind.fitness.values, i) for i, ind in enumerate(individuals)]

    nobj = len(individuals[0].fitness.values)

    for i in range(nobj):
        crowd.sort(key=lambda element: element[0][i])      # 按第 i 目标排序
        distances[crowd[0][1]] = float("inf")              # 边界赋 inf
        distances[crowd[-1][1]] = float("inf")
        if crowd[-1][0][i] == crowd[0][0][i]:              # 该维全相同，跳过
            continue
        norm = nobj * float(crowd[-1][0][i] - crowd[0][0][i])  # 归一化因子
        for prev, cur, nxt in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
            distances[cur[1]] += (nxt[0][i] - prev[0][i]) / norm

    for i, dist in enumerate(distances):
        individuals[i].fitness.crowding_dist = dist


# ───────────────────────── NSGA2 选择 ─────────────────────────

def selNSGA2(individuals, k, nd='standard'):
    """NSGA-II 选择：选 k 个个体。

    流程：
      1. 非支配排序分层
      2. 每层算拥挤距离
      3. 整层整层地取，直到最后一层放不下 → 按拥挤距离排序取前几个

    选择标准：先按前沿秩（rank 小的层优先），同层内按拥挤距离（大的优先）。
    这保证"接近前沿" + "分布均匀"。

    Args:
        individuals: 候选个体（通常 len > k，从父代+子代选）
        k:           选 k 个
        nd:          非支配排序方法（'standard'，'log' 未实现）
    Returns:
        chosen: 选中的 k 个个体
    """
    if nd == 'standard':
        pareto_fronts = sortNondominated(individuals, k)
    else:
        raise ValueError(f"selNSGA2: unknown nd method '{nd}'")

    for front in pareto_fronts:
        assignCrowdingDist(front)

    # 整层取前面的前沿
    chosen = list(chain(*pareto_fronts[:-1]))
    k = k - len(chosen)
    # 最后一层放不下，按拥挤距离排序取前 k 个
    if k > 0:
        sorted_front = sorted(pareto_fronts[-1],
                              key=attrgetter("fitness.crowding_dist"),
                              reverse=True)
        chosen.extend(sorted_front[:k])

    return chosen


# ───────────────────────── DCD 锦标赛选择 ─────────────────────────

def selTournamentDCD(individuals, k):
    """基于支配+拥挤距离的锦标赛选择（NSGA2 原版选择）。

    两个个体比较：
      1. 若一个支配另一个 → 选支配者
      2. 否则 → 选拥挤距离大的
      3. 都相同 → 随机

    要求个体已分配 crowding_dist（先调 assignCrowdingDist）。
    len(individuals) 必须是 4 的倍数且 k == len 时。

    Args:
        individuals: 候选个体（需有 crowding_dist）
        k:           选 k 个
    Returns:
        chosen: 选中的 k 个
    """
    if k > len(individuals):
        raise ValueError("selTournamentDCD: k must be <= len(individuals)")
    if k == len(individuals) and k % 4 != 0:
        raise ValueError("selTournamentDCD: k must be divisible by 4 if k == len")

    def tourn(ind1, ind2):
        if ind1.fitness.dominates(ind2.fitness):
            return ind1
        elif ind2.fitness.dominates(ind1.fitness):
            return ind2
        if ind1.fitness.crowding_dist < ind2.fitness.crowding_dist:
            return ind2
        elif ind1.fitness.crowding_dist > ind2.fitness.crowding_dist:
            return ind1
        return ind1 if random.random() <= 0.5 else ind2

    individuals_1 = random.sample(individuals, len(individuals))
    individuals_2 = random.sample(individuals, len(individuals))

    chosen = []
    for i in range(0, k, 4):
        chosen.append(tourn(individuals_1[i], individuals_1[i + 1]))
        chosen.append(tourn(individuals_1[i + 2], individuals_1[i + 3]))
        chosen.append(tourn(individuals_2[i], individuals_2[i + 1]))
        chosen.append(tourn(individuals_2[i + 2], individuals_2[i + 3]))

    return chosen


# ───────────────────────── SBX 交叉 + 多项式变异 ─────────────────────────
# NSGA2 的标准配套算子，适合有界连续空间。

def cxSimulatedBinaryBounded(ind1, ind2, eta, low, up):
    """模拟二进制交叉（SBX），有界版。

    eta 大 → 子代像父代；eta 小 → 子代差异大。
    low/up 是标量边界（各维相同）。in-place 修改 ind1, ind2。
    """
    size = min(len(ind1), len(ind2))
    for i in range(size):
        if random.random() <= 0.5:
            if abs(ind1[i] - ind2[i]) > 1e-14:
                x1 = min(ind1[i], ind2[i])
                x2 = max(ind1[i], ind2[i])
                rand = random.random()

                beta = 1.0 + (2.0 * (x1 - low) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

                beta = 1.0 + (2.0 * (up - x2) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

                c1 = min(max(c1, low), up)
                c2 = min(max(c2, low), up)

                if random.random() <= 0.5:
                    ind1[i] = c2
                    ind2[i] = c1
                else:
                    ind1[i] = c1
                    ind2[i] = c2

    return ind1, ind2


def mutPolynomialBounded(individual, eta, low, up, indpb):
    """多项式变异（有界版），NSGA-II 原版实现。

    eta 大 → 变异小；eta 小 → 变异大。
    low/up 标量边界。每维以 indpb 概率变异。in-place。
    """
    size = len(individual)
    for i in range(size):
        if random.random() <= indpb:
            x = individual[i]
            delta_1 = (x - low) / (up - low)
            delta_2 = (up - x) / (up - low)
            rand = random.random()
            mut_pow = 1.0 / (eta + 1.0)

            if rand < 0.5:
                xy = 1.0 - delta_1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta_2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                delta_q = 1.0 - val ** mut_pow

            x = x + delta_q * (up - low)
            x = min(max(x, low), up)
            individual[i] = x
    return individual,