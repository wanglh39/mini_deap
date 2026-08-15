"""tools.operators —— 进化算子（纯函数库）。

设计思想（对照 DEAP tools/{init,selection,crossover,mutation}.py）
====================================================================

1. 算子全是纯函数：接 individuals 返回 individuals。交叉/变异 **in-place 修改**输入
   个体并返回（deap 约定），配合 `toolbox.clone` 保证不污染原种群。

2. 通过 `toolbox.register` 组装，算法层不直接 import。见阶段 2。

3. 选择算子用 `attrgetter("fitness")` 取适应度，依赖个体有 `.fitness`（creator 保证）。
   比较走 Fitness 的 `__gt__` 等，自动处理 max/min（阶段 1 的 weights 统一方向）。

4. 交叉/变异返回**元组**：`(ind1, ind2)` 或 `(ind,)`。配合
   `offspring[i-1], offspring[i] = toolbox.mate(...)` 解包。

5. 关键算子给纯 Python + numpy 向量化两版对照（selRoulette / mutGaussian），
   注释讲清差异与适用场景。

简化（相比 deap/tools/）
-----------------------
- 保留核心：initRepeat/initIterate/initCycle、selRandom/selBest/selTournament/selRoulette、
  cxOnePoint/cxTwoPoint/cxUniform/cxBlend、mutGaussian/mutFlipBit/mutShuffleIndexes/mutUniformInt。
- 砍掉：cxPartialyMatched(PMX)/cxOrdered(OX) 等排列交叉（mutShuffleIndexes 够 TSP）、
  mutPolynomialBounded（NSGA2 专用，阶段 8 再加）、selDoubleTournament/selLexicase 等。
- numpy 向量化版作为对照（后缀 _np），不替换默认。
"""

import random
from collections.abc import Sequence
from itertools import repeat
from operator import attrgetter


# ==== 初始化 ====

def initRepeat(container, func, n):
    """调用 func n 次，结果装进 container 返回。

    用于初始化个体或种群::

        initRepeat(list, random.random, 5)            # [r, r, r, r, r]
        initRepeat(list, lambda: IndType(...), 100)   # 种群 100 个个体
    """
    return container(func() for _ in range(n))


def initIterate(container, generator):
    """调 generator() 拿可迭代，装进 container。用于排列个体。

        initIterate(list, partial(random.sample, range(10), 10))  # 0..9 的随机排列
    """
    return container(generator())


def initCycle(container, seq_func, n=1):
    """循环调 seq_func 里的函数 n 轮，结果装进 container。用于混合类型个体。

        initCycle(list, [lambda: random.random(), lambda: random.randint(0,1)], n=5)
    """
    return container(func() for _ in range(n) for func in seq_func)


# ==== 选择 ====

def selRandom(individuals, k):
    """随机有放回选 k 个。"""
    return [random.choice(individuals) for _ in range(k)]


def selBest(individuals, k, fit_attr="fitness"):
    """选最好的 k 个（按 fitness 降序）。"""
    return sorted(individuals, key=attrgetter(fit_attr), reverse=True)[:k]


def selTournament(individuals, k, tournsize, fit_attr="fitness"):
    """锦标赛选择：每次从 tournsize 个随机抽签者中取最优，重复 k 次。

    选择压力由 tournsize 控制：越大压力越高（更倾向最优）。
    tournsize=1 退化为随机选择；tournsize=len(pop) 退化为 selBest。
    """
    chosen = []
    for _ in range(k):
        aspirants = selRandom(individuals, tournsize)
        chosen.append(max(aspirants, key=attrgetter(fit_attr)))
    return chosen


def selRoulette(individuals, k, fit_attr="fitness"):
    """轮盘赌选择（纯 Python 版）：按 fitness 比例选。

    警告：仅适用于最大化且 fitness > 0。最小化或负 fitness 会出错。
    复杂度 O(k*N)（每次选循环累加）。
    """
    s_inds = sorted(individuals, key=attrgetter(fit_attr), reverse=True)
    sum_fits = sum(getattr(ind, fit_attr).values[0] for ind in individuals)
    chosen = []
    for _ in range(k):
        u = random.random() * sum_fits
        sum_ = 0.0
        for ind in s_inds:
            sum_ += getattr(ind, fit_attr).values[0]
            if sum_ > u:
                chosen.append(ind)
                break
    return chosen


def selRoulette_np(individuals, k, fit_attr="fitness"):
    """轮盘赌选择（numpy 向量化版）：cumsum 一次 + searchsorted 每次选。

    对照纯 Python 版：
    - 纯 Python：每次选循环累加，O(k*N)
    - numpy：cumsum 一次 O(N)，每次选 searchsorted O(log N)，总 O(N + k*log N)
    大种群下 numpy 版快；小种群 numpy 转换开销反而不划算。
    """
    import numpy as np
    fits = np.array([getattr(ind, fit_attr).values[0] for ind in individuals])
    cum = np.cumsum(fits)
    cum /= cum[-1]   # 归一化为累积概率
    chosen = []
    for _ in range(k):
        u = random.random()
        idx = int(np.searchsorted(cum, u))   # 找第一个 cum >= u 的位置
        chosen.append(individuals[idx])
    return chosen


# ==== 交叉 ====

def cxOnePoint(ind1, ind2):
    """单点交叉：随机选切点，交换后段。in-place 修改，返回 (ind1, ind2)。"""
    size = min(len(ind1), len(ind2))
    cxpoint = random.randint(1, size - 1)
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
    return ind1, ind2


def cxTwoPoint(ind1, ind2):
    """两点交叉：随机选两个切点，交换中间段。in-place。"""
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = \
        ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
    return ind1, ind2


def cxUniform(ind1, ind2, indpb):
    """均匀交叉：每位以 indpb 概率交换。in-place。"""
    size = min(len(ind1), len(ind2))
    for i in range(size):
        if random.random() < indpb:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2


def cxBlend(ind1, ind2, alpha):
    """混合交叉（实数 BLX-α 的简化）：

        child1 = parent1 + alpha * (parent2 - parent1)
        child2 = parent2 + alpha * (parent1 - parent2)

    alpha=0 不变，alpha=1 完全交换，alpha=0.5 取中点。in-place。
    """
    size = min(len(ind1), len(ind2))
    for i in range(size):
        c1 = ind1[i] + alpha * (ind2[i] - ind1[i])
        c2 = ind2[i] + alpha * (ind1[i] - ind2[i])
        ind1[i], ind2[i] = c1, c2
    return ind1, ind2


# ==== 变异 ====

def mutGaussian(individual, mu, sigma, indpb):
    """高斯变异（纯 Python 版）：每位以 indpb 概率加 N(mu, sigma)。

    mu/sigma 可以是标量（所有位同参）或序列（每位独立）。返回 (individual,)。
    """
    size = len(individual)
    if not isinstance(mu, Sequence):
        mu = repeat(mu, size)
    if not isinstance(sigma, Sequence):
        sigma = repeat(sigma, size)
    for i, m, s in zip(range(size), mu, sigma):
        if random.random() < indpb:
            individual[i] += random.gauss(m, s)
    return individual,


def mutGaussian_np(individual, mu, sigma, indpb):
    """高斯变异（numpy 向量化版）：批量生成掩码和噪声。

    对照纯 Python 版：
    - 纯 Python：逐位 random.random() + random.gauss()，Python 循环开销
    - numpy：np.random.random(size) 批量掩码 + np.random.normal 批量噪声，C 层
    高维个体下 numpy 版快。若 individual 本身是 ndarray，可直接 `individual += noise`
    去掉最后那个循环（此处保留循环以兼容 list 个体）。
    """
    import numpy as np
    size = len(individual)
    mask = np.random.random(size) < indpb
    mu_arr = np.full(size, mu) if not isinstance(mu, Sequence) else np.array(mu)
    sigma_arr = np.full(size, sigma) if not isinstance(sigma, Sequence) else np.array(sigma)
    noise = np.random.normal(mu_arr, sigma_arr, size) * mask
    for i in range(size):
        individual[i] += noise[i]
    return individual,


def mutFlipBit(individual, indpb):
    """位翻转变异（二进制）：每位以 indpb 概率取反。

    用 type(x)(not x) 保持元素类型（int 0/1 而非 bool）。
    """
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = type(individual[i])(not individual[i])
    return individual,


def mutShuffleIndexes(individual, indpb):
    """索引洗牌变异（排列）：每位以 indpb 概率与随机位交换。保持排列有效性。"""
    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            swap_idx = random.randint(0, size - 2)
            if swap_idx >= i:
                swap_idx += 1
            individual[i], individual[swap_idx] = individual[swap_idx], individual[i]
    return individual,


def mutUniformInt(individual, low, up, indpb):
    """均匀整数变异：每位以 indpb 概率重采 [low, up] 整数。"""
    size = len(individual)
    if not isinstance(low, Sequence):
        low = repeat(low, size)
    if not isinstance(up, Sequence):
        up = repeat(up, size)
    for i, lo, hi in zip(range(size), low, up):
        if random.random() < indpb:
            individual[i] = random.randint(lo, hi)
    return individual,