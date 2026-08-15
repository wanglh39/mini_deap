"""进化算法主循环骨架。

本模块把前 5 阶段（Fitness / Toolbox / Creator / 算子 / 统计组件）串成
可运行的算法。只含最常用的 5 个函数：

    varAnd            —— 交叉 AND 变异（标准 GA 的变异阶段）
    varOr             —— 交叉 OR 变异 OR 复制（(μ+λ)/(μ,λ) ES 的变异阶段）
    eaSimple          —— 简单 GA（Generational GA），用 varAnd
    eaMuPlusLambda    —— (μ+λ) 进化策略，用 varOr
    eaMuCommaLambda   —— (μ,λ) 进化策略，用 varOr

核心设计思想（和 deap 一致）：
  1. 惰性评估：只重算 fitness 无效的个体（not ind.fitness.valid），
     交叉/变异后 del ind.fitness.values 使其失效，避免重复评估黑盒。
  2. toolbox.map 并行：用 toolbox.map(toolbox.evaluate, invalid_ind) 批量评估，
     默认是内置 map（串行），注册 multiprocessing.Pool.map 即可并行。
  3. stats / halloffame 可选：算法里 if stats: / if halloffame is not None:
     守卫，不传也能跑，算法代码不被观测逻辑淹没。
  4. population[:] = offspring 原地替换：保持外部对 population 列表的引用，
     用切片赋值而非 population = offspring（那只会改局部变量）。
"""

import random

from . import tools


# ───────────────────────── 变异阶段 ─────────────────────────

def varAnd(population, toolbox, cxpb, mutpb):
    """交叉 AND 变异：先交叉后变异，两个操作都可能施加到同一个体。

    流程：
      1. 全体 clone 一份（不污染父代）
      2. 相邻两两配对，以 cxpb 概率交叉，交叉后 del fitness.values 失效
      3. 逐个以 mutpb 概率变异，变异后 del fitness.values 失效

    "And" 的含义：交叉和变异都施加（各自按概率），一个个体可能
    只被交叉、只被变异、两者都施加、或都不施加（原样复制）。

    Args:
        population: 父代种群（列表）
        toolbox:    注册了 mate / mutate / clone 的 Toolbox
        cxpb:       交叉概率 ∈ [0, 1]
        mutpb:      变异概率 ∈ [0, 1]
    Returns:
        offspring:  变异后的子代种群（独立于父代，已 clone）
    """
    offspring = [toolbox.clone(ind) for ind in population]

    # 交叉：步长 2，相邻配对 (0,1), (2,3), ...
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    # 变异：逐个判定
    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring


def varOr(population, toolbox, lambda_, cxpb, mutpb):
    """交叉 OR 变异 OR 复制：每个子代只由一种操作产生。

    流程（循环 lambda_ 次产子）：
      - 以 cxpb 概率：随机选两个父本 clone 后交叉，取第一个子代
      - 以 mutpb 概率：随机选一个父本 clone 后变异
      - 否则（1 - cxpb - mutpb）：随机选一个父本直接复制（不 clone，
        因为不会被修改；但后续若 in-place 改它会污染父代——deap 原版如此）

    "Or" 的含义：每个子代只来自一种操作，不会既交叉又变异。
    约束：cxpb + mutpb <= 1.0，余下概率是复制。

    Args:
        population: 父代种群
        toolbox:    注册了 mate / mutate / clone 的 Toolbox
        lambda_:    要产生的子代数量
        cxpb:       交叉概率
        mutpb:      变异概率
    Returns:
        offspring:  lambda_ 个子代
    """
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:                                    # 交叉
            ind1, ind2 = [toolbox.clone(i) for i in random.sample(population, 2)]
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:                          # 变异
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:                                                   # 复制
            offspring.append(random.choice(population))

    return offspring


# ───────────────────────── 评估辅助 ─────────────────────────

def _evaluate_invalid(population, toolbox):
    """惰性评估：只重算 fitness 无效的个体，返回评估数。

    核心优化：交叉/变异后 del ind.fitness.values 使 fitness.valid=False。
    这里只挑 invalid 的评估，复制的个体 fitness 仍有效，跳过省黑盒调用。
    toolbox.map 默认是 map（串行），注册 Pool.map 即并行。
    """
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    return len(invalid_ind)


# ───────────────────────── 算法主循环 ─────────────────────────

def eaSimple(population, toolbox, cxpb, mutpb, ngen,
             stats=None, halloffame=None, verbose=__debug__):
    """简单遗传算法（Generational GA）。

    伪代码：
        evaluate(population)
        for g in range(ngen):
            offspring = select(population, len(population))   # 选择
            offspring = varAnd(offspring, toolbox, cxpb, mutpb)  # 交叉+变异
            evaluate(offspring)                                # 惰性评估
            population = offspring                             # 1:1 替换

    特点：
      - 1:1 替换：子代完全取代父代，要求选择算子是随机的且允许重复选
        （如 selTournament / selRoulette），否则选 n 个从 n 个里等于没选。
      - 用 varAnd：交叉和变异都施加（各自按概率）。

    Args:
        population:  初始种群（会被原地修改）
        toolbox:     注册了 mate / mutate / select / evaluate / clone 的 Toolbox
        cxpb, mutpb: 交叉/变异概率
        ngen:        迭代代数
        stats:       可选 Statistics / MultiStatistics
        halloffame:  可选 HallOfFame
        verbose:     是否打印每代统计
    Returns:
        (population, logbook)
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # 初始种群的惰性评估
    nevals = _evaluate_invalid(population, toolbox)
    if halloffame is not None:
        halloffame.update(population)
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=nevals, **record)
    if verbose:
        print(logbook.stream)

    # 进化主循环
    for gen in range(1, ngen + 1):
        offspring = toolbox.select(population, len(population))
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        nevals = _evaluate_invalid(offspring, toolbox)
        if halloffame is not None:
            halloffame.update(offspring)
        population[:] = offspring
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=nevals, **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def eaGenerateUpdate(toolbox, ngen, halloffame=None, stats=None,
                     verbose=__debug__):
    """ask-tell 算法骨架（给 CMA-ES 等用）。

    伪代码：
        for g in range(ngen):
            population = toolbox.generate()    # ask：策略采样
            evaluate(population)
            toolbox.update(population)          # tell：策略更新

    CMA-ES 等基于策略的算法用这个骨架——generate 采样，update 更新策略。
    与 eaSimple/eaMuPlusLambda 不同：没有 select/mate/mutate，
    策略自己管采样和更新。

    Args:
        toolbox:     注册了 generate / evaluate / update 的 Toolbox
        ngen:        迭代代数
        stats:       可选 Statistics
        halloffame:  可选 HallOfFame
        verbose:     是否打印
    Returns:
        (population, logbook)
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    for gen in range(ngen):
        # ask：策略采样
        population = toolbox.generate()
        # 评估
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        # tell：策略更新
        toolbox.update(population)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    """(μ + λ) 进化策略：从 父代+子代 中选 μ 个。

    伪代码：
        evaluate(population)
        for g in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(population + offspring, mu)    # 父子合并选

    特点：
      - 精英保留：父代参与选择，最优个体可一直存活（除非被更优取代）。
      - 用 varOr：每个子代只来自一种操作。
      - lambda_ 是每代子代数，mu 是存活数，两者独立。

    Args:
        population:  初始种群（长度应为 mu）
        mu:          每代存活个体数
        lambda_:     每代产生子代数
        其余同 eaSimple
    Returns:
        (population, logbook)
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    nevals = _evaluate_invalid(population, toolbox)
    if halloffame is not None:
        halloffame.update(population)
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=nevals, **record)
    if verbose:
        print(logbook.stream)

    for gen in range(1, ngen + 1):
        # 变异产子
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
        # 惰性评估
        nevals = _evaluate_invalid(offspring, toolbox)
        # 名人堂
        if halloffame is not None:
            halloffame.update(offspring)
        # 父子合并选 mu 个（精英保留）
        population[:] = toolbox.select(population + offspring, mu)
        # 记录
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=nevals, **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def eaMuCommaLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                     stats=None, halloffame=None, verbose=__debug__):
    """(μ, λ) 进化策略：只从子代中选 μ 个，父代全部淘汰。

    伪代码：
        evaluate(population)
        for g in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(offspring, mu)                # 只从子代选

    特点：
      - 无精英保留：父代全部淘汰，避免早熟收敛，鼓励探索。
      - 要求 lambda_ >= mu（子代要多于存活数，否则没得选）。
      - 适合自适应参数变异（变异强度自身在进化，父代保留会锁死参数）。

    Args:
        population:  初始种群（长度应为 mu）
        mu:          每代存活个体数
        lambda_:     每代产生子代数（必须 >= mu）
        其余同 eaSimple
    Returns:
        (population, logbook)
    """
    assert lambda_ >= mu, "lambda must be greater or equal to mu."

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    nevals = _evaluate_invalid(population, toolbox)
    if halloffame is not None:
        halloffame.update(population)
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=nevals, **record)
    if verbose:
        print(logbook.stream)

    for gen in range(1, ngen + 1):
        # 变异产子
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
        # 惰性评估
        nevals = _evaluate_invalid(offspring, toolbox)
        # 名人堂
        if halloffame is not None:
            halloffame.update(offspring)
        # 只从子代选 mu 个（父代全淘汰）
        population[:] = toolbox.select(offspring, mu)
        # 记录
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=nevals, **record)
        if verbose:
            print(logbook.stream)

    return population, logbook