# 阶段 4：tools/operators —— 进化算子（纯函数库）

> 对应 DEAP：`deap/tools/{init,selection,crossover,mutation}.py`（原版合计 ~700 行，本阶段 ~230 行，保留核心算子）
> 产出：`mini_deap/tools/operators.py` + `tests/tools/test_operators.py`（22 测试全过）

---

## 一、背景：算子在进化算法里扮演什么角色？

进化算法的"进化"靠两类算子驱动：

- **变异算子**（交叉 + 变异）：产生新个体，探索搜索空间。交叉组合父代基因，变异引入新基因。
- **选择算子**：决定谁进入下一代，施加选择压力。压力太低→随机游走不收敛；太高→早熟收敛陷局部最优。

算子库的设计目标：**提供足够多样的算子让用户组合，同时保证算子可互换**。可互换的关键是**统一接口约定**——所有算子签名、返回值、副作用规则一致，这样 `toolbox.register` 能无差别组装，算法骨架无差别调用。

DEAP 的约定：

| 算子类型 | 签名 | 返回 | 副作用 |
|---|---|---|---|
| 初始化 | `init(container, func, n)` | 填好的 container | 无 |
| 选择 | `sel(individuals, k, ...)` | list[k 个个体] | 无（返回引用，不拷贝） |
| 交叉 | `cx(ind1, ind2)` | `(ind1, ind2)` | **in-place 改 ind1/ind2** |
| 变异 | `mut(ind, ...)` | `(ind,)` | **in-place 改 ind** |

交叉/变异返回元组是为了配合 `offspring[i-1], offspring[i] = toolbox.mate(...)` 解包。in-place 是为效率（不造新对象），但要求调用方先 `clone`（`varAnd` 里有 `[toolbox.clone(ind) for ind in population]`），否则污染原种群。

---

## 二、核心设计思想

### ① 纯函数 + in-place 约定 —— 算子无状态，副作用受控

算子全是模块级函数，不持有状态，不依赖全局。副作用只有"in-place 改输入个体"一种，且由调用方负责 clone 隔离。这让算子：

- **可自由组合**：任意选择 + 任意交叉 + 任意变异都能拼。
- **可并行**：无共享状态，`pool.map` 安全。
- **可单元测试**：给定输入确定输出，不依赖环境。

### ② `attrgetter("fitness")` 解耦 —— 算子不直接访问个体的 fitness 属性

```python
def selBest(individuals, k, fit_attr="fitness"):
    return sorted(individuals, key=attrgetter(fit_attr), reverse=True)[:k]
```

`fit_attr` 默认 `"fitness"`，但可传别的属性名。算子通过 `attrgetter` 间接取适应度，不硬编码 `ind.fitness`。这让算子能适配"fitness 存在别的属性名上"的个体（虽然很少这么干，但留了口子）。deap 全部选择算子都这么写。

### ③ 选择压力由参数连续控制 —— 锦标赛为例

```python
def selTournament(individuals, k, tournsize, fit_attr="fitness"):
    chosen = []
    for _ in range(k):
        aspirants = selRandom(individuals, tournsize)
        chosen.append(max(aspirants, key=attrgetter(fit_attr)))
    return chosen
```

`tournsize` 连续控制选择压力：
- `tournsize=1`：每次从 1 个抽签者取 max = 那个抽签者本身 → 退化为 `selRandom`，压力 0。
- `tournsize=len(pop)`：每次从全种群取 max → 退化为 `selBest`，压力无穷。
- `tournsize=2~5`：常用值，中等压力。

一个参数覆盖从随机到精英的全谱，不用换算子。这是锦标赛选择最优雅的地方。

---

### ④ 算子速查表

| 类别 | 算子 | 签名 | 适用个体 |
|---|---|---|---|
| 初始化 | `initRepeat` | `(container, func, n)` | 任意（重复填） |
| | `initIterate` | `(container, generator)` | 排列（一次性可迭代） |
| | `initCycle` | `(container, seq_func, n)` | 混合类型（循环填） |
| 选择 | `selRandom` | `(inds, k)` | 任意（基线） |
| | `selBest` | `(inds, k)` | 任意（精英） |
| | `selTournament` | `(inds, k, tournsize)` | 任意（压力可调） |
| | `selRoulette` | `(inds, k)` | 最大化 + 正 fitness |
| 交叉 | `cxOnePoint` | `(ind1, ind2)` | 序列（定长） |
| | `cxTwoPoint` | `(ind1, ind2)` | 序列（定长） |
| | `cxUniform` | `(ind1, ind2, indpb)` | 序列（逐位交换） |
| | `cxBlend` | `(ind1, ind2, alpha)` | 实数（凸组合） |
| 变异 | `mutGaussian` | `(ind, mu, sigma, indpb)` | 实数（加噪声） |
| | `mutFlipBit` | `(ind, indpb)` | 二进制（取反） |
| | `mutShuffleIndexes` | `(ind, indpb)` | 排列（交换位） |
| | `mutUniformInt` | `(ind, low, up, indpb)` | 整数（重采） |

### ⑤ 参数标量/序列双模式

`mutGaussian`、`mutUniformInt` 的 `mu`/`sigma`/`low`/`up` 既可传标量（所有位同参）也可传序列（每位独立）：

```python
mutGaussian(ind, mu=0, sigma=1, indpb=0.1)               # 标量：所有位同 N(0,1)
mutGaussian(ind, mu=[0,1,2], sigma=[1,2,3], indpb=0.1)   # 序列：位 i 用 N(i, i+1)
```

实现用 `isinstance(mu, Sequence)` 判断：标量则 `repeat(mu, size)` 造重复迭代器，序列则直接用。这让一个算子同时服务"均匀变异"和"异质变异"（不同维度不同强度），不用写两个函数。deap 全部数值算子都支持这个模式。

---

## 三、逐类精读

### 3.1 初始化：`initRepeat` 的生成器表达式

```python
def initRepeat(container, func, n):
    return container(func() for _ in range(n))
```

`container(func() for _ in range(n))`：传一个**生成器表达式**给 container。`list(生成器)` 会消费生成器填列表。关键好处：**惰性求值，不先造中间列表**。对 `n=100000` 的大种群，省一个临时列表的内存。

`container` 是类型（`list`/`set`/`creator.Individual`），所以 `initRepeat(list, random.random, 5)` 产 `[r,r,r,r,r]`，`initRepeat(Individual, lambda: initRepeat(list, random.random, 10), 100)` 产 100 个 10 维个体（嵌套调用）。

`initIterate` 用于排列（`initIterate(list, partial(random.sample, range(10), 10))` 产 0-9 的随机排列），`initCycle` 用于混合类型个体（交替填不同类型的基因）。

### 3.2 选择：`selRoulette` 纯 Python vs `selRoulette_np` numpy 对照

**纯 Python 版**（累积求和 + 线性搜索）：

```python
sum_fits = sum(getattr(ind, fit_attr).values[0] for ind in individuals)
for _ in range(k):
    u = random.random() * sum_fits
    sum_ = 0.0
    for ind in s_inds:              # 每次从头累加
        sum_ += getattr(ind, fit_attr).values[0]
        if sum_ > u:
            chosen.append(ind); break
```

每次选都从头线性累加直到超过随机数 u。复杂度 **O(k·N)**：选 k 次，每次最多扫 N 个。

**numpy 向量化版**（cumsum 一次 + 二分搜索）：

```python
fits = np.array([getattr(ind, fit_attr).values[0] for ind in individuals])
cum = np.cumsum(fits)          # 一次 O(N)
cum /= cum[-1]                 # 归一化
for _ in range(k):
    u = random.random()
    idx = int(np.searchsorted(cum, u))   # 二分 O(log N)
    chosen.append(individuals[idx])
```

`np.cumsum` 一次算好累积分布，之后每次选用 `np.searchsorted` 二分查找。复杂度 **O(N + k·log N)**。

**对照表**：

| 维度 | 纯 Python | numpy |
|---|---|---|
| 预处理 | O(N) sum | O(N) cumsum + 归一化 |
| 每次选 | O(N) 线性扫 | O(log N) 二分 |
| 总复杂度 | O(k·N) | O(N + k·log N) |
| 小种群 | 快（无 numpy 转换开销） | 略慢（数组转换） |
| 大种群 / 大 k | 慢 | 快 |
| 可读性 | 直观，看清原理 | 工程化，原理被 API 藏 |

教学价值：两版并排，**纯 Python 版让你看清"轮盘赌"的累积分布原理，numpy 版让你看工程上怎么优化**。

### 3.3 交叉：in-place 切片交换

```python
def cxOnePoint(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cxpoint = random.randint(1, size - 1)
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
    return ind1, ind2
```

`ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]` 是 Python 的**同时赋值**：右先求值成两个列表，再同时赋给左。不会出现"先赋 ind1 后 ind1 已变导致 ind2 拿错"的问题。这是 in-place 交叉的核心技巧。

`cxTwoPoint` 的切点处理：先随机两个点，若 `cxpoint2 >= cxpoint1` 则 `cxpoint2 += 1` 保证两点不同且有序。这避免了切点重合（退化为单点）和切点越界。

`cxBlend`（实数）：`c1 = ind1[i] + alpha*(ind2[i]-ind1[i])`，两个子代分别向对方偏移。`alpha=0.5` 取中点，`alpha>1` 外推（扩展搜索范围）。这是实数 GA 常用交叉。

### 3.4 变异：`mutGaussian` 纯 Python vs `mutGaussian_np` + 排列守恒

**纯 Python 版**：

```python
for i, m, s in zip(range(size), mu, sigma):
    if random.random() < indpb:
        individual[i] += random.gauss(m, s)
```

逐位：先掷 `random.random()` 决定是否变异，若是则加 `random.gauss(mu, sigma)`。`mu`/`sigma` 用 `repeat(mu, size)` 支持标量（所有位同参）或序列（每位独立）。

**numpy 向量化版**：

```python
mask = np.random.random(size) < indpb          # 批量掩码
noise = np.random.normal(mu_arr, sigma_arr, size) * mask   # 批量噪声
for i in range(size):
    individual[i] += noise[i]
```

`np.random.random(size)` 一次生成 size 个随机数，`< indpb` 得布尔掩码。`np.random.normal` 批量生成高斯噪声。乘掩码后未变异位噪声为 0。最后循环赋值（兼容 list 个体；若个体是 ndarray 可直接 `individual += noise`）。

**对照**：纯 Python 每位两次 `random` 调用（Python 层开销），numpy 批量在 C 层。高维个体（如 1000 维）numpy 快几倍；低维（如 10 维）numpy 转换开销反而不划算。

**`mutFlipBit` 的 `type(x)(not x)`**：

```python
individual[i] = type(individual[i])(not individual[i])
```

`not 0` 是 `True`（bool），但个体元素可能是 `int`。`type(0)(True)` = `int(True)` = `1`，保持 int 类型。若直接 `individual[i] = not individual[i]`，0 会变成 `True`（bool），类型不一致。

**`mutShuffleIndexes` 的排列守恒**：每位以 `indpb` 概率与另一个随机位交换。交换不改变元素集合，只打乱顺序 → 排列个体变异后仍是合法排列。这是 TSP 等排列问题的关键。

---

## 四、和 deap 原版对照

| 本教学版 | deap 原版 | 差异 |
|---|---|---|
| `initRepeat/initIterate/initCycle` | 同 | 完全保留 |
| `selRandom/selBest/selTournament/selRoulette` | 同 | 完全保留 |
| `selRoulette_np` | deap 无 | **新增**：numpy 向量化对照版 |
| `cxOnePoint/cxTwoPoint/cxUniform/cxBlend` | 同 | 完全保留 |
| `mutGaussian/mutFlipBit/mutShuffleIndexes/mutUniformInt` | 同 | 完全保留 |
| `mutGaussian_np` | deap 无 | **新增**：numpy 向量化对照版 |
| `cxPartialyMatched/cxOrdered/...` | 有 | 砍掉：排列交叉，mutShuffleIndexes 够 TSP |
| `mutPolynomialBounded` | 有 | 砍掉：NSGA2 专用，阶段 8 再加 |
| `selDoubleTournament/selLexicase/...` | 有 | 砍掉：高级选择，非核心 |

**保留全部常用算子**，新增两个 numpy 对照版，砍掉的是排列交叉和高级选择。

---

## 五、性能分析：纯 Python vs numpy 何时值得？

量化对照（`selRoulette`，种群 N，选 k=N）：

| 种群规模 N | 纯 Python O(k·N) | numpy O(N + k·log N) | 谁快 |
|---|---|---|---|
| 50 | 2500 次比较 | 50 + 50·6 ≈ 350 | 纯 Python（numpy 数组转换开销 ~1μs/元素占主导） |
| 1000 | 10⁶ 次比较 | 1000 + 1000·10 ≈ 11000 | numpy 快 ~100× |
| 100000 | 10¹⁰ 次比较 | 10⁵ + 10⁵·17 ≈ 1.8×10⁶ | numpy 快 ~5000× |

经验法则：**N < 100 用纯 Python，N > 1000 用 numpy**。中间看场景。教学版默认纯 Python（贴近 deap 原版），numpy 版作为大种群优化选项。

`mutGaussian` 类似：个体维度 D < 50 纯 Python 够，D > 500 numpy 值得。关键在 numpy 的"批量生成随机数 + 数组运算"省掉 Python 循环开销，但数组转换有固定成本，小规模不划算。

---

## 六、Python 语言特性备忘

- **生成器表达式** `(func() for _ in range(n))`：惰性求值，不造中间列表。`container(生成器)` 消费它填充。`initRepeat` 用它省内存。
- **同时赋值** `a, b = b, a`：右先整体求值再赋左。`ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]` 靠这个避免中间拷贝。拆成两行会错（第二行时 ind1 已变）。
- **`operator.attrgetter("fitness")`**：返回一个函数 `lambda obj: obj.fitness`。比 `key=lambda ind: ind.fitness` 略快（C 层）。`sorted`/`max` 的 key 参数常用。
- **`itertools.repeat(x, n)`**：返回重复 x n 次的迭代器，不造列表。`mutGaussian` 用它把标量 mu 当成"每位的 mu"。
- **`zip` 截断**：`zip(range(5), [1,2,3])` 只产 3 对，不报错。`mutGaussian` 若 mu 序列短于个体维度，高位静默不变异（deap 抛 IndexError，教学版简化）。
- **`np.searchsorted(cum, u)`**：二分查找 u 在有序数组 cum 中的插入位置。等价 `bisect.bisect_left` 但返回 numpy int。O(log N)。
- **`np.cumsum`**：累积和，`[1,2,3] → [1,3,6]`。C 层，比 `itertools.accumulate` 快。

---

## 七、常见陷阱

1. **`selRoulette` 用于最小化或负 fitness**：轮盘赌按 fitness 比例算概率，负值或最小化会出错（负概率/反向选择）。最小化用 `selTournament` 或给 fitness 加偏移。
2. **交叉/变异后忘 `del fitness.values`**：in-place 改了个体但 fitness 没失效，选择会拿旧值。`varAnd` 自动处理，自己写循环要记得。
3. **`cxOnePoint` 对不同长度个体**：`size = min(len(ind1), len(ind2))`，切点在较短者范围内，结果长度可能不保。定长个体才安全。
4. **`mutGaussian` 的 `mu`/`sigma` 序列长度不够**：若传序列且长度 < 个体维度，`zip` 会截断，高位不变异。deap 会抛 IndexError，教学版用 zip 静默截断（可接受的简化）。
5. **`selTournament` 的 `tournsize > len(pop)`**：`selRandom` 有放回抽样，`tournsize` 可以大于种群，但意义不大。通常 `tournsize` 远小于种群。
6. **numpy 版的随机数种子**：`np.random` 和 `random` 是独立的随机数流，`random.seed()` 不影响 `np.random`。要复现 numpy 版结果用 `np.random.seed()`。
7. **in-place 交叉污染原种群**：直接 `toolbox.mate(pop[i], pop[j])` 会改 pop。必须先 clone：`varAnd` 里 `offspring = [toolbox.clone(ind) for ind in population]`。

---

## 八、关键收获

1. **统一接口约定让算子可互换**：签名/返回/副作用一致 → `toolbox.register` 无差别组装，算法骨架无差别调用。
2. **纯函数 + in-place + 调用方 clone**：算子无状态可并行，副作用受控不污染。
3. **`attrgetter` 间接取属性**：算子不硬编码 `ind.fitness`，留适配口子。
4. **选择压力参数化**：`tournsize` 一个参数从随机到精英连续覆盖，不用换算子。
5. **纯 Python 看原理，numpy 看工程**：两版并排，前者讲清算法本质（累积分布/逐位变异），后者展示向量化优化（cumsum/batch），教学兼顾。
6. **排列算子要守恒**：`mutShuffleIndexes` 用交换不用替换，保证元素集合不变 → 排列合法性。

---

## 九、思考题

1. `selRoulette_np` 用 `np.searchsorted(cum, u)`，`cum` 是递增的累积概率。为什么用 `searchsorted` 而不是 `argmax(cum >= u)`？（提示：复杂度 log N vs N）
2. `cxOnePoint` 的 `ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]` 如果拆成两行 `ind1[cxpoint:] = ind2[cxpoint:]; ind2[cxpoint:] = ind1[cxpoint:]` 会怎样？（提示：第二行时 ind1 已变）
3. `mutGaussian_np` 最后用循环 `individual[i] += noise[i]` 而非 `individual += noise`，为什么？什么情况下可以直接 `+=`？（提示：list 不支持广播，ndarray 可以）
4. `selTournament` 的期望适应度：种群 fitness {1,2,3,4,5}，tournsize=2，期望被选中的 fitness 是多少？（提示：E[max of 2] = 95/25 = 3.8，对比随机期望 3.0）
5. 如果要加一个"锦标赛 + 轮盘赌混合"的选择算子，怎么组合现有算子？需要新写吗？

---

## 十、下一阶段预告

**阶段 5：tools/support.py** —— 统计与记录组件。`Statistics`（注册统计函数 + compile 种群）、`HallOfFame`（保留最优 k 个 + update）、`Logbook`（record 每代 + stream 输出）。这些是算法层的"观测插件"，让 `eaSimple` 能边跑边输出统计、保留精英。设计上它们是**可选回调**，算法层 `if stats: stats.compile(pop)`，不传也能跑。