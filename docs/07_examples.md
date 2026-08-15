# 阶段 7：examples —— 串联实战（把零件装成整机）

> 产出：`mini_deap/examples/{onemax, sphere, tsp}.py` + `docs/07_examples.md`
> 三个经典问题验证前 6 阶段的接口兼容性

---

## 一、背景：为什么需要 examples？

前 6 阶段造了零件：Fitness / Toolbox / Creator / 算子 / support / algorithms。但零件单独测过不代表能组装成整机——**接口兼容性**只有在实际串联时才暴露。examples 是"集成测试"：

| 例子 | 问题 | 验证什么 |
|---|---|---|
| `onemax.py` | 位串求和最大化 | eaSimple + cxOnePoint + mutFlipBit + selTournament |
| `sphere.py` | 连续函数最小化 | eaMuPlusLambda + cxBlend + mutGaussian + array.array 个体 |
| `tsp.py` | 旅行商（排列编码） | 自定义算子（OX 交叉 + 交换变异）+ eaSimple |

三个例子覆盖了三种编码（位串/连续/排列）、两种算法（eaSimple/eaMuPlusLambda）、两种优化方向（max/min）。

---

## 二、One-Max：最简单的 GA

### 2.1 问题定义

个体是 N 位 0/1 串，fitness = 1 的个数，最大化。最优解全 1，fitness = N。

### 2.2 代码结构

```python
creator.create("FitMax", Fitness, weights=(1.0,))      # 最大化
creator.create("Ind", list, fitness=creator.FitMax)     # 个体是 list

tb = Toolbox()
tb.register("attr_bool", random.randint, 0, 1)          # 0/1 随机生成器
tb.register("individual", initRepeat, creator.Ind, tb.attr_bool, 50)  # 50 位
tb.register("population", initRepeat, list, tb.individual)
tb.register("evaluate", lambda ind: (sum(ind),))        # fitness = sum，返回 tuple
tb.register("mate", cxOnePoint)                         # 单点交叉
tb.register("mutate", mutFlipBit, indpb=1.0/50)         # 每位 1/50 翻转概率
tb.register("select", selTournament, tournsize=3)       # 锦标赛选择

eaSimple(pop, tb, 0.5, 0.2, 40, stats=stats, halloffame=hof)
```

### 2.3 运行结果

```
gen  nevals  avg      max
0    100     25.30    34.00
...
25   64      48.84    50.00
...
40   62      49.77    50.00

最优 fitness = 50 / 50
```

**收敛分析**：
- gen 0：avg=25.3（随机初始化约一半 1），max=34
- gen 25：max=50（已找到最优），avg=48.84（种群趋同）
- gen 40：avg=49.77（几乎全收敛到最优）

**nevals 的惰性评估**：gen 1 的 nevals=60（非 100），因为 varAnd 的 clone 保留 fitness，只有交叉/变异的个体失效。cxpb=0.5+mutpb=0.2，约 60% 个体被改 → 60 个需重评估。

### 2.4 关键点

- **`indpb=1.0/50`**：每位翻转概率 1/N，期望每位变异 1 次。太大会破坏已优化的解，太小探索不足。
- **`evaluate` 返回 tuple**：`(sum(ind),)` 不是 `sum(ind)`。fitness.values 要求可迭代且长度匹配 weights。
- **`weights=(1.0,)`**：正权 = 最大化。Fitness 的比较逻辑靠 weights 正负编码方向。

---

## 三、Sphere：连续优化 + (μ+λ) ES

### 3.1 问题定义

Sphere 函数 f(x) = Σx_i²，x ∈ [-5.12, 5.12]^n。全局最小 f(0,...,0) = 0。这是连续优化的经典测试函数，单峰、各向同性。

### 3.2 代码结构

```python
creator.create("FitMin", Fitness, weights=(-1.0,))     # 最小化：负权
creator.create("Ind", array.array, typecode="d", fitness=creator.FitMin)  # float 数组

tb.register("attr_float", random.uniform, -5.12, 5.12)
tb.register("individual", initRepeat, creator.Ind, tb.attr_float, 10)  # 10 维
tb.register("evaluate", lambda ind: (sum(x*x for x in ind),))          # Σx²
tb.register("mate", cxBlend, alpha=0.5)                  # 混合交叉
tb.register("mutate", mutGaussian, mu=0, sigma=0.2, indpb=1.0/10)  # 高斯变异
tb.register("select", selBest)                           # 确定性选最优

eaMuPlusLambda(pop, tb, 50, 100, 0.5, 0.2, 100, ...)    # μ=50, λ=100
```

### 3.3 运行结果

```
gen  nevals  avg      min
0    50      88.80    48.77
...
50   70      0.0088   0.0082
...
100  70      0.0004   0.0004

最优 f(x) = 0.000362 (目标 0.0)
```

**收敛分析**：
- gen 0：avg=88.8（随机点远离原点），min=48.77
- gen 10：min=0.14（已接近原点）
- gen 100：min=0.0004（高精度收敛）

**为什么 (μ+λ) 而非 eaSimple？** 连续优化用精英保留更高效——好的解不丢失，持续产生好子代。eaSimple 的 1:1 替换可能丢最优。

**为什么 selBest 可用？** (μ+λ) 有 lambda_=100 个子代注入多样性，即使 selBest 确定性选最优，也不会退化（子代持续探索）。eaSimple 用 selBest 会退化（1:1 替换无多样性注入）。

### 3.4 关键点

- **`array.array` 个体**：`typecode="d"` 双精度浮点数组，比 list 省内存。测试了 Creator 的 `class_replacers`（array.array 的 C 层 deepcopy 不拷 __dict__，deap 用 replacers 修正）。
- **`weights=(-1.0,)`**：负权 = 最小化。Fitness 比较时 wvalues = weights × values，负权把最小化转成"越大越好"。
- **`sigma=0.2`**：高斯变异的标准差。太大跳过最优点，太小收敛慢。后期应衰减（自适应变异，本例固定）。
- **`cxBlend`**：混合交叉 BLX-α，两个父本的凸组合产生子代，适合连续空间。

---

## 四、TSP：排列编码 + 自定义算子

### 4.1 问题定义

n 个城市坐标已知，找最短哈密顿回路（访问每城一次并返回起点）。个体是城市排列 [0,1,...,n-1] 的一个置换，fitness = 回路总长度。

### 4.2 自定义算子

TSP 的排列编码不能用通用算子（cxOnePoint 会破坏排列合法性，产生重复城市）。需要专用算子：

**顺序交叉 OX（cxOrdered）**：
```python
def cxOrdered(ind1, ind2):
    size = len(ind1)
    a, b = sorted(random.sample(range(size), 2))    # 随机切两段点
    hole = set(ind1[a:b])                            # ind1[a:b] 保留
    rest = [x for x in ind2 if x not in hole]        # 其余从 ind2 按序填
    ind1[:] = rest[:a] + ind1[a:b] + rest[a:]        # 拼接
    # 对称处理 ind2
    ...
```

保留 ind1 的一段 [a:b]，其余位置从 ind2 按序填入（跳过已保留的城市）。保证子代仍是合法排列。这是 TSP 最常用的交叉之一。

**交换变异（mutSwap）**：
```python
def mutSwap(ind, indpb=0.1):
    if random.random() < indpb:
        i, j = random.sample(range(len(ind)), 2)
        ind[i], ind[j] = ind[j], ind[i]              # 交换两个城市
    return ind,
```

以 indpb 概率交换两个随机位置。简单但有效——局部扰动排列。

### 4.3 距离矩阵预计算

```python
def make_distance(cities):
    n = len(cities)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = math.hypot(cities[i][0]-cities[j][0], cities[i][1]-cities[j][1])
            dist[i][j] = dist[j][i] = d
    return dist
```

预计算 n×n 距离矩阵，评估时直接查表 O(n)，避免每次重算 sqrt（O(n) 次 hypot）。对 n=30、pop=100、ngen=50，共 5000 次评估 × 30 殡段 = 150000 次 hypot，预计算只算 30×29/2=435 次。

### 4.4 运行结果

```
gen  nevals  avg        min
0    100     1568.15    1143.56
...
25   78      844.78     829.20
...
50   85      776.70     773.05

最短回路 = 773.05
路径 = [5, 23, 13, 6, 21, 11, 28, 26, 24, 18, 19, 10, 2, 27, 17, 12, 3, 9, 20, 0, 7, 14, 15, 16, 8, 22, 1, 4, 25, 29]
```

**收敛分析**：
- gen 0：avg=1568（随机回路很长），min=1143
- gen 25：min=829（下降 27%）
- gen 50：min=773（下降 32%），但 avg=776（种群趋同，收敛停滞）

**收敛停滞**：gen 25-50 min 只从 829 降到 773，avg 趋同（844→776）。种群多样性丧失，需要更大的变异率或更复杂的算子（如 2-opt 局部优化）才能继续改进。这是简单 GA 在 TSP 上的典型瓶颈。

### 4.5 关键点

- **`lambda: creator.Ind(random.sample(range(n), n))`**：直接用 lambda 建个体，不用 initRepeat 包一层（initRepeat 会产生双层嵌套 `[[...]]`）。
- **`tour_length` 返回 tuple**：`(total,)`，和所有 evaluate 一样。
- **`weights=(-1.0,)`**：最小化距离。
- **自定义算子要 in-place**：`ind1[:] = ...` 改内容不改对象，`return ind1, ind2` 返回元组。这是 deap 算子约定。

---

## 五、如何扩展：用 mini_deap 跑自定义问题

三个例子展示了用 mini_deap 的标准流程：

1. **定义个体类**：`creator.create("Fit", Fitness, weights=...)` + `creator.create("Ind", <基类>, fitness=...)`
2. **建 Toolbox**：注册 6 个算子——`attr`（基因生成器）、`individual`、`population`、`evaluate`、`mate`、`mutate`、`select`
3. **建观测组件**（可选）：`Statistics` + `HallOfFame`
4. **选算法**：`eaSimple`（1:1 GA）/ `eaMuPlusLambda`（精英 ES）/ `eaMuCommaLambda`（无精英 ES）
5. **跑**：`pop, log = algorithm(pop, tb, ..., stats=stats, halloffame=hof)`

**自定义问题的扩展点**：
- **新编码**：换个体基类（list/array/numpy.ndarray）和 attr 生成器
- **新算子**：写函数，签名遵循约定（交叉返回 `(ind1, ind2)`，变异返回 `(ind,)`，in-place 修改）
- **多目标**：weights 多维 `(1.0, -1.0,)`，evaluate 返回多维 tuple，用 ParetoFront 替代 HallOfFame（阶段 8）

---

## 六、三个例子的设计权衡

**为什么 One-Max 用 eaSimple 而 Sphere 用 eaMuPlusLambda？**
One-Max 是离散问题，1:1 替换 + 锦标赛选择足够（多样性靠变异维持）。Sphere 是连续优化，精英保留更高效——好的解不丢失，持续产生好子代。连续问题的梯度信息隐含在种群分布中，精英保留维持好的分布。

**为什么 TSP 用 eaSimple 而非 (μ+λ)？**
TSP 是组合优化，两种都可用。本例用 eaSimple 展示它对排列编码的适用性。实际 TSP 常用 (μ+λ) + 更复杂的算子（如 2-opt、EAX 交叉）。

**为什么 Sphere 的 sigma=0.2 固定而非自适应？**
自适应变异（sigma 自身在进化）需要 (μ,λ) 全替换策略（阶段 10 CMA-ES）。本例用固定 sigma + (μ+λ) 精英保留，简单但有效。后期 sigma 应衰减（如 sigma *= 0.99 每代），本例省略。

---

## 六点五、三个例子的收敛对比

| 例子 | 编码 | 算法 | 方向 | 初始 | 最终 | 收敛率 |
|---|---|---|---|---|---|---|
| One-Max | 位串(50位) | eaSimple | max | 25.3/50 | 50/50 | 100% |
| Sphere | 连续(10维) | (μ+λ) | min | 88.8 | 0.0004 | ≈100% |
| TSP | 排列(30城) | eaSimple | min | 1143 | 773 | 32%↓ |

**收敛速度差异**：
- One-Max 最快（gen 25 找到最优）：离散空间小，选择压力大
- Sphere 中等（gen 10 已 0.14）：连续空间平滑，高斯变异有效
- TSP 最慢（gen 50 仍停滞）：组合空间巨大（30! ≈ 10³²），简单 GA 不够，需局部优化

**nevals 对比**（惰性评估效果）：
- One-Max gen 1：nevals=60/100（60% 个体被交叉/变异）
- Sphere gen 1：nevals=64/100（(μ+λ) 的 varOr 复制分支省评估）
- TSP gen 1：nevals=84/100（高交叉率 0.7 → 更多个体失效）

---

## 七、常见陷阱（集成时才暴露）

1. **evaluate 返回 int 而非 tuple**：`register("evaluate", sum)` 返回 int，`fitness.values = int` 报 `TypeError: 'int' has no len()`。必须 `lambda ind: (sum(ind),)`。
2. **initRepeat 双层嵌套**：`initRepeat(Ind, func, 1)` 产生 `Ind([func()])` = `[[...]]`。若 func 返回列表，用 `lambda: Ind(func())` 直接建。
3. **weights 正负搞反**：最大化用 `(1.0,)`，最小化用 `(-1.0,)`。搞反会往错误方向进化（One-Max 收敛到全 0）。
4. **TSP 用 cxOnePoint 破坏排列**：通用交叉不保排列合法性，产生重复城市。必须用 OX/PMX 等排列专用交叉。
5. **忘 `del fitness.values` 使惰性评估失效**：自定义算子若改了个体但没 del fitness.values，算法不重评估，用旧 fitness 选择 → 错误。deap 的 varAnd/varOr 已处理，自定义算法要自己 del。

---

## 八、关键收获

1. **接口兼容性验证通过**：三个例子覆盖三种编码/两种算法/两种方向，前 6 阶段的零件能组装成可运行整机。
2. **"注册 6 算子 + 选算法"是统一流程**：不管什么问题，都是建 Toolbox + 注册算子 + 调 algorithm。框架的扩展性在此。
3. **自定义算子只需遵循约定**：交叉返回 `(ind1, ind2)`，变异返回 `(ind,)`，in-place 修改。无需继承任何类。
4. **惰性评估在实践中生效**：One-Max gen 1 的 nevals=60（非 100），省 40% 评估。
5. **收敛瓶颈提示后续改进**：TSP 在 gen 25 后停滞，需要更复杂算子（2-opt）或自适应变异——这是阶段 8-10 进阶内容的动机。

---

## 九、思考题

1. One-Max 的 `indpb=1.0/50`（每位 1/50 翻转概率）。若改成 `indpb=0.5` 会怎样？（提示：每位 50% 翻转，破坏已优化的解）
2. Sphere 用 `selBest`（确定性）+ (μ+λ) 能收敛。若用 `selBest` + eaSimple 会怎样？（提示：1:1 替换无多样性注入，退化）
3. TSP 的 OX 交叉保留 ind1[a:b] 段。为什么从 ind2 按序填而非随机填？（提示：保留 ind2 的相对顺序信息）
4. 三个例子都用了 `random.seed(42)`。为什么？（提示：可复现——调试/对比算法时结果一致）
5. TSP 预计算距离矩阵省 sqrt。若 n=1000 城市，矩阵多大？内存多少？（提示：1000×1000×8B = 8MB）

---

## 十、本阶段总结

阶段 7 是**集成验证**——三个例子证明前 6 阶段的零件能组装成可运行整机。这是教学项目的关键里程碑：从"每个模块单独测过"到"整体能跑通"。三个例子覆盖了：
- **三种编码**：位串（list of int）、连续（array.array of float）、排列（list of int）
- **两种算法**：eaSimple（1:1 GA）、eaMuPlusLambda（精英 ES）
- **两种方向**：最大化（One-Max）、最小化（Sphere/TSP）
- **自定义算子**：TSP 的 OX 交叉 + 交换变异，展示框架扩展性

所有例子用 `random.seed(42)` 保证可复现，可直接运行验证。

---

## 十一、下一阶段预告

**阶段 8：tools/emo.py + NSGA2** —— 多目标进化算法。前 7 阶段都是单目标，阶段 8 进入多目标：
- **非支配排序**（fast_non_dominated_sort）：把种群按 Pareto 支配关系分层
- **拥挤距离**（crowding_distance）：同层内按拥挤度排序，维持前沿分布均匀
- **selNSGA2**：锦标赛选择 + 支配秩 + 拥挤距离
- **实战**：ZDT1 双目标测试函数

多目标的核心难点：**没有全局最优，而是一个 Pareto 前沿**。选择算子要同时考虑"接近前沿"（支配秩）和"分布均匀"（拥挤距离）。