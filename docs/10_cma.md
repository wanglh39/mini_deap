# 阶段 10：cma.py —— CMA-ES 协方差矩阵自适应进化策略

> 对应 DEAP：`deap/cma.py`（原版 868 行含 Strategy/StrategyOnePlusLambda/StrategyLQ，本阶段约 170 行保留核心 Strategy）
> 产出：`mini_deap/cma.py` + `tests/test_cma.py`（8 测试全过）+ `examples/cma_es.py`

---

## 一、背景：为什么 CMA-ES 是"黄金标准"？

前 9 阶段的连续优化用简单高斯变异（mutGaussian + 固定 sigma）。问题：
- **sigma 太大**：跳过最优点，不收敛
- **sigma 太小**：收敛慢，陷入局部
- **各维独立**：无法处理变量间相关性（如狭长斜谷）

CMA-ES 解决这三个问题：
1. **步长 sigma 自适应**：成功时增大（探索），失败时减小（开发）
2. **协方差矩阵 C 学习相关性**：C 编码变量间的协方差，采样椭圆对齐问题几何
3. **无需调参**：所有参数自适应，默认值通用

**CMA-ES 的地位**：在 BBOB（黑盒优化基准）上，CMA-ES 是连续优化中表现最好的进化算法，几乎成为"默认选择"。

---

## 二、核心设计思想

### ① ask-tell 接口 —— 策略与算法分离

```python
strategy = Strategy(centroid, sigma)     # 初始化策略
for gen in range(ngen):
    pop = strategy.generate(ind_init)    # ask：策略采样
    evaluate(pop)
    strategy.update(pop)                 # tell：策略更新
```

**ask（generate）**：从 N(mean, sigma²·C) 采样 lambda_ 个子代。
**tell（update）**：根据成功子代更新 mean/sigma/C。

**为什么用 ask-tell 而非 select/mate/mutate？** CMA-ES 没有传统意义的交叉/变异/选择——策略自己管采样和更新。ask-tell 接口更自然：策略是状态机，ask 产出，tell 反馈。

### ② 协方差矩阵 C —— 学习问题几何

```
采样：z ~ N(0, I)，x = mean + sigma · B·D·z
其中 C = B·D²·B^T（特征分解）
```

**C 的含义**：C 是协方差矩阵，C[i][j] 编码变量 i 和 j 的相关性。
- C = I（单位阵）：各维独立，采样球形
- C 有非对角项：各维相关，采样椭圆，椭圆轴对齐问题几何

**B·D 的作用**：B 是特征向量矩阵（旋转），D 是特征值平方根（缩放）。B·D·z 把标准正态 z 变换成服从 N(0, C) 的样本。

**学习过程**：update 根据成功子代更新 C——如果成功子代沿某方向集中，C 在该方向增大（采样更聚焦）；如果分散，C 在该方向减小（探索更广）。

### ③ 进化路径 —— 累积成功方向

```python
self.ps = (1 - self.cs) * self.ps + sqrt(...) * ...   # 步长路径
self.pc = (1 - self.cc) * self.pc + sqrt(...) * c_diff  # 协方差路径
```

**进化路径（cumulation path）**：不是只看当代成功方向，而是**指数加权累积历史**。`(1-cs)` 是衰减率，cs 大→忘得快（只看当代），cs 小→记得久（平滑）。

**两条路径**：
- **ps（步长路径）**：累积均值移动方向，用于更新 sigma。如果路径长（持续同向移动）→ sigma 增大（加速）。
- **pc（协方差路径）**：累积均值移动方向，用于更新 C 的 rank-1 项。路径方向 → C 在该方向增大。

### ④ rank-1 + rank-mu 更新 —— 协方差矩阵学习

```python
self.C = (1 - ccov1 - ccovmu) * self.C \        # 衰减旧 C
    + ccov1 * outer(pc, pc) \                    # rank-1：进化路径
    + ccovmu * dot(weights * artmp.T, artmp)     # rank-mu：成功子代
```

**rank-1 更新**：`outer(pc, pc)` 是进化路径的外积。如果 pc 沿某方向持续大，C 在该方向增大。单步更新只用一个方向。

**rank-mu 更新**：`artmp` 是 mu 个成功子代减均值。用多个子代的协方差更新 C。比 rank-1 信息更丰富。

**ccov1/ccovmu**：学习率。大→学得快但不稳，小→学得慢但稳。默认值由 dim 和 mueff 决定。

---

## 三、逐函数精读

### 3.1 Strategy.__init__ —— 初始化

```python
self.centroid = numpy.array(centroid)    # 均值（搜索中心）
self.sigma = sigma                       # 步长（全局缩放）
self.pc = numpy.zeros(self.dim)          # 协方差路径
self.ps = numpy.zeros(self.dim)          # 步长路径
self.C = numpy.identity(self.dim)        # 协方差矩阵（初始各维独立）
self.diagD, self.B = numpy.linalg.eigh(self.C)  # 特征分解 C = B·D²·B^T
self.lambda_ = int(4 + 3 * log(self.dim))       # 子代数
```

**初始 C = I**：各维独立，采样球形。随着 update，C 学习问题几何，变成椭圆。

**lambda_ = 4 + 3·ln(N)**：子代数默认值。N 大时 lambda_ 适当增大（更多采样覆盖高维空间）。

**chiN**：N(0,I) 的期望范数 `sqrt(N)·(1 - 1/(4N) + ...)`。用于步长更新的归一化——`||ps|| / chiN` 衡量当前路径长度相对期望的倍数。

### 3.2 generate —— ask 采样

```python
def generate(self, ind_init):
    arz = numpy.random.standard_normal((self.lambda_, self.dim))  # z ~ N(0,I)
    arz = self.centroid + self.sigma * numpy.dot(arz, self.BD.T)  # x = mean + sigma·BD·z
    return [ind_init(a) for a in arz]
```

**采样公式** `x = mean + sigma · B·D · z`：
1. `z ~ N(0, I)`：标准正态
2. `B·D·z`：旋转（B）+ 缩放（D），变成 N(0, C)
3. `sigma ·`：全局步长缩放
4. `+ mean`：平移到均值

**BD 预计算**：`self.BD = self.B * self.diagD`，避免每次 generate 重算。

### 3.3 update —— tell 策略更新

```python
def update(self, population):
    population.sort(key=lambda ind: ind.fitness, reverse=True)  # 按 fitness 降序
    self.centroid = numpy.dot(self.weights, population[0:self.mu])  # 加权均值
    # 更新 ps（步长路径）
    # 更新 pc（协方差路径）
    # 更新 C（rank-1 + rank-mu）
    # 更新 sigma（步长）
    # 特征分解 C = B·D²·B^T
```

**排序取前 mu 个**：只用好子代更新策略。weights 是超线性递减权重——最好的子代权重最大。

**更新顺序**：centroid → ps → pc → C → sigma → 特征分解。每步依赖前一步。

**特征分解**：每次 update 后 `numpy.linalg.eigh(self.C)` 重算 B/D。这是最贵的操作 O(N³)，但保证下次 generate 用更新后的 C。

### 3.4 computeParams —— 参数计算

```python
self.weights = log(mu + 0.5) - log(arange(1, mu+1))  # 超线性权重
self.mueff = 1. / sum(weights ** 2)                    # 有效种群大小
self.cs = (mueff + 2) / (dim + mueff + 3)             # 步长累积率
self.cc = 4. / (dim + 4)                              # 协方差累积率
self.ccov1 = 2. / ((dim + 1.3)**2 + mueff)            # rank-1 学习率
self.ccovmu = 2*(mueff-2+1/mueff) / ((dim+2)**2+mueff)  # rank-mu 学习率
```

**超线性权重** `log(mu+0.5) - log(i)`：最好的子代权重最大，衰减超线性。比线性权重更强调精英。

**mueff（有效种群大小）**：`1/Σw²`。weights 越集中（好子代权重越大），mueff 越小（等效于更少的有效样本）。mueff 影响所有学习率。

---

## 四、eaGenerateUpdate —— ask-tell 算法骨架

```python
def eaGenerateUpdate(toolbox, ngen, ...):
    for gen in range(ngen):
        population = toolbox.generate()       # ask
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        toolbox.update(population)            # tell
```

**与 eaSimple 的区别**：没有 select/mate/mutate，只有 generate/evaluate/update。策略自己管采样和更新，算法骨架只管循环和评估。

**为什么单独一个骨架？** CMA-ES 的流程和传统 GA 完全不同——没有选择/交叉/变异，只有策略采样+更新。eaSimple/eaMuPlusLambda 的 select/mate/mutate 模型不适用。

---

## 五、实战：Sphere + Rosenbrock

### 5.1 Sphere（单峰各向同性）

```
gen  min       avg
0    137.41    262.32
...
50   0.157     0.293
...
99   0.0001    0.0003

最优 f(x) = 0.00009885
```

**100 代收敛到 0.0001**。CMA-ES 在单峰问题上极高效——sigma 快速衰减，centroid 直奔最优点。

### 5.2 Rosenbrock（非凸狭长谷）

```
gen  min       avg
0    127.95    350.69
...
50   8.02      8.26
...
99   6.33      6.51

最优 f(x) = 6.332
```

**100 代收敛到 6.33**。Rosenbrock 是著名的难优化函数——狭长弯曲谷，需要 CMA-ES 学习协方差矩阵对齐谷方向。100 代不够（通常需 500+ 代到 0），但下降趋势正确。

**对比**：Sphere 100 代到 0.0001，Rosenbrock 100 代到 6.33。差距源于问题难度——Sphere 单峰各向同性（易），Rosenbrock 非凸病态（难）。

---

## 六、和 deap 原版对照

| 本教学版 | deap `cma.py` | 差异 |
|---|---|---|
| `Strategy` | 同 | 完全保留核心 |
| `generate/update/computeParams` | 同 | 完全保留 |
| `StrategyOnePlusLambda` | deap 有 | 砍掉：(1+λ)-CMA-ES 变体 |
| `StrategyLQ` | deap 有 | 砍掉：线性代数优化版 |
| `eaGenerateUpdate` | 同 | 完全保留（加到 algorithms.py） |

**保留 CMA-ES 核心 Strategy**，砍掉变体（OnePlusLambda/LQ）。教学版用标准 Strategy 足够。

---

## 七、常见陷阱

1. **numpy.random.seed vs random.seed**：CMA 的 generate 用 `numpy.random`，要 `numpy.random.seed(42)` 而非 `random.seed(42)` 才能复现。
2. **centroid 维度**：centroid 的长度决定问题维度。`Strategy([5.0]*10, 1.0)` 是 10 维。
3. **sigma 初始值**：太大发散，太小收敛慢。经验值：约为搜索范围的 1/3。
4. **lambda_ 太小**：默认 `4+3·ln(N)` 是下限。难问题可增大 lambda_（如 10·N）。
5. **Rosenbrock 收敛慢**：狭长谷需要 CMA-ES 学习协方差，几百代才到 0。不是 bug。
6. **C 的数值稳定性**：update 后 C 可能非正定（数值误差）。deap 不处理，靠 eigh 的鲁棒性。生产级需加正则化。

---

## 八、Python 语言特性备忘

- **`numpy.linalg.eigh(C)`**：对称矩阵特征分解，返回 (特征值, 特征向量)。比 eig 快且保证实数。
- **`numpy.outer(a, b)`**：外积，`a[i]·b[j]`。rank-1 更新用。
- **`numpy.dot(weights, population[0:mu])`**：加权求和。weights 是 1D，population 是 2D，dot 沿第一维加权。
- **`numpy.argsort`**：返回排序后的索引（不移动数据）。用于重排特征值/向量。
- **`float(bool)`**：`float(True)=1.0, float(False)=0.0`。hsig 是布尔转浮点（用于 C 更新的系数）。

---

## 九、关键收获

1. **CMA-ES 学习问题几何**：协方差矩阵 C 编码变量间相关性，采样椭圆对齐问题等高线。无需手动调参。
2. **ask-tell 接口**：策略与算法分离。generate 采样，update 更新策略。比 select/mate/mutate 更适合基于策略的算法。
3. **进化路径累积历史**：ps/pc 指数加权累积成功方向，不只看当代。路径长→sigma 增大（加速），路径短→sigma 减小（聚焦）。
4. **rank-1 + rank-mu 更新**：rank-1 用进化路径（单方向），rank-mu 用成功子代（多方向）。两者结合学习协方差。
5. **特征分解是瓶颈**：每次 update 后 eigh(C) 是 O(N³)。高维问题这是主要开销。
6. **CMA-ES 是连续优化默认选择**：BBOB 基准上表现最好，几乎无需调参。

---

## 十、思考题

1. CMA-ES 的 C 初始为单位阵。如果初始 C 已知问题的大致几何（如各维方差不同），怎么设？（提示：`cmatrix` 参数）
2. `lambda_ = 4 + 3·ln(N)`。为什么用对数而非线性？（提示：高维采样覆盖效率）
3. Sphere 100 代到 0.0001，Rosenbrock 100 代到 6.33。为什么差距这么大？（提示：问题条件数——Sphere 各向同性，Rosenbrock 病态）
4. ask-tell 接口和 eaSimple 的 select/mate/mutate 有什么本质区别？（提示：策略有状态 vs 无状态）
5. update 里 `population.sort(key=lambda ind: ind.fitness, reverse=True)`。为什么按 fitness 降序？（提示：取前 mu 个最好的）

---

## 十一、项目总结

**mini_deap 10 阶段全部完成**：

| 阶段 | 模块 | 内容 | 测试 |
|---|---|---|---|
| 0 | 骨架 | 包结构 | - |
| 1 | base/fitness | Fitness 类 | 20 |
| 2 | base/toolbox | Toolbox 类 | 13 |
| 3 | base/creator | 元编程建类 | 15 |
| 4 | tools/operators | 16 个算子 | 22 |
| 5 | tools/support | 统计/名人堂/日志 | 13 |
| 6 | algorithms | 算法骨架 | 24 |
| 7 | examples | 3 个实战例子 | - |
| 8 | tools/emo | NSGA-II 多目标 | 18 |
| 9 | gp | 遗传规划 | 25 |
| 10 | cma | CMA-ES | 8 |

**总计**：约 2000 行代码 + 158 个测试（全过）+ 10 篇文档（300+ 行/篇）。

**覆盖的进化算法**：
- 简单 GA（eaSimple）
- (μ+λ) / (μ,λ) 进化策略
- NSGA-II 多目标
- 遗传规划（GP）
- CMA-ES

**DEAP 的 5 大设计思想全已体现**：
1. 数据与算法解耦（Toolbox 粘合）
2. weights 正负统一 max/min
3. 惰性求值（fitness.valid）
4. 元编程建类（creator.create）
5. 并行/拷贝友好（toolbox.map + __deepcopy__）