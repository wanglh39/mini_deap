# 阶段 6：algorithms —— 进化算法主循环骨架（把前 5 阶段串成可运行算法）

> 对应 DEAP：`deap/algorithms.py`（原版 503 行，本阶段约 210 行，保留 5 个核心函数）
> 产出：`mini_deap/algorithms.py` + `tests/test_algorithms.py`（24 测试全过）

---

## 一、背景：为什么需要"算法骨架"？

前 5 阶段我们造好了零件：
- **Fitness**（阶段 1）：适应度容器，weights 编码方向
- **Toolbox**（阶段 2）：算子粘合剂，register 冻参
- **Creator**（阶段 3）：元编程建个体类
- **算子**（阶段 4）：选择/交叉/变异/初始化
- **support**（阶段 5）：Statistics/HallOfFame/Logbook 观测组件

但零件不会自己跑。进化算法有固定的骨架——**选择 → 交叉 → 变异 → 评估 → 替换** 的循环。这个骨架把零件串起来。不同算法的骨架差异在于：
- **变异策略**：交叉和变异都施加（varAnd）vs 每个个体只选一种操作（varOr）
- **替换策略**：子代完全取代父代（eaSimple）vs 父子合并选（(μ+λ)）vs 只从子代选（(μ,λ)）

本阶段实现 5 个函数，覆盖最常用的 3 种算法：

| 函数 | 职责 | 用在哪 |
|---|---|---|
| `varAnd` | 交叉 AND 变异 | eaSimple 的变异阶段 |
| `varOr` | 交叉 OR 变异 OR 复制 | (μ+λ)/(μ,λ) 的变异阶段 |
| `eaSimple` | 简单 GA（Generational） | One-Max、位串问题 |
| `eaMuPlusLambda` | (μ+λ) 进化策略 | 连续优化、精英保留 |
| `eaMuCommaLambda` | (μ,λ) 进化策略 | 自适应参数变异 |

---

## 二、核心设计思想

### ① 惰性评估 —— 只重算无效的个体

进化算法最贵的操作是评估黑盒函数。但不是每个个体都需要重新评估——**复制**（未交叉未变异）的个体 fitness 仍然有效。deap 的解法：

```python
# 交叉/变异后使 fitness 失效
del ind.fitness.values      # → fitness.valid = False

# 评估时只挑无效的
invalid_ind = [ind for ind in population if not ind.fitness.valid]
fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
for ind, fit in zip(invalid_ind, fitnesses):
    ind.fitness.values = fit
```

`del ind.fitness.values` 触发 Fitness 的 `delValues`，把 `self.values = ()` 并 `self.valid = False`。评估时只处理 `not valid` 的个体。在 varOr 的复制分支（else），个体没被 clone 也没被改，fitness 仍有效，跳过评估省黑盒调用。

**为什么不用脏标记而用 del？** `del` 是 Python 原生的"使属性不存在"操作，Fitness 把它重载为"使 values 归空 + valid 置 False"。比设一个 `_dirty = True` 更优雅，且访问 `ind.fitness.values` 时能立刻发现是空的（报错比静默用旧值好）。

### ② toolbox.map 并行 —— 评估是可并行化的瓶颈

```python
fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
```

`toolbox.map` 默认注册为内置 `map`（串行懒序列）。但评估是"对每个个体独立调黑盒"，天然可并行。用户注册 `toolbox.register("map", multiprocessing.Pool().map)` 即可并行，算法代码不用改。

**为什么用 map 而不是列表推导？** 列表推导 `[toolbox.evaluate(ind) for ind in invalid_ind]` 无法并行（GIL + 串行语法）。`map` 是函数式接口，换成 `Pool.map` 即并行，零代码改动。这是 deap 的并行化设计：**算法只认 toolbox.map 这个接口，串行/并行是注册时的选择**。

### ③ stats / halloffame 可选 —— 观测做成守卫插件

```python
if halloffame is not None:
    halloffame.update(offspring)
record = stats.compile(population) if stats else {}
logbook.record(gen=gen, nevals=nevals, **record)
```

算法里 `if stats:` / `if halloffame is not None:` 守卫，不传也能跑。这让算法骨架纯净——只管进化逻辑，观测是可选的旁路。传了 stats 就算统计，传了 hof 就更新名人堂，都不传就纯跑进化。

**为什么用 `if stats else {}` 而非 `if stats is not None`？** `if stats` 对 `None` 和空对象都 False，更宽松。但 `if halloffame is not None` 用 `is not None` 因为 HallOfFame 可能是空列表（`[]` 是 falsy 但有效）。deap 原版混用两种，本教学版保持一致。

### ④ population[:] = offspring —— 原地替换保持引用

```python
population[:] = offspring    # 切片赋值：改内容，不改对象
# vs
population = offspring        # 只改局部变量，外部引用不变！
```

`population[:] = offspring` 把 offspring 的内容**复制进** population 列表对象（改内容不改 id）。`population = offspring` 只让局部变量 population 指向 offspring，**外部的 population 列表对象没变**。

为什么重要？调用方可能持有 population 的引用（如 `my_pop = [...]; eaSimple(my_pop, ...); # my_pop 要是结果`）。用 `[:]` 保证调用方看到的列表被原地更新。这是 deap 的约定：**算法原地修改 population**。

### ⑤ varAnd vs varOr —— 两种变异策略

**varAnd（交叉 AND 变异）**：
```python
offspring = [toolbox.clone(ind) for ind in population]  # 全体 clone
for i in range(1, len(offspring), 2):                   # 相邻配对交叉
    if random.random() < cxpb:
        offspring[i-1], offspring[i] = toolbox.mate(...)
for i in range(len(offspring)):                         # 逐个变异
    if random.random() < mutpb:
        offspring[i], = toolbox.mutate(...)
```
一个个体可能既被交叉又被变异（两个操作独立施加）。子代数 = 父代数（1:1）。

**varOr（交叉 OR 变异 OR 复制）**：
```python
for _ in range(lambda_):                # 产 lambda_ 个子代
    op = random.random()
    if op < cxpb:                       # 交叉
        ind1, ind2 = clone(随机选2个); mate(ind1, ind2); offspring.append(ind1)
    elif op < cxpb + mutpb:             # 变异
        ind = clone(随机选1个); mutate(ind); offspring.append(ind)
    else:                               # 复制
        offspring.append(随机选1个)
```
每个子代只来自一种操作。子代数 lambda_ 与父代数 mu 独立。约束 `cxpb + mutpb <= 1.0`（余下是复制概率）。

**为什么有两种？** varAnd 适合 Generational GA（子代数 = 父代数，1:1 替换）。varOr 适合 (μ+λ)/(μ,λ) ES（子代数 lambda_ 独立于父代数 mu，每个子代只来自一种操作，便于控制算子比例）。

---

## 三、逐函数精读

### 3.1 varAnd —— 交叉 AND 变异

```python
def varAnd(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]   # ① 全体 clone

    for i in range(1, len(offspring), 2):                   # ② 相邻配对交叉
        if random.random() < cxpb:
            offspring[i-1], offspring[i] = toolbox.mate(offspring[i-1], offspring[i])
            del offspring[i-1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):                         # ③ 逐个变异
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring
```

**① 全体 clone**：`toolbox.clone` 默认用 copy.deepcopy。不 clone 的话交叉/变异会 in-place 改父代个体，污染种群。clone 后 offspring 与 population 完全独立。

**② 相邻配对**：`range(1, len, 2)` 产生 1, 3, 5, ...，配对 (0,1), (2,3), ...。若种群长度是奇数，最后一个个体不配对（不交叉）。`del fitness.values` 使两个子代都失效，强制重评估。

**③ 逐个变异**：`offspring[i], = toolbox.mutate(...)` 注意逗号——`mutate` 返回元组 `(ind,)`，用解包取第一个。变异后 `del fitness.values` 失效。

**为什么先交叉后变异？** 顺序可换，但先交叉后变异是惯例：交叉产生大范围重组，变异在小范围扰动。先大后小，子代既有重组又有扰动。

### 3.2 varOr —— 交叉 OR 变异 OR 复制

```python
def varOr(population, toolbox, lambda_, cxpb, mutpb):
    assert (cxpb + mutpb) <= 1.0, "..."                    # ① 概率和约束

    offspring = []
    for _ in range(lambda_):                               # ② 产 lambda_ 个
        op_choice = random.random()
        if op_choice < cxpb:                               # ③ 交叉
            ind1, ind2 = [toolbox.clone(i) for i in random.sample(population, 2)]
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)                         # 只取第一个子代
        elif op_choice < cxpb + mutpb:                     # ④ 变异
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:                                              # ⑤ 复制
            offspring.append(random.choice(population))    # 不 clone！
    return offspring
```

**① 概率和约束**：`cxpb + mutpb <= 1.0`，余下 `1 - cxpb - mutpb` 是复制概率。三段式划分概率空间。

**② 产 lambda_ 个**：子代数独立于父代数。这是 (μ+λ)/(μ,λ) 的特点——mu 个父代产 lambda_ 个子代，lambda_ 可大于或小于 mu。

**③ 交叉只取第一个子代**：`mate` 返回两个子代，但 varOr 只 append 第一个（`ind1`），第二个丢弃。这是 (μ+λ) 的约定——每次交叉只贡献一个子代，控制算子比例更精确。

**⑤ 复制不 clone**：`random.choice(population)` 直接 append 引用，不 clone。因为复制分支不会修改个体，共享引用安全。且复制的个体 fitness 仍有效，惰性评估会跳过——省 clone 开销 + 省评估开销。但要注意：后续若有人 in-place 改了这个个体，会污染父代（deap 原版的风险）。

### 3.3 _evaluate_invalid —— 惰性评估辅助

```python
def _evaluate_invalid(population, toolbox):
    invalid_ind = [ind for ind in population if not ind.fitness.valid]  # 挑无效
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)              # 批量评估
    for ind, fit in zip(invalid_ind, fitnesses):                        # 回填
        ind.fitness.values = fit
    return len(invalid_ind)                                             # 返回评估数
```

本教学版把这段重复逻辑提成辅助函数（deap 原版在每个算法里重复写）。返回 `nevals` 供 Logbook 记录。

**惰性评估的收益**：varOr 复制分支的个体 fitness 有效，跳过评估。若 cxpb=0.5, mutpb=0.3，复制概率 0.2，约 20% 个体免评估。评估是黑盒最贵的部分，省 20% 是实打实的加速。

### 3.4 eaSimple —— 简单 GA

```python
def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    nevals = _evaluate_invalid(population, toolbox)        # 初始评估
    if halloffame is not None: halloffame.update(population)
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=nevals, **record)
    if verbose: print(logbook.stream)

    for gen in range(1, ngen + 1):
        offspring = toolbox.select(population, len(population))    # 选择
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)        # 交叉+变异
        nevals = _evaluate_invalid(offspring, toolbox)             # 惰性评估
        if halloffame is not None: halloffame.update(offspring)
        population[:] = offspring                                  # 1:1 原地替换
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=nevals, **record)
        if verbose: print(logbook.stream)

    return population, logbook
```

**流程**：初始评估 → 循环 ngen 代 { 选择 → varAnd → 评估 → 更新 hof → 原地替换 → 记录 }。

**1:1 替换**：`population[:] = offspring`，子代完全取代父代。要求选择算子是**随机的且允许重复选**（如 selTournament），否则选 n 个从 n 个里等于没选（如 selBest 选 n 个最优，下一代全是同一个体，交叉无意义）。

**gen=0 的记录**：初始种群也算一代（gen=0），记录其统计。所以 ngen=5 → logbook 有 6 条（gen 0..5）。

### 3.5 eaMuPlusLambda —— (μ+λ) 精英保留

```python
for gen in range(1, ngen + 1):
    offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)   # 产 λ 子代
    nevals = _evaluate_invalid(offspring, toolbox)
    if halloffame is not None: halloffame.update(offspring)
    population[:] = toolbox.select(population + offspring, mu)     # 父子合并选 μ
```

**精英保留**：`population + offspring` 合并后选 mu 个。父代参与选择，最优个体可一直存活（除非被更优取代）。这保证**历代最优不会退化**——测试 `test_elitism_best_not_regress` 验证此性质。

**mu vs lambda_**：mu 是存活数，lambda_ 是子代数，两者独立。常见配置 lambda_ = 7 * mu（进化策略的经验值）。

### 3.6 eaMuCommaLambda —— (μ,λ) 无精英

```python
assert lambda_ >= mu, "lambda must be greater or equal to mu."
...
    population[:] = toolbox.select(offspring, mu)                 # 只从子代选
```

**无精英保留**：`select(offspring, mu)` 只从子代选，父代全淘汰。最优可能丢失（测试 `test_no_elitism_can_regress` 不做严格断言，因随机性）。

**为什么要求 lambda_ >= mu？** 子代数要 >= 存活数，否则没得选。assert 在函数入口检查。

**适合自适应参数变异**：(μ,λ) 常用于变异强度自身在进化的场景（如 CMA-ES）。父代保留会锁死参数——如果变异强度大的父代一直存活，它会持续产生大扰动子代，参数无法收敛。全淘汰让参数自由探索。

---

## 四、和 deap 原版对照

| 本教学版 | deap `algorithms.py` | 差异 |
|---|---|---|
| `varAnd` | 同 | 完全保留 |
| `varOr` | 同 | 完全保留 |
| `eaSimple` | 同 | 完全保留 |
| `eaMuPlusLambda` | 同 | 完全保留 |
| `eaMuCommaLambda` | 同 | 完全保留 |
| `_evaluate_invalid` 辅助 | deap 在每个算法里重复写 | 提成函数复用（教学版简化） |
| `eaGenerateUpdate` | deap 有 | 留到阶段 10 CMA-ES（ask-tell 模型） |

**保留全部常用算法**，只把 eaGenerateUpdate 推迟到阶段 10（它专门给 CMA-ES 用）。`_evaluate_invalid` 是本教学版的改进——deap 原版在 3 个算法里重复写这段惰性评估逻辑，我们提成函数复用。

---

## 五、运行示例：One-Max 问题

```python
import random
from mini_deap.base.fitness import Fitness
from mini_deap.base.toolbox import Toolbox
import mini_deap.base.creator as creator
from mini_deap.tools.operators import initRepeat, selTournament, cxOnePoint, mutFlipBit
from mini_deap.tools.support import Statistics, HallOfFame
from mini_deap.algorithms import eaSimple

creator.create("FitMax", Fitness, weights=(1.0,))
creator.create("Ind", list, fitness=creator.FitMax)

tb = Toolbox()
tb.register("attr_bool", random.randint, 0, 1)
tb.register("individual", initRepeat, creator.Ind, tb.attr_bool, 10)
tb.register("population", initRepeat, list, tb.individual)
tb.register("evaluate", lambda ind: (sum(ind),))
tb.register("mate", cxOnePoint)
tb.register("mutate", mutFlipBit, indpb=0.1)
tb.register("select", selTournament, tournsize=3)

pop = tb.population(n=20)
hof = HallOfFame(maxsize=1)
stats = Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("avg", lambda x: sum(x)/len(x))
stats.register("max", max)

eaSimple(pop, tb, 0.5, 0.2, 10, stats=stats, halloffame=hof, verbose=True)
print("最优解:", sum(hof[0]))   # 应接近 10（全 1）
```

输出形如：
```
gen  nevals  avg     max
0    20      5.2     8
1    16      5.8     9
...
10   17      8.5     10
```

---

## 六、常见陷阱

1. **evaluate 必须返回 tuple**：`fitness.values` 期望可迭代且长度匹配 weights。`register("evaluate", sum)` 返回 int 会报 `TypeError: object of type 'int' has no len()`。要 `lambda ind: (sum(ind),)` 包成 tuple。
2. **eaSimple 用 selBest 会退化**：1:1 替换 + 确定性选择 = 下一代全是同一个体，交叉无意义。必须用随机选择（selTournament/selRoulette）。
3. **忘 `population[:] = offspring` 用 `=` 赋值**：外部引用不更新，调用方看不到结果。这是 deap 最常见的集成 bug。
4. **varOr 复制分支不 clone**：复制的个体与父代共享引用。若后续 in-place 改子代，会污染父代。deap 原版如此，风险已知。
5. **(μ,λ) 的 lambda_ < mu**：子代不够选，assert 报错。常见配置 lambda_ = 7 * mu。
6. **verbose=__debug__ 的坑**：Python 运行时 `__debug__` 默认 True（除非用 `-O` 优化模式）。要静默得显式传 `verbose=False`。
7. **stats 的 key 返回标量 vs tuple**：单目标 `key=lambda ind: ind.fitness.values[0]` 取标量；多目标 `key=attrgetter("fitness.values")` 取 tuple。混用会导致统计函数报错。

---

## 七、Python 语言特性备忘

- **`del ind.fitness.values`**：触发 Fitness 的 `__delattr__` 或 property 的 deleter。Fitness 重载为"使 values 归空 + valid 置 False"，比设 `_dirty = True` 更优雅。
- **`population[:] = offspring`**：切片赋值改内容不改 id。`a = [1,2,3]; b = a; a[:] = [4,5]; assert b == [4,5]`。vs `a = [4,5]; assert b == [1,2,3]`（b 不变）。
- **`offspring[i], = toolbox.mutate(...)`**：逗号解包——`mutate` 返回元组 `(ind,)`，`x, = t` 等价 `x = t[0]`。比 `x = t[0]` 更显式地表达"我知道这是单元素元组"。
- **`random.sample(pop, 2)`**：无放回抽 2 个不同元素。vs `random.choices(pop, k=2)` 有放回可重复。
- **`__debug__`**：Python 内置常量，`-O` 优化模式下为 False。deap 用它做 verbose 默认值——开发时自动打印，生产用 `-O` 自动静默。
- **`toolbox.map`**：默认是 `map`（返回迭代器）。`for ind, fit in zip(invalid_ind, fitnesses)` 消费迭代器。换 `Pool.map` 返回 list 也能 zip，接口兼容。

---

## 八、设计权衡：为什么这样切分

**为什么 varAnd/varOr 是独立函数而非内联在算法里？**
两种变异策略被多个算法共享：varAnd 被 eaSimple 用，varOr 被 (μ+λ)/(μ,λ) 用。提成函数复用 + 可独立测试。且用户写自定义算法时也能调它们组合新算法。

**为什么 _evaluate_invalid 提成辅助函数？**
deap 原版在 3 个算法里重复写这段逻辑（约 4 行 × 3 = 12 行重复）。教学版提成函数，减少重复 + 突出"惰性评估"是统一模式。代价：多一层函数调用（微开销）。

**为什么 eaSimple 的选择在 varAnd 之前？**
`select(population, len)` 先选，再对选中个体 varAnd。若先 varAnd 再 select，子代数会翻倍（clone 后变 2N），select 再缩回 N，浪费。先 select 缩到 N，再 varAnd 保持 N，高效。

**为什么 (μ+λ) 合并后 select 而非分两步？**
`select(population + offspring, mu)` 一次性从 mu+lambda_ 个里选 mu 个。若分两步（先从 offspring 选，再从 population 补），无法保证全局最优——可能 offspring 的最差比 population 的最优好，分步会错过。合并后 select 让选择算子看到全部候选，保证精英保留。

---

## 九、关键收获

1. **惰性评估是进化算法的核心优化**：`del fitness.values` 失效 + `not valid` 过滤，只重算被交叉/变异改过的个体，复制的个体免评估。
2. **toolbox.map 是并行化接口**：算法只认 map，串行/并行是注册时的选择，零代码改动换并行。
3. **观测做成可选插件**：`if stats:` / `if hof:` 守卫，算法骨架纯净，不传也能跑。
4. **population[:] = offspring 原地替换**：保持外部引用，调用方看到结果。这是 deap 的约定。
5. **varAnd vs varOr 两种变异策略**：And 适合 1:1 Generational GA，Or 适合 (μ+λ)/(μ,λ) ES（子代数独立）。
6. **(μ+λ) 精英保留 vs (μ,λ) 全替换**：+ 保留父代防退化，, 全淘汰鼓励探索（适合自适应参数）。

---

## 十、思考题

1. `varOr` 的复制分支不 clone，直接 append 引用。若后续对这个个体 in-place 操作会怎样？为什么 deap 仍这么做？（提示：复制分支不会被改，但共享引用有风险）
2. `eaSimple` 用 `selBest`（确定性选最优）会退化。为什么 `eaMuPlusLambda` 用 `selBest` 不会？（提示：(μ+λ) 有 lambda_ 个子代注入多样性）
3. `_evaluate_invalid` 返回 nevals。若 varOr 的 cxpb=0.5, mutpb=0.3，lambda_=100，期望 nevals 是多少？（提示：复制概率 0.2，约 80）
4. `toolbox.map` 默认是 `map`（返回迭代器）。若换成 `Pool.map`（返回 list），`zip(invalid_ind, fitnesses)` 还能正常工作吗？（提示：zip 接受任意可迭代）
5. (μ,λ) 要求 `lambda_ >= mu`。若 lambda_ = mu 会怎样？（提示：子代全选，无选择压力，等于随机游走）

---

## 十一、下一阶段预告

**阶段 7：examples/** —— 串联实战。用阶段 0-6 的全部组件跑三个经典问题：
- **One-Max**：位串求和（验证 eaSimple）
- **Sphere**：连续函数优化（验证 eaMuPlusLambda + mutGaussian）
- **TSP**：旅行商（验证自定义个体/交叉/变异 + eaSimple）

这是"把零件装成整机"的验证阶段——如果前 6 阶段有接口不兼容，examples 会暴露。每个例子都是可独立运行的完整脚本。