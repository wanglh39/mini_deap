# 阶段 8：tools/emo.py —— NSGA-II 多目标进化算法

> 对应 DEAP：`deap/tools/emo.py`（原版 500+ 行含 log 版排序，本阶段约 300 行保留 standard 版）
> 产出：`mini_deap/tools/emo.py` + `tests/tools/test_emo.py`（18 测试全过）+ `examples/nsga2_zdt1.py`

---

## 一、背景：多目标优化有什么不同？

前 7 阶段都是单目标——有一个标量 fitness，比大小就行。多目标优化完全不同：

**例子**：买车。要便宜（目标1 min）又要性能好（目标2 max）。最便宜的车性能差，性能最好的车贵。没有"两全其美"的最优解，而是一组**互不支配**的折中解——Pareto 前沿。

**Pareto 支配**：解 A 支配 B，当且仅当 A 在所有目标上不差于 B，且至少一个目标严格更好。
**Pareto 前沿**：所有非支配解的集合——没有任何解能同时在所有目标上改进它们。

**多目标进化的目标**：
1. **接近前沿**：找到的解尽量接近真实 Pareto 前沿
2. **分布均匀**：前沿上的解分布均匀，不挤成一团

NSGA-II（Deb 2002）用两个机制实现这两点：
- **非支配排序**（前沿秩）→ 接近前沿
- **拥挤距离** → 分布均匀

---

## 二、核心设计思想

### ① 非支配排序 —— 把种群分层

```python
def sortNondominated(individuals, k, first_front_only=False):
    # 1. 统计每个 fitness 被多少个支配、支配哪些
    for i, fit_i in enumerate(fits):
        for fit_j in fits[i+1:]:
            if fit_i.dominates(fit_j):
                dominating_fits[fit_j] += 1        # fit_j 被支配数+1
                dominated_fits[fit_i].append(fit_j) # fit_i 支配的列表
            elif fit_j.dominates(fit_i):
                ...
    # 2. 被支配数==0 的进第一前沿
    # 3. 第一前沿支配的个体，被支配数-1，减到0进下一前沿
```

**算法**（Deb 的快速非支配排序）：
1. 两两比较所有 fitness，统计每个被多少个支配（`dominating_fits`），及它支配哪些（`dominated_fits`）
2. 被支配数 == 0 的进第一前沿（Pareto 最优）
3. 对第一前沿的每个个体，把它支配的个体的被支配数减 1，减到 0 的进下一前沿
4. 重复直到排完 k 个

**复杂度** O(M·N²)，M 目标数，N 个体数。两两比较是 N²，每次比较是 M 维。

**为什么按 fitness 去重？** `map_fit_ind = defaultdict(list)` 把相同 fitness 的个体合并。支配关系只取决于 fitness 值，相同 fitness 的个体共享支配关系，避免重复比较。

### ② 拥挤距离 —— 同层内的密度估计

```python
def assignCrowdingDist(individuals):
    for i in range(nobj):                           # 对每个目标维
        crowd.sort(key=lambda e: e[0][i])           # 按该维排序
        distances[crowd[0][1]] = float("inf")       # 边界赋 inf
        distances[crowd[-1][1]] = float("inf")
        norm = nobj * (crowd[-1][0][i] - crowd[0][0][i])  # 归一化
        for prev, cur, nxt in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
            distances[cur[1]] += (nxt[0][i] - prev[0][i]) / norm  # 两侧邻居距离
```

**拥挤距离**：个体在目标空间中两侧邻居的矩形包围盒周长。
- **边界个体**（每维的 min/max）距离 = inf，保证不被淘汰（维持前沿范围）
- **中间个体**距离 = Σ_维 (f_{i+1} - f_{i-1}) / (f_max - f_min)，归一化后累加
- 距离大 → 周围稀疏 → 选择优先（维持分布均匀）

**为什么用 inf 而非大数？** inf 保证边界个体在任何情况下优先于中间个体。用大数可能被多个中间个体的累加超过。

### ③ NSGA2 选择 —— 前沿秩 + 拥挤距离

```python
def selNSGA2(individuals, k):
    pareto_fronts = sortNondominated(individuals, k)  # 分层
    for front in pareto_fronts:
        assignCrowdingDist(front)                     # 每层算拥挤距离
    chosen = list(chain(*pareto_fronts[:-1]))         # 整层取前面的
    k = k - len(chosen)
    if k > 0:                                         # 最后一层放不下
        sorted_front = sorted(pareto_fronts[-1],
                              key=attrgetter("fitness.crowding_dist"), reverse=True)
        chosen.extend(sorted_front[:k])               # 按拥挤距离取前 k 个
```

**选择策略**：
1. 整层整层地取前面的前沿（前沿秩优先）
2. 最后一层放不下时，按拥挤距离排序取前几个（同层内拥挤距离优先）

**为什么这样选？** 前沿秩保证"接近前沿"（第1层比第2层优），拥挤距离保证"分布均匀"（同层内稀疏的优先）。两个目标分治。

---

## 三、逐函数精读

### 3.1 sortNondominated 的分层过程

```python
# 第一前沿：被支配数==0
for i, fit_i in enumerate(fits):
    for fit_j in fits[i+1:]:
        if fit_i.dominates(fit_j): ...
    if dominating_fits[fit_i] == 0:
        current_front.append(fit_i)

# 后续前沿：当前前沿支配的个体，被支配数减到0
while pareto_sorted < N:
    for fit_p in current_front:
        for fit_d in dominated_fits[fit_p]:
            dominating_fits[fit_d] -= 1
            if dominating_fits[fit_d] == 0:
                next_front.append(fit_d)
```

**关键优化**：不重复比较。排第一前沿时已建好 `dominated_fits`（每个 fitness 支配哪些），后续前沿只需遍历当前前沿的 `dominated_fits`，把被支配数减 1。减到 0 说明它的所有支配者都已进入前面的前沿，它属于下一前沿。

**k 截断**：排到第 k 个个体就停，不排完所有层。NSGA2 只需选 k 个，后面的层不用排。

### 3.2 assignCrowdingDist 的归一化

```python
norm = nobj * float(crowd[-1][0][i] - crowd[0][0][i])  # 该维范围 × nobj
for prev, cur, nxt in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
    distances[cur[1]] += (nxt[0][i] - prev[0][i]) / norm
```

**归一化因子** `nobj * (f_max - f_min)`：
- `(f_max - f_min)` 把各维缩放到 [0,1]（不同目标量纲可能差很大）
- `nobj` 使总距离在 [0, 1] 范围（每维贡献 1/nobj）

**zip 三元组** `crowd[:-2], crowd[1:-1], crowd[2:]`：对每个中间个体，取它的前一个和后一个（按该维排序）。`(nxt - prev)` 是两侧邻居在该维的距离——包围盒边长。

### 3.3 selNSGA2 的整层取 + 最后一层切

```python
chosen = list(chain(*pareto_fronts[:-1]))   # 前面的层全取
k = k - len(chosen)
if k > 0:
    sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"), reverse=True)
    chosen.extend(sorted_front[:k])         # 最后一层按拥挤距离取
```

**chain(*fronts[:-1])**：把前面的前沿展平成一个列表。`[:-1]` 排除最后一层（可能放不下）。

**最后一层切**：按 `fitness.crowding_dist` 降序排，取前 k 个。拥挤距离大的优先——维持前沿分布均匀。

### 3.4 selTournamentDCD —— 支配+拥挤距离锦标赛

```python
def tourn(ind1, ind2):
    if ind1.fitness.dominates(ind2.fitness): return ind1    # 支配优先
    elif ind2.fitness.dominates(ind1.fitness): return ind2
    if ind1.fitness.crowding_dist < ind2.fitness.crowding_dist: return ind2  # 拥挤距离次之
    elif ind1.fitness.crowding_dist > ind2.fitness.crowding_dist: return ind1
    return ind1 if random.random() <= 0.5 else ind2         # 都相同随机
```

**比较优先级**：支配关系 > 拥挤距离 > 随机。这是 NSGA2 原版的锦标赛选择。

**为什么需要它？** selNSGA2 是"一次性选 k 个"的批量选择。selTournamentDCD 是"两两比赛"的锦标赛选择，适合 eaSimple（每代 select(pop, len(pop))）。两种选择方式对应不同算法流程。

---

## 四、SBX 交叉 + 多项式变异

NSGA2 的标准配套算子，适合有界连续空间。

### 4.1 cxSimulatedBinaryBounded（SBX）

模拟二进制交叉：在连续空间模拟二进制串交叉的效果。`eta` 控制子代与父代的相似度——大 eta 子代像父代，小 eta 子代差异大。

核心：对每维，以 0.5 概率交叉。用 beta 分布生成子代位置，裁剪到 [low, up]。

### 4.2 mutPolynomialBounded（多项式变异）

多项式变异：在个体附近做多项式分布的扰动。`eta` 控制变异强度——大 eta 变异小，小 eta 变异大。每维以 `indpb` 概率变异，裁剪到 [low, up]。

**为什么用这两个而非 cxBlend/mutGaussian？** SBX 和多项式变异是 NSGA2 原版实现，有理论保证（分布形状与前沿几何匹配）。cxBlend/mutGaussian 也能用但收敛性差些。

---

## 五、ZDT1 实战

### 5.1 问题定义

ZDT1（Zitzler-Deb-Thiele 1）：
```
f1(x) = x1
g(x) = 1 + 9/(n-1) * sum(x[1:])
f2(x) = g * (1 - sqrt(f1/g))
```
x ∈ [0,1]^30，最小化 (f1, f2)。理论前沿：f2 = 1 - sqrt(f1), f1 ∈ [0,1]。

### 5.2 运行结果

```
gen  f1_min    f2_min
0    0.029     2.620
...
50   0.000     0.333

Pareto 前沿采样：
  f1=0.081, f2=0.994
  f1=0.745, f2=0.333
  f1=0.000, f2=1.408
```

**前沿形状**：f1 从 0 到 0.745，f2 从 0.33 到 1.41。理论前沿 f2 = 1 - sqrt(f1)：
- f1=0 → f2=1（前沿起点）
- f1=1 → f2=0（前沿终点）

50 代的收敛还不够深（f1=0 时 f2 应≈1，实际 1.41；f1=0.745 时 f2 应≈0.137，实际 0.33），但前沿的**形状和方向**已正确显现。增加 ngen 可逼近理论前沿。

### 5.3 为什么用 eaMuPlusLambda + selNSGA2？

NSGA2 的流程 = 合并父子 → 非支配排序 → 按前沿秩+拥挤距离选。`eaMuPlusLambda` 的"父子合并选"正好是这个时机——`select(population + offspring, mu)` 时调 `selNSGA2`，对合并种群分层选。

---

## 六、和 deap 原版对照

| 本教学版 | deap `emo.py` | 差异 |
|---|---|---|
| `sortNondominated` | 同 | 完全保留（standard 版） |
| `assignCrowdingDist` | 同 | 完全保留 |
| `selNSGA2` | 同 | 完全保留（只支持 nd='standard'） |
| `selTournamentDCD` | 同 | 完全保留 |
| `cxSimulatedBinaryBounded` | 同 | 简化：low/up 标量，非序列 |
| `mutPolynomialBounded` | 同 | 简化：low/up 标量，非序列 |
| `sortLogNondominated` | deap 有 | 砍掉：log 版是 O(M·N·logN) 优化，复杂度高，教学版用 standard 够了 |

**保留全部常用功能**，只砍掉 log 版非支配排序（性能优化，实现复杂，教学价值低）。

---

## 七、常见陷阱

1. **weights 正负决定支配方向**：`weights=(-1.0,-1.0)` 最小化，`dominates` 内部用 wvalues = weights × values 比较。搞反会往错误方向分层。
2. **忘先调 assignCrowdingDist 就用 selTournamentDCD**：`crowding_dist` 属性不存在 → AttributeError。selNSGA2 内部自动调，selTournamentDCD 需手动调。
3. **相同 fitness 的个体**：sortNondominated 按 fitness 去重，相同 fitness 的个体共享支配关系。assignCrowdingDist 对相同 fitness 会 `continue`（该维范围==0）。
4. **k > len(individuals)**：selNSGA2 不会报错但返回的列表可能不足 k 个（前沿不够多）。
5. **SBX 的 eta 太大**：eta=20 子代几乎等于父代，收敛极慢。常用 eta=15-20。
6. **多项式变异的 indpb**：通常 `1/n`（每维期望变异1次）。太大破坏前沿分布。

---

## 八、Python 语言特性备忘

- **`defaultdict(list)`**：`map_fit_ind[fit].append(ind)`，不存在的 key 自动建空 list。省去 `if fit not in d: d[fit] = []`。
- **`itertools.chain(*lists)`**：把多个列表展平成一个迭代器。`chain(*fronts[:-1])` 把前面的前沿拼成一个列表。
- **`attrgetter("fitness.crowding_dist")`**：点号路径属性获取，等价 `lambda x: x.fitness.crowding_dist`。sorted 的 key 函数。
- **`zip(a[:-2], a[1:-1], a[2:])`**：三元滑动窗口。对每个中间元素取前一个和后一个。经典用法。
- **`float("inf")`**：正无穷。比任何有限数大。边界个体的拥挤距离设 inf 保证优先。

---

## 八点五、设计权衡

**为什么只实现 standard 版非支配排序，砍掉 log 版？**
log 版（sortLogNondominated）用分治+中位数切分，复杂度 O(M·N·logN)，比 standard 的 O(M·N²) 快。但实现极复杂（splitA/sweepA/sortNDHelperB 等十几个辅助函数），教学价值低。standard 版直接两两比较，逻辑清晰，足以理解非支配排序的思想。性能上 N<1000 时 standard 够用。

**为什么 selNSGA2 整层取而非逐个选？**
整层取保证前沿完整性——第1层全选，第2层全选，...，最后一层切。逐个选可能把第1层的某些个体漏选（如果按拥挤距离逐个排，边界个体可能被中间个体超过）。整层取保证前沿秩的严格优先。

**为什么拥挤距离而非 ε-支配或参考点？**
拥挤距离是 NSGA2 原版方法，简单直观（两侧邻居距离）。ε-支配（SPEA2）和参考点（NSGA3）更先进但复杂。教学版用拥挤距离足够。

---

## 九、关键收获

1. **多目标没有全局最优，只有 Pareto 前沿**：解互不支配，是不同目标间的折中。
2. **非支配排序分层**：O(M·N²) 把种群按支配关系分层，第一层是 Pareto 最优。用被支配数减1的技巧避免重复比较。
3. **拥挤距离维持分布均匀**：边界个体 inf 保证不被淘汰，中间个体按两侧邻居距离排序。和前沿秩分治"接近前沿"和"分布均匀"。
4. **NSGA2 = eaMuPlusLambda + selNSGA2**：父子合并 → 非支配排序 → 按前沿秩+拥挤距离选。eaMuPlusLambda 的合并选择时机正好。
5. **SBX + 多项式变异是 NSGA2 标准配套**：有理论保证，eta 控制子代与父代相似度。
6. **按 fitness 去重省比较**：相同 fitness 的个体共享支配关系，避免 N² 重复比较。

---

## 十、思考题

1. `sortNondominated` 用被支配数减1的技巧排后续前沿。如果直接对每个个体重新数被支配数会怎样？（提示：O(N²) 每层 vs O(N) 每层）
2. `assignCrowdingDist` 的边界个体赋 inf。若两个边界个体在同一维都是 min，会怎样？（提示：都赋 inf，但该维范围==0 时 continue）
3. `selNSGA2` 的最后一层按拥挤距离切。若最后一层全挤在一起（拥挤距离都≈0），切前 k 个和随机切有区别吗？（提示：几乎没区别，距离都≈0）
4. ZDT1 的理论前沿 f2 = 1 - sqrt(f1)。50 代后 f1=0 时 f2=1.41（应≈1）。为什么？（提示：g(x) 还没收敛到 1，增加 ngen）
5. `sortNondominated` 按 fitness 去重。若两个个体内容不同但 fitness 相同，它们会被分到同一层吗？（提示：会，支配关系只看 fitness）

---

## 十一、下一阶段预告

**阶段 9：gp.py + 符号回归** —— 遗传规划。前 8 阶段个体都是定长向量（位串/浮点/排列），阶段 9 个体是**变长树结构**（语法树）：
- **PrimitiveSet**：定义函数集 + 终端集（变量/常量）
- **PrimitiveTree**：树个体，继承 list，元素是 Primitive/Terminal
- **genHalfAndHalf**：子树生成（满树/半树混合）
- **cxOnePoint（树版）**：交换两棵子树
- **mutUniform（树版）**：替换一个子树
- **实战**：符号回归——用 GP 拟合 sin(x) 等函数

GP 的核心难点：**树结构无固定长度**，交叉/变异在子树层面操作，要处理类型一致性（子树替换要类型匹配）。