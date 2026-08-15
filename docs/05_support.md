# 阶段 5：tools/support —— 统计与记录组件（算法层的观测插件）

> 对应 DEAP：`deap/tools/support.py`（原版 652 行，本阶段约 235 行，保留核心组件）
> 产出：`mini_deap/tools/support.py` + `tests/tools/test_support.py`（13 测试全过）

---

## 一、背景：为什么需要"观测插件"？算法跑起来要看什么？

进化算法跑几十上百代，过程中你要回答：
- **收敛了吗？** 每代最优/平均 fitness 在变好吗？
- **历代最优是谁？** 算法可能因选择压力波动丢掉曾见过的最优解，要单独存。
- **每代花了多少评估？** 评估是黑盒最贵的部分，要统计 nevals。

如果把这些写进算法骨架，算法代码被观测逻辑淹没。DEAP 的解法：**做成可选回调插件**，算法里 `if stats: record = stats.compile(pop)`，不传也能跑。三个组件各管一摊：

| 组件 | 职责 | 何时调 |
|---|---|---|
| `Statistics` | 算种群统计量（avg/max/min...） | 每代 `stats.compile(pop)` |
| `HallOfFame` | 保留历代最优 k 个 | 每代 `hof.update(pop)` |
| `Logbook` | 记录每代、输出表格 | 每代 `log.record(gen=..., **stats)` |

它们是**正交的**：可以只用 Statistics 不用 HallOfFame，或反之。算法层用 `if stats:` 守卫，不传就不调。

---

## 二、核心设计思想

### ① 注册式统计 —— `Statistics.register` 同 Toolbox 一脉相承

```python
class Statistics:
    def __init__(self, key=identity):
        self.key = key
        self.functions = dict()
    def register(self, name, function, *args, **kargs):
        self.functions[name] = partial(function, *args, **kargs)
    def compile(self, data):
        values = tuple(self.key(elem) for elem in data)
        return {name: func(values) for name, func in self.functions.items()}
```

和 `Toolbox.register` 同款用 `partial` 冻参。`key` 先把每个个体映射成要统计的值（如 `attrgetter("fitness.values")` 取适应度元组），再对所有注册函数批量应用。一次 `key` 提取，多函数复用。

### ② `bisect` 维持有序 —— HallOfFame 的插入排序

```python
def insert(self, item):
    item = deepcopy(item)
    i = bisect_right(self.keys, item.fitness)
    self.items.insert(len(self) - i, item)
    self.keys.insert(i, item.fitness)
```

`self.keys` 维持**升序**的 fitness 列表，`bisect_right` 二分找插入位置 → O(log n) 插入。`self.items` 是对应的个体列表，**降序**（`[0]` 最优），所以 `items.insert(len(self) - i, ...)`。双列表反向对应是 deap 的技巧：keys 升序方便 bisect，items 降序方便 `hof[0]` 取最优。

为什么不用 `sorted` 每次重排？O(n log n) 每次 vs O(log n) 每次。名人堂每代 update 整个种群，高频插入，bisect 省一个 log 因子。

### ③ `deepcopy` 隔离 —— 名人堂不污染种群

```python
def insert(self, item):
    item = deepcopy(item)   # 关键
    ...
```

名人堂存的是个体的**深拷贝**。若存引用，后续种群里的变异会改名人堂里的个体（同一对象），历代最优被污染。deepcopy 隔离后，名人堂里的个体是冻结的快照。

### ④ `stream` 增量输出 —— Logbook 只打印新行

```python
@property
def stream(self):
    startindex, self.buffindex = self.buffindex, len(self)
    return self.__str__(startindex)
```

`buffindex` 记住上次输出到哪，`stream` 只格式化 `[startindex:]` 的新行。这样算法里每代 `print(log.stream)` 只输出当代一行（首次含表头），不重复打印历史。比每代 `print(log)` 重打整个表高效。

---

## 三、逐类精读

### 3.1 `Statistics` —— key 提取 + partial 冻参 + compile 批量

```python
def compile(self, data):
    values = tuple(self.key(elem) for elem in data)   # 一次 key 提取
    return {name: func(values) for name, func in self.functions.items()}
```

`values = tuple(self.key(elem) for elem in data)`：先把种群映射成要统计的值序列。如 `key=attrgetter("fitness.values")`，`values = ((3.2,), (5.0,), ...)`。然后每个注册函数 `func(values)` 算一个统计量。**key 只跑一次，所有函数复用同一 values**，避免每个函数各自遍历种群。

`register` 用 `partial`：`stats.register("mean", numpy.mean, axis=0)` 冻住 `axis=0`。和 Toolbox.register 完全同款。

### 3.2 `MultiStatistics` —— 多键复合，dict 继承

```python
class MultiStatistics(dict):
    def compile(self, data):
        return {name: stats.compile(data) for name, stats in self.items()}
    def register(self, name, function, *args, **kargs):
        for stats in self.values():
            stats.register(name, function, *args, **kargs)
```

继承 `dict`：`MultiStatistics(fitness=stats1, size=stats2)` 本质是 `{"fitness": stats1, "size": stats2}`。`compile` 对每个子 Statistics 分别 compile，返回嵌套 dict。`register` 对所有子 Statistics 同时注册同一函数（如同时给 fitness 和 size 注册 mean）。

**为什么继承 dict 而非组合？** 让 `mstats["fitness"]` 直接访问子 Statistics，且 `mstats.fields` 能用 `sorted(self.keys())`。dict 的接口免费复用。

### 3.3 `Logbook` —— record/select/stream + 列宽对齐

```python
class Logbook(list):
    def record(self, **infos):
        self.append(infos)   # 每条记录是一个 dict
```

继承 `list`：每条记录是一个 dict 追加到末尾。`log[0]` 是第一代记录，`log[-1]` 是最新。

```python
def select(self, *names):
    if len(names) == 1:
        return [entry.get(names[0], None) for entry in self]
    return tuple([entry.get(name, None) for entry in self] for name in names)
```

`select("avg")` 返回所有代的 avg 列（一个列表）。`select("gen", "max")` 返回两列（元组）。用于事后画收敛曲线：`gen, avg = log.select("gen", "avg"); plt.plot(gen, avg)`。

`__str__` 的列宽对齐：先初始化 `columns_len` 为列名长度，遍历内容时取 `max(列名长度, 内容长度)`，最后用 `"{:<%i}" % l` 左对齐模板格式化。首次输出（`startindex==0`）插表头行。

### 3.4 `HallOfFame` —— update 的去重 + 替换逻辑

```python
def update(self, population):
    for ind in population:
        if len(self) == 0 and self.maxsize != 0:
            self.insert(population[0]); continue      # 空堂特判
        if ind.fitness > self[-1].fitness or len(self) < self.maxsize:
            for hofer in self:
                if self.similar(ind, hofer):
                    break                               # 已有重复，跳过
            else:
                if len(self) >= self.maxsize:
                    self.remove(-1)                     # 满了先删最差
                self.insert(ind)
```

逐个判断：
1. **空堂特判**：`self[-1]` 在空堂时越界，单独处理第一个。
2. **候选条件**：`ind.fitness > self[-1].fitness`（比当前最差优）或 `len(self) < maxsize`（还没满）。不满足直接跳过（不够格）。
3. **去重**：`for hofer in self: if similar(ind, hofer): break`。`similar` 默认 `operator.eq`（内容完全相同算重复）。`for...else` 的 `else` 在**没 break**时执行（无重复）。
4. **替换**：满了先 `remove(-1)` 删最差，再 `insert`。

`similar` 可自定义：如 GP 树用结构等价而非内容相等判断重复。

### 3.5 `ParetoFront` —— 三标志位非支配过滤

```python
def update(self, population):
    for ind in population:
        is_dominated = False
        dominates_one = False
        has_twin = False
        to_remove = []
        for i, hofer in enumerate(self):
            if not dominates_one and hofer.fitness.dominates(ind.fitness):
                is_dominated = True; break        # ind 被现有成员支配 -> 丢弃
            elif ind.fitness.dominates(hofer.fitness):
                dominates_one = True
                to_remove.append(i)                # ind 支配现有成员 -> 标记删除
            elif ind.fitness == hofer.fitness and self.similar(ind, hofer):
                has_twin = True; break             # 已有等价成员 -> 丢弃
        for i in reversed(to_remove):              # 从后往前删，避免索引错位
            self.remove(i)
        if not is_dominated and not has_twin:
            self.insert(ind)
```

三个标志：
- `is_dominated`：ind 被某个现有成员支配 → 不加入（break 提前退出）。
- `dominates_one`：ind 支配某些现有成员 → 标记待删（`to_remove`）。
- `has_twin`：已有等价成员 → 不加入（避免重复）。

`to_remove` 从后往前删（`reversed`）：列表删除会移位，从后往前删保证前面的索引不变。

`maxsize=None`（无上限）：Pareto 前沿大小由问题决定，连续问题可能无限大，`similar` 限制密度。

---

## 三点五、运行示例：三组件如何协同

把 Statistics + HallOfFame + Logbook 串起来看一个完整用法（阶段 7 会真正跑）：

```python
import numpy
from operator import attrgetter
from mini_deap.tools.support import Statistics, MultiStatistics, HallOfFame, Logbook

# 1. 建统计器：对 fitness.values 算多个统计量
stats = Statistics(key=attrgetter("fitness.values"))
stats.register("avg", numpy.mean)
stats.register("min", numpy.min)
stats.register("max", numpy.max)

# 2. 建名人堂：保留历代最优 5 个
hof = HallOfFame(maxsize=5)

# 3. 建日志本
log = Logbook()

# 算法主循环里（伪代码）
for gen in range(ngen):
    # ... 选择、交叉、变异、评估 ...
    hof.update(population)                    # 先更新名人堂
    record = stats.compile(population)       # 再算统计
    log.record(gen=gen, nevals=len(invalid), **record)  # 记录
    print(log.stream)                        # 增量输出当代一行
```

**顺序很重要**：`hof.update` 要在 `stats.compile` 之前或后都行（它们独立），但都要在评估之后（fitness 得是有效的）。`log.record` 的 `**record` 把 stats 的 dict 展开成 `avg=..., min=..., max=...` 关键字参数，和 `gen`/`nevals` 合并成一条记录。

**MultiStatistics 的多目标场景**：

```python
mstats = MultiStatistics(fitness=Statistics(key=attrgetter("fitness.values")),
                         size=Statistics(key=len))
mstats.register("avg", numpy.mean)
mstats.register("max", numpy.max)
# compile 返回 {"fitness": {"avg":..., "max":...}, "size": {"avg":..., "max":...}}
record = mstats.compile(population)
log.record(**record)   # Logbook 能处理嵌套 dict
```

`MultiStatistics` 对每个子 Statistics 分别 compile，Logbook 的 `__str__` 会把嵌套 dict 展平成 `fitness-avg`、`fitness-max`、`size-avg` 等列名。

---

## 四、和 deap 原版对照

| 本教学版 | deap `support.py` | 差异 |
|---|---|---|
| `Statistics` | 同 | 完全保留 |
| `MultiStatistics` | 同 | 完全保留 |
| `Logbook` record/select/stream | 同 | 完全保留 |
| `Logbook.__txt__` chapter 格式化 | deap 有复杂 chapter 机制 | 简化：单层表格，无 chapter |
| `HallOfFame` | 同 | 完全保留 |
| `ParetoFront` | 同 | 完全保留 |
| `History`（家谱树） | deap 有 | 砍掉：依赖 networkx，非核心 |

**保留全部常用组件**，砍掉 History 和 Logbook 的 chapter 机制（MultiStatistics 的子日志输出）。

---

## 五、常见陷阱

1. **`HallOfFame` 的 `similar=eq` 对浮点个体**：两个 `[1.0, 2.0]` 内容相等算重复，但浮点经变异后微小差异就不等了。要按"结构等价"去重得自定义 similar。
2. **忘 `hof.update(offspring)` 只 update 初始种群**：历代最优要每代 update 新种群，否则名人堂只记初始。
3. **`Logbook.stream` 忘消费**：`log.stream` 是 property，访问一次就推进 `buffindex`。若 `record` 后不 `print(log.stream)`，下次 stream 会一次输出多行。
4. **`Statistics` 的 key 返回多目标元组**：`key=attrgetter("fitness.values")` 对多目标返回 `(f1, f2)`，`numpy.mean` 会按 axis 算。单目标 `key=lambda ind: ind.fitness.values[0]` 取标量更安全。
5. **`ParetoFront` 无上限膨胀**：连续问题 Pareto 前沿可能无限大，内存炸。用 `similar` 限制密度（如按精度取整后比较）。
6. **`HallOfFame.insert` 不检查 maxsize**：`insert` 直接插入不检查 maxsize，`update` 负责检查。若直接调 `insert` 会超容。API 上 insert 是半公开的。
7. **`MultiStatistics` 的 `fields` 排序**：`sorted(self.keys())` 按字典序，列顺序可能不合预期。显式设 `log.header` 控制。

---

## 六、Python 语言特性备忘

- **`bisect.bisect_right(a, x)`**：返回 x 应插入 a（升序）的位置，使 a 仍有序。a 中已有等于 x 的元素时插到右边。O(log n)。
- **`functools.partial`**：同 Toolbox，冻参造半成品函数。Statistics.register 用它绑 `axis` 等参数。
- **继承 `dict`/`list`**：`MultiStatistics(dict)` 免费获得 `mstats["key"]`、`len`、`in` 等。`Logbook(list)` 获得 `log[0]`、`len`、迭代。代价：要小心基类方法冲突。
- **`for...else`**：`else` 在循环**正常结束（没 break）**时执行。HallOfFame 用它判"无重复 = 没 break"。
- **`@property`**：`stream` 是只读 property，访问时执行计算并推进 `buffindex`。每次访问有副作用（改 buffindex），不是纯 property——这是 deap 的设计，方便算法里 `print(log.stream)`。

---

## 六点五、设计权衡：为什么这样切分组件

**为什么不把 Statistics/HallOfFame/Logbook 合成一个 Recorder 类？**
三者职责正交且可选：你可能只要 stats 不要 hof（纯收敛分析），或只要 hof 不要 stats（只要最优解不要过程）。合成一个类就得传一堆 `None` 开关，且强制耦合。deap 的选择是三个独立类 + 算法里 `if stats:` / `if hof:` 守卫，调用方按需组合。

**为什么 HallOfFame 用双列表（keys + items）而非一个 (fitness, item) 元组列表？**
元组列表也能 bisect（按元组首元素排序）。但双列表的好处：`self.items[0]` 直接拿个体，不用解包；`self.keys` 是纯 fitness 列表，bisect 比较时不用构造元组。微优化，但在高频插入（每代 update 整个种群）下有意义。

**为什么 Logbook 继承 list 而非用 deque？**
record 是 append（尾加），select 是全表遍历，没有头部删除需求。list 的 `__getitem__`/`__str__`/迭代接口更通用，且 `log[0]` 取第一代直观。deque 的 popleft 优势这里用不上。

**为什么 ParetoFront 继承 HallOfFame 而非独立实现？**
共享 `insert`/`remove`/`maxsize`/`similar` 机制，只重写 `update`（非支配过滤 vs 简单择优）。继承复用插入逻辑，只改更新策略。代价：ParetoFront 的 `maxsize` 语义弱化（前沿大小由问题决定，maxsize 常设 None）。

---

## 七、关键收获

1. **观测做成可选回调插件**：算法 `if stats: stats.compile(pop)`，不传也能跑，算法代码不被观测逻辑淹没。
2. **注册式统计同 Toolbox 一脉相承**：`partial` 冻参 + key 提取 + 批量应用，一套模式贯穿全库。
3. **`bisect` 维持有序列表**：O(log n) 插入 vs O(n log n) 重排，高频插入场景的关键优化。keys 升序 + items 降序双列表反向对应是巧技。
4. **`deepcopy` 隔离防污染**：名人堂存快照不存引用，后续变异不影响历代最优记录。
5. **`stream` 增量输出**：`buffindex` 记住输出位置，每次只打新行，不重打全表。
6. **ParetoFront 三标志位过滤**：`is_dominated`/`dominates_one`/`has_twin` 分流，`to_remove` 从后往前删避免索引错位。

---

## 八、思考题

1. `HallOfFame` 用 `bisect` 维持 `keys` 升序，`items` 降序。为什么不用一个 `sorted` 列表每次重排？（提示：插入频率）
2. `Logbook` 继承 `list`，`record` 就是 `append`。如果改用组合（内含一个 list）而非继承，会失去什么？（提示：`log[0]`、`len`、迭代）
3. `ParetoFront.update` 的 `to_remove` 为什么 `reversed` 删？（提示：列表删元素会移位）
4. `Statistics` 的 `key` 提取一次，所有函数复用。如果每个函数各自遍历种群提 key，开销差多少？（提示：N 函数 × M 种群 vs M 种群 + N 函数）
5. `HallOfFame` 的 `similar=eq` 对 GP 树个体：两棵不同结构但相同语义的树，`eq` 判不等（结构不同）→ 都进名人堂。要按语义去重怎么改 `similar`？

---

## 九、下一阶段预告

**阶段 6：algorithms.py** —— 算法骨架。`varAnd`（交叉 AND 变异）、`varOr`（交叉 OR 变异 OR 复制）、`eaSimple`（简单 GA）、`eaMuPlusLambda`（(μ+λ) ES）。核心：惰性评估（只重算 `invalid_ind`）、`toolbox.map` 并行、`stats`/`halloffame` 可选接入。这是把前 5 阶段（Fitness/Toolbox/Creator/算子/统计）串成可运行算法的"主循环"。