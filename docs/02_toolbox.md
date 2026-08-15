# 阶段 2：Toolbox —— 算子容器（进化算法的调度中心）

> 对应 DEAP：`deap/base.py` 的 `Toolbox` 类（原版约 90 行，本阶段同量级，核心全保留）
> 产出：`mini_deap/base/toolbox.py` + `tests/base/test_toolbox.py`（13 测试全过）

---

## 一、背景：为什么需要一个 Toolbox？算子直接调用不行吗？

假设没有 Toolbox，你写一个 GA 大概是这样：

```python
from my_operators import selTournament, cxTwoPoint, mutGaussian

def ea_simple(pop, ngen, cxpb, mutpb):
    for g in range(ngen):
        pop = selTournament(pop, len(pop), tournsize=3)          # 直接调
        offspring = [copy.deepcopy(ind) for ind in pop]
        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:
                offspring[i-1], offspring[i] = cxTwoPoint(offspring[i-1], offspring[i])
        for i in range(len(offspring)):
            if random.random() < mutpb:
                offspring[i], = mutGaussian(offspring[i], mu=0, sigma=1, indpb=0.2)
        fitnesses = [evaluate(ind) for ind in offspring]          # 串行评估
        ...
```

痛点逐条展开：

1. **算法和具体算子硬绑**：想把 `selTournament` 换成 `selRoulette`，得改算法源码。算法不可复用。
2. **默认参数散落算法里**：`tournsize=3`、`mu=0`、`sigma=1`、`indpb=0.2` 全写死在算法中。换组参数又得改算法。
3. **没法并行**：评估是 `[evaluate(ind) for ind in ...]`，想换 `pool.map` 得改算法循环结构。
4. **没法复用算法骨架**：这套 `ea_simple` 只能配这几个算子，换一套（如 ES 的 (μ+λ)）就得重写整个循环。
5. **横切逻辑无处安放**：想给交叉加"限制个体大小"、给评估加"超时"、统计算子调用次数，得侵入算子源码或算法源码。

DEAP 的解法是**引入 Toolbox 当中间层**：所有算子先 `register` 到 toolbox 上取个别名（`mate`/`mutate`/`select`/`evaluate`），算法只认别名。这本质是**依赖倒置原则**：算法依赖抽象的"算子接口"（别名），具体算子由调用方注入。

```
调用方:  toolbox.register("mate", cxTwoPoint)          ← 算子配置在这
         toolbox.register("select", selTournament, tournsize=3)
              ↓
         Toolbox (粘合剂，存 partial 对象)
              ↓
算法:    toolbox.mate(a, b)                            ← 算法只认别名
         toolbox.select(pop, k=len(pop))
```

换算子、换参数、换并行 map、加装饰器，全在 register 处改，**算法代码一行不动**。这就是 DEAP 能用同一套 `eaSimple` 跑 GA、ES、GP、PSO 的根因。

---

## 二、核心设计思想

### ① `functools.partial` —— 冻结部分参数，造一个"半成品函数"

`register(alias, function, *args, **kargs)` 的核心就一行：

```python
pfunc = partial(function, *args, **kargs)
setattr(self, alias, pfunc)
```

`partial(func, 2, c=4)` 产生一个新可调用对象，调用它时等价于 `func(2, *你的参数, c=4, **你的kw)`。即**把 `2` 和 `c=4` 冻进函数里**，调用时只传剩余参数。

```python
tb.register("myFunc", func, 2, c=4)
tb.myFunc(3)        # → func(2, 3, c=4) → (2, 3, 4)
tb.myFunc(3, c=10)  # → func(2, 3, c=10) → (2, 3, 10)   # 调用时能覆盖冻参
```

**partial vs lambda 对比**（都能冻参，但有关键差异）：

| 维度 | `partial(func, 2, c=4)` | `lambda x: func(2, x, c=4)` |
|---|---|---|
| 反向解包 | ✅ `.func`/`.args`/`.keywords` 可拿回原函数和冻参 | ❌ 闭包捕获，无法反向拿回 |
| pickle | ✅ 实现了 `__reduce__`，可跨进程 | ❌ 闭包不可 pickle |
| `decorate` 可行 | ✅ 解包后套装饰器再重绑 | ❌ 拿不回原函数和冻参 |
| 元信息 | 可设 `__name__`/`__doc__` | 同 |
| 调用开销 | C 层，略快 | 字节码，略慢 |

`decorate` 能工作的前提就是 partial **可解包**：`pfunc.func` 拿原函数，`pfunc.args`/`pfunc.keywords` 拿冻参，套装饰器后重新 `register` 绑回。lambda 丢掉原函数，decorate 没法做。这是 DEAP 选 partial 而非 lambda 的根因。

### ② 默认注册 `clone=deepcopy` 和 `map=map` —— 两个"钩子"

```python
def __init__(self):
    self.register("clone", deepcopy)
    self.register("map", map)
```

这两个不是算子，是**算法骨架依赖的两个钩子**：

- **`clone`**：`varAnd` 里 `offspring = [toolbox.clone(ind) for ind in population]`，每代克隆整个种群。默认 `deepcopy`，但个体若是 numpy 数组，可换成 `numpy.copy` 提速。
- **`map`**：`fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)`，批量评估。默认内置 `map`（串行），换成 `multiprocessing.Pool().map` 即**并行评估，算法代码一行不改**。

这是 DEAP 并行设计的精髓：**并行不是算法的事，是 toolbox 配置的事。**

```python
# 串行
toolbox.register("map", map)
pop, log = eaSimple(pop, toolbox, ...)

# 并行（只改这一行）
pool = multiprocessing.Pool()
toolbox.register("map", pool.map)
pop, log = eaSimple(pop, toolbox, ...)   # 算法代码完全一样
pool.close(); pool.join()
```

注意 `pool` 生命周期要在算法外管理（`close` 停止接收任务、`join` 等子进程结束）。`pool.map` 会把 `toolbox.evaluate`（partial 对象）和每个 individual pickle 到子进程，partial 可 pickle（见上表），individual 可 pickle（阶段 3 的 `__reduce__` 保了），所以链路通。

### ③ `decorate` —— 给已注册算子套装饰器，保留原绑参

实战里常需要给算子加横切逻辑。`decorate` 干这个：

```python
def decorate(self, alias, *decorators):
    pfunc = getattr(self, alias)
    function, args, kargs = pfunc.func, pfunc.args, pfunc.keywords   # 解包 partial
    for decorator in decorators:
        function = decorator(function)                                # 逐层套
    self.register(alias, function, *args, **kargs)                    # 重新绑回原参
```

关键在**解包 partial**：装饰器套在**原函数**上（不是套在 partial 外），再重新 register 把原绑参绑回去。这样装饰器看到的是"完整参数调用"，绑参不丢。

**三个实战场景**：

```python
# 场景1：限制 GP 交叉后个体大小（防 bloat）
def limit_height(max_h):
    def decorator(func):
        def wrapper(ind1, ind2):
            (ind1, ind2) = func(ind1, ind2)
            if ind1.height > max_h: ind1[:] = prune(ind1, max_h)
            if ind2.height > max_h: ind2[:] = prune(ind2, max_h)
            return ind1, ind2
        return wrapper
    return decorator
toolbox.decorate("mate", limit_height(17))

# 场景2：统计变异次数
counter = {"mutate": 0}
def count_call(func):
    def wrapper(*args, **kw):
        counter["mutate"] += 1
        return func(*args, **kw)
    return wrapper
toolbox.decorate("mutate", count_call)

# 场景3：给评估加超时（需 signal，仅 Unix）
def timeout(seconds):
    def decorator(func):
        def wrapper(ind):
            with alarm(seconds): return func(ind)
        return wrapper
    return decorator
toolbox.decorate("evaluate", timeout(10))
```

**多装饰器顺序**：`for decorator in decorators: function = decorator(function)` 是左到右逐个套，后套的在最外层。`decorate("f", a, b)` → `b(a(orig))`，调用时 `b` 先执行。deap docstring 说"the last decorator decorating all the others"即此意。

---

## 三、逐段精读 `mini_deap/base/toolbox.py`

### 3.1 `__init__` —— 默认钩子

```python
def __init__(self):
    self.register("clone", deepcopy)
    self.register("map", map)
```

注意 `clone` 和 `map` 也是用 `register` 注册的，和普通算子同机制 —— **用户随时可覆盖**：`toolbox.register("clone", my_fast_clone)`。没有特殊待遇，统一接口。这避免了"默认钩子走特殊路径"的分支复杂度。

### 3.2 `register` —— partial + 元信息同步

```python
pfunc = partial(function, *args, **kargs)
pfunc.__name__ = alias
pfunc.__doc__ = function.__doc__

if hasattr(function, "__dict__") and not isinstance(function, type):
    pfunc.__dict__.update(function.__dict__.copy())

setattr(self, alias, pfunc)
```

逐行：
1. **partial 冻参**：核心。`partial(function, *args, **kargs)` 把 args/kargs 冻进 function。
2. **`__name__ = alias`**：让 `toolbox.mate.__name__ == "mate"`，日志/调试/Stats 里显示别名而非原函数名，可读性好。
3. **`__doc__` 同步**：`help(toolbox.mate)` 能看到原函数文档。
4. **`__dict__` 拷贝**：原函数上挂的自定义属性（如 `func.counter = 0`）拷到别名上，别名能访问。`isinstance(function, type)` 排除类：类的 `__dict__` 是 mappingproxy 且语义不对（类属性不该当实例属性拷给 partial）。
5. **`setattr`**：把 partial 挂成实例属性，`toolbox.mate` 直接可调用。

**完整调用追踪**：`tb.register("myFunc", func, 2, c=4)` 后 `tb.myFunc(3)` 发生了什么：
```
tb.myFunc → 实例属性查找 → partial(func, 2, c=4) 对象
partial.__call__(3) → func(2, 3, c=4) → (2, 3, 4)
```
中间无额外字典查找、无分支，partial 的 `__call__` 是 C 层实现，开销极低。

### 3.3 `unregister` —— 一行

```python
def unregister(self, alias):
    delattr(self, alias)
```

直接删实例属性。不存在则抛 `AttributeError`（Python 默认行为，不额外处理）。

### 3.4 `decorate` —— 解包 + 重套 + 重绑

见第二节 ③，核心是 `pfunc.func`/`pfunc.args`/`pfunc.keywords` 三件套解包 partial。

**完整调用追踪**：`tb.register("myFunc", func, 2, c=4)` 后 `tb.decorate("myFunc", trace)`：
```
pfunc = tb.myFunc = partial(func, 2, c=4)
function, args, kargs = func, (2,), {"c": 4}      # 解包
function = trace(func)                              # 套装饰器 → wrapper
register("myFunc", wrapper, 2, c=4)                 # 重新绑
tb.myFunc = partial(wrapper, 2, c=4)
```
之后 `tb.myFunc(3)` → `partial(wrapper, 2, c=4)(3)` → `wrapper(2, 3, c=4)` → `trace` 记录后调 `func(2, 3, c=4)`。wrapper 看到完整参数 `(2, 3)` 和 `c=4`，绑参没丢。

**注意**：装饰后的 `function` 是普通函数（非 partial），重新 register 时再包一层 partial。如果装饰器用了闭包（如 `def wrapper(...)` 捕获外层变量），在 `multiprocessing` 下可能不可 pickle（Python pickle 不支持闭包）。deap docstring 明确警告。需并行时改用手动 `@decorator` 装饰原函数后再 register。

---

## 四、性能分析

- **partial 调用开销**：partial 的 `__call__` 是 C 层（`functools._partial`），比 lambda 的字节码调用略快（约 10-20ns 级差异，微优化）。真正价值在可解包和可 pickle，不在这点速度。
- **`setattr` 属性查找**：`toolbox.mate` 走实例 `__dict__` 查找，O(1) 哈希表。每代调用几千次，开销可忽略。
- **并行收益**：评估通常是最贵的黑盒（毫秒~秒级）。`pool.map` 8 进程并行，评估阶段近 8× 加速。算法其余部分（选择/交叉/变异）是纯 Python 快操作，串行即可。DEAP 的设计让你只并行最贵的部分，且改动只在一行 register。

---

## 五、Toolbox 在整个 DEAP 架构中的位置

```
    ┌─────────────┐     register        ┌──────────┐
    │ 纯函数算子   │ ──────────────────→ │ Toolbox  │
    │ (tools/*)   │   (partial 冻参)    │ (粘合剂)  │
    └─────────────┘                     └────┬─────┘
                                              │ mate/mutate/select/evaluate
                                              ↓
    ┌─────────────┐     create          ┌──────────┐
    │ creator     │ ──────────────────→ │ 个体类    │
    │ (动态建类)  │                     │ (list+fit)│
    └─────────────┘                     └────┬─────┘
                                              │ 种群 = [个体]
                                              ↓
                                       ┌──────────────┐
                                       │ algorithms   │  ← 只依赖 toolbox 别名 + 个体
                                       │ (eaSimple等) │
                                       └──────────────┘
```

Toolbox 是算子和算法之间**唯一的耦合点**。算法不 import 任何具体算子，算子不 import 算法。换算法不用动算子，换算子不用动算法。这是 DEAP 可组合性的根基。

---

## 六、和 deap 原版对照

| 本教学版 | deap `base.py` | 差异 |
|---|---|---|
| `register` 全部逻辑 | 同 | 完全保留 |
| `__dict__` 拷贝 + `isinstance(function, type)` 排除 | 同 | 完全保留 |
| `decorate` 解包重套 | 同 | 完全保留 |
| `unregister` | 同 | 完全保留 |

**本阶段无实质简化** —— Toolbox 本身约 90 行，核心紧凑，全保留。唯一差异是 docstring 教学化重写。

---

## 七、常见陷阱

1. **`register("map", pool.map)` 后忘记关 pool**：并行评估完 `pool.close()`/`pool.join()`，否则进程残留。通常在算法外管理 pool 生命周期。
2. **decorate 后不可 pickle**：装饰器闭包不能 pickle，`multiprocessing` 下会炸。需并行就手动 `@decorator` 装饰原函数后再 register。
3. **别名和算子名混淆**：`toolbox.mate` 是别名，`cxTwoPoint` 是原函数。日志里 `toolbox.mate.__name__ == "mate"`，要追溯原函数用 `toolbox.mate.func`。
4. **`register` 覆盖了默认 `clone`/`map` 却忘了**：`toolbox.register("map", ...)` 会静默覆盖默认 map，调试时可能困惑。deap 不警告，靠自觉。
5. **partial 的冻参在调用时能被覆盖**：`register("f", func, c=4)` 后 `toolbox.f(c=10)` 会用 10 不是 4。这是 partial 的特性（kw 覆盖），有时是惊喜有时是坑。
6. **`pool.map` 的 chunksize**：默认按任务数均分，对极不均匀的评估（有的几秒有的几毫秒）会负载不均。传 `chunksize=1` 让任务逐个分发。
7. **子进程要能 import 到 evaluate 函数**：`pool.map(evaluate, ...)` 把 evaluate pickle 到子进程，子进程要能 `import` 到 evaluate 的定义。lambda/局部函数不可 pickle，evaluate 必须是模块级函数。

---

## 八、Python 语言特性备忘

- **`functools.partial(func, *args, **kw)`**：返回一个 `partial` 对象，调用它等价于 `func(*args, *新args, **kw, **新kw)`。属性 `.func`/`.args`/`.keywords` 可反向解包。partial 对象可 pickle（只要 func 和参数可 pickle），所以 `register` 的产物能跨进程。
- **`setattr(obj, name, value)`**：动态设属性。等价于 `obj.__dict__[name] = value`（对普通对象）。Toolbox 用它把 partial 挂成可调用属性。
- **函数的 `__dict__`**：函数对象有个 `__dict__`，可以挂任意自定义属性（`func.counter = 0`）。这是 Python 函数作为"一等对象"的体现。`register` 把它拷给别名，让别名也带这些属性。
- **`isinstance(function, type)`**：`type` 是所有类的元类。`isinstance(int, type)` True（int 是类）。这行判断"传进来的是不是类本身"。
- **依赖倒置原则**：高层模块（算法）不应依赖低层模块（具体算子），二者都应依赖抽象（toolbox 别名）。Toolbox 是这个抽象的具体载体。

---

## 九、关键收获

1. **依赖倒置用中间层**：算法依赖别名，具体算子由调用方注入 → 算法骨架与算子解耦，一套骨架跑多种算法。
2. **`partial` 是"配置即代码"的利器**：把参数冻进函数，造配置好的半成品，调用处更干净；且 partial 可解包，支持后续装饰；可 pickle，支持并行。
3. **把并行做成可替换钩子（`map`）**：并行是配置问题不是算法问题，换 map 即换并行度，算法无感。
4. **装饰器解包重套保留绑参**：`pfunc.func/.args/.keywords` 三件套是 partial 可逆的关键，让 decorate 不丢配置。
5. **默认钩子也走 register**：`clone`/`map` 无特殊待遇，统一可覆盖，接口一致。
6. **Toolbox 是架构唯一的耦合点**：算子和算法互不 import，全靠 toolbox 连接，可组合性的根基。

---

## 十、思考题

1. 如果要把"统计每个算子被调用次数"加进 Toolbox，最少改动怎么改？（提示：decorate 一个计数装饰器，register 后自动套）
2. `register("map", pool.map)` 后，`toolbox.map(evaluate, invalid_ind)` 里 `evaluate` 本身是 partial，`pool.map` 把它序列化到子进程 —— partial 能 pickle 吗？为什么？（提示：partial 实现了 `__reduce__`）
3. 为什么 `decorate` 要解包 partial 套在原函数上，而不是直接套在 partial 外？（提示：套在 partial 外，装饰器看到的是"调用时的剩余参数"，看不到冻参，且重新 register 会双重冻结）
4. 如果算法层直接 `import` 算子而非走 toolbox，会丢失哪些能力？（换算子/换参/换并行/装饰全受阻）
5. `pool.map` vs `pool.imap`：前者一次性返回列表（占内存），后者惰性迭代。对极大种群哪个更合适？DEAP 默认用 `map`（同步），怎么换成 `imap`？

---

## 十一、下一阶段预告

**阶段 3：Creator** —— `create(name, base, **kargs)` 动态建类。一行 `create("Individual", list, fitness=FitnessMax)` 等价于"定义一个带 fitness 属性的 list 子类"。关键技巧：`**kargs` 里值是类型（`fitness=FitnessMax`）→ 实例化时调 `FitnessMax()` 存为实例属性；值是普通对象（`spam=1`）→ 存为类属性。靠 `MetaCreator` 元类改写 `__init__` 实现。这是 DEAP 适配任意个体表示（list/set/array/numpy/GP 树）的根。
