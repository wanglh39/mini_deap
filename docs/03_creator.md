# 阶段 3：Creator —— 动态建类（元编程工厂）

> 对应 DEAP：`deap/creator.py`（原版 193 行，本阶段同量级，核心全保留）
> 产出：`mini_deap/base/creator.py` + `tests/base/test_creator.py`（15 测试全过）

---

## 一、背景：为什么不直接 `class Individual(list): ...`，非要动态建类？

进化算法的个体可以是任意容器：`list`（实数/二进制 GA）、`set`（集合 GA）、`array.array`（紧凑数值）、`numpy.ndarray`（向量化）、`PrimitiveTree`（遗传编程的语法树）。每种都要挂一个 `fitness` 属性，且 fitness 类型也可能变（`FitnessMax`/`FitnessMin`/`FitnessMulti`）。

**静态写法的组合爆炸**：容器类型 × fitness 类型 = N × M 个子类要手写：

```python
class IndividualListMax(list):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.fitness = FitnessMax()

class IndividualListMin(list):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.fitness = FitnessMin()

class IndividualSetMax(set):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.fitness = FitnessMax()

class IndividualArrayMax(array.array):
    def __new__(cls, seq=()):                          # array 要 __new__
        return super().__new__(cls, 'f', seq)
    def __deepcopy__(self, memo):                      # array 的 deepcopy 坑
        ...
# ... 还有 numpy、GP 树、Multi 各种组合
```

每换一个组合就手写一个子类，重复且易错（尤其 `array.array`/`numpy.ndarray` 的 `__new__`/`__deepcopy__` 坑每个都得重处理）。

**DEAP 的解法**：用一个工厂函数 `create`，一行声明就造好类：

```python
creator.create("FitnessMax", Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=FitnessMax)      # list + fitness
creator.create("Individual", set, fitness=FitnessMax)       # 换成 set，一行
creator.create("Individual", array.array, typecode="f", fitness=FitnessMax)
creator.create("Individual", numpy.ndarray, fitness=FitnessMax)
```

这是**元编程**：用代码生成代码（类）。`create` 是"类的工厂"，配置（基类 + 属性）当参数传进去，产出定制类。本质是把"写一个类"这件事数据化、参数化，消除组合爆炸。

---

## 二、核心设计思想

### ① `isinstance(obj, type)` 区分实例属性 vs 类属性（最关键的技巧）

`create(name, base, **kargs)` 的 `kargs` 里，值有两种：

| 值的类型 | 例子 | 处理 | 结果 |
|---|---|---|---|
| 是类型（`type` 的实例） | `fitness=FitnessMax` | 实例化时调 `FitnessMax()` 存到 `self.fitness` | **实例属性**，每实例独立 |
| 不是类型 | `weights=(1.0,)`、`size=42` | 直接挂类上 | **类属性**，所有实例共享 |

为什么这么分？因为**类型需要"每实例一份新对象"，普通值可以共享**：
- `fitness=FitnessMax`：每个个体要有自己的 fitness 实例（值各不同），所以实例化时调 `FitnessMax()` 造新的。
- `weights=(1.0,)`：所有同类 fitness 共享同一组 weights（不可变，共享安全），挂类属性即可。
- `size=42`：常量配置，共享。

判断依据就是 `isinstance(obj, type)`：`type` 是所有类的元类，`isinstance(FitnessMax, type)` 为 True（FitnessMax 是类），`isinstance((1.0,), type)` 为 False（元组不是类）。

### ② 元类 `MetaCreator` 改写 `__init__` 注入分发逻辑

**两种动态建类方式对比**：

```python
# 方式A：type() 三参数（不用元类）
Individual = type("Individual", (list,), {"fitness": FitnessMax})  # fitness 变类属性，错！

# 方式B：元类（DEAP 用的）
class MetaCreator(type):
    def __init__(cls, name, base, dct):
        # 拦截，改写 __init__，把类型值挪到实例化时调
        ...
Individual = MetaCreator("Individual", list, {"fitness": FitnessMax})
```

方式 A 的问题：`type()` 三参数把 `dct` 全挂成类属性，`fitness=FitnessMax` 会变成类属性（所有实例共享同一个 FitnessMax **类**，不是实例）。且没法在实例化时调 `FitnessMax()` 造实例。元类能拦截 `__init__` 注入定制逻辑，方式 A 不能。这是 DEAP 必须用元类而非 `type()` 的根因。

元类改写后的 `init_type`：

```python
def init_type(self, *args, **kargs):
    for obj_name, obj in dict_inst.items():      # dict_inst = {fitness: FitnessMax}
        setattr(self, obj_name, obj())           # self.fitness = FitnessMax()
    if base.__init__ is not object.__init__:
        base.__init__(self, *args, **kargs)       # 调 list.__init__(self, *args)

cls.__init__ = init_type
```

之后 `Individual([1,2,3])` 就走这个 `init_type`：先造 fitness 实例，再调 `list.__init__` 初始化容器内容。

### ③ `class_replacers` 修正 `array.array`/`numpy.ndarray` 的 deepcopy 坑

`array.array` 和 `numpy.ndarray` 的 `__deepcopy__` 是 C 层实现，**只拷数据不拷 `__dict__`**（即不拷 Python 层挂的属性如 `fitness`）。直接继承会导致 `deepcopy(ind).fitness` 抛 `AttributeError`。

**为什么 C 层不拷 `__dict__`？** CPython 的 `array.array`/`numpy.ndarray` 在 C 层定义，`__deepcopy__` 走 C 的 `__reduce_ex__`/`__copy__` 路径，该路径只处理 C 结构里的数据缓冲区，不知道 Python 层的 `__dict__`（子类才加的）。这是"内置类型子类化"的经典陷阱。

DEAP 的解法：定义替换子类 `_array`/`_numpy_array` 重写 `__deepcopy__` 拷贝 `__dict__`，注册进 `class_replacers`。`create` 时若 `base` 在表里，换成替换类再继承：

```python
if base in class_replacers:
    base = class_replacers[base]    # array.array → _array
```

### ④ `__reduce__` + `copyreg.pickle` 让动态类可序列化

`multiprocessing` 并行评估要 pickle 个体，pickle 个体要先 pickle 它的类。但动态创建的类不在任何模块的源码里，pickle 默认按"模块名.类名"查找，找不到 → 抛错。

解法：给 `MetaCreator` 定义 `__reduce__` 返回 `(meta_create, (name, base, dct))`，告诉 pickle"重建这个类就调 `meta_create(name, base, dct)`"。`copyreg.pickle(MetaCreator, ...)` 注册让 pickle 知道对 MetaCreator 创建的类用这个 reduce。于是 pickle 时序列化的是"重建指令"，loads 时重新 `meta_create` 造出类并注册回全局。

---

## 三、逐段精读 `mini_deap/base/creator.py`

### 3.1 `class_replacers` 与 `_array` / `_numpy_array`

```python
class_replacers = {}   # base 类 → 替换类
```

`_array(array.array)` 关键方法：

```python
@staticmethod
def __new__(cls, seq=()):
    return super(_array, cls).__new__(cls, cls.typecode, seq)
```

`array.array` 需要 `typecode`（如 `"f"` 单精度浮点）。`cls.typecode` 从子类的类属性取 —— 所以 `create("Ind", array.array, typecode="f", ...)` 时 `typecode="f"` 是字符串（非类型）→ 进 `dict_cls` → 类属性。`__new__` 时 `cls.typecode` 拿到 `"f"`。

```python
def __deepcopy__(self, memo):
    cls = self.__class__
    copy_ = cls.__new__(cls, self)              # 用 __new__ 造同类型空数组
    memo[id(self)] = copy_                       # 登记，防循环引用
    copy_.__dict__.update(copy.deepcopy(self.__dict__, memo))  # 拷 Python 层属性
    return copy_
```

`memo[id(self)] = copy_` 必须在拷贝 `__dict__` **之前**登记，否则 `__dict__` 里若有指回 self 的引用会无限递归。`memo` 是 `{id(源): 副本}` 字典，deepcopy 协议用它打破循环。

`_numpy_array` 同理，额外用 `numpy.array(list(iterable)).view(cls)` 把数组 view 成子类（保留 numpy 的向量化操作 + 子类属性）。

### 3.2 `MetaCreator` 元类

```python
class MetaCreator(type):
    def __new__(cls, name, base, dct):
        return super(MetaCreator, cls).__new__(cls, name, (base,), dct)

    def __init__(cls, name, base, dct):
        ...   # 分发 + 改写 __init__
```

**`__new__` vs `__init__`（元类版）**：
- `__new__` 造出类对象本身（分配内存、建类）。这里调 `type.__new__` 造一个继承 `(base,)` 的类。
- `__init__` 配置已造好的类（设属性、改写方法）。这里分发 `dct`、改写 `cls.__init__`。

注意元类的 `cls` 参数是**正在创建的类**（不是类的实例），`name/base/dct` 是 `MetaCreator(name, base, dct)` 调用时的参数。

**`init_type` 闭包变量捕获**：`init_type` 定义在 `MetaCreator.__init__` 内部，捕获外层的 `dict_inst` 和 `base`。每次 `MetaCreator.__init__` 调用（即每次 create）产生新的 `dict_inst`/`base`，对应一个新的 `init_type` 闭包。赋给 `cls.__init__` 后，该类的实例化就走这个定制 init。不同动态类的 `__init__` 互不干扰（各自闭包）。

**`base.__init__ is not object.__init__` 的所有情况**：
- `base=list`：`list.__init__` 自定义过 → True → 调 `list.__init__(self, *args)` 初始化列表内容。
- `base=object`：`object.__init__` 是默认 → False → 不调（避免传参炸，`object.__init__` 不接受额外参数）。
- `base=用户类且没写 __init__`：继承 `object.__init__` → `is` 判断为 True（同一函数对象）→ 不调。
- `base=用户类且写了 __init__`：→ True → 调。

### 3.3 `meta_create` + `globals()` 注册

```python
def meta_create(name, base, dct):
    class_ = MetaCreator(name, base, dct)
    globals()[name] = class_      # 注册到本模块全局
    return class_
```

`globals()[name] = class_` 让动态类能通过 `creator.Individual` 访问（因为模块全局 = 模块属性）。也让 `pickle.loads` 重建类时，`meta_create` 把类重新注册回全局，后续引用能找到。

### 3.4 `create` 入口

```python
def create(name, base, **kargs):
    if name in globals():
        warnings.warn(...)              # 重复创建警告
    if base in class_replacers:
        base = class_replacers[base]    # array/numpy 替换
    meta_create(name, base, kargs)
```

重复创建同名类会**覆盖**（不是报错），deap 选择警告而非禁止，因为交互式使用中重定义常见。警告让你意识到旧类被覆盖了（已存在的实例仍用旧类，新实例用新类，可能混淆）。

**完整调用追踪**：`create("Individual", list, fitness=FitnessMax)`：
```
create("Individual", list, fitness=FitnessMax)
  → base=list 不在 class_replacers，不替换
  → meta_create("Individual", list, {"fitness": FitnessMax})
    → MetaCreator("Individual", list, {"fitness": FitnessMax})
      → __new__: type.__new__(MetaCreator, "Individual", (list,), dct) 造类对象
      → __init__: 分发 dct
        → FitnessMax 是类型 → dict_inst["fitness"] = FitnessMax
        → 定义 init_type 捕获 dict_inst={fitness: FitnessMax}, base=list
        → cls.__init__ = init_type
        → cls.reduce_args = ("Individual", list, {"fitness": FitnessMax})
    → globals()["Individual"] = cls
```
之后 `Individual([1,2,3])`：
```
Individual([1,2,3]) → init_type(self, [1,2,3])
  → setattr(self, "fitness", FitnessMax())   # self.fitness = FitnessMax()
  → list.__init__(self, [1,2,3])              # self = [1,2,3]
```

---

## 四、和 deap 原版对照

| 本教学版 | deap `creator.py` | 差异 |
|---|---|---|
| `MetaCreator` 全部逻辑 | 同 | 完全保留 |
| `_array` / `_numpy_array` | 同 | 完全保留 |
| `__reduce__` + `copyreg.pickle` | 同 | 完全保留 |
| 重复警告消息 | 略短 | 语义一致 |

**本阶段无实质简化**，creator 核心紧凑且全部是机制性代码，全保留。

---

## 五、常见陷阱

1. **`create("Ind", array.array, fitness=...)` 忘传 `typecode`**：`_array.__new__` 访问 `cls.typecode` 抛 `AttributeError`。必须 `typecode="f"`/`"i"` 等。
2. **`create("Ind", numpy.ndarray, fitness=...)` 后用 `Ind([1,2,3])`**：能跑，但 numpy 数组操作返回的是裸 `numpy.ndarray` 不是 `Ind`（除非用 `view`）。算子里注意。
3. **重复 `create` 同名类**：旧实例仍用旧类（类对象本身没变，只是全局名指向新类），`isinstance` 判断可能失效。生产代码避免重名。
4. **动态类在 multiprocessing 里**：依赖 `__reduce__` + `copyreg.pickle`。如果子进程没 import creator 模块，pickle 重建时找不到 `meta_create` → 抛错。确保子进程导入了 `mini_deap.base.creator`。
5. **`fitness=FitnessMax` vs `fitness=FitnessMax()`**：前者（传类型）每实例造新 fitness；后者（传实例）所有实例共享同一个 fitness 实例 → 改一个全变，几乎肯定不是你想要的。**永远传类型，不传实例。**
6. **`weights` 必须是 tuple 不是 list**：`create("Fit", Fitness, weights=[1.0])` 能跑（list 是 Sequence），但可变，被意外改了全类实例都受影响。约定 tuple。
7. **元类冲突**：如果 base 自己用了别的元类（如 ABCMeta），`MetaCreator` 和它可能冲突（Python 3 要求元类是基类元类的子类）。DEAP 的 Fitness 没用 ABCMeta 正是为此避坑（见阶段 1）。

---

## 六、Python 语言特性备忘

- **元类**：`class Foo(metaclass=Meta)` 时，`Foo` 由 `Meta("Foo", bases, dict)` 创建。元类拦截类的创建过程，可改写类的方法、注入属性。`type` 是默认元类。
- **`type.__new__(cls, name, bases, dct)`**：造一个新类。`name` 是类名（字符串），`bases` 是基类元组，`dct` 是类命名空间字典。
- **`isinstance(obj, type)`**：判断 `obj` 是不是"一个类"。`isinstance(int, type)` True（int 是类），`isinstance(42, type)` False（42 是 int 的实例，不是类）。
- **`__new__` vs `__init__`**：`__new__` 造实例（分配内存），`__init__` 初始化实例。对不可变类型（tuple/str/array）必须用 `__new__` 因为 `__init__` 时数据已定。元类的 `__new__`/`__init__` 同理但作用于"类对象"。
- **`copyreg.pickle(cls, reduce_func)`**：注册 `cls` 类型的对象 pickle 时用 `reduce_func`。`reduce_func(obj)` 返回 `(callable, args)`，pickle 时存这个对，loads 时调 `callable(*args)` 重建。
- **`globals()`**：返回当前模块的全局命名空间字典。`globals()[name] = cls` 等价于在模块顶层写 `name = cls`，但动态。
- **闭包捕获**：内层函数引用外层变量时，捕获的是**变量绑定的值**（Python 3 闭包捕获变量本身，late binding）。`init_type` 捕获 `dict_inst`/`base`，每次 `MetaCreator.__init__` 调用产生新闭包。

---

## 七、关键收获

1. **元编程把"写类"数据化**：配置（基类 + 属性）当参数，工厂产出定制类，消除组合爆炸。
2. **用 `isinstance(obj, type)` 区分"要实例化的类型"和"直接存的值"** —— 一个判断统一了实例属性/类属性的分发，是 `create` 的灵魂。
3. **元类改写 `__init__` 注入定制初始化**：`init_type` 闭包捕获分发配置，每个动态类有独立 init。`type()` 三参数做不到（无法注入实例化逻辑）。
4. **C 层类型的 deepcopy 坑用替换子类修**：`array`/`numpy` 不拷 `__dict__`，定义子类重写 `__deepcopy__`，用 `class_replacers` 在 create 时透明替换。
5. **动态类要可 pickle 必须实现 `__reduce__`**：返回重建指令 `(meta_create, args)`，配合 `copyreg.pickle` 注册，让并行序列化能工作。
6. **传类型不传实例**（`fitness=FitnessMax` 不是 `fitness=FitnessMax()`）—— 前者每实例独立，后者共享，99% 场景要前者。

---

## 八、思考题

1. 如果想给 `create` 加一个"实例属性也支持传实例时 deepcopy 一份"的模式（即 `fitness=some_instance` 时每实例 deepcopy 一份），怎么改 `init_type`？
2. `base.__init__ is not object.__init__` 这个判断在 `base` 是 `list` 时为 True（list 重写了 __init__），在 `base` 是 `object` 时为 False。如果 base 是用户自定义类且没写 `__init__`，这个判断结果是什么？会出错吗？
3. 为什么 `memo[id(self)] = copy_` 必须在 deepcopy `__dict__` 之前登记？（提示：自引用）
4. `create` 把类注册到 `creator` 模块的 `globals()`。如果两个独立项目都用 `create("Individual", ...)`，会冲突吗？怎么隔离？（提示：deap 没隔离，靠自觉起唯一名）
5. `type()` 三参数造类 vs 元类造类：前者 `type("Foo", (list,), dct)` 把 dct 全挂类属性，后者能改写 `__init__` 注入实例化逻辑。如果只想加类属性（不要实例属性），用 `type()` 三参数够吗？DEAP 为什么统一用元类而非混用？

---

## 九、实战示例与性能

### 9.1 create 的各种用法

```python
import array, numpy
from mini_deap.base.fitness import Fitness
from mini_deap.base.creator import create

# 适应度类型
create("FitnessMax", Fitness, weights=(1.0,))           # 单目标最大化
create("FitnessMin", Fitness, weights=(-1.0,))          # 单目标最小化
create("FitnessMulti", Fitness, weights=(1.0, -1.0))    # 双目标：max f1, min f2

# 个体类型（同一套算法可跑任意表示）
create("IndList", list, fitness=FitnessMax)             # 实数/二进制 GA
create("IndSet", set, fitness=FitnessMax)               # 集合 GA
create("IndArr", array.array, typecode="f", fitness=FitnessMax)  # 紧凑浮点
create("IndNp", numpy.ndarray, fitness=FitnessMax)      # 向量化

# 用法
ind = IndList([1.0, 2.0, 3.0]); ind.fitness.values = (6.0,)
pop = [IndList([random.random() for _ in range(5)]) for _ in range(100)]
```

### 9.2 性能：元类创建开销 vs 手写类

- **元类创建类**：`create(...)` 调一次，开销约几微秒（元类 `__new__`+`__init__`+`globals` 注册）。**一次性成本**，程序启动时调几次，之后不再调。
- **实例化**：`IndList([1,2,3])` 走 `init_type`：一个 `setattr` + 一次 `list.__init__`。比手写类的 `__init__` 多一个 `setattr` 循环（实例属性数 × setattr），但实例属性通常就 1-2 个（fitness），开销纳秒级，可忽略。
- **真正贵的不是 create，是评估**：元类开销相对评估黑盒（毫秒~秒）是 0。create 的价值在消除手写子类的重复，不在运行时性能。

### 9.3 调试技巧

- `creator.Individual` 查看动态类（`globals()` 注册后可直接访问）。
- `Individual.__init__` 是 `init_type` 闭包，`Individual.__init__.__code__.co_freevars` 能看到捕获的变量名（`dict_inst`/`base`）。
- `Individual.reduce_args` 存了 `(name, base, dct)`，pickle 重建时用它。

---

## 十、下一阶段预告

**阶段 4：tools/operators.py** —— 纯函数算子库。`initRepeat`（初始化种群）、`selTournament`/`selBest`（选择）、`cxOnePoint`/`cxTwoPoint`（交叉）、`mutGaussian`/`mutFlipBit`（变异）。全部是**纯函数**（接 individuals 返回 individuals，无副作用），通过 `toolbox.register` 组装。关键算子给纯 Python + numpy 向量化两版对照。这是 DEAP "算子是纯函数" 思想的集中体现。
