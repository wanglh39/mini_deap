# 阶段 1：Fitness —— 适应度抽象基类（深入版）

> 对应 DEAP：`deap/base.py` 的 `Fitness` 类（原版 358 行含 `ConstrainedFitness`，本阶段 ~110 行）
> 产出：`mini_deap/base/fitness.py` + `tests/base/test_fitness.py`（20 测试全过）

---

## 一、背景：适应度在进化算法中到底扮演什么角色

先把视野拉到整个进化算法（EA）的通用骨架，看清 fitness 处在哪个位置：

```
    初始化种群 P
    评估 P 中每个个体的 fitness          ← fitness 在这里被赋值
    while 未收敛:
        P_parent = 选择(P)               ← 选择需要"比较"fitness
        P_child  = 交叉变异(P_parent)    ← 变异后 fitness 失效
        评估 P_child 中失效的个体          ← 只重算失效的，fitness 在这里被重新赋值
        P = 替换(P_parent, P_child)
```

fitness 有**三个职责**，这决定了它的 API 形状：

1. **衡量质量**：存目标函数值（单目标是一个数，多目标是多个数）。
2. **驱动选择**：选择算子要能对两个个体比大小（`a > b`）。
3. **标记是否已评估**：变异后原值作废，不能拿旧 fitness 去做选择，需要一个"有效/失效"标志。

一个 EA 库的 fitness 设计，本质是在回答三个问题：
- **怎么统一处理最大化/最小化/多目标？**（方向问题）
- **比较是热路径，怎么让比较最快？**（性能问题）
- **怎么标记失效且不重复评估？**（惰性问题）

DEAP 的 Fitness 类把这三个问题一次性解决，而且解法很优雅。下面逐段精读。

---

## 二、DEAP 的设计哲学：用数据编码方向，而不是用逻辑分支

先看一个**反面写法**（很多新手会这么写）：

```python
class FitnessBad:
    def __init__(self, value, maximize=True):
        self.value = value
        self.maximize = maximize
    def __gt__(self, other):
        if self.maximize:
            return self.value > other.value
        else:
            return self.value < other.value   # ← 每次比较都分支
```

问题：
- 每次比较都走 `if maximize` 分支，选择算子每代 O(N log N) 次比较，全是冗余分支。
- 多目标时分支爆炸：`if max1 and max2 ... elif max1 and not max2 ...`。
- `maximize` 是实例属性，每个个体都存一份，浪费内存。

**DEAP 的解法**：把方向编码进 `weights` 的正负，设值时一次性乘进去存成 `wvalues`，之后比较**永远只比 wvalues 的字典序，永远"越大越好"**，零分支：

```python
# 最小化问题，weights = (-1.0,)
# 原值 1 和 5：
#   wvalues = 1 * -1 = -1
#   wvalues = 5 * -1 = -5
# 比较：-1 > -5  →  原值 1 的 fitness "更大"  →  选择取 max 自然选中更小原值
```

这是一个普遍适用的设计思想：**能把条件编码进数据，就别写进控制流。** 类似的手法：用 `1`/`-1` 当方向系数、用补码统一加减法、用齐次坐标统一平移和线性变换。

---

## 三、逐段精读 `mini_deap/base/fitness.py`

### 3.1 类属性 `weights` 与 `wvalues` —— 抽象基类 + 双重含义空元组

```python
class Fitness:
    weights = None      # 类属性：子类必须覆盖。None = 抽象基类
    wvalues = ()        # 实例属性：加权值。空元组 = 未评估/已失效
```

**为什么 `weights` 是类属性而不是实例属性？**
- 同一种适应度（如 `FitnessMax`）的所有实例共享同一组 weights，存一份就够。
- 类属性在 `__init__` 之前就存在，`isinstance(self.weights, Sequence)` 检查可以在实例化时立刻拦截错误配置。
- 类属性天然支持"子类定义即配置"的声明式风格：`class FitnessMax(Fitness): weights = (1.0,)` 一行就声明了一个具体适应度类型。

**为什么不用 `abc.ABCMeta` + `abstractmethod`？**
- DEAP 选择更轻量的方式：`weights = None` 当哨兵，`__init__` 里检查。这避免了 ABCMeta 的元类冲突（`creator.py` 已经用了自己的元类 `MetaCreator`，再叠 ABCMeta 容易打架）。
- 教学上更直白：读 `weights = None` 就知道"这是抽象的，子类要填"。

**`wvalues = ()` 的双重含义**：同一个空元组既表示"从未评估过"，也表示"评估过但被 `del` 失效了"。`valid = len(wvalues) != 0` 不区分这两种情况——因为对算法层来说，**只要无效就要重算，原因不重要**。这是一个"合并状态"的简化：与其用两个标志（`evaluated` + `stale`），不如用一个空/非空。

### 3.2 `__init__` —— 抽象基类保护与类型校验

```python
def __init__(self, values=()):
    if self.weights is None:
        raise TypeError("Can't instantiate abstract %r ..." % self.__class__)
    if not isinstance(self.weights, Sequence):
        raise TypeError("Attribute weights of %r must be a sequence." % self.__class__)
    if len(values) > 0:
        self.values = values
```

逐行：
1. **哨兵检查**：`weights is None` 说明子类没定义 weights，直接 `Fitness()` 或 `class Bad(Fitness): pass` 都会在这里炸。错误消息用 `%r` 带上类名，便于定位是哪个子类配错了。
2. **类型校验**：weights 必须是序列（tuple/list），不能是 `int`、`None` 之外的单值。防止 `weights = 1.0` 这种把单目标写成标量的错误。
3. **可选初值**：`values=()` 默认空，表示"构造时还没评估"。传了值就走 `self.values = values` 触发 property 的 setter 完成加权。

**为什么 `len(values) > 0` 而不是 `values`？** 因为 `values` 可能是合法的空元组 `()`（表示未评估），`if values` 对空元组为 False 行为一致，但 `len() > 0` 更明确表达"有值才设"的意图。deap 原版也是这么写。

### 3.3 `values` property 三件套 —— 对外原值，对内加权值

这是整个类最精巧的部分。对外看 `fitness.values` 是原值，对内 `wvalues` 才是真正存的。用 property 做读写转换：

```python
def getValues(self):
    return tuple(map(truediv, self.wvalues, self.weights))   # 还原 = wvalues / weights

def setValues(self, values):
    assert len(values) == len(self.weights), "..."
    self.wvalues = tuple(map(mul, values, self.weights))     # 加权 = values * weights

def delValues(self):
    self.wvalues = ()                                        # 失效

values = property(getValues, setValues, delValues, doc="...")
```

**为什么用 `map(mul, ...)` 而不是列表推导 `[v*w for v,w in zip(...)]`？**
- `map` + `operator.mul` 是 C 层实现，比 Python 字节码循环快一点（微优化，但 deap 处处这么干）。
- `tuple(map(...))` 直接产元组，匹配 `wvalues` 的不可变语义。

**为什么用 `operator.truediv` 而不是 `/`？** `truediv` 是 `operator` 模块的函数，能塞进 `map`；`/` 是语法不是函数。`map(truediv, a, b)` 等价于 `[x/y for x,y in zip(a,b)]`。

**性能账（惰性加权的本质）**：
- 设：1 次乘法（N 维目标 N 次乘法）。
- 比较：0 次乘法，纯元组字典序比较（C 层）。
- 一个个体生命周期内：设值 1~几次（每次变异后重设），比较 几十~几百次（每代选择）。
- **乘法挪到 set，把"每次比较都乘"降成"总共乘几次"。** 对种群 100、代数 100、每次选择比较 ~500 次，省掉约 100×100×500 = 5,000,000 次乘法（单目标）。

**为什么用 `property()` 内置而不是 `@property` 装饰器？** 因为要同时定义 get/set/del 三个。`@property` 装饰器语法只能顺滑定义 get，set/del 要再写 `@values.setter`、`@values.deleter`，对一个三件套反而更绕。`property(get, set, del)` 一行全包，deap 原版也这么写。

**`assert` vs `raise`**：教学版用 `assert` 检查长度，简洁。deap 原版用 `try/except TypeError` + 重建带上下文的错误消息（更友好的生产级报错）。`assert` 在 `python -O` 下会被去掉，生产代码应改 `raise ValueError`，教学场景可接受。

### 3.4 `valid` —— 用空元组当标志

```python
@property
def valid(self):
    return len(self.wvalues) != 0
```

不另设 `self._valid = False` 布尔标志，而是**从 `wvalues` 是否为空派生**。好处：
- 状态来源单一：失效就是 `wvalues = ()`，不会有"标志忘了同步"的 bug。
- 省一个属性槽。

代价：`len()` 调用有微小开销，但 `len(())` 是 O(1) 且 C 层，可忽略。

### 3.5 `dominates` —— Pareto 支配的严格定义与实现

**数学定义**：解 a 支配解 b（记 a ≻ b）⟺
- ∀i: a_i ≥ b_i（每维不劣）
- ∃j: a_j > b_j（至少一维严格更优）

```python
def dominates(self, other, obj=slice(None)):
    not_equal = False
    for self_wv, other_wv in zip(self.wvalues[obj], other.wvalues[obj]):
        if self_wv > other_wv:
            not_equal = True       # 记下"有严格更优的维度"
        elif self_wv < other_wv:
            return False            # 有严格更劣的维度 → 直接判负
    return not_equal
```

逐行：
- `obj=slice(None)`：默认比较所有目标。传 `obj=slice(0,2)` 可只在前两维判支配（部分支配），NSGA3 会用到。
- `not_equal` 标志：记录是否出现过严格更优。只有"全不劣"且"至少一维严格更优"才支配。
- 遇到严格更劣**立即返回 False**：短路，平均情况下不用比完所有维度。
- 比较 `wvalues` 而非 `values`：最小化目标的 wvalues 已经反号，所以这里不用关心方向，**和比较运算符同一套逻辑**。

**为什么不用 `all(...) and any(...)` 一行写？** 
```python
return all(s >= o for s, o in zip(...)) and any(s > o for s, o in zip(...))
```
这要遍历两遍，且不能短路。显式循环一遍 + 短路，性能更好。deap 原版也是显式循环。

### 3.6 比较运算符 —— 只实现两个，其余反推

```python
def __le__(self, other): return self.wvalues <= other.wvalues
def __lt__(self, other): return self.wvalues < other.wvalues
def __gt__(self, other): return not self.__le__(other)    # 反推
def __ge__(self, other): return not self.__lt__(other)    # 反推
def __eq__(self, other): return self.wvalues == other.wvalues
def __ne__(self, other): return not self.__eq__(other)
```

**为什么 `__gt__ = not __le__` 而不是 `wvalues > other.wvalues`？**
- 逻辑等价：`a > b ⟺ ¬(a <= b)`。
- 好处：比较的"真值来源"只有 `__lt__` 和 `__le__` 两个，改一处全改。如果六个运算符各写一遍 `self.wvalues <op> other.wvalues`，将来要换比较逻辑（如加约束处理）得改六处。
- deap 原版完全同款。

**Python 比较协议补充**：
- Python 3 没有 `__cmp__`，必须分别定义富比较运算符。
- 定义了 `__eq__` 就该定义 `__hash__`（否则实例变不可哈希，不能放进 set/当 dict key）。这里 `__hash__ = hash(self.wvalues)`，和 `__eq__` 一致（wvalues 相等 → hash 相等）。
- `__le__`/`__lt__` 用元组的字典序：`(1,9) > (1,5)` 因为第一维相等看第二维。**注意：字典序比较 ≠ Pareto 支配**。`(2,1)` 和 `(1,2)` 字典序 `(2,1) > (1,2)`，但 Pareto 上两者互不支配。选择算子用 `max`/`sorted` 走的是字典序，多目标要真正按 Pareto 得用 `selNSGA2`（阶段 8）。

### 3.7 `__deepcopy__` —— 为什么不能直接用通用 deepcopy

```python
def __deepcopy__(self, memo):
    copy_ = self.__class__()
    copy_.wvalues = self.wvalues
    return copy_
```

**通用 `copy.deepcopy` 会做什么？** 递归遍历对象的 `__dict__` 和类属性，逐个拷贝。对 Fitness 这会：
- 试图拷贝类属性 `weights`（其实是 tuple，不可变，但通用 deepcopy 不一定知道）。
- 试图处理 property 描述符对象。
- 走 `__reduce__`/`__getstate__` 协议探测。

这些都是浪费：Fitness 真正的可变状态只有 `wvalues`（一个不可变 tuple）。

**自定义版本**：`self.__class__()` 造一个同类的空实例（`weights` 自动从类继承），再把 `wvalues` 赋过去。tuple 赋值是引用复制（不可变所以安全），零递归。

**`memo` 参数**：deepcopy 协议要求接收 `memo` 字典，用来处理循环引用（`memo[id(self)] = copy`）。这里 wvalues 是不可变 tuple 不可能循环引用，所以不用写 memo。但如果 fitness 将来持有可变对象，就得 `memo[id(self)] = copy_` 在赋值前先登记。

**进化算法里 clone 多频繁？** `varAnd` 里 `offspring = [toolbox.clone(ind) for ind in population]`，每代克隆整个种群。种群 100、代数 100 → 10000 次克隆，每次省下的递归叠加起来可观。

### 3.8 `__hash__` / `__str__` / `__repr__`

```python
def __hash__(self): return hash(self.wvalues)
def __str__(self):  return str(self.values if self.valid else tuple())
def __repr__(self): return "%s.%s(%r)" % (self.__module__, self.__class__.__name__, self.values if self.valid else tuple())
```

- `__hash__` 用 wvalues：和 `__eq__`（也比 wvalues）一致，满足 Python 约定"相等对象 hash 相等"。
- `__repr__` 带 `__module__` 和 `__class__.__name__`：为了 `eval(repr(f))` 能重建对象（配合 `creator` 动态建类时，类名注册在 `creator` 模块命名空间）。教学测试里只检查字符串包含类名和值，不做 eval（eval 依赖模块导入，脆弱）。

---

## 四、和 deap 原版逐行对照

| 本教学版 | deap `base.py` | 差异说明 |
|---|---|---|
| `from collections.abc import Sequence` | `try: from collections.abc import Sequence except ImportError: from collections import Sequence` | 砍掉 Py2/老 Py3 兼容 |
| `assert len(values) == len(self.weights)` | `assert len(values) == len(self.weights), "..."` | 保留，消息略简 |
| `self.wvalues = tuple(map(mul, values, self.weights))` | 同 + `try/except TypeError` + `sys.exc_info` + `with_traceback` 重建 | 砍掉生产级错误重建 |
| 无 `ConstrainedFitness` | 有（~80 行，含 `constraint_violation` + 重写 6 个比较符 + dominates） | 整个砍掉，留进阶 |
| `__deepcopy__` 同 | 同 | 完全保留 |
| `dominates` 同 | 同 | 完全保留 |

**保留的全部核心**：weights/wvalues/values property、valid、6 个比较符、dominates、__deepcopy__、__hash__、抽象基类保护。**核心思想零损失，代码量减 70%。**

---

## 五、简化决策详解（每项为什么能砍）

| 砍掉 | 为什么能砍 | 砍掉的风险 |
|---|---|---|
| `ConstrainedFitness` | 约束处理是正交特性，不影响核心 fitness 语义；阶段 6 算法层不依赖它 | 想跑约束优化时没有，得自己加 |
| `setValues` 的 `with_traceback` 错误重建 | 教学场景错误消息够用就行，assert 抛的 AssertionError 已能定位 | 生产里错误信息不够友好 |
| `collections.abc` 的 ImportError 兼容 | 目标 Python 3.13，`collections.abc` 一定存在 | 不能在 Py2 跑（无所谓） |

---

## 六、常见陷阱

1. **忘 `del values` 导致旧 fitness 被复用**：变异后不失效，选择会拿旧值选，等于"没变异"。算法层 `varAnd` 会自动 `del`，自己写算法要记得。
2. **weights 用 list 而非 tuple**：`weights = [1.0]` 也能跑（list 是 Sequence），但可变，被意外改了全类实例都受影响。约定用 tuple。
3. **多目标维度不匹配**：`weights = (1.0, -1.0)` 但 `values = (3.0,)`，setValues 的 assert 会炸。这是好事，早暴露。
4. **直接 `Fitness((1.0,))`**：基类 weights=None，抛 TypeError。必须先子类化或用 `creator.create`。
5. **以为 `>` 是 Pareto 支配**：`>` 是字典序，多目标要用 `dominates`。见 3.6 末尾。

---

## 七、Python 语言特性备忘（本阶段用到的）

- **property 是描述符**：`property(get, set, del)` 返回一个描述符对象，访问 `instance.values` 时触发 `__get__`→get，赋值触发 `__set__`→set，`del` 触发 `__delete__`→del。
- **类属性 vs 实例属性**：`weights` 在类上定义，所有实例共享、通过 `self.weights` 读取（沿 MRO 查找）。`wvalues` 在 `__init__` 里通过 `self.values = ...` 间接写入实例 `__dict__`。实例属性遮蔽同名类属性。
- **`__deepcopy__(self, memo)` 协议**：`copy.deepcopy` 优先调用对象的 `__deepcopy__`，没定义才走通用递归。`memo` 是 `{id(源): 副本}` 字典，处理循环引用。
- **富比较协议**：Python 3 要分别定义 `__lt__/__le__/__gt__/__ge__/__eq__/__ne__`，没有 `__cmp__`。定义 `__eq__` 后默认 `__hash__` 被置 None（不可哈希），需手动定义。

---

## 八、关键收获（带走这几点）

1. **把方向/条件编码进数据（weights 正负），而不是写进控制流（if else）** —— 消除热路径分支的通用手法。
2. **把昂贵计算挪到冷路径（set 时乘），让热路径（比较）变纯比较** —— 惰性预处理的经典应用。
3. **用空/非空当状态标志，合并冗余状态** —— `wvalues=()` 同时表达"未评估"和"失效"。
4. **富比较只实现两个，其余反推** —— 单一真值来源，易维护。
5. **自定义 `__deepcopy__` 跳过通用递归** —— 高频克隆场景的性能要点。
6. **抽象基类用哨兵属性 + __init__ 检查，不用 ABCMeta** —— 避免元类冲突的轻量方案。

---

## 九、思考题

1. 如果 weights 允许含 0（如 `(1.0, 0.0)`，第二目标"不在乎"），`getValues` 的 `truediv` 会怎样？该怎么处理？
2. `__gt__ = not __le__` 在浮点 NaN 存在时还正确吗？（提示：NaN 的比较全返回 False，`not False` = True，可能反直觉）
3. 如果要把"评估次数计数"加进 Fitness，最少改动怎么改？放哪一层？
4. 多目标字典序比较（`>`）和 Pareto 支配（`dominates`）分别适合什么场景的选择算子？

---

## 十、下一阶段预告

**阶段 2：Toolbox** —— 算子容器。核心是 `register(alias, function, *args, **kargs)` 用 `functools.partial` 把纯函数绑上默认参数挂成对象方法，让算法层只认 `toolbox.mate/mutate/select/evaluate` 别名。这是 DEAP "数据与算法解耦" 的另一半：Fitness 统一了"怎么比较"，Toolbox 统一了"怎么调用算子"。
