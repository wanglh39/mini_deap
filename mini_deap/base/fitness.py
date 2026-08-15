"""base.fitness —— 适应度抽象基类。

设计思想（对照 DEAP base.py 的 Fitness）
========================================

1. 统一最大化/最小化：用类属性 ``weights`` 的正负编码方向
   - ``(1.0,)``      单目标最大化
   - ``(-1.0,)``     单目标最小化
   - ``(1.0, -1.0)`` 双目标：第一目标最大化、第二目标最小化
   子类必须定义 weights；本基类 weights=None，实例化即抛 TypeError —— 抽象基类保护。

2. 惰性加权：设 ``values`` 时立刻乘 weights 存入 ``wvalues``，之后所有比较只比 wvalues。
   - 好处：比较是热路径（选择算子每代 O(N log N) 次比较），一次乘法换无数次比较里的分支判断。
   - 还原原值：get 时再除以 weights（冷路径，只在读取时算一次）。

3. 惰性求值配合算法层：变异/交叉后 ``del ind.fitness.values`` 清空 wvalues → ``valid=False``，
   算法每代只重算 invalid 个体。valid 就是 ``len(wvalues) != 0``。

4. 比较永远"越大越好"：因为 wvalues = values * weights，最小化问题 weights 为负，
   原值越小 → wvalues 越大 → 选择算子取 max 自然选中更优解，零分支。

5. Pareto 支配 ``dominates``：用于多目标 NSGA2，逐维比较 wvalues。

简化（相比 deap/base.py）
------------------------
- 砍掉 ConstrainedFitness（约束适应度，约 80 行），约束处理留到进阶阶段。
- 砍掉 setValues 里 sys.exc_info/with_traceback 的复杂错误重建，用 assert + 原生异常。
- 砍掉 collections.abc 的 try/except ImportError 兼容（Python 3.13 直接导入）。
"""

from collections.abc import Sequence
from operator import mul, truediv


class Fitness:
    """适应度基类。子类须定义类属性 ``weights``（序列，正=最大化，负=最小化）。

    用法（通常配合 creator 动态建类，这里直接子类化演示）::

        class FitnessMax(Fitness):
            weights = (1.0,)

        f = FitnessMax((3.14,))
        f.values          # (3.14,)
        f.valid           # True
        del f.values      # 失效
        f.valid           # False
    """

    # 类属性：子类必须覆盖。None 表示抽象基类，实例化即抛错。
    weights = None

    # 实例属性：加权后的值。空元组表示"未评估/已失效"。
    wvalues = ()

    def __init__(self, values=()):
        # 抽象基类保护：weights 未定义不能实例化
        if self.weights is None:
            raise TypeError(
                "Can't instantiate abstract %r with abstract attribute weights."
                % self.__class__
            )
        if not isinstance(self.weights, Sequence):
            raise TypeError(
                "Attribute weights of %r must be a sequence." % self.__class__
            )
        if len(values) > 0:
            self.values = values  # 走 setValues，立刻乘 weights

    # ---- values property：对外的接口，内部转 wvalues ----
    def getValues(self):
        # 还原原值 = wvalues / weights（冷路径，只在读取时算）
        return tuple(map(truediv, self.wvalues, self.weights))

    def setValues(self, values):
        # 长度必须匹配 weights（多目标维度一致）
        assert len(values) == len(self.weights), (
            "Assigned values have not the same length as fitness weights"
        )
        # 热路径：一次乘法，之后比较不再分支
        self.wvalues = tuple(map(mul, values, self.weights))

    def delValues(self):
        # 失效：清空 wvalues → valid 变 False
        self.wvalues = ()

    values = property(
        getValues, setValues, delValues,
        doc="适应度原值。set 时乘 weights 存为 wvalues；del 使其失效。",
    )

    # ---- valid：是否已评估 ----
    @property
    def valid(self):
        """是否有效（已评估且未失效）。"""
        return len(self.wvalues) != 0

    # ---- Pareto 支配（多目标用）----
    def dominates(self, other, obj=slice(None)):
        """self 是否 Pareto 支配 other：每维不劣且至少一维严格更优。

        :param obj: 切片，指定在哪些目标上判断，默认全部。
        """
        not_equal = False
        for self_wv, other_wv in zip(self.wvalues[obj], other.wvalues[obj]):
            if self_wv > other_wv:
                not_equal = True       # 有一维严格更优
            elif self_wv < other_wv:
                return False            # 有一维严格更劣 → 不支配
        return not_equal

    # ---- 比较运算符：全部基于 wvalues 字典序，永远"越大越好" ----
    # 技巧：只实现 < 和 <=，其余用 not 反推，避免逻辑重复（deap 同款写法）。
    def __hash__(self):
        return hash(self.wvalues)

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

    def __le__(self, other):
        return self.wvalues <= other.wvalues

    def __lt__(self, other):
        return self.wvalues < other.wvalues

    def __eq__(self, other):
        return self.wvalues == other.wvalues

    def __ne__(self, other):
        return not self.__eq__(other)

    # ---- 自定义 deepcopy：只拷 wvalues，weights 是类属性共享不用拷 ----
    def __deepcopy__(self, memo):
        # 比通用 deepcopy 快：跳过 weights（类属性）和 property 描述符的递归拷贝
        copy_ = self.__class__()
        copy_.wvalues = self.wvalues
        return copy_

    def __str__(self):
        return str(self.values if self.valid else tuple())

    def __repr__(self):
        return "%s.%s(%r)" % (
            self.__module__, self.__class__.__name__,
            self.values if self.valid else tuple(),
        )