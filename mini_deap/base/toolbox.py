"""base.toolbox —— 算子容器（进化算法的调度中心）。

设计思想（对照 DEAP base.py 的 Toolbox）
========================================

1. 算子是纯函数，Toolbox 是粘合剂：register(alias, func, *args, **kargs) 用
   functools.partial 把 func 绑上默认参数，挂成 self.<alias>。算法层只认
   toolbox.mate / mutate / select / evaluate 这几个别名，不直接 import 具体算子 →
   **数据与算法解耦**：换算子只改 register，算法代码一行不动。

2. 默认注册 clone=deepcopy, map=map：
   - clone 用于 varAnd 里复制个体（每代克隆整个种群，高频）。
   - map 用于批量评估 fitnesses = toolbox.map(evaluate, invalid_ind)。
     换成 multiprocessing.Pool().map 即可并行评估，**算法代码一行不改** —— 这是
     DEAP 并行设计的精髓：并行不是算法的事，是 toolbox 配置的事。

3. decorate(alias, *dec) 给已注册算子套装饰器（如限制个体大小、统计调用次数、
   记时）。原理：解包 partial 拿到原 func + 已绑参数，套装饰器后重新 register，
   保留原绑参。

简化（相比 deap/base.py）
------------------------
- 几乎无简化，Toolbox 本身约 90 行，核心全保留。
- register 里 __dict__ 拷贝的 isinstance(function, type) 判断保留（注释讲清为何排除类）。
"""

from copy import deepcopy
from functools import partial


class Toolbox:
    """算子容器。register 注册算子，算法层通过 toolbox.<alias> 调用。

    用法::

        toolbox = Toolbox()
        toolbox.register("select", selTournament, tournsize=3)
        selected = toolbox.select(population, k=50)
        # 等价于 selTournament(population, k=50, tournsize=3)
    """

    def __init__(self):
        # 默认算子：clone 用 deepcopy，map 用内置 map（可换并行 map）
        self.register("clone", deepcopy)
        self.register("map", map)

    def register(self, alias, function, *args, **kargs):
        """把 function 绑定默认参数 *args/**kargs 后，挂到 self.<alias>。

        :param alias: 算子别名（如 "mate"）。已存在则覆盖。
        :param function: 被注册的函数。
        :param args/kargs: 调用 alias 时自动前置的默认参数，可在调用时覆盖。

        例::
            def func(a, b, c=3): return (a, b, c)
            tb = Toolbox()
            tb.register("myFunc", func, 2, c=4)
            tb.myFunc(3)        # → (2, 3, 4)   # 2 是绑的默认，3 是调用传入，c=4 是绑的默认
        """
        # partial 把 *args/**kargs 冻进 function，调用时只需传剩余参数
        pfunc = partial(function, *args, **kargs)
        pfunc.__name__ = alias
        pfunc.__doc__ = function.__doc__

        # 拷贝原函数的实例字典（如函数上挂的自定义属性），让别名能访问到。
        # 排除类（type）：类的 __dict__ 是 mappingproxy 且语义不对（类属性不该当实例属性拷）。
        if hasattr(function, "__dict__") and not isinstance(function, type):
            pfunc.__dict__.update(function.__dict__.copy())

        setattr(self, alias, pfunc)

    def unregister(self, alias):
        """移除已注册的算子。"""
        delattr(self, alias)

    def decorate(self, alias, *decorators):
        """给已注册算子套装饰器（按顺序套，序列末尾的装饰器在最外层最先执行）。

        :param alias: 已注册的算子名。
        :param decorators: 一个或多个装饰器。

        例：限制交叉后个体大小::

            def limit_size(func):
                def wrapper(ind1, ind2):
                    (ind1, ind2) = func(ind1, ind2)
                    if len(ind1) > 10: del ind1[10:]
                    if len(ind2) > 10: del ind2[10:]
                    return ind1, ind2
                return wrapper
            toolbox.decorate("mate", limit_size)

        注意：装饰后函数变成普通函数（不再是 partial），在 multiprocessing 下
        可能不可 pickle。如需并行，用手动 @ 装饰后再 register。
        """
        pfunc = getattr(self, alias)
        # 解包 partial：拿到原 function 和已绑的参数
        function, args, kargs = pfunc.func, pfunc.args, pfunc.keywords
        # 逐个套装饰器：后套的在最外层
        for decorator in decorators:
            function = decorator(function)
        # 重新 register，保留原绑参
        self.register(alias, function, *args, **kargs)