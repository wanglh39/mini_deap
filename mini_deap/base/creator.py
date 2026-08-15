"""base.creator —— 动态建类（元编程工厂）。

设计思想（对照 DEAP creator.py）
================================

1. 一行 ``create("Individual", list, fitness=FitnessMax)`` 等价于定义一个带 fitness
   属性的 list 子类。个体可以是 list/set/array/numpy.ndarray/GP树，同一套算法全通用。

2. **kargs 的值分两类（关键技巧）：
   - 值是【类型】（如 fitness=FitnessMax）→ 实例化时调 FitnessMax() 存为【实例属性】
   - 值是【普通对象】（如 weights=(1.0,)）→ 存为【类属性】（所有实例共享）
   靠元类 MetaCreator 改写 __init__ 实现这个区分。

3. class_replacers：array.array / numpy.ndarray 的 deepcopy 不拷 __dict__（C 层实现
   忽略 Python 层属性），deap 定义替换子类重写 __deepcopy__ 修正。create 时若 base
   在 replacers 里，换成替换类再继承。

4. __reduce__ + copyreg.pickle：让动态创建的类可 pickle（multiprocessing 并行需要
   序列化个体，个体类是动态建的，类本身要可 pickle）。

简化（相比 deap/creator.py）
---------------------------
- 保留全部核心：create / MetaCreator / meta_create / class_replacers / _array / _numpy_array。
- 砍掉 deprecation warning 的细节措辞，保留警告本身。
- 注释教学化。
"""

import array
import copy
import copyreg
import warnings

class_replacers = {}
"""base 类 → 替换类 的映射。create 时若 base 在此，用替换类代替。
array.array / numpy.ndarray 的 deepcopy 不拷 __dict__，需替换。"""


# ---- numpy.ndarray 替换：修正 deepcopy 不拷 __dict__ ----
try:
    import numpy

    class _numpy_array(numpy.ndarray):
        """numpy.ndarray 子类：重写 deepcopy/Reduce 以拷贝 __dict__（含 fitness 等）。"""

        @staticmethod
        def __new__(cls, iterable):
            # 从可迭代造数组，再 view 成子类
            return numpy.array(list(iterable)).view(cls)

        def __deepcopy__(self, memo):
            copy_ = numpy.ndarray.copy(self)
            copy_.__dict__.update(copy.deepcopy(self.__dict__, memo))
            return copy_

        def __setstate__(self, state):
            self.__dict__.update(state)

        def __reduce__(self):
            return (self.__class__, (list(self),), self.__dict__)

    class_replacers[numpy.ndarray] = _numpy_array
except ImportError:
    pass


# ---- array.array 替换：同上 ----
class _array(array.array):
    """array.array 子类：重写 deepcopy/Reduce 以拷贝 __dict__。"""

    @staticmethod
    def __new__(cls, seq=()):
        # array.array 需要 typecode，由子类的类属性 typecode 提供（create 时传 typecode="f"）
        return super(_array, cls).__new__(cls, cls.typecode, seq)

    def __deepcopy__(self, memo):
        cls = self.__class__
        copy_ = cls.__new__(cls, self)
        memo[id(self)] = copy_
        copy_.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return copy_

    def __reduce__(self):
        return (self.__class__, (list(self),), self.__dict__)


class_replacers[array.array] = _array


# ---- 元类：改写 __init__ 区分类属性/实例属性 ----
class MetaCreator(type):
    """元类：把 create 的 **kargs 分为实例属性（值是类型）和类属性（值不是类型）。

    元类是"类的类"：普通类实例化产生对象，元类"实例化"产生类。
    ``class Foo(metaclass=MetaCreator)`` 等价于 ``MetaCreator("Foo", bases, dict)``。
    这里 MetaCreator 拦截类的创建，改写 __init__ 注入属性分发逻辑。
    """

    def __new__(cls, name, base, dct):
        # 造出类对象本身（单继承 base）
        return super(MetaCreator, cls).__new__(cls, name, (base,), dct)

    def __init__(cls, name, base, dct):
        dict_inst = {}   # 值是类型 → 实例属性（实例化时调 类型() 生成）
        dict_cls = {}    # 值不是类型 → 类属性（直接挂类上，所有实例共享）
        for obj_name, obj in dct.items():
            if isinstance(obj, type):
                dict_inst[obj_name] = obj
            else:
                dict_cls[obj_name] = obj

        def init_type(self, *args, **kargs):
            # 实例属性：每个实例独立生成（如 self.fitness = FitnessMax()）
            for obj_name, obj in dict_inst.items():
                setattr(self, obj_name, obj())
            # 调原 base 的 __init__（如果它自定义过，而非 object 的默认空 __init__）
            if base.__init__ is not object.__init__:
                base.__init__(self, *args, **kargs)

        cls.__init__ = init_type
        cls.reduce_args = (name, base, dct)   # 存起来给 __reduce__ 用
        # 把类属性挂上（dict_cls），实例属性靠 init_type 动态生成
        super(MetaCreator, cls).__init__(name, (base,), dict_cls)

    def __reduce__(cls):
        # pickle 动态类时：用 meta_create + (name, base, dct) 重建
        return (meta_create, cls.reduce_args)


# 注册：让 pickle 知道 MetaCreator 创建的类怎么序列化
copyreg.pickle(MetaCreator, MetaCreator.__reduce__)


def meta_create(name, base, dct):
    """用元类造类，并注册到本模块全局命名空间（让 eval(repr)/pickle 能按名找到）。"""
    class_ = MetaCreator(name, base, dct)
    globals()[name] = class_
    return class_


def create(name, base, **kargs):
    """动态创建类 ``name`` 继承 ``base``，属性由 ``**kargs`` 定义。

    :param name: 类名（注册到 creator 模块全局，可用 ``creator.<name>`` 访问）。
    :param base: 基类（list/set/array.array/numpy.ndarray/Fitness/...）。
    :param kargs: 属性。值是类型 → 实例属性（实例化时调 类型()）；否则 → 类属性。

    例::

        create("FitnessMax", Fitness, weights=(1.0,))   # weights 是 tuple → 类属性
        create("Individual", list, fitness=FitnessMax)  # fitness 是类型 → 实例属性

        ind = Individual([1, 2, 3])
        ind.fitness          # FitnessMax() 实例
        ind                  # [1, 2, 3]
    """
    if name in globals():
        warnings.warn(
            "A class named '%s' has already been created and it will be overwritten. "
            "Consider deleting previous creation or rename it." % name,
            RuntimeWarning,
        )

    # base 若在替换表里，换成替换类（修正 deepcopy 不拷 __dict__ 的问题）
    if base in class_replacers:
        base = class_replacers[base]
    meta_create(name, base, kargs)