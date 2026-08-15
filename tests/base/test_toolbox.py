"""tests/base/test_toolbox.py —— Toolbox 单元测试（镜像 mini_deap/base/toolbox.py）。

覆盖：register/partial 绑定、调用覆盖默认参、__name__/__doc__、覆盖已有别名、
      __dict__ 拷贝、unregister、decorate（单/多装饰器、保留绑参）、
      默认 clone=deepcopy、默认 map=map、map 可替换（模拟并行）。
"""

import pytest

from mini_deap.base.toolbox import Toolbox


def _func(a, b, c=3):
    """测试用函数：返回 (a, b, c)。"""
    return (a, b, c)


class TestRegister:
    """register / partial 绑定 / 别名调用。"""

    def test_register_basic(self):
        tb = Toolbox()
        tb.register("myFunc", _func, 2, c=4)
        # _func(2, 3, c=4) → (2, 3, 4)
        assert tb.myFunc(3) == (2, 3, 4)

    def test_call_overrides_default(self):
        tb = Toolbox()
        tb.register("myFunc", _func, 2, c=4)
        # 调用时覆盖 c
        assert tb.myFunc(3, c=10) == (2, 3, 10)

    def test_register_sets_name_and_doc(self):
        tb = Toolbox()
        tb.register("myFunc", _func)
        assert tb.myFunc.__name__ == "myFunc"
        assert tb.myFunc.__doc__ == _func.__doc__

    def test_register_overwrites_existing(self):
        tb = Toolbox()
        tb.register("myFunc", _func, 2)
        tb.register("myFunc", _func, 5)   # 覆盖
        assert tb.myFunc(1) == (5, 1, 3)

    def test_dict_attribute_copied(self):
        _func.custom_attr = 42
        try:
            tb = Toolbox()
            tb.register("myFunc", _func)
            assert tb.myFunc.custom_attr == 42
        finally:
            del _func.custom_attr


class TestUnregister:
    def test_unregister_removes_alias(self):
        tb = Toolbox()
        tb.register("myFunc", _func)
        tb.unregister("myFunc")
        assert not hasattr(tb, "myFunc")

    def test_unregister_nonexistent_raises(self):
        tb = Toolbox()
        with pytest.raises(AttributeError):
            tb.unregister("nope")


class TestDecorate:
    def test_decorate_wraps_function(self):
        def double(func):
            def wrapper(*args, **kw):
                return func(*args, **kw) * 2
            return wrapper

        tb = Toolbox()
        tb.register("square", lambda x: x * x)
        tb.decorate("square", double)
        assert tb.square(3) == 18   # (3*3)*2

    def test_decorate_preserves_bound_args(self):
        # decorate 应保留 register 时绑的默认参数
        calls = []

        def trace(func):
            def wrapper(*args, **kw):
                calls.append((args, kw))
                return func(*args, **kw)
            return wrapper

        tb = Toolbox()
        tb.register("myFunc", _func, 2, c=4)
        tb.decorate("myFunc", trace)
        result = tb.myFunc(3)
        # wrapper 看到 partial 解包后的参数：(2, 3) 和 c=4
        assert result == (2, 3, 4)
        assert calls == [((2, 3), {'c': 4})]

    def test_decorate_multiple_last_is_outermost(self):
        # 多装饰器按顺序套，序列末尾的最外层最先执行
        order = []

        def dec(name):
            def decorator(func):
                def wrapper(*args, **kw):
                    order.append(name)
                    return func(*args, **kw)
                return wrapper
            return decorator

        tb = Toolbox()
        tb.register("f", lambda x: x)
        tb.decorate("f", dec("a"), dec("b"))
        tb.f(1)
        # 先套 a → a(orig)；再套 b → b(a(orig))；调用时 b 先执行
        assert order == ["b", "a"]


class TestDefaults:
    def test_default_clone_is_deepcopy(self):
        tb = Toolbox()
        obj = [[1, 2], [3]]
        clone = tb.clone(obj)
        assert clone == obj
        assert clone is not obj
        assert clone[0] is not obj[0]   # 深拷

    def test_default_map_is_builtin_map(self):
        tb = Toolbox()
        result = list(tb.map(lambda x: x * 2, [1, 2, 3]))
        assert result == [2, 4, 6]

    def test_map_replaceable_for_parallel(self):
        # 模拟换成并行 map（这里用列表推导代替 multiprocessing.Pool.map）
        tb = Toolbox()
        tb.register("map", lambda f, it: [f(x) for x in it])
        result = tb.map(lambda x: x + 1, [1, 2, 3])
        assert result == [2, 3, 4]