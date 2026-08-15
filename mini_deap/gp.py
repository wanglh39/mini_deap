"""gp.py —— 遗传规划（Genetic Programming）。

GP 的个体是**变长语法树**（非定长向量）。树用扁平列表表示（深度优先序）：
    [mul, x, add, y, 1]  表示  mul(x, add(y, 1))

核心组件：
    Primitive / Terminal    树节点
    PrimitiveSet            函数集 + 终端集
    PrimitiveTree           树个体（继承 list）
    genFull/genGrow/genHalfAndHalf   树生成
    compile                 编译树为可执行函数
    cxOnePoint              子树交换交叉
    mutUniform              子树替换变异
    mutNodeReplacement      同 arity 节点替换
    mutShrink               删一个子树

参考：Koza, "Genetic Programming: On the Programming of Computers
by Means of Natural Selection", 1992.
"""

import copy
import random
from collections import defaultdict


__type__ = object   # 无类型 GP 的统一类型标记


# ───────────────────────── 树节点 ─────────────────────────

class Primitive:
    """函数节点：有 name、arity（参数数）、args（参数类型）、ret（返回类型）。

    format(*args) 生成调用代码，如 Primitive("mul",2) → "mul({0}, {1})"。
    """

    __slots__ = ('name', 'arity', 'args', 'ret', 'seq')

    def __init__(self, name, args, ret):
        self.name = name
        self.arity = len(args)
        self.args = args
        self.ret = ret
        arg_str = ", ".join("{{{0}}}".format(i) for i in range(self.arity))
        self.seq = "{name}({args})".format(name=self.name, args=arg_str)

    def format(self, *args):
        return self.seq.format(*args)

    def __eq__(self, other):
        if type(self) is type(other):
            return all(getattr(self, s) == getattr(other, s)
                       for s in self.__slots__)
        return NotImplemented

    def __hash__(self):
        return hash(self.name)


class Terminal:
    """终端节点：变量或常量，arity=0。"""

    __slots__ = ('name', 'value', 'ret', 'conv_fct')

    def __init__(self, terminal, symbolic, ret):
        self.ret = ret
        self.value = terminal
        self.name = str(terminal)
        self.conv_fct = str if symbolic else repr

    @property
    def arity(self):
        return 0

    def format(self):
        return self.conv_fct(self.value)

    def __eq__(self, other):
        if type(self) is type(other):
            return all(getattr(self, s) == getattr(other, s)
                       for s in self.__slots__)
        return NotImplemented

    def __hash__(self):
        return hash(self.name)


# ───────────────────────── PrimitiveSet ─────────────────────────

class PrimitiveSet:
    """无类型 GP 的函数集 + 终端集。

    用法：
        pset = PrimitiveSet("MAIN", arity=1)      # 1 个输入变量 ARG0
        pset.addPrimitive(operator.add, 2, name="add")
        pset.addTerminal(1.0, name="one")
    """

    def __init__(self, name, arity, prefix="ARG"):
        self.terminals = defaultdict(list)
        self.primitives = defaultdict(list)
        self.arguments = []
        self.context = {"__builtins__": None}    # eval 上下文
        self.terms_count = 0
        self.prims_count = 0
        self.name = name
        self.ret = __type__
        self.ins = [__type__] * arity

        # 自动添加输入变量 ARG0, ARG1, ...
        for i in range(arity):
            arg_str = "{prefix}{i}".format(prefix=prefix, i=i)
            self.arguments.append(arg_str)
            term = Terminal(arg_str, True, __type__)
            self._add_terminal(term)

    def addPrimitive(self, func, arity, name=None):
        """添加函数节点。func 是可调用，arity 是参数数，name 是显示名。"""
        assert arity > 0, "arity should be >= 1"
        name = name or func.__name__
        prim = Primitive(name, [__type__] * arity, __type__)
        self.primitives[__type__].append(prim)
        self.context[name] = func
        self.prims_count += 1

    def addTerminal(self, terminal, name=None):
        """添加终端（常量或 0 参函数）。"""
        if callable(terminal) and not isinstance(terminal, (int, float, str)):
            # 0 参函数
            name = name or terminal.__name__
            self.context[name] = terminal
            term = Terminal(name, True, __type__)
        else:
            # 常量
            name = name or repr(terminal)
            term = Terminal(terminal, False, __type__)
        self._add_terminal(term)

    def _add_terminal(self, term):
        self.terminals[__type__].append(term)
        self.terms_count += 1

    @property
    def terminalRatio(self):
        """终端数 / (终端数 + 函数数)，用于 genGrow 的停止概率。"""
        return self.terms_count / float(self.terms_count + self.prims_count)


# ───────────────────────── PrimitiveTree ─────────────────────────

class PrimitiveTree(list):
    """树个体：扁平列表表示（深度优先序）。

    [mul, x, add, y, 1] 表示 mul(x, add(y, 1))。
    每个节点有 arity 属性，searchSubtree 用 arity 求子树范围。
    """

    def __init__(self, content):
        list.__init__(self, content)

    def __deepcopy__(self, memo):
        new = self.__class__(self)
        new.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return new

    def __str__(self):
        """递归格式化为可执行代码字符串。"""
        string = ""
        stack = []
        for node in self:
            stack.append((node, []))
            while len(stack[-1][1]) == stack[-1][0].arity:
                prim, args = stack.pop()
                string = prim.format(*args)
                if len(stack) == 0:
                    break
                stack[-1][1].append(string)
        return string

    @property
    def height(self):
        """树高（根到最深叶的距离）。"""
        stack = [0]
        max_depth = 0
        for elem in self:
            depth = stack.pop()
            max_depth = max(max_depth, depth)
            stack.extend([depth + 1] * elem.arity)
        return max_depth

    @property
    def root(self):
        """根节点（self[0]）。"""
        return self[0]

    def searchSubtree(self, begin):
        """返回 begin 位置子树的 slice(begin, end)。

        用 arity 累积：子树 = begin 自己 + 所有参数子树。
        total = arity，每加一个节点 total += arity - 1（消耗1个槽位，提供arity个）。
        total==0 时子树闭合。
        """
        end = begin + 1
        total = self[begin].arity
        while total > 0:
            total += self[end].arity - 1
            end += 1
        return slice(begin, end)


# ───────────────────────── 树生成 ─────────────────────────

def generate(pset, min_, max_, condition, type_=None):
    """树生成核心：从根到叶深度优先构建。

    condition(height, depth) 决定何时停止生长（放终端）。
    genFull 的 condition: depth == height（满树）
    genGrow 的 condition: depth == height 或随机选终端
    """
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while stack:
        depth, type_ = stack.pop()
        if condition(height, depth):
            term = random.choice(pset.terminals[type_])
            expr.append(term)
        else:
            prim = random.choice(pset.primitives[type_])
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr


def genFull(pset, min_, max_, type_=None):
    """满树生成：所有叶在同一深度。"""
    return generate(pset, min_, max_, lambda h, d: d == h, type_)


def genGrow(pset, min_, max_, type_=None):
    """生长树：叶可在不同深度（达到 min_ 后随机选终端）。"""
    return generate(pset, min_, max_,
                    lambda h, d: d == h or (d >= min_ and random.random() < pset.terminalRatio),
                    type_)


def genHalfAndHalf(pset, min_, max_, type_=None):
    """半半混合：一半概率满树，一半概率生长树。"""
    method = random.choice((genGrow, genFull))
    return method(pset, min_, max_, type_)


# ───────────────────────── 编译 ─────────────────────────

def compile(expr, pset):
    """把树编译成可执行函数。

    str(expr) → "mul(ARG0, add(ARG0, 1.0))"
    有参数时 → lambda ARG0: mul(ARG0, add(ARG0, 1.0))
    eval 在 pset.context 里查函数名。
    """
    code = str(expr)
    if len(pset.arguments) > 0:
        args = ",".join(pset.arguments)
        code = "lambda {args}: {code}".format(args=args, code=code)
    return eval(code, pset.context, {})


# ───────────────────────── 交叉 ─────────────────────────

def cxOnePoint(ind1, ind2):
    """单点交叉：随机选两棵树的子树交换。

    流程：
      1. 各随机选一个非根节点（索引 1..len-1）
      2. searchSubtree 找子树范围
      3. 交换两个 slice
    """
    if len(ind1) < 2 or len(ind2) < 2:
        return ind1, ind2   # 单节点树不交叉

    # 无类型 GP：所有节点类型相同，任意子树可交换
    index1 = random.randint(1, len(ind1) - 1)
    index2 = random.randint(1, len(ind2) - 1)

    slice1 = ind1.searchSubtree(index1)
    slice2 = ind2.searchSubtree(index2)

    ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2


# ───────────────────────── 变异 ─────────────────────────

def mutUniform(individual, expr, pset):
    """均匀变异：随机选一个节点，用 expr 生成的新子树替换。"""
    index = random.randrange(len(individual))
    slice_ = individual.searchSubtree(index)
    individual[slice_] = expr(pset=pset)
    return individual,


def mutNodeReplacement(individual, pset):
    """节点替换：随机选一个节点，用同 arity 的另一个节点替换。"""
    if len(individual) < 2:
        return individual,

    index = random.randint(1, len(individual) - 1)
    node = individual[index]

    if node.arity == 0:   # 终端
        term = random.choice(pset.terminals[__type__])
        individual[index] = term
    else:                 # 函数：找同 arity 的
        prims = [p for p in pset.primitives[__type__] if p.arity == node.arity]
        if prims:
            individual[index] = random.choice(prims)

    return individual,


def mutShrink(individual):
    """收缩变异：随机选一个非终端节点，用其第一个子树替换（删一层）。"""
    if len(individual) < 3:
        return individual,

    # 找所有非终端节点（arity > 0），排除根
    prim_indices = [i for i, node in enumerate(individual[1:], 1)
                    if node.arity > 0]
    if not prim_indices:
        return individual,

    index = random.choice(prim_indices)
    slice_ = individual.searchSubtree(index)
    # 用该子树的第一个子树替换（删掉函数节点，保留第一个参数子树）
    sub = individual[slice_]
    if sub[0].arity == 0:
        return individual,
    # 找第一个参数子树
    first_child_slice = individual.searchSubtree(index + 1)
    individual[slice_] = individual[first_child_slice]

    return individual,