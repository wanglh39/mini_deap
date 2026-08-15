"""tools.support —— 统计与记录组件（算法层的观测插件）。

设计思想（对照 DEAP tools/support.py）
========================================

1. 这些是算法层的"可选回调"：算法里 ``if stats: record = stats.compile(pop)``，
   不传也能跑。让 eaSimple 能边跑边输出统计、保留精英，但不强制。

2. Statistics：注册统计函数（mean/max/min...），compile(pop) 返回 {name: value}。
   key 参数提取要统计的值（如 attrgetter("fitness.values")）。

3. HallOfFame：保留历代最优 k 个个体。用 bisect 维持按 fitness 降序，
   update(pop) 把 pop 里更优的换进来。deepcopy 隔离，不污染种群。

4. Logbook：record(gen=0, **stats) 记录每代，stream 属性输出未打印的新行。
   继承 list，每条记录是一个 dict。

5. ParetoFront：多目标版 HallOfFame，保留所有非支配个体。阶段 8 NSGA2 用。

简化（相比 deap/tools/support.py）
---------------------------------
- 砍掉 History（家谱树，依赖 networkx，非核心）。
- 砍掉 Logbook 的 chapters 机制（MultiStatistics 的子日志），简化为单层表格输出。
- 保留 Statistics/MultiStatistics/HallOfFame/ParetoFront/Logbook 核心。
"""

from bisect import bisect_right
from copy import deepcopy
from functools import partial
from operator import eq


def identity(obj):
    """恒等函数，Statistics 的默认 key。"""
    return obj


class Statistics:
    """统计聚合器：注册统计函数，compile(data) 返回 {name: value}。

        stats = Statistics(key=attrgetter("fitness.values"))
        stats.register("avg", numpy.mean)
        stats.register("max", numpy.max)
        stats.compile(population)  # {'avg': 3.2, 'max': 5.0}
    """

    def __init__(self, key=identity):
        self.key = key
        self.functions = dict()
        self.fields = []

    def register(self, name, function, *args, **kargs):
        """注册统计函数。name 是结果字典里的键。可绑默认参数（同 toolbox.register）。"""
        self.functions[name] = partial(function, *args, **kargs)
        self.fields.append(name)

    def compile(self, data):
        """对 data 应用所有注册的统计函数，返回 {name: value}。"""
        values = tuple(self.key(elem) for elem in data)
        return {name: func(values) for name, func in self.functions.items()}


class MultiStatistics(dict):
    """多键统计：对同一数据按不同 key 分别统计。

        mstats = MultiStatistics(fitness=Statistics(key=attrgetter("fitness.values")),
                                 size=Statistics(key=len))
        mstats.register("mean", numpy.mean)
        mstats.compile(pop)  # {'fitness': {'mean': ...}, 'size': {'mean': ...}}
    """

    def compile(self, data):
        return {name: stats.compile(data) for name, stats in self.items()}

    @property
    def fields(self):
        return sorted(self.keys())

    def register(self, name, function, *args, **kargs):
        """对所有子 Statistics 同时注册同一函数。"""
        for stats in self.values():
            stats.register(name, function, *args, **kargs)


class Logbook(list):
    """进化日志：每代 record 一条，stream 输出未打印的新行。继承 list。

        log = Logbook()
        log.header = ["gen", "nevals", "avg", "max"]
        log.record(gen=0, nevals=100, avg=3.2, max=5.0)
        print(log.stream)  # 输出表头 + 第一行
    """

    def __init__(self):
        self.buffindex = 0       # stream 已输出到哪
        self.header = None       # 列顺序
        self.columns_len = None  # 各列宽度
        self.log_header = True   # 是否输出表头

    def record(self, **infos):
        """记录一条（追加到末尾）。"""
        self.append(infos)

    def select(self, *names):
        """按列名取历史值。select("avg") → [3.2, 3.5, ...]"""
        if len(names) == 1:
            return [entry.get(names[0], None) for entry in self]
        return tuple([entry.get(name, None) for entry in self] for name in names)

    @property
    def stream(self):
        """输出自上次 stream 后新增的行（首次含表头）。"""
        startindex, self.buffindex = self.buffindex, len(self)
        return self.__str__(startindex)

    def __str__(self, startindex=0):
        # 简化版表格输出（不带 chapter）
        columns = self.header
        if not columns:
            columns = sorted(self[0].keys()) if len(self) > 0 else []
        # 列宽初始化为列名长度，遍历内容时取 max
        self.columns_len = [len(str(c)) for c in columns]

        str_matrix = []
        for line in self[startindex:]:
            str_line = []
            for j, name in enumerate(columns):
                value = line.get(name, "")
                col = "{:.4f}".format(value) if isinstance(value, float) else str(value)
                self.columns_len[j] = max(self.columns_len[j], len(col))
                str_line.append(col)
            str_matrix.append(str_line)

        # 首次输出且要表头：表头行插到最前
        if startindex == 0 and self.log_header:
            str_matrix = [[str(name) for name in columns]] + str_matrix

        template = "  ".join("{:<%i}" % l for l in self.columns_len)
        text = [template.format(*line) for line in str_matrix]
        return "\n".join(text)


class HallOfFame:
    """名人堂：保留历代最优 maxsize 个个体。按 fitness 降序，[0] 最优。

        hof = HallOfFame(maxsize=10)
        hof.update(population)  # 每代调
        hof[0]  # 历代最优个体
    """

    def __init__(self, maxsize, similar=eq):
        self.maxsize = maxsize
        self.keys = list()    # fitness 列表，升序（用 bisect 维持）
        self.items = list()   # 个体列表，降序（[0] 最优）
        self.similar = similar

    def update(self, population):
        """把 population 里更优且不重复的个体换进名人堂。"""
        for ind in population:
            if len(self) == 0 and self.maxsize != 0:
                # 空名人堂的特判（否则 self[-1] 越界）
                self.insert(population[0])
                continue
            # 比当前最差优，或还没满：候选
            if ind.fitness > self[-1].fitness or len(self) < self.maxsize:
                for hofer in self:
                    if self.similar(ind, hofer):
                        break   # 已有重复，跳过
                else:
                    # 无重复：满了先删最差，再插入
                    if len(self) >= self.maxsize:
                        self.remove(-1)
                    self.insert(ind)

    def insert(self, item):
        """用 bisect 维持按 fitness 降序插入。deepcopy 隔离。"""
        item = deepcopy(item)
        i = bisect_right(self.keys, item.fitness)
        self.items.insert(len(self) - i, item)
        self.keys.insert(i, item.fitness)

    def remove(self, index):
        """删除指定位置。index=-1 删最差（末尾）。"""
        del self.keys[len(self) - (index % len(self) + 1)]
        del self.items[index]

    def clear(self):
        del self.items[:]
        del self.keys[:]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def __iter__(self):
        return iter(self.items)

    def __reversed__(self):
        return reversed(self.items)

    def __str__(self):
        return str(self.items)


class ParetoFront(HallOfFame):
    """Pareto 前沿：保留所有非支配个体（多目标版 HallOfFame）。maxsize=None 无上限。

    阶段 8 NSGA2 用。update 时移除被新个体支配的旧成员，加入非支配的新个体。
    """

    def __init__(self, similar=eq):
        HallOfFame.__init__(self, None, similar)

    def update(self, population):
        for ind in population:
            is_dominated = False
            dominates_one = False
            has_twin = False
            to_remove = []
            for i, hofer in enumerate(self):
                if not dominates_one and hofer.fitness.dominates(ind.fitness):
                    is_dominated = True
                    break
                elif ind.fitness.dominates(hofer.fitness):
                    dominates_one = True
                    to_remove.append(i)
                elif ind.fitness == hofer.fitness and self.similar(ind, hofer):
                    has_twin = True
                    break
            for i in reversed(to_remove):   # 从后往前删，避免索引错位
                self.remove(i)
            if not is_dominated and not has_twin:
                self.insert(ind)
