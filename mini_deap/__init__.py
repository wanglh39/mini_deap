"""mini_deap —— DEAP 进化算法库的教学简化重写。

本包按阶段重构 DEAP 的核心设计，保留思想、简化非教学细节：
    base/       Fitness / Toolbox / Creator   (类型与调度抽象)
    tools/      算子(选择/交叉/变异/初始化) + 统计组件 + 多目标(emo)
    algorithms/ 通用算法骨架 (eaSimple / eaMuPlusLambda / ...)
    gp.py       遗传编程 (Primitive / PrimitiveTree / compile)
    cma.py      CMA-ES 协方差矩阵自适应进化策略
    examples/   串联实战 (One-Max / Sphere / TSP / ZDT1 / 符号回归)

设计原点见各模块 docstring。
"""

__version__ = "0.1.0"