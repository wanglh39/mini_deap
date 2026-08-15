"""base 子包：类型抽象与算子调度中心。

阶段 1: fitness.Fitness       适应度（统一最大/最小化、惰性加权、Pareto 支配）
阶段 2: toolbox.Toolbox       算子容器（register/partial 绑定）
阶段 3: creator.create        动态建类（元编程）
"""