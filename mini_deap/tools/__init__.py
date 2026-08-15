"""tools 子包：进化算子与辅助组件。

阶段 4: operators    纯函数算子（init/selection/crossover/mutation），含 numpy 向量化对照版
阶段 5: support      Statistics / HallOfFame / Logbook
阶段 8: emo          多目标选择（NSGA2 非支配排序 + 拥挤距离）
"""

from .operators import *
from .support import *
from .emo import *