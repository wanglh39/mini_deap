# mini_deap —— DEAP 进化算法库教学重写 · 总览

> 目标：把 DEAP 的核心源码**简化重写**一遍，保留设计思想，砍掉非教学细节。
> 每阶段一个模块 + 一份 markdown 讲解 + 单测，一阶段一确认，最后串联实战。

## 一、DEAP 的 5 大核心设计思想（全项目主线）

| # | 思想 | 体现位置 | 一句话 |
|---|---|---|---|
| 1 | **数据与算法解耦** | `Toolbox` | 个体是纯数据，算子是纯函数，Toolbox 用 `partial` 粘合，算法只认别名 |
| 2 | **统一 max/min** | `Fitness.weights` | weights 正负编码方向，比较只比 `wvalues`，零分支 |
| 3 | **惰性求值** | `fitness.valid` | 变异后 `del values` 失效，每代只重算 invalid 个体 |
| 4 | **元编程建类** | `creator.create` | 一行动态建"带 fitness 的 list 子类"，适配任意表示 |
| 5 | **并行/拷贝友好** | `toolbox.map` / `__deepcopy__` | 换 map 即并行，自定义 deepcopy 提速 |

## 二、10 阶段任务规划与进度

| 阶段 | 模块 | 对应 deap | 状态 |
|---|---|---|---|
| 0 | 项目骨架 | — | ✅ 完成 |
| **1** | `base/fitness.py` | `base.Fitness` | ✅ 完成 |
| 2 | `base/toolbox.py` | `base.Toolbox` | ⬜ 待做 |
| 3 | `base/creator.py` | `creator.create` | ⬜ 待做 |
| 4 | `tools/operators.py` | `tools/{init,selection,crossover,mutation}` | ⬜ 待做 |
| 5 | `tools/support.py` | `tools.support` | ⬜ 待做 |
| 6 | `algorithms.py` | `algorithms` | ⬜ 待做 |
| 7 | `examples/` One-Max+Sphere+TSP | — | ⬜ 待做（串联） |
| 8 | `tools/emo.py` + ZDT1 | `tools.emo` NSGA2 | ⬜ 待做（进阶） |
| 9 | `gp.py` + 符号回归 | `gp` | ⬜ 待做（进阶） |
| 10 | `cma.py` + Sphere | `cma` | ⬜ 待做（进阶） |

## 三、目录结构

```
deap/
├── mini_deap/                 # 教学库
│   ├── base/                  # 阶段1-3: Fitness / Toolbox / creator
│   ├── tools/                 # 阶段4-5,8: 算子 / 统计 / 多目标
│   ├── algorithms.py          # 阶段6: 算法骨架
│   ├── gp.py                  # 阶段9: 遗传编程
│   ├── cma.py                 # 阶段10: CMA-ES
│   └── examples/              # 阶段7-10: 实战
├── tests/                     # 镜像测试 (pytest)
├── docs/                      # 本文档目录：每阶段一份讲解
│   ├── 00_overview.md
│   ├── 01_fitness.md
│   └── ...
└── pytest.ini
```

## 四、如何运行

```powershell
# Python 解释器：conda base 的 Python 3.13.5
$py = "python"

# 跑全部测试
& $py -m pytest

# 跑某阶段测试
& $py -m pytest tests/base/test_fitness.py -v

# 跑某实战例子
& $py -m mini_deap.examples.one_max
```

## 五、阅读顺序建议

1. 读 `docs/01_fitness.md` → 打开 `mini_deap/base/fitness.py` 对照
2. 依次推进，每阶段先读 docs 再读代码再读测试
3. 阶段 7 三个例子是全库串联的"验收点"，重点看