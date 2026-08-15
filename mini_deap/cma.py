"""cma.py —— 协方差矩阵自适应进化策略（CMA-ES）。

CMA-ES 是连续优化的"黄金标准"——黑盒优化中表现最好的算法之一。
核心思想：**学习问题的几何结构**——协方差矩阵 C 编码变量间相关性，
步长 sigma 自适应控制全局探索 vs 局部开发。

流程（ask-tell 模型）：
    strategy = Strategy(centroid, sigma)   # 初始化策略
    for gen in range(ngen):
        pop = strategy.generate(ind_init)  # ask：从 N(mean, sigma²·C) 采样
        evaluate(pop)                      # 评估
        strategy.update(pop)               # tell：更新 mean/sigma/C

核心组件：
    Strategy    维护均值/步长/协方差矩阵/进化路径
    generate    ask：采样 lambda_ 个子代
    update      tell：根据成功子代更新策略参数

参考：Hansen & Ostermeier, 2001. Completely Derandomized Self-Adaptation
in Evolution Strategies.
"""

import numpy
from math import sqrt, log


class Strategy:
    """CMA-ES 策略：维护均值/步长/协方差矩阵/进化路径。

    Args:
        centroid: 初始均值（搜索起点）
        sigma:    初始步长（全局缩放）
        **kargs:  可选参数（lambda_/mu/weights/cs/damps/ccum/ccov1/ccovmu）
    """

    def __init__(self, centroid, sigma, **kargs):
        self.params = kargs

        self.centroid = numpy.array(centroid)
        self.dim = len(self.centroid)
        self.sigma = sigma

        # 进化路径（cumulation paths）
        self.pc = numpy.zeros(self.dim)   # 协方差路径
        self.ps = numpy.zeros(self.dim)   # 步长路径

        # chiN：N(0,I) 的期望范数（用于步长更新的归一化）
        self.chiN = sqrt(self.dim) * (1 - 1. / (4. * self.dim)
                                      + 1. / (21. * self.dim ** 2))

        # 协方差矩阵 C = B·D²·B^T（特征分解）
        self.C = self.params.get("cmatrix", numpy.identity(self.dim))
        self.diagD, self.B = numpy.linalg.eigh(self.C)
        indx = numpy.argsort(self.diagD)
        self.diagD = self.diagD[indx] ** 0.5
        self.B = self.B[:, indx]
        self.BD = self.B * self.diagD

        # 子代数 lambda_，默认 4 + 3*ln(N)
        self.lambda_ = self.params.get("lambda_", int(4 + 3 * log(self.dim)))
        self.update_count = 0
        self.computeParams(self.params)

    def generate(self, ind_init):
        """ask：从 N(mean, sigma²·C) 采样 lambda_ 个子代。

        采样：z ~ N(0, I)，x = mean + sigma · B·D·z
        B·D 是 C 的平方根（C = BD·(BD)^T）。
        """
        arz = numpy.random.standard_normal((self.lambda_, self.dim))
        arz = self.centroid + self.sigma * numpy.dot(arz, self.BD.T)
        return [ind_init(a) for a in arz]

    def update(self, population):
        """tell：根据成功子代更新 mean/sigma/C。

        流程：
          1. 按 fitness 排序，取前 mu 个
          2. 更新均值 centroid = Σ weights[i] · pop[i]
          3. 更新进化路径 ps（步长）、pc（协方差）
          4. 更新协方差矩阵 C（rank-1 + rank-mu 更新）
          5. 更新步长 sigma
          6. 特征分解 C = B·D²·B^T
        """
        population.sort(key=lambda ind: ind.fitness, reverse=True)

        old_centroid = self.centroid
        self.centroid = numpy.dot(self.weights, population[0:self.mu])
        c_diff = self.centroid - old_centroid

        # 步长路径 ps
        self.ps = (1 - self.cs) * self.ps \
            + sqrt(self.cs * (2 - self.cs) * self.mueff) / self.sigma \
            * numpy.dot(self.B, (1. / self.diagD)
                        * numpy.dot(self.B.T, c_diff))

        # hsig：检测是否在期望范围内（防 C 更新偏 bias）
        hsig = float((numpy.linalg.norm(self.ps)
                      / sqrt(1. - (1. - self.cs) ** (2. * (self.update_count + 1.)))
                      / self.chiN < (1.4 + 2. / (self.dim + 1.))))
        self.update_count += 1

        # 协方差路径 pc
        self.pc = (1 - self.cc) * self.pc + hsig \
            * sqrt(self.cc * (2 - self.cc) * self.mueff) / self.sigma \
            * c_diff

        # 协方差矩阵 C 更新（rank-1 + rank-mu）
        artmp = population[0:self.mu] - old_centroid
        self.C = (1 - self.ccov1 - self.ccovmu + (1 - hsig)
                  * self.ccov1 * self.cc * (2 - self.cc)) * self.C \
            + self.ccov1 * numpy.outer(self.pc, self.pc) \
            + self.ccovmu * numpy.dot((self.weights * artmp.T), artmp) \
            / self.sigma ** 2

        # 步长 sigma 更新
        self.sigma *= numpy.exp((numpy.linalg.norm(self.ps) / self.chiN - 1.)
                                * self.cs / self.damps)

        # 特征分解 C = B·D²·B^T
        self.diagD, self.B = numpy.linalg.eigh(self.C)
        indx = numpy.argsort(self.diagD)
        self.diagD = self.diagD[indx] ** 0.5
        self.B = self.B[:, indx]
        self.BD = self.B * self.diagD

    def computeParams(self, params):
        """计算依赖 lambda_ 的参数：mu/weights/mueff/cs/cc/ccov1/ccovmu/damps。"""
        self.mu = params.get("mu", int(self.lambda_ / 2))
        rweights = params.get("weights", "superlinear")
        if rweights == "superlinear":
            self.weights = log(self.mu + 0.5) - \
                numpy.log(numpy.arange(1, self.mu + 1))
        elif rweights == "linear":
            self.weights = self.mu + 0.5 - numpy.arange(1, self.mu + 1)
        elif rweights == "equal":
            self.weights = numpy.ones(self.mu)
        else:
            raise RuntimeError("Unknown weights : %s" % rweights)

        self.weights /= sum(self.weights)
        self.mueff = 1. / sum(self.weights ** 2)

        self.cc = params.get("ccum", 4. / (self.dim + 4.))
        self.cs = params.get("cs", (self.mueff + 2.)
                             / (self.dim + self.mueff + 3.))
        self.ccov1 = params.get("ccov1", 2. / ((self.dim + 1.3) ** 2
                                                + self.mueff))
        self.ccovmu = params.get("ccovmu", 2 * (self.mueff - 2 + 1. / self.mueff)
                                 / ((self.dim + 2.) ** 2 + self.mueff))
        self.ccov1 = min(self.ccov1, 1)
        self.ccovmu = min(self.ccovmu, 1 - self.ccov1)
        self.damps = params.get("damps", 1 + 2 * max(0, sqrt((self.mueff - 1.)
                                                              / (self.dim + 1.)) - 1.) + self.cs)