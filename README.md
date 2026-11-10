# VRPTW 智能优化实验

当前以 Python 完成任务一至任务四的实验流程；长预算的性能结论仍需进一步实验。

## 目录

- `config/config.py`：数据与结果路径。
- `problems/vrptw.py`：Solomon 解析、路径解码、可行性检查与评价。
- `algorithms/`：初始解、公共配置与状态、邻域、SA、TS 和混合算法。
- `experiments/verify.py`：精确验证；`runner.py`：统一算法调用；`benchmark.py` 和 `statistics.py`：批量实验与统计。
- `visualization/distribution.py`：客户分布图。
- `tests/`：约束、解析和解码测试。
- `results/`：生成的图表。
- `任务一汇报.md`：任务一方法、结果和复现说明。

## 运行

Python 3.8 及以上，在项目根目录运行：

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python main.py
```

本机也可使用 `py -3.13` 替换 `python`。

## 任务二算法演示

```powershell
python main.py --algorithm hybrid3 --instance C101 --customers 10 --iterations 200 --seed 0
```

算法名称：`sa`、`ts`、`hybrid1`、`hybrid2`、`hybrid3`。通过 `experiments.runner.run(problem, name, SearchConfig(...))` 可在代码中使用统一接口；返回值包含最优路径、车辆数、距离、收敛历史和耗时。详细说明见 `任务二汇报.md`。

## 任务三基准实验

```powershell
python -m experiments.benchmark --out results/benchmark --repetitions 30 --seconds 1 --iterations 1000 --candidates 30
```

默认六个实例、五种算法和种子 0～29；每次运行限时与最大迭代数相同。完成一次即写入原始结果，相同参数再次执行可续跑；参数变化时使用新的输出目录。文件包括 `config.json`、`runs.jsonl`、`summary.csv`、`distance_by_vehicles.csv`、`statistical_tests.csv`、`convergence.png`、`best_route.png` 和 `best_route.json`。

GAP 只对完整 100 客户、且车数与任务书参考值相同的结果填写。距离均值和标准差按车数分组；配对 t 检验只使用同一种子且车数相同的结果，并列出各组配对数，样本不足或方差为零时不输出 p 值。此筛选依赖结果，可能引入选择偏差，p 值需要结合车数分布和样本量谨慎解释。时间上限在迭代边界检查，可能略有超时；各算法也可能提前停止，因此应同时查看实际运行时间和迭代次数。启用墙钟时间限制后，即使种子固定，实际迭代次数也可能波动。

## 任务四敏感性与行为分析

```powershell
python -m experiments.sensitivity --out results/sensitivity --repetitions 30 --seconds 1 --iterations 1000
```

实验代码在 `experiments/sensitivity.py` 和 `experiments/behavior.py`，绘图代码在 `visualization/`，不修改算法模块。默认 R101，比较禁忌长度、候选数量和三种邻域组合；指定 `--instance R104` 可切换实例。结果逐条保存，可用相同参数续跑。行为分析复用任务三数据，可通过 `--benchmark` 指定其目录。数据解释与本次短预算结果见 `任务四汇报.md`。
