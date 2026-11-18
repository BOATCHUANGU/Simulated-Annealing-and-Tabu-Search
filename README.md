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

### RC101 完整实例

```powershell
py -3.13 main.py --algorithm hybrid3 --instance RC101 --customers 100 --seconds 10 --seed 8
py -3.13 main.py --algorithm sa --instance RC101 --customers 100 --seconds 140 --seed 24 --fleet-repair --target-vehicles 14 --target-distance 1696.95 --output results/rc101_replay.json
```

程序打印各构造方法的车辆数、距离、可行性，以及最终路线和停止原因。开启限时后默认按墙钟预算运行，显式指定 `--iterations` 可以同时设置迭代上限。退火降至温度下限后继续搜索。

车辆消除尝试将整条路线的客户重新分配，允许逐出其他客户再插回。跨路线重组包括后缀交换和移出多点再插入。插入位置用前后向时间窗标签检查，候选只重算发生变化的路线。

`--fleet-repair` 在长预算搜索停滞时启用软约束退火，通过暂时允许迟到和超载来探索更少车辆数的排布。候选通过原始约束检查后才交回搜索。设置 `--target-vehicles` 后，达到该车数时继续缩短距离。长预算模式可同时重排多条路线的尾部，最后用原始约束复核。给定 `--target-distance` 后，两项指标达到字典序目标时停止。启发式结果随种子和实际迭代次数波动。

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
