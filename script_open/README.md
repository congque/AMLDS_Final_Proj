# script_open

这是当前项目准备给队友直接阅读和运行的最小代码包。日常只需要看下面 4 个文件：

1. `run_basic_repro.py`
2. `osl_repro/model.py`
3. `osl_repro/evaluation.py`
4. `osl_repro/datasets.py`

## 代码主链

整体流程和原论文保持一致：

1. 从 `data/data_processed/*.npz` 读取统一格式的数据 `X`
2. 求稀疏目标码 `Y*`
3. 学稀疏 lifting operator `W*`
4. 用 `W*` 对新样本编码
5. 用 `precision@k` 评测原空间和输出空间的近邻保持情况

最重要的代码位置：

- `run_basic_repro.py`
  实验入口，负责加载数据、采样、训练、评测、写结果
- `osl_repro/model.py -> solve_optimal_sparse_codes`
  原论文 step 1，求 `Y*`
- `osl_repro/model.py -> learn_sparse_lifting_operator`
  原论文 step 2，学 `W*`
- `osl_repro/model.py -> encode_with_lifting_operator`
  推理阶段，用 `W` 编码
- `osl_repro/evaluation.py -> precision_at_k_from_features`
  近邻保持评测

## 和原论文一致的地方

- 训练还是两阶段：先解 `Y*`，再解 `W*`
- step 1 还是让 `Y^T Y` 逼近 `X^T X`
- step 2 还是让 `W X` 逼近 `Y*`
- 编码还是 `W x` 后做 top-k 二值化
- 评测还是看原空间近邻和输出空间近邻的重合度

## 我们刻意做的简化

- 这不是原论文官方代码，而是按论文逻辑整理的复现版
- 求解器用的是轻量的 Frank-Wolfe 风格更新 + hard top-k 投影
- 没有逐字实现论文里的完整正则项和全部约束
- 默认实验规模偏小，目的是让链路容易跑通、容易检查
- 数据统一落成 `data/data_processed/*.npz`，后续实验不再重复 raw 预处理

## Proposal 的改动

我们的真正改动只发生在第二阶段 `W*` 的学习上。

原始复现版：

- step 1: 求 `Y*`
- step 2: 用 Euclidean 损失学 `W*`
  `||W X - Y*||_F^2`

Proposal 版：

- step 1 保持不变
- step 2 把 Euclidean 损失替换成 Mahalanobis 加权损失

对应代码：

- `osl_repro/model.py -> estimate_mahalanobis_statistics`
- `osl_repro/model.py -> learn_sparse_lifting_operator`

当前实现选择：

- `Sigma` 在 lifted code space 中由 `Y*` 估计
- 用 `Sigma + lambda I` 做正则化
- 支持三种模式：
  - `euclidean`
  - `mahalanobis_diag`
  - `mahalanobis_full`

这样链路很干净：

- `Y*` 不变
- 推理不变
- 只有 `W*` 的学习规则变了

## 运行

先跑原始 Euclidean 版：

```bash
python script_open/run_basic_repro.py \
  --dataset mnist \
  --operator-metric euclidean
```

再跑 proposal 的 diagonal Mahalanobis 版：

```bash
python script_open/run_basic_repro.py \
  --dataset mnist \
  --operator-metric mahalanobis_diag \
  --covariance-reg 1e-2
```

如果要跑高光谱数据：

```bash
python script_open/run_basic_repro.py \
  --dataset paviaU \
  --operator-metric mahalanobis_full \
  --covariance-reg 1e-2
```

如果只想看方法实现，看 `osl_repro/model.py`。  
如果只想看完整实验链路，看 `run_basic_repro.py`。
