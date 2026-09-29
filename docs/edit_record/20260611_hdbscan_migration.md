# HDBSCAN 聚类迁移记录

- **日期**: 2026-06-11
- **影响范围**: `embodied_mixup/clustering.py`, `embodied_mixup/config.py`, `embodied_mixup/cli.py`

## 变更动机

原有 DBSCAN 聚类（eps=0.04, min_samples=3）在 StackCube-v1 数据集上产出了 **1,204 个簇**，远超预期。可视化观察发现大量簇在空间中距离非常小，彼此之间近似联通，属于典型的 DBSCAN 过分割问题。

根本原因：DBSCAN 使用全局统一密度阈值，无法适应轨迹交点在不同空间区域密度差异较大的实际情况。轨迹交叉密集区（如物体堆叠点附近）交点密度远高于稀疏区域，单一 eps 无法同时处理两者。

## 解决方案：HDBSCAN

HDBSCAN 通过构建聚类层次结构，自动适应变密度数据，是 DBSCAN 的直接升级方案。

### 新增依赖

- `hdbscan>=0.8.0`（已安装 0.8.44）

## 聚类对象说明

**重要：聚类对象是交点对的 6D 中点，而非拆分的单个点。**

每个 neighbor pair 包含 anchor 帧的 EEF pose 和 neighbor 帧的 EEF pose：
```
pair = {"anchor_pose": [x_a, y_a, z_a, ax_a, ay_a, az_a],
        "neighbor_pose": [x_n, y_n, z_n, ax_n, ay_n, az_n]}
```

聚类的原始数据点为两者的中点：
```
midpoint = (anchor_pose + neighbor_pose) / 2  # shape: (6,)
```

14,801 个 neighbor pairs → 14,801 个 6D 中点作为聚类输入。

**这样做而不是拆分点的原因：**
- 中点代表了轨迹交点的空间位置，语义上是"两条轨迹相遇的位置"
- 每个 neighbor pair 已经是一个模糊匹配的实例，不能将 anchor_pose 和 neighbor_pose 各自拆出作为独立点重新聚类 — 原始配对信息会丢失
- DBSCAN/HDBSCAN 需要的是"模糊区域"的空间密度分布，即哪些区域有大量轨迹在此交汇，中点很好地表达了这一信息

## 参数迁移对照

| DBSCAN | HDBSCAN | 值 |
|--------|---------|-----|
| eps=0.04 | cluster_selection_epsilon=0.05 | 后处理合并阈值 |
| min_samples=3 | min_cluster_size=int(N*0.01) | 簇最小大小（总点数的 1%） |
| - | min_samples=3 | 核心点最小邻居数（保留原 DBSCAN 语义） |

### 新增 CLI 参数

- `--cluster_selection_epsilon`: 控制后处理簇合并阈值（默认 0.05）
- `--hdbscan_alpha`: HDBSCAN 簇选择稳定性（默认 1.0）
- `--hdbscan_min_cluster_size_ratio`: min_cluster_size 相对于总点数的比例（默认 0.01 = 1%）

### 移除 CLI 参数

- `--dbscan_min_samples`: 被 HDBSCAN 的 min_samples 替代
- 保留 `--dbscan_eps` 作为 `--cluster_selection_epsilon` 的别名以兼容旧脚本

## HDBSCAN 参数说明

| 参数 | 含义 | 对"千簇问题"的作用 |
|------|------|-------------------|
| min_cluster_size | 成为簇的最小样本数 | 过滤微簇：小于此值的不能独立成簇 |
| min_samples | 成为核心点的最小邻居数 | 控制密度敏感度：越小越激进 |
| cluster_selection_epsilon | 树图截断合并阈值 | **核心参数**：合并距离<=此值的近联通簇 |
| alpha | 簇选择稳定性 | 越大簇越大越稳健 |
