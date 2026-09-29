# DEPRECATED: This module uses brute-force O(n^2) intersection detection and
# KMeans clustering. It has been superseded by the embodied_mixup package
# which uses KD-Tree dual-threshold search + DBSCAN clustering + Plotly
# interactive visualization. See:
#   - embodied_mixup.index_eef for neighbor lookup
#   - embodied_mixup.clustering for DBSCAN clustering
#   - embodied_mixup.visualization for Plotly-based visualization
#   - embodied_mixup.cli for command-line entry points
#
# This file is retained for reference and comparison only. Do not use for
# production work. Use `python -m embodied_mixup build` and
# `python -m embodied_mixup visualize` instead.

import matplotlib.pyplot as plt
import torch
import numpy as np
import umap
import seaborn as sns
import argparse

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import default_collate
from tqdm import tqdm
from itertools import combinations

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ====================== 全局设置：seaborn 风格 ======================
sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "white"


class EpisodeLeRobotDataset(torch.utils.data.Dataset):
    def __init__(self, lerobot_dataset):
        self.dataset = lerobot_dataset
        self.num_episodes = lerobot_dataset.num_episodes

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, episode_index):
        from_idx = self.dataset.meta.episodes["dataset_from_index"][episode_index]
        to_idx = self.dataset.meta.episodes["dataset_to_index"][episode_index]
        return default_collate([self.dataset[i] for i in range(from_idx, to_idx)])


def find_intersection_details(
    all_actions,
    episode_data_list,
    xyz_threshold=0.02,    # 位置阈值
    rpy_threshold=0.05,    # 姿态阈值
):
    """
    双阈值交点检测：
    1. 先判断 xyz 距离 < xyz_threshold
    2. 再判断 rpy 距离 < rpy_threshold
    3. 同时满足才算交点
    """
    intersections = []

    for (ep_i, ep_j) in combinations(range(len(all_actions)), 2):
        act_i = all_actions[ep_i]
        act_j = all_actions[ep_j]
        data_i = episode_data_list[ep_i]
        data_j = episode_data_list[ep_j]

        traj_i_xyz = act_i[:, :3]
        traj_i_rpy = act_i[:, 3:6]

        traj_j_xyz = act_j[:, :3]
        traj_j_rpy = act_j[:, 3:6]

        for idx_p, (p_xyz, p_rpy) in enumerate(zip(traj_i_xyz, traj_i_rpy)):
            for idx_q, (q_xyz, q_rpy) in enumerate(zip(traj_j_xyz, traj_j_rpy)):
                # 1. 计算 XYZ 距离
                dist_xyz = np.linalg.norm(p_xyz - q_xyz)
                if dist_xyz > xyz_threshold:
                    continue

                # 2. 计算 RPY 距离
                dist_rpy = np.linalg.norm(p_rpy - q_rpy)
                if dist_rpy > rpy_threshold:
                    continue

                # 3. 同时满足，保存交点
                img_i_base = data_i['observation.images.base_camera'][idx_p].cpu().numpy().transpose(1, 2, 0)
                img_i_hand = data_i['observation.images.hand_camera'][idx_p].cpu().numpy().transpose(1, 2, 0)
                img_j_base = data_j['observation.images.base_camera'][idx_q].cpu().numpy().transpose(1, 2, 0)
                img_j_hand = data_j['observation.images.hand_camera'][idx_q].cpu().numpy().transpose(1, 2, 0)

                intersections.append({
                    'episode_pair': (ep_i, ep_j),
                    'point_indices': (idx_p, idx_q),
                    'intersect_point_3d': (p_xyz + q_xyz) / 2,
                    'p': np.concatenate([p_xyz, p_rpy]),
                    'q': np.concatenate([q_xyz, q_rpy]),
                    'distance_xyz': dist_xyz,
                    'distance_rpy': dist_rpy,
                    'images': {
                        'ep_i_base': img_i_base,
                        'ep_i_hand': img_i_hand,
                        'ep_j_base': img_j_base,
                        'ep_j_hand': img_j_hand,
                    }
                })

    return intersections


def deduplicate_intersections(intersections):
    if not intersections:
        return []

    from collections import defaultdict
    pair_groups = defaultdict(list)
    for inter in intersections:
        epi, epj = inter['episode_pair']
        p, q = inter['point_indices']
        pair_groups[(epi, epj)].append((p, q, inter))

    cleaned = []
    window_size = 3

    for (epi, epj), group in pair_groups.items():
        if not group:
            continue

        group_sorted = sorted(group, key=lambda x: (x[0], x[1]))
        n = len(group_sorted)
        visited = [False] * n

        for i in range(n):
            if visited[i]:
                continue

            p_i, q_i, inter_i = group_sorted[i]
            current_cluster = [inter_i]
            visited[i] = True

            for j in range(i + 1, n):
                if visited[j]:
                    continue
                p_j, q_j, inter_j = group_sorted[j]

                if (abs(p_j - p_i) <= window_size) or (abs(q_j - q_i) <= window_size):
                    current_cluster.append(inter_j)
                    visited[j] = True
                    p_i, q_i = p_j, q_j

            best = min(current_cluster, key=lambda x: x['distance_xyz'])
            cleaned.append(best)

    return cleaned


def cluster_and_select_intersections(
    intersections,
    n_clusters=3,
):
    if len(intersections) < n_clusters:
        print(f"交点数 {len(intersections)} 小于聚类数 {n_clusters}，调整聚类数")
        n_clusters = max(1, len(intersections))

    features = []
    for inter in intersections:
        feat = (inter['p'] + inter['q']) / 2
        features.append(feat)

    X = np.array(features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    selected = []
    cluster_info = []

    for cluster_id in range(n_clusters):
        mask = (labels == cluster_id)
        cluster_indices = np.where(mask)[0]
        cluster_size = len(cluster_indices)

        if cluster_size == 0:
            continue

        cluster_center = kmeans.cluster_centers_[cluster_id]
        dists = np.linalg.norm(X_scaled[mask] - cluster_center, axis=1)
        best_idx_in_cluster = cluster_indices[np.argmin(dists)]

        best_inter = intersections[best_idx_in_cluster]
        best_inter['cluster_id'] = cluster_id
        best_inter['cluster_size'] = cluster_size

        selected.append(best_inter)
        cluster_info.append({
            'cluster_id': cluster_id,
            'size': cluster_size,
            'representative': best_inter
        })

    return selected, cluster_info, labels


def plot_intersection_images_by_cluster(
    selected,
    cluster_info,
    max_display_per_cluster=1,
    save_path=None,
    show=True
):
    n_clusters = len(cluster_info)
    if n_clusters == 0:
        print("无交点可展示")
        return

    fig, axes = plt.subplots(
        n_clusters, 4,
        figsize=(16, 3.5 * n_clusters),
        gridspec_kw={'hspace': 0.5, 'wspace': 0.05}
    )
    if n_clusters == 1:
        axes = axes[np.newaxis, :]

    for row_idx, info in enumerate(cluster_info):
        cid = info['cluster_id']
        size = info['size']
        inter = info['representative']

        ep_i, ep_j = inter['episode_pair']
        idx_p, idx_q = inter['point_indices']
        d_xyz = inter['distance_xyz']
        d_rpy = inter['distance_rpy']

        imgs = inter['images']
        titles = [
            f'Cluster {cid} (Size={size})\nEp{ep_i} Base Pt{idx_p}\nxyz={d_xyz:.3f}',
            f'Ep{ep_i} Hand Pt{idx_p}\nrpy={d_rpy:.3f}',
            f'Ep{ep_j} Base Pt{idx_q}',
            f'Ep{ep_j} Hand Pt{idx_q}'
        ]
        img_list = [imgs['ep_i_base'], imgs['ep_i_hand'], imgs['ep_j_base'], imgs['ep_j_hand']]

        for col in range(4):
            ax = axes[row_idx, col]
            img = img_list[col]
            ax.imshow(img)
            ax.set_title(titles[col], fontsize=9)
            ax.axis('off')

    plt.suptitle(f'Intersection Clusters (Total {n_clusters} types)', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_2d_projection_with_clusters(
    all_actions,
    intersections,
    cluster_labels,
    method='pca',
    save_path=None,
    show=True
):
    episode_labels = []
    points_all = []
    inter_points = []

    for ep_idx, act in enumerate(all_actions):
        pts = act[:, :6]
        points_all.append(pts)
        episode_labels.extend([ep_idx] * len(pts))

    for inter in intersections:
        p = inter['p']
        q = inter['q']
        inter_points.append(p)
        inter_points.append(q)

    X = np.concatenate(points_all, axis=0)
    inter_points = np.array(inter_points) if len(inter_points) > 0 else None

    print(f"\n正在使用 {method.upper()} 降维到2D...")

    if inter_points is not None and len(inter_points) > 0:
        combined = np.concatenate([X, inter_points], axis=0)
    else:
        combined = X

    if method == 'pca':
        model = PCA(n_components=2, random_state=42)
    elif method == 'tsne':
        model = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
    elif method == 'umap':
        model = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("method must be pca / tsne / umap")

    combined_2d = model.fit_transform(combined)

    Z_normal = combined_2d[:len(X)]
    Z_inter = combined_2d[len(X):] if inter_points is not None and len(inter_points) > 0 else np.array([])

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_actions)))

    ptr = 0
    for ep_idx, pts in enumerate(all_actions):
        n = len(pts)
        z_ep = Z_normal[ptr:ptr + n]
        ptr += n
        ax.plot(z_ep[:, 0], z_ep[:, 1], color=colors[ep_idx], linewidth=1.2, alpha=0.5, label=f"Ep{ep_idx}" if ep_idx < 10 else "")
        ax.scatter(z_ep[0, 0], z_ep[0, 1], color=colors[ep_idx], s=50, edgecolors='k', alpha=0.7)
        ax.scatter(z_ep[-1, 0], z_ep[-1, 1], color=colors[ep_idx], s=50, marker='x', alpha=0.7)

    if len(Z_inter) > 0:
        unique_labels = np.unique(cluster_labels)
        cluster_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

        for i, cid in enumerate(unique_labels):
            mask = np.array([cid == l for l in cluster_labels for _ in range(2)])
            z_c = Z_inter[mask]
            ax.scatter(z_c[:, 0], z_c[:, 1],
                       color=cluster_colors[i], s=160, marker='*', edgecolors='k', linewidth=2,
                       label=f'Cluster {cid}')

    ax.set_title(f"2D {method.upper()} Projection (XYZ+RPY) | Clustered Intersections", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


# ======================== 主程序 ========================
def main():
    parser = argparse.ArgumentParser(description="轨迹交点分析与聚类可视化（双阈值版）")

    # 数据集参数
    parser.add_argument('--repo_id', type=str, help='LeRobot数据集 repo_id')
    parser.add_argument('--root', type=str, help='数据集本地路径')

    # 🔥 双阈值参数
    parser.add_argument('--xyz_threshold', type=float, default=0.02, help='XYZ位置距离阈值')
    parser.add_argument('--rpy_threshold', type=float, default=0.01, help='RPY姿态距离阈值')

    # 降维参数
    parser.add_argument('--projection_method', type=str, default='tsne', choices=['pca', 'tsne', 'umap'], help='降维方法')

    # 聚类参数
    parser.add_argument('--n_clusters', type=int, default=10, help='聚类数量')

    # 保存路径
    parser.add_argument('--save_traj_path', type=str, default='traj_2d_clusters.png', help='2D轨迹图保存路径')
    parser.add_argument('--save_img_path', type=str, default='intersection_clusters.png', help='交点聚类图保存路径')

    # 显示控制
    parser.add_argument('--show', action='store_true', default=True, help='显示图片')
    parser.add_argument('--no_show', action='store_false', dest='show', help='不显示图片')

    args = parser.parse_args()

    # 加载数据集
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
    )
    episode_dataset = EpisodeLeRobotDataset(dataset)

    episode_data_list = []
    all_actions = []

    print("Loading episodes...")
    for episode_data in tqdm(episode_dataset):
        episode_data_list.append(episode_data)
        act = episode_data['action'].cpu().numpy()[:, :6]
        all_actions.append(act)

    # 1. 双阈值找交点
    raw_intersections = find_intersection_details(
        all_actions,
        episode_data_list,
        xyz_threshold=args.xyz_threshold,
        rpy_threshold=args.rpy_threshold,
    )

    # 2. 去重
    intersections = deduplicate_intersections(raw_intersections)

    print(f"\n原始交点数: {len(raw_intersections)}")
    print(f"去重后交点数: {len(intersections)}")

    # 3. 聚类
    if intersections:
        selected, cluster_info, cluster_labels = cluster_and_select_intersections(
            intersections,
            n_clusters=args.n_clusters,
        )

        print(f"\n===== 聚类分析结果 =====")
        for info in cluster_info:
            cid = info['cluster_id']
            size = info['size']
            inter = info['representative']
            epi, epj = inter['episode_pair']
            pi, pj = inter['point_indices']
            d_xyz = inter['distance_xyz']
            d_rpy = inter['distance_rpy']
            print(f"簇 {cid}: 共 {size} 个 | 代表: Ep{epi}Pt{pi}<->Ep{epj}Pt{pj} | xyz={d_xyz:.3f} rpy={d_rpy:.3f}")

        # 4. 绘图
        plot_2d_projection_with_clusters(
            all_actions,
            intersections,
            cluster_labels,
            method=args.projection_method,
            save_path=args.save_traj_path,
            show=args.show
        )

        plot_intersection_images_by_cluster(
            selected,
            cluster_info,
            save_path=args.save_img_path,
            show=args.show
        )
    else:
        print("未找到满足双阈值的交点")


if __name__ == "__main__":
    main()