import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import load_workbook

data = pd.read_excel("LR2.xlsx")
points = data[['x', 'y']].values

def distance(a, b, metric='euclidean'): # тут все эти расстояния 
    if metric == 'euclidean':
        return np.sqrt(np.sum((a - b)**2))
    elif metric == 'manhattan':
        return np.sum(np.abs(a - b))
    elif metric == 'chebyshev':
        return np.max(np.abs(a - b))
    else:
        raise ValueError("Unknown metric")

def k_means(points, k, metric='euclidean', max_iter=100, tol=1e-4): # туту к-минс
    np.random.seed(42)
    centers = points[np.random.choice(points.shape[0], k, replace=False)]
    
    for tochka in range(max_iter):
        clusters = [[] for tochka in range(k)] # Назначаем каждую точку к ближайшему центру
        for p in points:
            dists = [distance(p, c, metric) for c in centers]
            cluster_idx = np.argmin(dists)
            clusters[cluster_idx].append(p)

        clusters = [np.array(c) for c in clusters if len(c) > 0] # Удаляем пустые кластеры, если есть
        if len(clusters) < k:
            centers = points[np.random.choice(points.shape[0], k, replace=False)] # Если кластеров стало меньше из-за пустоты, заново инициализируем
            continue

        new_centers = [] # Пересчитываем центры как среднее точек в кластере
        for c in clusters:
            new_centers.append(np.mean(c, axis=0))
        new_centers = np.array(new_centers)

        shift = np.sqrt(np.sum((centers - new_centers)**2)) # Проверка сходимости
        centers = new_centers
        if shift < tol:
            break
    
    return clusters, centers

def cluster_statistics(clusters, centers, metric='euclidean'):
    total_variance = 0.0 # Ето суммарного квадратичного отклонения
    cluster_stats = []
    for i, c in enumerate(clusters):
        dists = np.array([distance(p, centers[i], metric) for p in c])
        variance = np.mean(dists**2)      # тута дисперсия
        radius = np.max(dists)            # а тута радиус
        total_variance_cluster = np.sum(dists**2) # здеся суммарное квадратичное отклонение
        total_variance += total_variance_cluster
        cluster_stats.append({
            'cluster': i+1,
            'variance': variance,
            'radius': radius
        })
    return cluster_stats, total_variance

k_values = [2, 3, 4]
metrics = ['euclidean', 'manhattan', 'chebyshev']
fig, axes = plt.subplots(len(metrics), len(k_values), figsize=(12, 12))
fig.suptitle('K-means Clustering Results for Different Distances and K', fontsize=16)
results = []

for i, metric in enumerate(metrics):
    for j, k in enumerate(k_values):
        clusters, centers = k_means(points, k, metric=metric)
        c_stats, total_var = cluster_statistics(clusters, centers, metric=metric)

        results.append({
            'metric': metric,
            'k': k,
            'cluster_statistics': c_stats,
            'total_variance': total_var,
            'centers': centers
        })

        ax = axes[i, j] # Визуализация графиков типа ага
        colors = plt.cm.get_cmap('tab10', k)
        for idx, c in enumerate(clusters):
            ax.scatter(c[:,0], c[:,1], s=20, color=colors(idx), label=f'Cluster {idx+1}')
        ax.scatter(centers[:,0], centers[:,1], s=200, color='black', marker='x', linewidths=1.0, label='Centroids')
        ax.set_title(f'{metric.capitalize()}, k={k}')
        ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

for res in results: # результаты после графиков
    print(f"=== Metric: {res['metric'].capitalize()}, K={res['k']} ===")
    print("Centers:")
    for idx, center in enumerate(res['centers']):
        print(f"  Centroid {idx+1}: x={center[0]:.4f}, y={center[1]:.4f}")
    for c_stat in res['cluster_statistics']:
        print(f"Cluster {c_stat['cluster']}: Variance={c_stat['variance']:.4f}, Radius={c_stat['radius']:.4f}")
    print(f"Total Variance (Sum of squared distances)={res['total_variance']:.4f}\n")
