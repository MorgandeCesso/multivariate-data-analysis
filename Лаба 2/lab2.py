import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Загрузка данных
data = pd.read_excel("Lab2Data.xlsx")
points = data[['x', 'y']].values

# Функция вычисления расстояния
def distance(a, b, metric='euclidean'):
    if metric == 'euclidean':
        return np.sqrt(np.sum((a - b)**2))
    elif metric == 'manhattan':
        return np.sum(np.abs(a - b))
    elif metric == 'chebyshev':
        return np.max(np.abs(a - b))
    else:
        raise ValueError("Неизвестная метрика")

# Алгоритм k-средних
def k_means(points, k, metric='euclidean', max_iter=100, tol=1e-4):
    np.random.seed(42)
    centers = points[np.random.choice(points.shape[0], k, replace=False)]
    
    for iteration in range(max_iter):
        clusters = [[] for _ in range(k)]
        for p in points:
            dists = [distance(p, c, metric) for c in centers]
            cluster_idx = np.argmin(dists)
            clusters[cluster_idx].append(p)

        clusters = [np.array(c) for c in clusters if len(c) > 0]
        if len(clusters) < k:
            centers = points[np.random.choice(points.shape[0], k, replace=False)]
            continue

        new_centers = np.array([np.mean(c, axis=0) for c in clusters])
        shift = np.sqrt(np.sum((centers - new_centers)**2))
        centers = new_centers
        if shift < tol:
            break

    return clusters, centers

# Расчет статистики кластеров
def cluster_statistics(clusters, centers, metric='euclidean'):
    total_variance = 0.0
    cluster_stats = []
    
    for i, cluster in enumerate(clusters):
        dists = np.array([distance(p, centers[i], metric) for p in cluster])
        variance = np.mean(dists**2)
        radius = np.max(dists)
        total_variance += np.sum(dists**2)
        cluster_stats.append({
            'cluster': i+1,
            'variance': variance,
            'radius': radius
        })

    return cluster_stats, total_variance

# Визуализация результатов и расчет метрик
k_values = [2, 3, 4]
metrics = ['euclidean', 'manhattan', 'chebyshev']
results = []

# Сначала выполняем все вычисления и выводим результаты
for i, metric in enumerate(metrics):
    for j, k in enumerate(k_values):
        clusters, centers = k_means(points, k, metric=metric)
        c_stats, total_var = cluster_statistics(clusters, centers, metric=metric)

        silhouette = silhouette_score(points, np.concatenate([[idx] * len(c) for idx, c in enumerate(clusters)]))
        
        results.append({
            'metric': metric,
            'k': k,
            'cluster_statistics': c_stats,
            'total_variance': total_var,
            'silhouette': silhouette,
            'centers': centers,
            'clusters': clusters  # Сохраняем кластеры для последующей визуализации
        })

# Вывод результатов
for res in results:
    print(f"=== Метрика: {res['metric'].capitalize()}, K={res['k']} ===")
    print("Центроиды:")
    for idx, center in enumerate(res['centers']):
        print(f"  Центроид {idx+1}: x={center[0]:.4f}, y={center[1]:.4f}")
    for c_stat in res['cluster_statistics']:
        print(f"Кластер {c_stat['cluster']}: Дисперсия={c_stat['variance']:.4f}, Радиус={c_stat['radius']:.4f}")
    print(f"Суммарное квадратичное отклонение={res['total_variance']:.4f}")
    print(f"Силуэтный коэффициент={res['silhouette']:.4f}\n")

# Затем создаем визуализацию
fig, axes = plt.subplots(len(metrics), len(k_values), figsize=(15, 12))
fig.suptitle('Результаты кластеризации K-средних для разных метрик', fontsize=16)

for idx, res in enumerate(results):
    i = idx // len(k_values)
    j = idx % len(k_values)
    
    ax = axes[i, j]
    colors = plt.colormaps.get_cmap('tab10')(np.linspace(0, 1, res['k']))
    for cluster_idx, cluster in enumerate(res['clusters']):
        ax.scatter(cluster[:, 0], cluster[:, 1], s=20, color=colors[cluster_idx], label=f'Кластер {cluster_idx+1}')
    ax.scatter(res['centers'][:, 0], res['centers'][:, 1], s=200, color='black', marker='x', label='Центроиды')
    ax.set_title(f'Метрика: {"Евклидова" if res["metric"] == "euclidean" else "Манхэттена" if res["metric"] == "manhattan" else "Чебышева"}, k={res["k"]}\nОбщий коэффициент силуэта={res["silhouette"]:.2f}')
    ax.legend()

plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.show()
