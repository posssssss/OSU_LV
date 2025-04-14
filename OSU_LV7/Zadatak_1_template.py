import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering


def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X
#1)
# Generiranje i prikaz podataka za različite vrijednosti flagc
for flagc in range(1, 6):
    X = generate_data(500, flagc)
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c='cyan')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(f'podatkovni primjeri za flagc={flagc}')
    plt.show()

#2)
# Generiranje podataka (primjer za flagc=2)
X = generate_data(500, 2)

# Postavite broj klastera (K)
K = 3  # Promijenite vrijednost K kako biste vidjeli različite rezultate

# Primjena K-means algoritma
kmeans = KMeans(n_clusters=K, random_state=0)
labels = kmeans.fit_predict(X)

# Vizualizacija grupiranja
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title(f'K-means clustering with K={K}')
plt.colorbar(label='Cluster')
plt.show()

