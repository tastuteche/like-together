from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


def compute_elbow(km, X):
    centroids = km.cluster_centers_
    dt_trans = X.toarray()
    D = cdist(dt_trans, centroids, 'euclidean')
    cIdx = np.argmin(D, axis=1)
    dist = np.min(D, axis=1)
    avgWithinSS = sum(dist) / dt_trans.shape[0]

    wcss = sum(dist**2)
    tss = sum(pdist(dt_trans)**2) / dt_trans.shape[0]
    bss = tss - wcss
    return (avgWithinSS, bss / tss * 100)


def figure_elbow(X):
    K = range(1, 50)
    KM = [KMeans(n_clusters=k).fit(X) for k in K]
    centroids = [k.cluster_centers_ for k in KM]

    dt_trans = X.toarray()
    D_k = [cdist(dt_trans, cent, 'euclidean') for cent in centroids]
    cIdx = [np.argmin(D, axis=1) for D in D_k]
    dist = [np.min(D, axis=1) for D in D_k]
    avgWithinSS = [sum(d) / dt_trans.shape[0] for d in dist]

    # Total with-in sum of square
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(dt_trans)**2) / dt_trans.shape[0]
    bss = tss - wcss

    kIdx = 10 - 1

    # elbow curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(K, avgWithinSS, 'b*-')
    ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12,
            markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow for KMeans clustering')
    plt.savefig('figures/kmeans_elbow_avg_sum_squares.png', dpi=200)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(K, bss / tss * 100, 'b*-')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Percentage of variance explained')
    plt.title('Elbow for KMeans clustering')
    plt.savefig('figures/kmeans_elbow_percentage_variance.png', dpi=200)


def main():
    import pickle
    doc_feat = pickle.load(open("doc_feat.pickle", "rb"))
    filename_list = pickle.load(open("filename_list.pickle", "rb"))
    figure_elbow(doc_feat)


if __name__ == '__main__':
    main()
