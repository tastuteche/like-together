import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict
from .bic import compute_bic, figure_bic
from .silhouette import figure_sil


def save_kmeans(km, titles, n_cluster, sample):
    with open("figures/km_%s.txt" % n_cluster, "w") as f:
        k = 0
        clusters = defaultdict(list)
        for i in km.labels_:
            clusters[i].append(titles[k])
            k += 1

        s_clusters = sorted(clusters.values(), key=lambda l: -len(l))
        print('==============', file=f)
        for cluster in s_clusters:
            print('Cluster [%s]:' % len(cluster), file=f)
            if len(cluster) > sample:
                cluster = random.sample(cluster, sample)
            for title in cluster:
                print(title, file=f)
            print('--------', file=f)


def all_k(X, titles, sample):
    dic = {}
    bics = []
    s = []
    for n_cluster in range(2, 59):
        km = KMeans(n_clusters=n_cluster).fit(X)
        label = km.labels_
        sil_coeff = silhouette_score(X, label, metric='euclidean')
        s.append(sil_coeff)
        bic = compute_bic(km, X)
        bics.append(bic)
        print("For n_clusters={}, The Silhouette Coefficient is {}|{}".format(
            n_cluster, sil_coeff, bic))
        dic[n_cluster] = sil_coeff
        save_kmeans(km, titles, n_cluster, sample)

    figure_bic(bics)
    figure_sil(s)

    for (k, v) in sorted(dic.items(), key=lambda x: x[1], reverse=True):
        return k
