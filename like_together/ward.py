import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def figure_ward(doc_feat, titles):
    dist = 1 - cosine_similarity(doc_feat)
    from scipy.cluster.hierarchy import ward, dendrogram

    # define the linkage_matrix using ward clustering pre-computed distances
    linkage_matrix = ward(dist)

    fig, ax = plt.subplots(figsize=(15, 20))  # set size
    ax = dendrogram(linkage_matrix, orientation="right", labels=titles)

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')

    plt.tight_layout()  # show plot with tight layout

    # uncomment below to save figure
    # save figure as ward_clusters
    plt.savefig('figures/ward_clusters.png', dpi=200)
