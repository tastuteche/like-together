import matplotlib.pyplot as plt
import seaborn as sns


def figure_sil(sil_coeffs):
    plt.plot(sil_coeffs)
    plt.ylabel("Silouette")
    plt.xlabel("k")
    plt.title("Silouette for K-means cell's behaviour")
    sns.despine()
    plt.savefig('figures/K-means-Silhouette.pdf', format='pdf',
                dpi=330, bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()
