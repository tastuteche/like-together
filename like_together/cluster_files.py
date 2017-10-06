"""
Groups (clusters) similar files together from a dir
using k-means clustering algorithm.

Also does some simple cleaning (such as removing white space and replacing numbers with (N)).

Example:

    python cluster_files.py --clusters 20 invalid_dates.txt

Required libs:

    click
    sklearn
"""
# https://gist.github.com/vaultah/d63cb4c86be2774377aa674b009f759a
import sys
import importlib
from pathlib import Path


def import_parents(level=1):
    global __package__
    file = Path(__file__).resolve()
    parent, top = file.parent, file.parents[level]

    sys.path.append(str(top))
    try:
        sys.path.remove(str(parent))
    except ValueError:  # already removed
        pass

    __package__ = '.'.join(parent.parts[len(top.parts):])
    importlib.import_module(__package__)  # won't be needed after that


if __name__ == '__main__' and __package__ is None:
    import_parents(level=1)


import click


import matplotlib
matplotlib.use('QT4Agg', force=True)


@click.command()
@click.argument('dir_name')
@click.option('--num_clusters', default=10, help='Number of clusters')
@click.option('--sample', default=100, help='Number of samples to print')
def cluster_files(dir_name, num_clusters, sample):
    from .tfidf import get_tfidf, load_tfidf
    # get_tfidf(dir_name)
    doc_feat, filename_list = load_tfidf()

    from .kmeans import all_k
    max_sil_k = all_k(doc_feat, filename_list, sample)
    print("max_sil_k:", max_sil_k)

    from .ward import figure_ward
    figure_ward(doc_feat, filename_list)


if __name__ == '__main__':
    cluster_files()
