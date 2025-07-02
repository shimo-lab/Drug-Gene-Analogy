import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from sklearn.decomposition import PCA


def main():

    model_file = 'output/embeddings/concept_skipgram.pkl'
    with open(model_file, 'rb') as f:
        model = pkl.load(f)

    rel_df = pd.read_csv('output/relation/bcv_drug2genes_relation.csv')
    all_rels = []
    for _, row in rel_df.iterrows():
        _, _, drug_concept, _, gene_concepts = row
        gene_concepts = eval(gene_concepts)
        for gene_concept in gene_concepts:
            all_rels.append((drug_concept, gene_concept))
    assert len(all_rels) == len(set(all_rels))

    # sample 200 relations
    all_rels = np.array(all_rels)
    seed = 0
    np.random.seed(seed)
    idx = np.random.choice(len(all_rels), 200, replace=False)
    rels = all_rels[idx]

    drug_cpts = []
    gene_cpts = []
    for drug_cpt, gene_cpt in rels:
        drug_cpts.append(drug_cpt)
        gene_cpts.append(gene_cpt)

    dvecs = []
    for cpt in drug_cpts:
        dvecs.append(np.array(model[cpt]))
    gvecs = []
    for cpt in gene_cpts:
        gvecs.append(np.array(model[cpt]))
    vecs = np.array(dvecs + gvecs)
    pca = PCA(n_components=2)
    pca.fit(vecs)
    vecs = pca.transform(vecs)

    all_dvecs = []
    all_gvecs = []
    for drug_cpt, gene_cpt in all_rels:
        all_dvecs.append(np.array(model[drug_cpt]))
        all_gvecs.append(np.array(model[gene_cpt]))
    mean_dvec = np.mean(np.array(all_dvecs), axis=0)
    mean_dvec = pca.transform(mean_dvec.reshape(1, -1))[0]
    mean_gvec = np.mean(np.array(all_gvecs), axis=0)
    mean_gvec = pca.transform(mean_gvec.reshape(1, -1))[0]

    cpt2vec = dict()
    for cpt, vec in zip(drug_cpts + gene_cpts, vecs):
        cpt2vec[cpt] = vec

    plt.style.use("ggplot")
    plt.subplots_adjust(wspace=0, hspace=0)
    fig = plt.figure(figsize=(10, 8))
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.15, top=0.99)
    plot = plt.subplot()
    plot.xaxis.set_major_locator(MultipleLocator(1.0))
    plot.yaxis.set_major_locator(MultipleLocator(1.0))

    # drug
    legend_flag = True
    for cpt in drug_cpts:
        vec = cpt2vec[cpt]
        if legend_flag:
            plot.scatter(vec[0], vec[1], marker='^',
                         color='blue', s=125, label='Drug')
            legend_flag = False
        else:
            plot.scatter(vec[0], vec[1], marker='^', color='blue', s=125)

    # gene
    legend_flag = True
    for cpt in gene_cpts:
        vec = cpt2vec[cpt]
        if legend_flag:
            plot.scatter(vec[0], vec[1], marker='o',
                         color='orange', s=125, label='Gene')
            legend_flag = False
        else:
            plot.scatter(vec[0], vec[1], marker='o', color='orange', s=125)

    # relation
    for drug_cpt, gene_cpt in rels:
        vec1 = cpt2vec[drug_cpt]
        vec2 = cpt2vec[gene_cpt]
        plot.plot([vec1[0], vec2[0]], [vec1[1], vec2[1]], c='black',
                  linewidth=0.25)

    # mean vector
    plot.scatter(mean_dvec[0], mean_dvec[1], marker='*',
                 color='red', s=250, label='Drug Mean', zorder=20)
    plot.scatter(mean_gvec[0], mean_gvec[1], marker='s',
                 color='red', s=200, label='Gene Mean', zorder=20)
    plot.plot([mean_dvec[0], mean_gvec[0]],
              [mean_dvec[1], mean_gvec[1]],
              c='red',
              linewidth=5, linestyle='--',)

    # equal aspect ratio
    plot.set_aspect('equal')
    plot.tick_params(labelsize=25)
    plot.set_xlabel('PC1', fontsize=25)
    plot.set_ylabel('PC2', fontsize=25)
    plt.xlim(-2.9, 3.5)
    plt.ylim(-2.5, 2.5)
    plot.legend(loc='lower right', fontsize=20,
                facecolor='white', framealpha=1.0)
    # no padding
    plt.tight_layout()
    Path('output/images').mkdir(parents=True, exist_ok=True)
    fig.savefig(f'output/images/relation_samples_seed{seed}.pdf',
                pad_inches=0.0)
    plt.close()


if __name__ == '__main__':
    main()
