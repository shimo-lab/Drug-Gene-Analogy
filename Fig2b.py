import pickle as pkl
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from sklearn.decomposition import PCA

from utils import readable_name


def main():

    embed = 'bcv'
    hsa = 'hsa04012'

    output_dir = Path('output/images')
    output_dir.mkdir(parents=True, exist_ok=True)

    # load model
    model_file = 'output/embeddings/concept_skipgram.pkl'
    with open(model_file, 'rb') as f:
        model = pkl.load(f)

    drug_df = pd.read_csv(f'output/pathway/{embed}_drug_pathway.csv')
    hsa_drug_df = drug_df.query(f'hsa=="{hsa}"')
    cal_Dp = list(hsa_drug_df['drug_concept'].values)
    drug_names = list(hsa_drug_df['drug_name'].values)
    assert len(cal_Dp) == len(set(cal_Dp))
    assert len(drug_names) == len(set(drug_names))
    drug_cpt2drug_name = dict(zip(cal_Dp, drug_names))

    gene_df = pd.read_csv(f'output/pathway/{embed}_gene_pathway.csv')
    hsa_gene_df = gene_df.query(f'hsa=="{hsa}"')
    cal_Gp = list(hsa_gene_df['gene_concept'].values)
    gene_names = list(hsa_gene_df['gene_name'].values)
    assert len(cal_Gp) == len(set(cal_Gp))
    assert len(gene_names) == len(set(gene_names))
    gene_cpt2gene_name = dict(zip(cal_Gp, gene_names))

    pathway_pkl = f'output/pathway/{embed}_hsa_to_drug2genes.pkl'
    with open(pathway_pkl, 'rb') as f:
        hsa2query_name_and_query_concept_and_tgt_name_concept_pair = \
            pkl.load(f)

    cal_Rp = []
    drug_and_gene_names = []
    for query_name, query_cpt, tgt_name_concept_pair in \
            hsa2query_name_and_query_concept_and_tgt_name_concept_pair[hsa]:
        for tgt_name, tgt_cpt in tgt_name_concept_pair:
            cal_Rp.append((query_cpt, tgt_cpt))
            drug_and_gene_names.append(
                f'({readable_name(query_name)}, {tgt_name})')
    assert len(cal_Rp) == len(set(cal_Rp))
    assert len(drug_and_gene_names) == len(set(drug_and_gene_names))
    drug_and_gene_names.sort()

    drug_cpt2gene_cpts = defaultdict(set)
    for query_cpt, tgt_cpt in cal_Rp:
        drug_cpt2gene_cpts[query_cpt].add(tgt_cpt)

    print('Drug set:', len(drug_names), [readable_name(drug_name)
                                         for drug_name in drug_names])
    print('Gene set:', len(gene_names), gene_names)
    print('Relation set:', len(drug_and_gene_names), drug_and_gene_names)

    # PCA
    dvecs = []
    for cpt in cal_Dp:
        dvecs.append(np.array(model[cpt]))
    gvecs = []
    for cpt in cal_Gp:
        gvecs.append(np.array(model[cpt]))

    vecs = np.array(dvecs + gvecs)
    pca = PCA(n_components=2)
    pca.fit(vecs)
    vecs = pca.transform(vecs)

    cpt2vec = dict()
    for cpt, vec in zip(cal_Dp + cal_Gp, vecs):
        cpt2vec[cpt] = vec

    plt.style.use("ggplot")
    plt.subplots_adjust(wspace=0, hspace=0)
    fig = plt.figure(figsize=(10, 8))
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.15, top=0.99)
    plot = plt.subplot()
    plot.xaxis.set_major_locator(MultipleLocator(1.0))
    plot.yaxis.set_major_locator(MultipleLocator(1.0))

    fs = 10

    # drug
    cal_Dp = sorted(cal_Dp, key=lambda cpt: -len(drug_cpt2drug_name[cpt]))
    for cpt in cal_Dp:
        vec = cpt2vec[cpt]
        if cpt not in drug_cpt2gene_cpts:
            plot.text(vec[0], vec[1], readable_name(drug_cpt2drug_name[cpt]),
                      fontsize=fs, color='blue', horizontalalignment='center')
        else:
            plot.text(vec[0], vec[1], readable_name(drug_cpt2drug_name[cpt]),
                      fontsize=fs, color='blue', horizontalalignment='center',
                      zorder=10, bbox=dict(facecolor='white',
                                           edgecolor='black',
                                           boxstyle='round,pad=0.2'))

    # gene
    cal_Gp = sorted(cal_Gp, key=lambda cpt: -len(gene_cpt2gene_name[cpt]))
    for cpt in cal_Gp:
        vec = cpt2vec[cpt]
        flag = False
        for drug_cpt in drug_cpt2gene_cpts:
            if cpt in drug_cpt2gene_cpts[drug_cpt]:
                flag = True
                break
        if flag:
            plot.text(vec[0], vec[1], gene_cpt2gene_name[cpt], fontsize=fs,
                      color='orange', horizontalalignment='center', zorder=10,
                      bbox=dict(facecolor='white', edgecolor='black',
                                boxstyle='round,pad=0.2'))
        else:
            plot.text(vec[0], vec[1], gene_cpt2gene_name[cpt], fontsize=fs,
                      color='orange', horizontalalignment='center')

    # relation
    count = 0
    for drug_cpt in drug_cpt2gene_cpts:
        vec1 = cpt2vec[drug_cpt]
        for gene_cpt in drug_cpt2gene_cpts[drug_cpt]:
            vec2 = cpt2vec[gene_cpt]
            count += 1
            plot.plot([vec1[0], vec2[0]], [vec1[1], vec2[1]], c='black',
                      linewidth=0.25)

    drug_vec_mean = []
    gene_vec_mean = []
    for drug_cpt in drug_cpt2gene_cpts:
        for gene_cpt in drug_cpt2gene_cpts[drug_cpt]:
            vec1 = cpt2vec[drug_cpt]
            vec2 = cpt2vec[gene_cpt]
            drug_vec_mean.append(vec1)
            gene_vec_mean.append(vec2)
    drug_vec_mean = np.mean(np.array(drug_vec_mean), axis=0)
    gene_vec_mean = np.mean(np.array(gene_vec_mean), axis=0)

    plot.scatter(drug_vec_mean[0], drug_vec_mean[1], marker='*', c='red',
                 s=250, zorder=20, label='Drug Mean')
    plot.scatter(gene_vec_mean[0], gene_vec_mean[1], marker='s', c='red',
                 s=200, zorder=20, label='Gene Mean')
    plot.plot([drug_vec_mean[0], gene_vec_mean[0]],
              [drug_vec_mean[1], gene_vec_mean[1]], c='red',
              linewidth=5, zorder=20, linestyle='--')

    # equal aspect ratio
    plot.set_aspect('equal')
    plot.tick_params(labelsize=25)
    plot.set_xlabel('PC1', fontsize=25)
    plot.set_ylabel('PC2', fontsize=25)
    plt.xlim(-1.9, 4.5)
    plt.ylim(-2.25, 2.75)
    plot.legend(loc='lower right', fontsize=20,
                facecolor='white', framealpha=1.0)

    plt.tight_layout()
    fig.savefig(str(output_dir / f'{embed}_{hsa}_pca.pdf'), pad_inches=0.0)
    plt.close()


if __name__ == '__main__':
    import fire
    fire.Fire(main)
