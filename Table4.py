import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Model, readable_name


def main(embed='homemade'):

    hsa = 'hsa04012'
    top = 10

    # load model
    if embed == 'bcv':
        model_file = 'output/embeddings/concept_skipgram.pkl'
        with open(model_file, 'rb') as f:
            model = pkl.load(f)

    elif embed == 'homemade':
        model_file = 'data/embeddings/years/1781-2023/min_count30/'\
            'embedding_e10s1e-05l0.025w5n5m30d300'
        model = np.fromfile(model_file, dtype=np.float64)
        model = model.reshape(-1, 300)
        word_to_id_path = \
            'data/embeddings/years/1781-2023/min_count30/word_to_id.pkl'
        with open(word_to_id_path, 'rb') as f:
            word_to_id = pkl.load(f)
        assert len(model) == len(word_to_id)
        model = Model(model, word_to_id)
    else:
        raise NotImplementedError

    # gene concept convert
    gene_conv_csv_path = 'output/gene_cpt2name/gene_cpt2name.csv'
    gene_df = pd.read_csv(gene_conv_csv_path)
    if embed == 'homemade':
        gene_df['concept'] = gene_df['concept'].apply(lambda x: x.lower())
    gene_concept2gene_name = dict(zip(gene_df['concept'], gene_df['name']))

    # query directions
    query_direction = 'drug2gene'
    centering = True

    query_prefix = 'Chemical_MESH_'
    cand_prefix = 'Gene'

    if embed == 'homemade':
        query_prefix = query_prefix.lower()
        cand_prefix = cand_prefix.lower()

    domains = []
    cands = []
    to_id = dict()
    to_word = []
    for k, v in model.items():
        if k[:len(query_prefix)] == query_prefix:
            domains.append(v)
        if k[:len(cand_prefix)] == cand_prefix:
            cands.append(v)
            to_id[k] = len(to_id)
            to_word.append(k)

    domains = np.array(domains)
    cands = np.array(cands)

    # pathway
    pathway_pkl = \
        f'output/pathway/{embed}_hsa_to_{query_direction}s.pkl'

    with open(pathway_pkl, 'rb') as f:
        hsa2relations = pkl.load(f)

    hsa_query_cpt_set = set()
    query_cpt2query_name = dict()
    for query_name, query_cpt, anss in hsa2relations[hsa]:
        hsa_query_cpt_set.add(query_cpt)
        query_cpt2query_name[query_cpt] = query_name

    # prepare queries
    rel_df = pd.read_csv(
        f'output/relation/{embed}_{query_direction}s_relation.csv')

    query_cpt_col = 'drug_concept'
    ans_cpts_col = 'gene_concepts'
    ans_names_col = 'gene_names'
    assert len(rel_df) == len(rel_df[query_cpt_col].unique())

    # define query and answer set
    query_cpt2ans_cpts = dict()
    query_cpt2ans_names = dict()
    total_ans_list = []
    for _, row in rel_df.iterrows():
        query_cpt = row[query_cpt_col]
        ans_cpts = eval(row[ans_cpts_col])
        query_cpt2ans_cpts[query_cpt] = ans_cpts
        total_ans_list.append(ans_cpts)
        query_cpt2ans_names[query_cpt] = eval(row[ans_names_col])

    # calcuate candidate norms for cosine similarity
    cands_norm = np.linalg.norm(cands, axis=-1)
    cands_norm = np.maximum(cands_norm, 1e-8)
    cands_mean = np.mean(cands, axis=0)
    cands_c = cands - cands_mean[np.newaxis, ...]
    cands_norm_c = np.linalg.norm(cands_c, axis=-1)
    cands_norm_c = np.maximum(cands_norm_c, 1e-8)

    def nearest_words(pred, n=top, centering=False):
        # size: vocab_num
        if centering:
            pred = pred - cands_mean
            cos = np.dot(cands_c, pred) / (
                np.linalg.norm(pred) * cands_norm_c)
        else:
            cos = np.dot(cands, pred) / \
                (np.linalg.norm(pred) * cands_norm)
        # top n index
        index = np.argpartition(-cos, n)[:n]
        top = index[np.argsort(-cos[index])]
        word_list = []
        sim_list = []
        for word_id in top:
            word = to_word[word_id]
            sim = cos[word_id]
            word_list.append(word)
            sim_list.append(sim)
        return word_list, sim_list

    # analogy
    # calc relation vector
    v = []
    for query_cpt, ans_cpts in query_cpt2ans_cpts.items():
        for ans_cpt in ans_cpts:
            v.append(model[ans_cpt] - model[query_cpt])
    v = np.mean(v, axis=0)

    data = []
    # prediction
    for query_cpt, ans_cpts in tqdm(
            query_cpt2ans_cpts.items()):

        if query_cpt not in hsa_query_cpt_set:
            continue

        pred = model[query_cpt] + v
        cand_cpts, _ = nearest_words(
            pred, centering=centering)

        row = {'setting': 'G',
               'drug_name': query_cpt2query_name[query_cpt],
               'ans_gene_names': query_cpt2ans_names[query_cpt]}
        for cdx, cand_cpt in enumerate(cand_cpts):
            if cand_cpt in gene_concept2gene_name:
                cand_name = gene_concept2gene_name[cand_cpt]
            else:
                cand_name = ''
            row[f'top{cdx+1}_gene_name'] = cand_name
        data.append(row)

    # pathway
    # calc relation vector
    vp = []
    for _, query_cpt, anss in hsa2relations[hsa]:
        for _, ans_cpt in anss:
            vp.append(model[ans_cpt] -
                      model[query_cpt])
    vp = np.mean(vp, axis=0)

    # prediction
    for _, query_cpt, anss in hsa2relations[hsa]:
        pred = model[query_cpt] + vp
        cand_cpts, _ = nearest_words(
            pred, centering=centering)
        p1_ans_cpts = [ans_cpt for _, ans_cpt in anss]
        p1_ans_names = [gene_concept2gene_name[ans_cpt]
                        for ans_cpt in p1_ans_cpts]
        p2_ans_names = query_cpt2ans_names[query_cpt]

        row = {'setting': 'P1',
               'drug_name': query_cpt2query_name[query_cpt],
               'ans_gene_names': p1_ans_names}
        for cdx, cand_cpt in enumerate(cand_cpts):
            if cand_cpt in gene_concept2gene_name:
                cand_name = gene_concept2gene_name[cand_cpt]
            else:
                cand_name = ''
            row[f'top{cdx+1}_gene_name'] = cand_name
        data.append(row)

        row = {'setting': 'P2',
               'drug_name': query_cpt2query_name[query_cpt],
               'ans_gene_names': p2_ans_names}
        for cdx, cand_cpt in enumerate(cand_cpts):
            if cand_cpt in gene_concept2gene_name:
                cand_name = gene_concept2gene_name[cand_cpt]
            else:
                cand_name = ''
            row[f'top{cdx+1}_gene_name'] = cand_name
        data.append(row)

    df = pd.DataFrame(data)

    # len(ans_gene_names) > 1 where setting == 'G'
    _df = df.copy()
    _df = _df[_df['setting'] == 'G']
    _df = _df[_df['ans_gene_names'].apply(lambda x: len(x) > 1)]
    drug_names = _df['drug_name'].unique()

    df = df[df['drug_name'].isin(drug_names)]
    df['drug_name'] = df['drug_name'].apply(lambda x: readable_name(x))
    df = df.sort_values(['drug_name', 'setting']).reset_index(drop=True)
    new_columns = ['drug_name', 'setting'] + [f'top{i}_gene_name'
                                              for i in range(1, top+1)] + \
        ['ans_gene_names']
    df = df[new_columns]
    output_dir = Path('output/ErbB_signaling_pathway')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{embed}_ErbB_signaling_pathway.csv'
    print(f'Saving results to {output_path}')
    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
