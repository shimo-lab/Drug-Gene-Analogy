import pickle as pkl
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def main(embed):

    rel_tsv_path = 'data/relation/human_KEGG_drug_names.tsv'
    rel_df = pd.read_table(rel_tsv_path)

    assert embed in ['bcv', 'homemade']

    if embed == 'bcv':
        model_file = 'output/embeddings/concept_skipgram.pkl'
    elif embed == 'homemade':
        model_file = \
            'data/embeddings/years/1781-2023/min_count30/word_to_id.pkl'
    else:
        raise NotImplementedError()

    with open(model_file, 'rb') as f:
        model = pkl.load(f)

    drug_conv_csv_path = 'output/kegg2mesh/kegg2mesh_df.csv'
    drug_df = pd.read_csv(drug_conv_csv_path)
    drug_df = drug_df[drug_df['conv_method'] == 'exact_heading']
    drug_df['concept'] = 'Chemical_MESH_' + drug_df['mesh_id']
    if embed == 'homemade':
        drug_df['concept'] = drug_df['concept'].apply(lambda x: x.lower())
    assert len(drug_df['kegg_id']) == len(set(drug_df['kegg_id']))
    kegg_id2drug_concept = dict(zip(drug_df['kegg_id'], drug_df['concept']))
    kegg_id2drug_name = dict(zip(drug_df['kegg_id'], drug_df['mesh_label']))

    gene_conv_csv_path = 'output/gene_cpt2name/gene_cpt2name.csv'
    gene_df = pd.read_csv(gene_conv_csv_path)
    if embed == 'homemade':
        gene_df['concept'] = gene_df['concept'].apply(lambda x: x.lower())
    gene_concept2gene_name = dict(zip(gene_df['concept'], gene_df['name']))

    # drug to genes
    data = []
    for _, row in tqdm(list(rel_df.iterrows())):
        kegg_id, _, _, gene_ids = row

        if kegg_id not in kegg_id2drug_concept:
            continue

        drug_concept = kegg_id2drug_concept[kegg_id]
        if drug_concept not in model:
            continue
        drug_name = kegg_id2drug_name[kegg_id]

        gene_ids = gene_ids.split('/')
        gene_names = []
        gene_concepts = []
        for gene_id in gene_ids:
            gene_concept = f'Gene_{gene_id}'
            if embed == 'homemade':
                gene_concept = gene_concept.lower()

            if gene_concept not in gene_concept2gene_name:
                continue

            if gene_concept not in model:
                continue

            gene_names.append(gene_concept2gene_name[gene_concept])
            gene_concepts.append(gene_concept)

        if len(gene_names) == 0:
            continue

        data.append((kegg_id, drug_name, drug_concept,
                     gene_names, gene_concepts))

    df = pd.DataFrame(data, columns=('kegg_id', 'drug_name', 'drug_concept',
                      'gene_names', 'gene_concepts'))
    before = len(df)
    df = df.drop_duplicates(subset=['drug_concept'])
    after = len(df)
    print(f'remove duplication: {before} -> {after}')

    Path('output/relation').mkdir(parents=True, exist_ok=True)
    df.to_csv(f'output/relation/{embed}_drug2genes_relation.csv', index=False)

    # gene to drugs
    gene_concept2gene_name = dict()
    gene_concept2drug_tuples = defaultdict(list)
    for _, row in df.iterrows():
        kegg_id, drug_name, drug_concept, gene_names, gene_concepts = row

        for gene_name, gene_concept in zip(gene_names, gene_concepts):
            gene_concept2gene_name[gene_concept] = gene_name
            gene_concept2drug_tuples[gene_concept].append(
                (kegg_id, drug_name, drug_concept))

    data = []
    for gene_concept, drug_tuples in gene_concept2drug_tuples.items():
        gene_name = gene_concept2gene_name[gene_concept]

        kegg_ids = []
        drug_names = []
        drug_concepts = []
        for kegg_id, drug_name, drug_concept in drug_tuples:
            kegg_ids.append(kegg_id)
            drug_names.append(drug_name)
            drug_concepts.append(drug_concept)
        data.append((gene_name, gene_concept,
                     kegg_ids, drug_names, drug_concepts))

    df = pd.DataFrame(data, columns=('gene_name', 'gene_concept', 'kegg_ids',
                      'drug_names', 'drug_concepts'))
    assert len(df['gene_name']) == len(set(df['gene_name']))
    df.to_csv(f'output/relation/{embed}_gene2drugs_relation.csv', index=False)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
