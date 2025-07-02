import json
import pickle as pkl

import pandas as pd
from tqdm import tqdm


def main(embed):

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

    drug2genes_csv_path = f'output/relation/{embed}_drug2genes_relation.csv'
    drug2genes_df = pd.read_csv(drug2genes_csv_path)
    assert len(drug2genes_df['kegg_id']) == len(set(drug2genes_df['kegg_id']))
    kegg_id2drug_concept = dict(
        zip(drug2genes_df['kegg_id'], drug2genes_df['drug_concept']))
    kegg_id2drug_name = dict(
        zip(drug2genes_df['kegg_id'], drug2genes_df['drug_name']))

    gene_conv_csv_path = 'output/gene_cpt2name/gene_cpt2name.csv'
    gene_conv_df = pd.read_csv(gene_conv_csv_path)
    if embed == 'homemade':
        gene_conv_df['concept'] = \
            gene_conv_df['concept'].apply(lambda x: x.lower())
    gene_concept2gene_name = dict(zip(gene_conv_df['concept'],
                                      gene_conv_df['name']))

    tsv_path = 'data/pathway/hsa.tsv'
    df = pd.read_table(tsv_path, header=None, names=['hsa', 'name'])

    drug_data = []
    gene_data = []
    hsas = df['hsa'].values
    for hsa in tqdm(sorted(hsas)):

        tmp_drug_data = []
        tmp_gene_data = []

        data_path = f'output/pathway/json/{hsa}.json'
        with open(data_path, 'r') as f:
            dic = json.load(f)

        if not ('DRUG' in dic and 'GENE' in dic):
            continue

        for gene_dic in dic['GENE']:

            gene_concept = f"Gene_{gene_dic['ID']}"
            if embed == 'homemade':
                gene_concept = gene_concept.lower()

            if gene_concept in gene_concept2gene_name and \
                    gene_concept in model:
                gene_name = gene_concept2gene_name[gene_concept]
                tmp_gene_data.append((hsa, gene_name, gene_concept))

        if len(tmp_gene_data) == 0:
            continue

        for drug_dic in dic['DRUG']:
            kegg_id = f"dr:{drug_dic['ID']}"

            if kegg_id not in kegg_id2drug_concept:
                continue

            drug_concept = kegg_id2drug_concept[kegg_id]
            if drug_concept not in model:
                continue
            drug_name = kegg_id2drug_name[kegg_id]
            tmp_drug_data.append((hsa, kegg_id, drug_name, drug_concept))

        if len(tmp_drug_data):
            gene_data += tmp_gene_data
            drug_data += tmp_drug_data

    drug_df = pd.DataFrame(drug_data, columns=(
        'hsa', 'kegg_id', 'drug_name', 'drug_concept'))
    gene_df = pd.DataFrame(gene_data, columns=(
        'hsa', 'gene_name', 'gene_concept'))

    for state, state_df in [['drug', drug_df],
                            ['gene', gene_df]]:
        data = []
        hsas = list(set(state_df['hsa'].values))
        hsas.sort()
        for hsa in hsas:
            hsa_df = state_df.query(f'hsa=="{hsa}"').reset_index(drop=True)
            assert len(hsa_df) == len(set(hsa_df[f'{state}_concept']))
            for _, row in hsa_df.iterrows():
                data.append(row)

        res_df = pd.DataFrame(data)
        assert len(res_df) == len(res_df.drop_duplicates(
            ['hsa', f'{state}_concept']))
        output_path = f'output/pathway/{embed}_{state}_pathway.csv'
        res_df.to_csv(output_path, index=False)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
