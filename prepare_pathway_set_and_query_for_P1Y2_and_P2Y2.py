import json
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Model


def main():

    # load model
    year_dir = Path('data/embeddings/years/1781-2023/min_count30')
    model_file = year_dir / 'embedding_e10s1e-05l0.025w5n5m30d300'  # noqa
    model = np.fromfile(str(model_file), dtype=np.float64)
    model = model.reshape(-1, 300)
    word_to_id_path = year_dir / 'word_to_id.pkl'
    with open(word_to_id_path, 'rb') as f:
        word_to_id = pkl.load(f)
    model = Model(model, word_to_id)

    gene_json_path = 'data/year/Gene.json'
    drug_json_path = 'data/year/Chemical.json'

    with open(gene_json_path, 'r') as f:
        gene_json = json.load(f)
    with open(drug_json_path, 'r') as f:
        drug_json = json.load(f)

    cpt2year = {}
    for gene_id, year in gene_json.items():
        if f'gene_{gene_id}' not in model:
            continue
        cpt2year[f'gene_{gene_id}'] = int(year)
    for mesh_id, year in drug_json.items():
        if mesh_id == '_':
            continue
        mesh_id = mesh_id.lower()
        if f'chemical_{mesh_id}' not in model:
            continue
        cpt2year[f'chemical_{mesh_id}'] = int(year)

    drug_df = pd.read_csv('output/pathway/homemade_drug_pathway.csv')
    drug_df['year'] = drug_df['drug_concept'].apply(
        lambda x: cpt2year[x.lower()])
    drug_df = drug_df[drug_df['year'] != '']

    gene_df = pd.read_csv('output/pathway/homemade_gene_pathway.csv')
    gene_df['year'] = gene_df['gene_concept'].apply(
        lambda x: cpt2year[x.lower()])
    gene_df = gene_df[gene_df['year'] != '']

    years = list(range(1975, 2021, 5))
    for year in years:
        # load model
        year_dir = Path(f'data/embeddings/years/1781-{year}/min_count30')
        model_file = year_dir / 'embedding_e10s1e-05l0.025w5n5m30d300'  # noqa
        model = np.fromfile(str(model_file), dtype=np.float64)
        model = model.reshape(-1, 300)
        word_to_id_path = year_dir / 'word_to_id.pkl'
        with open(word_to_id_path, 'rb') as f:
            word_to_id = pkl.load(f)
        model = Model(model, word_to_id)

        year_drug_df = drug_df.query(f'year <= {year}').reset_index(drop=True)
        data = []
        for _, row in year_drug_df.iterrows():
            cpt = row['drug_concept']
            if cpt not in model:
                continue
            data.append(row)
        year_drug_df = pd.DataFrame(data)
        year_drug_df.to_csv(
            f'output/year/{year}_drug_pathway_for_P1Y2_and_P2Y2.csv',
            index=False)

        year_gene_df = gene_df.query(f'year <= {year}').reset_index(drop=True)
        data = []
        for _, row in year_gene_df.iterrows():
            cpt = row['gene_concept']
            if cpt not in model:
                continue
            data.append(row)
        year_gene_df = pd.DataFrame(data)
        year_gene_df.to_csv(
            f'output/year/{year}_gene_pathway_for_P1Y2_and_P2Y2.csv',
            index=False)

        for query_direction in ('drug2gene', 'gene2drug'):
            rel_df = pd.read_csv(
                f'output/year/{year}_{query_direction}s_relation_for_Y2.csv')

            if query_direction == 'drug2gene':
                domain_df = year_drug_df
                cand_df = year_gene_df
                query_type = 'drug'
                ans_type = 'gene'
            else:
                domain_df = year_gene_df
                cand_df = year_drug_df
                query_type = 'gene'
                ans_type = 'drug'

            hsas = sorted(list(set(domain_df['hsa'].values)))
            hsa2queries = dict()
            for hsa in tqdm(hsas):
                hsa_domain_df = domain_df.query(f'hsa=="{hsa}"'
                                                ).reset_index(drop=True)
                tmp = []
                for query_cpt in sorted(list(set(
                        hsa_domain_df[f'{query_type}_concept'].values))):
                    if query_cpt not in rel_df['query_cpt'].values:
                        continue
                    row = rel_df.query(
                        f'query_cpt=="{query_cpt}"'
                    ).reset_index(drop=True).iloc[0]
                    tmp.append(dict(row))

                if len(tmp) == 0:
                    continue

                query_cpts = set([row['query_cpt'] for row in tmp])
                ans_cpts = set()
                for row in tmp:
                    ans_cpts.update(row['Ly_ans_cpts'])
                    ans_cpts.update(row['Uy_ans_cpts'])

                if len(query_cpts) < 2 or len(ans_cpts) < 2:
                    continue

                hsa2queries[hsa] = tmp

            print(f'year: {year}, query_direction: {query_direction}')
            print(f'P2Y2: {len(hsa2queries)}')

            with open(f'output/year/{year}_{query_direction}_P2Y2.pkl', 'wb'
                      ) as f:
                pkl.dump(hsa2queries, f)

            hsa2queries = dict()
            for hsa in tqdm(hsas):
                hsa_domain_df = domain_df.query(f'hsa=="{hsa}"'
                                                ).reset_index(drop=True)
                hsa_cand_df = cand_df.query(f'hsa=="{hsa}"'
                                            ).reset_index(drop=True)
                cand_cpts = set(hsa_cand_df[f'{ans_type}_concept'].values)
                tmp = []
                for query_cpt in sorted(list(set(
                        hsa_domain_df[f'{query_type}_concept'].values))):
                    if query_cpt not in rel_df['query_cpt'].values:
                        continue
                    row = rel_df.query(
                        f'query_cpt=="{query_cpt}"'
                    ).reset_index(drop=True).iloc[0]
                    row = dict(row)

                    tmp_Ly_ans_cpts = []
                    tmp_Ly_ans_names = []
                    tmp_Ly_ans_years = []
                    tmp_Ly_rel_years = []
                    for ans_cpt, ans_name, ans_year, rel_year in zip(
                        eval(row['Ly_ans_cpts']), eval(row['Ly_ans_names']),
                        eval(row['Ly_ans_years']), eval(row['Ly_rel_years'])
                    ):
                        assert ans_cpt in model
                        if ans_cpt not in cand_cpts:
                            continue
                        tmp_Ly_ans_cpts.append(ans_cpt)
                        tmp_Ly_ans_names.append(ans_name)
                        tmp_Ly_ans_years.append(ans_year)
                        tmp_Ly_rel_years.append(rel_year)

                    tmp_Uy_ans_cpts = []
                    tmp_Uy_ans_names = []
                    tmp_Uy_ans_years = []
                    tmp_Uy_rel_years = []
                    for ans_cpt, ans_name, ans_year, rel_year in zip(
                        eval(row['Uy_ans_cpts']), eval(row['Uy_ans_names']),
                        eval(row['Uy_ans_years']), eval(row['Uy_rel_years'])
                    ):
                        assert ans_cpt in model
                        if ans_cpt not in cand_cpts:
                            continue
                        tmp_Uy_ans_cpts.append(ans_cpt)
                        tmp_Uy_ans_names.append(ans_name)
                        tmp_Uy_ans_years.append(ans_year)
                        tmp_Uy_rel_years.append(rel_year)

                    if len(tmp_Ly_ans_cpts) == 0 and len(tmp_Uy_ans_cpts) == 0:
                        continue

                    row['Ly_ans_cpts'] = tmp_Ly_ans_cpts
                    row['Ly_ans_names'] = tmp_Ly_ans_names
                    row['Ly_ans_years'] = tmp_Ly_ans_years
                    row['Ly_rel_years'] = tmp_Ly_rel_years
                    row['Uy_ans_cpts'] = tmp_Uy_ans_cpts
                    row['Uy_ans_names'] = tmp_Uy_ans_names
                    row['Uy_ans_years'] = tmp_Uy_ans_years
                    row['Uy_rel_years'] = tmp_Uy_rel_years
                    tmp.append(row)

                if len(tmp) == 0:
                    continue

                query_cpts = set([row['query_cpt'] for row in tmp])
                ans_cpts = set()
                for row in tmp:
                    ans_cpts.update(row['Ly_ans_cpts'])
                    ans_cpts.update(row['Uy_ans_cpts'])
                if len(query_cpts) < 2 or len(ans_cpts) < 2:
                    continue

                hsa2queries[hsa] = tmp

            print(f'year: {year}, query_direction: {query_direction}')
            print(f'P1Y2: {len(hsa2queries)}')
            with open(
                f'output/year/{year}_{query_direction}_P1Y2.pkl', 'wb'
            ) as f:
                pkl.dump(hsa2queries, f)


if __name__ == '__main__':
    main()
