import json
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Model


def main():
    # load model
    model_file = 'data/embeddings/years/1781-2023/min_count30/'\
        'embedding_e10s1e-05l0.025w5n5m30d300'
    model = np.fromfile(str(model_file), dtype=np.float64)
    model = model.reshape(-1, 300)
    word_to_id_path = \
        'data/embeddings/years/1781-2023/min_count30/word_to_id.pkl'
    with open(word_to_id_path, 'rb') as f:
        word_to_id = pkl.load(f)
    model = Model(model, word_to_id)

    gene_json_path = 'data/year/Gene.json'
    drug_json_path = 'data/year/Chemical.json'
    gene_drug_json_path = 'data/year/Gene==Chemical.json'

    with open(gene_json_path, 'r') as f:
        gene_json = json.load(f)
    with open(drug_json_path, 'r') as f:
        drug_json = json.load(f)
    with open(gene_drug_json_path, 'r') as f:
        gene_drug_json = json.load(f)

    cpt2year = {}
    for gene_id, year in gene_json.items():
        if f'gene_{gene_id}' not in model:
            continue
        cpt2year[f'gene_{gene_id}'] = year
    for mesh_id, year in drug_json.items():
        if mesh_id == '_':
            continue
        mesh_id = mesh_id.lower()
        if f'chemical_{mesh_id}' not in model:
            continue
        cpt2year[f'chemical_{mesh_id}'] = year

    rel_df = pd.read_csv('output/relation/homemade_drug2genes_relation.csv')
    assert len(rel_df) == len(rel_df['drug_concept'].unique())

    rel2name = {}
    for _, row in rel_df.iterrows():
        query_name = row['drug_name']
        query_cpt = row['drug_concept']
        ans_names = eval(row['gene_names'])
        ans_cpts = eval(row['gene_concepts'])

        for ans_cpt, ans_name in zip(ans_cpts, ans_names):
            rel2name[(query_cpt, ans_cpt)] = (query_name, ans_name)

    data = []
    for gene_drug, year in tqdm(list(gene_drug_json.items())):
        gene_id, mesh_id = gene_drug.split('==')

        query_cpt = f'chemical_{mesh_id}'.lower()
        ans_cpt = f'gene_{gene_id}'.lower()

        if query_cpt == '_' or ans_cpt == '_':
            continue

        if query_cpt not in model or ans_cpt not in model:
            continue

        if (query_cpt, ans_cpt) not in rel2name:
            continue

        query_year = cpt2year[query_cpt]
        ans_year = cpt2year[ans_cpt]
        query_name, ans_name = rel2name[(query_cpt, ans_cpt)]

        data.append({
            'drug_cpt': query_cpt,
            'drug_name': query_name,
            'drug_year': int(query_year),
            'gene_cpt': ans_cpt,
            'gene_name': ans_name,
            'gene_year': int(ans_year),
            'rel_year': int(year),
        })

    df = pd.DataFrame(data)
    Path('output/year').mkdir(exist_ok=True, parents=True)
    df.to_csv('output/year/relation_for_Y2.csv', index=False)

    years = list(range(1975, 2021, 5)) + [2023]

    # query directions
    for query_direction in ('drug2gene', 'gene2drug'):
        if query_direction == 'drug2gene':
            query_cpt_col = 'drug_cpt'
            query_name_col = 'drug_name'
            query_year_col = 'drug_year'
            ans_cpt_col = 'gene_cpt'
            ans_name_col = 'gene_name'
            ans_year_col = 'gene_year'
        else:
            query_cpt_col = 'gene_cpt'
            query_name_col = 'gene_name'
            query_year_col = 'gene_year'
            ans_cpt_col = 'drug_cpt'
            ans_name_col = 'drug_name'
            ans_year_col = 'drug_year'
        rel_year_col = 'rel_year'

        for year in years:

            # load model
            year_dir = Path(f'data/embeddings/years/1781-{year}/min_count30')
            model_file = year_dir / 'embedding_e10s1e-05l0.025w5n5m30d300'
            model = np.fromfile(str(model_file), dtype=np.float64)
            model = model.reshape(-1, 300)
            word_to_id_path = year_dir / 'word_to_id.pkl'
            with open(word_to_id_path, 'rb') as f:
                word_to_id = pkl.load(f)
            model = Model(model, word_to_id)

            year_df = df.query(
                f'({query_year_col} <= {year}) & ({ans_year_col} <= {year})')

            # query cpt
            query_cpts = year_df[query_cpt_col].unique()
            data = []
            for query_cpt in query_cpts:

                if query_cpt not in model:
                    continue

                query_df = year_df.query(f'{query_cpt_col} == "{query_cpt}"')
                query_name = query_df[query_name_col].iloc[0]
                query_year = query_df[query_year_col].iloc[0]

                Ly_ans_cpts = []
                Ly_ans_names = []
                Ly_ans_years = []
                Ly_rel_years = []

                Uy_ans_cpts = []
                Uy_ans_names = []
                Uy_ans_years = []
                Uy_rel_years = []
                for _, row in query_df.iterrows():
                    ans_cpt = row[ans_cpt_col]

                    if ans_cpt not in model:
                        continue

                    ans_name = row[ans_name_col]
                    ans_year = row[ans_year_col]
                    rel_year = row[rel_year_col]

                    if rel_year <= year:
                        Ly_ans_cpts.append(ans_cpt)
                        Ly_ans_names.append(ans_name)
                        Ly_ans_years.append(ans_year)
                        Ly_rel_years.append(rel_year)
                    else:
                        Uy_ans_cpts.append(ans_cpt)
                        Uy_ans_names.append(ans_name)
                        Uy_ans_years.append(ans_year)
                        Uy_rel_years.append(rel_year)

                if len(Ly_ans_cpts) == 0 and len(Uy_ans_cpts) == 0:
                    continue

                data.append({
                    'query_cpt': query_cpt,
                    'query_name': query_name,
                    'query_year': int(query_year),
                    'Ly_ans_cpts': Ly_ans_cpts,
                    'Ly_ans_names': Ly_ans_names,
                    'Ly_ans_years': Ly_ans_years,
                    'Ly_rel_years': Ly_rel_years,
                    'Uy_ans_cpts': Uy_ans_cpts,
                    'Uy_ans_names': Uy_ans_names,
                    'Uy_ans_years': Uy_ans_years,
                    'Uy_rel_years': Uy_rel_years,
                })

            year_rel_df = pd.DataFrame(data)
            year_rel_df.to_csv(
                f'output/year/{year}_{query_direction}s_relation_for_Y2.csv',
                index=False)


if __name__ == '__main__':
    main()
