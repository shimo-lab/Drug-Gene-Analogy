import pickle as pkl
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Model


def main():

    # load model
    year_dir = Path('data/embeddings/years/1781-2023/min_count30')
    model_file = year_dir / 'embedding_e10s1e-05l0.025w5n5m30d300'
    model = np.fromfile(str(model_file), dtype=np.float64)
    model = model.reshape(-1, 300)
    word_to_id_path = year_dir / 'word_to_id.pkl'
    with open(word_to_id_path, 'rb') as f:
        word_to_id = pkl.load(f)
    model = Model(model, word_to_id)

    drug_df = pd.read_csv('output/pathway/homemade_drug_pathway.csv')
    gene_df = pd.read_csv('output/pathway/homemade_gene_pathway.csv')

    years = list(range(1975, 2021, 5)) + [2023]
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

        data = []
        for _, row in drug_df.iterrows():
            cpt = row['drug_concept']
            if cpt not in model:
                continue
            data.append(row)
        year_drug_df = pd.DataFrame(data)
        year_drug_df.to_csv(
            f'output/year/{year}_drug_pathway_for_P1Y1_and_P2Y1.csv',
            index=False)

        data = []
        for _, row in gene_df.iterrows():
            cpt = row['gene_concept']
            if cpt not in model:
                continue
            data.append(row)
        year_gene_df = pd.DataFrame(data)
        year_gene_df.to_csv(
            f'output/year/{year}_gene_pathway_for_P1Y1_and_P2Y1.csv',
            index=False)

        for query_direction in ('drug2gene', 'gene2drug'):
            rel_df = pd.read_csv(
                f'output/relation/homemade_{query_direction}s_relation.csv')

            query_concept_To_query_name_and_tgt_concept2tgt_name = dict()
            for _, row in rel_df.iterrows():

                if query_direction == 'drug2gene':
                    _, query_name, query_concept, tgt_names, tgt_concepts = row
                else:
                    query_name, query_concept, _, tgt_names, tgt_concepts = row

                if query_concept not in model:
                    continue

                tmp_tgt_names = eval(tgt_names)
                tmp_tgt_concepts = eval(tgt_concepts)
                tgt_names = []
                tgt_concepts = []
                for tmp_tgt_name, tmp_tgt_concept in zip(tmp_tgt_names,
                                                         tmp_tgt_concepts):
                    if tmp_tgt_concept not in model:
                        continue
                    tgt_names.append(tmp_tgt_name)
                    tgt_concepts.append(tmp_tgt_concept)
                if len(tgt_concepts) == 0:
                    continue

                tgt_concept2tgt_name = dict()
                for tgt_name, tgt_concept in zip(tgt_names, tgt_concepts):
                    tgt_concept2tgt_name[tgt_concept] = tgt_name

                query_concept_To_query_name_and_tgt_concept2tgt_name[
                    query_concept] = (query_name, tgt_concept2tgt_name)

            if query_direction == 'drug2gene':
                query_df = year_drug_df
                query_type = 'drug'
                tgt_df = year_gene_df
                tgt_type = 'gene'
            else:
                query_df = year_gene_df
                query_type = 'gene'
                tgt_df = year_drug_df
                tgt_type = 'drug'

            hsa2query_name_and_query_concept_and_tgt_name_concept_pair = \
                defaultdict(list)
            hsas = sorted(list(set(query_df['hsa'].values)))
            for hsa in tqdm(hsas):
                hsa_query_df = query_df.query(f'hsa=="{hsa}"'
                                              ).reset_index(drop=True)
                hsa_tgt_df = tgt_df.query(f'hsa=="{hsa}"'
                                          ).reset_index(drop=True)

                hsa_query_concepts = sorted(
                    list(set(hsa_query_df[f'{query_type}_concept'].values)))
                hsa_tgt_concepts = sorted(
                    list(set(hsa_tgt_df[f'{tgt_type}_concept'].values)))

                query_concept2tgt_name_concept_pair = defaultdict(list)
                for query_concept in hsa_query_concepts:
                    if query_concept not in query_concept_To_query_name_and_tgt_concept2tgt_name:  # noqa
                        continue
                    query_name, tgt_concept2tgt_name = \
                        query_concept_To_query_name_and_tgt_concept2tgt_name[
                            query_concept]
                    for tgt_concept in hsa_tgt_concepts:
                        if tgt_concept not in tgt_concept2tgt_name:
                            continue
                        tgt_name = tgt_concept2tgt_name[tgt_concept]
                        query_concept2tgt_name_concept_pair[
                            query_concept].append((tgt_name, tgt_concept))

                for query_concept, tgt_name_concept_pair in \
                        query_concept2tgt_name_concept_pair.items():
                    query_name, _ = \
                        query_concept_To_query_name_and_tgt_concept2tgt_name[
                            query_concept]
                    hsa2query_name_and_query_concept_and_tgt_name_concept_pair[
                        hsa].append(
                            (query_name, query_concept, tgt_name_concept_pair))

            # we assume that the number of drugs and genes is more than 1
            hsa_to_direct = dict()
            for hsa in tqdm(hsas):
                query_sets = set()
                tgt_sets = set()
                for _, query_concept, tgt_name_concept_pair in \
                    hsa2query_name_and_query_concept_and_tgt_name_concept_pair[
                        hsa]:
                    query_sets.add(query_concept)
                    for _, tgt_concept in tgt_name_concept_pair:
                        tgt_sets.add(tgt_concept)
                if len(query_sets) < 2 or len(tgt_sets) < 2:
                    continue
                hsa_to_direct[hsa] = \
                    hsa2query_name_and_query_concept_and_tgt_name_concept_pair[
                        hsa]

            print(f'{query_direction} - #pathway: {len(hsa_to_direct)}')
            with open(f'output/year/{year}_hsa_to_{query_direction}.pkl',
                      'wb') as f:
                pkl.dump(hsa_to_direct, f)


if __name__ == '__main__':
    main()
