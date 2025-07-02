import pickle as pkl
from collections import defaultdict

import pandas as pd
from tqdm import tqdm


def main(embed):

    assert embed in ['bcv', 'homemade']

    for direct in ['drug2genes', 'gene2drugs']:

        rel_df = pd.read_csv(f'output/relation/{embed}_{direct}_relation.csv')

        if direct == 'drug2genes':
            query = 'drug'
            tgt = 'gene'
        else:
            query = 'gene'
            tgt = 'drug'

        query_concept_To_query_name_and_tgt_concept2tgt_name = dict()
        for _, row in rel_df.iterrows():

            if direct == 'drug2genes':
                _, query_name, query_concept, tgt_names, tgt_concepts = row
            else:
                query_name, query_concept, _, tgt_names, tgt_concepts = row

            tgt_names = eval(tgt_names)
            tgt_concepts = eval(tgt_concepts)
            tgt_concept2tgt_name = dict()
            for tgt_name, tgt_concept in zip(tgt_names, tgt_concepts):
                tgt_concept2tgt_name[tgt_concept] = tgt_name
            query_concept_To_query_name_and_tgt_concept2tgt_name[
                query_concept] = (query_name, tgt_concept2tgt_name)

        query_df = pd.read_csv(f'output/pathway/{embed}_{query}_pathway.csv')
        tgt_df = pd.read_csv(f'output/pathway/{embed}_{tgt}_pathway.csv')

        assert len(set(query_df['hsa'])) == len(set(tgt_df['hsa']))

        hsa2query_name_and_query_concept_and_tgt_name_concept_pair = \
            defaultdict(list)
        hsas = sorted(list(set(query_df['hsa'].values)))
        for hsa in tqdm(hsas):
            hsa_query_df = query_df.query(f'hsa=="{hsa}"'
                                          ).reset_index(drop=True)
            hsa_tgt_df = tgt_df.query(f'hsa=="{hsa}"').reset_index(drop=True)

            hsa_query_concepts = sorted(
                list(set(hsa_query_df[f'{query}_concept'].values)))
            hsa_tgt_concepts = sorted(
                list(set(hsa_tgt_df[f'{tgt}_concept'].values)))

            query_concept2tgt_name_concept_pair = defaultdict(list)
            for query_concept in hsa_query_concepts:
                if query_concept not in \
                        query_concept_To_query_name_and_tgt_concept2tgt_name:
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

        print(f'{direct} - #pathway: {len(hsa_to_direct)}')

        with open(f'output/pathway/{embed}_hsa_to_{direct}.pkl', 'wb') as f:
            pkl.dump(hsa_to_direct, f)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
