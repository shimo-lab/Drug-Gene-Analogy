import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from utils import get_logger


def get_result_row(model, embed, query_direction, method, acc):
    result_row = {'model': model,
                  'embed': embed,
                  'query_direction': query_direction,
                  'method': method, }
    for ith, acc_ in enumerate(acc):
        result_row[f'top{ith+1}_acc'] = acc_
    return result_row


def calc_acc(rows, top):
    tots = []
    for row in rows:
        ans_names = set(row['ans_names'])
        n = len(row['cand_names'])
        c = [1]*min(n, top)
        for k in range(len(c)):
            cand_name = row['cand_names'][k]
            if cand_name.upper() in ans_names or \
                    cand_name.lower() in ans_names:
                break
            else:
                c[k] = 0

        if len(c) < top:
            last = c[-1]
            c += [last]*(top - len(c))

        assert len(c) == top

        tots.append(c)
    accs = np.mean(np.array(tots), axis=0)
    return list(accs)


def logging_calc_acc(ans_names, cands):
    ans_names = set(ans_names)

    isfound = None
    for k, cand_name in enumerate(cands):
        if cand_name.upper() in ans_names or \
                cand_name.lower() in ans_names:
            isfound = k + 1
            break

    if isfound is None:
        text = 'not found'
    else:
        text = f'found in top {isfound}'

    return text


def get_api_result(client, api_result_path, query_name, model, top):

    if api_result_path.exists():
        with open(api_result_path, 'rb') as f:
            api_result = pkl.load(f)
    else:
        api_result = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system",
                 "content": "You are a biologist specializing in drug-target interactions. Your responses should be formatted as JSON, with data listed under the key 'target_genes'."},  # noqa
                {"role": "user",
                 "content": f"List the top {top} potential target genes for the drug {query_name}."},  # noqa
            ],
            temperature=0.0,
        )
        with open(api_result_path, 'wb') as f:
            pkl.dump(api_result, f)

    return api_result


def main():

    top = 10

    output_dir = Path('output/analogy_API')
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / 'output.log'
    logger = get_logger(log_file)

    results = []

    # query directions
    query_type = 'drug'
    ans_type = 'gene'
    query_direction = f'{query_type}2{ans_type}'

    client = OpenAI()
    for model in ("gpt-3.5-turbo-0125",
                  "gpt-4-turbo-2024-04-09",
                  "gpt-4o-2024-05-13"):
        model_output_dir = output_dir / model
        model_output_dir.mkdir(parents=True, exist_ok=True)

        for embed in ('bcv', 'homemade'):

            # drug_cpt2drug_name
            drug_conv_csv_path = 'output/kegg2mesh/kegg2mesh_df.csv'
            drug_df = pd.read_csv(drug_conv_csv_path)
            drug_df = drug_df[drug_df['conv_method'] == 'exact_heading']
            drug_df['concept'] = 'Chemical_MESH_' + drug_df['mesh_id']
            if embed == 'homemade':
                drug_df['concept'] = drug_df['concept'].apply(
                    lambda x: x.lower())
            assert len(drug_df['kegg_id']) == len(set(drug_df['kegg_id']))
            drug_cpt2drug_name = dict(
                zip(drug_df['concept'], drug_df['mesh_label']))

            # gene_cpt2gene_name
            gene_conv_csv_path = 'output/gene_cpt2name/gene_cpt2name.csv'
            gene_conv_df = pd.read_csv(gene_conv_csv_path)
            if embed == 'homemade':
                gene_conv_df['concept'] = \
                    gene_conv_df['concept'].apply(lambda x: x.lower())
            gene_cpt2gene_name = dict(zip(gene_conv_df['concept'],
                                          gene_conv_df['name']))
            # prepare queries
            rel_df = pd.read_csv(
                f'output/relation/{embed}_{query_direction}s_relation.csv')

            query_cpt_col = f'{query_type}_concept'
            ans_cpts_col = f'{ans_type}_concepts'
            assert len(rel_df) == len(rel_df[query_cpt_col].unique())

            # define query and answer set
            query_cpts = set(rel_df[query_cpt_col].values)
            query_name2ans_names = dict()
            for _, row in rel_df.iterrows():
                query_cpt = row[query_cpt_col]
                ans_cpts = eval(row[ans_cpts_col])
                query_name = drug_cpt2drug_name[query_cpt]
                ans_names = \
                    [gene_cpt2gene_name[ans_cpt] for ans_cpt in ans_cpts]
                query_name2ans_names[query_name] = ans_names

            # setting G
            logger.info(f'{model} {embed} {query_direction} G')
            rows = []
            for query_cpt in tqdm(query_cpts):
                api_result_path = model_output_dir / \
                    f'{query_cpt.upper()}.pkl'
                query_name = drug_cpt2drug_name[query_cpt]
                api_result = get_api_result(client, api_result_path,
                                            query_name, model, top)
                try:
                    cands = eval(api_result.choices[0].message.content
                                 )['target_genes']
                except Exception as e:
                    logger.info(f'{query_name}: invalid response {e}')
                    continue
                ans_names = query_name2ans_names[query_name]
                # logger.info(f'{query_name}: cands={cands}, ans={ans_names}')
                # logger.info(logging_calc_acc(ans_names, cands))
                row = {'query_name': query_name,
                       'ans_names': ans_names,
                       'cand_names': cands}
                rows.append(row)
            result_row = get_result_row(model, embed, query_direction,
                                        'G', calc_acc(rows, top))
            result_row['query_num'] = len(rows)
            # logger.info(result_row)
            results.append(result_row)

            # pathway-wise setting
            pathway_pkl = \
                f'output/pathway/{embed}_hsa_to_{query_direction}s.pkl'

            with open(pathway_pkl, 'rb') as f:
                hsa2relations = pkl.load(f)
            hsas = sorted(list(hsa2relations.keys()))

            # setting P1
            logger.info(f'{model} {embed} {query_direction} P1')
            rows = []
            query_num = 0
            for hsa in tqdm(hsas):

                # prediction
                hsa_rows = []
                for _, query_cpt, anss in hsa2relations[hsa]:
                    query_num += 1
                    api_result_path = \
                        model_output_dir / f'{query_cpt.upper()}.pkl'
                    query_name = drug_cpt2drug_name[query_cpt]
                    api_result = get_api_result(client, api_result_path,
                                                query_name, model, top)
                    try:
                        cands = eval(api_result.choices[0].message.content
                                     )['target_genes']
                    except Exception as e:
                        logger.info(f'{query_name}: invalid response {e}')
                        continue

                    ans_cpts = [ans_cpt for _, ans_cpt in anss]
                    ans_names = [gene_cpt2gene_name[ans_cpt]
                                 for _, ans_cpt in anss]
                    # logger.info(
                    #     f'{query_name}: cands={cands}, ans={ans_names}')
                    # logger.info(logging_calc_acc(ans_names, cands))
                    row = {'query_name': query_name,
                           'ans_names': ans_names,
                           'cand_names': cands}

                    hsa_rows.append(row)
                rows += hsa_rows

            result_row = get_result_row(model, embed, query_direction,
                                        'P1', calc_acc(rows, top))
            result_row['query_num'] = query_num
            # logger.info(result_row)
            results.append(result_row)

            # prepare queries for setting P2
            all_query_cpt_set = set(rel_df[query_cpt_col].values)

            domain_df = pd.read_csv(
                f'output/pathway/{embed}_{query_type}_pathway.csv')

            hsa2query_cpt_set = dict()
            for hsa in tqdm(hsas):
                hsa2query_cpt_set[hsa] = set(
                    domain_df.query(f'hsa=="{hsa}"')[
                        f'{query_type}_concept'].values) & all_query_cpt_set

            # setting P2
            logger.info(f'{model} {embed} {query_direction} P2')
            rows = []
            query_num = 0
            for hsa in tqdm(hsas):

                query_cpt_set = hsa2query_cpt_set[hsa]

                # prediction
                hsa_rows = []
                for query_cpt in query_cpt_set:
                    query_num += 1
                    query_name = drug_cpt2drug_name[query_cpt]
                    api_result_path = \
                        model_output_dir / f'{query_cpt.upper()}.pkl'
                    api_result = get_api_result(client, api_result_path,
                                                query_name, model, top)
                    try:
                        cands = eval(api_result.choices[0].message.content
                                     )['target_genes']
                    except Exception as e:
                        logger.info(f'{query_name}: invalid response {e}')
                        continue
                    ans_names = query_name2ans_names[query_name]
                    # logger.info(
                    #     f'{query_name}: cands={cands}, ans={ans_names}')
                    # logger.info(logging_calc_acc(ans_names, cands))
                    row = {'query_name': query_name,
                           'ans_names': ans_names,
                           'cand_names': cands}
                    hsa_rows.append(row)
                rows += hsa_rows

            result_row = get_result_row(model, embed, query_direction,
                                        'P2', calc_acc(rows, top))
            result_row['query_num'] = query_num
            # logger.info(result_row)
            results.append(result_row)

    # save results
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_dir / 'result.csv', index=False)


if __name__ == '__main__':
    main()
