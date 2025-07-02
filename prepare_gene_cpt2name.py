import json
from pathlib import Path
from time import sleep

import pandas as pd
import requests
from tqdm import tqdm


def gene_nameizer(model, output_path):

    def get_cpt2name_list(b_idx, batch_queries, output_dir):

        cache_path = output_dir / f'{b_idx}.txt'
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                data = f.read()
        else:
            url = f"http://rest.kegg.jp/list/{batch_queries}"
            response = requests.get(url)
            sleep(1)
            if response.ok:
                data = response.text
            else:
                data = ''
            with open(cache_path, 'w') as f:
                f.write(data)

        cpt2name_list = []
        for line in data.split('\n'):
            if line == '':
                continue
            hsa, *others = line.split('\t')
            names = ' '.join(others).split(';')[0]
            name = [s.strip() for s in names.split(',')][0]
            gene_id = hsa.split(':')[1]
            cpt = f'Gene_{gene_id}'
            cpt2name_list.append({'concept': cpt, 'name': name})
        return cpt2name_list

    queries = []
    for concept in tqdm(model.keys()):
        if concept[:len('Gene_')] == 'Gene_':
            kegg_id = concept[len('Gene_'):]
            try:
                kegg_id = int(kegg_id)
            except Exception as e:
                print(f'Invalid KEGG ID: {kegg_id}, error: {e}')
                continue
            query = f'hsa:{kegg_id}'
            queries.append(query)
    queries.sort(key=lambda x: int(x.split(':')[1]))

    output_dir = Path('output/gene_cpt2name/txt')
    output_dir.mkdir(parents=True, exist_ok=True)

    query_num = len(queries)
    data = []
    batch_size = 100
    for b_idx in tqdm(list(range(query_num//batch_size + 1))):
        start = b_idx * batch_size
        end = min((b_idx + 1) * batch_size, query_num)
        batch_queries = queries[start:end]
        batch_queries = '+'.join(batch_queries)
        cpt2name_list = get_cpt2name_list(b_idx, batch_queries, output_dir)
        data.extend(cpt2name_list)

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)


def main():

    # load embedding
    embedding_path = 'data/embeddings/concept_skipgram.json'
    with open(embedding_path, 'r') as f:
        model = json.load(f)
    output_path = 'output/gene_cpt2name/gene_cpt2name.csv'

    gene_nameizer(model, output_path)


if __name__ == '__main__':
    main()
