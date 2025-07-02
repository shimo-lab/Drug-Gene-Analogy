import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def text2dict(raw_text):

    ID_FIRST = [
        'ENTRY',
        'NETWORK',
        'ELEMENT',
        'DRUG',
        'ORGANISM',
        'GENE',
        'COMPOUND',
        'REL_PATHWAY',
    ]

    lines = [line for line in raw_text.split('\n')
             if len(line.strip()) > 0 and line != '///']

    dic = defaultdict(list)
    ratj = None
    for line in lines:
        if len(line.strip()) == 0:
            continue
        if line[:2] == '  ':
            line = line[2:]
        if line[:len('        ')] != '        ':
            # attribute line
            splits = [x for x in line.split() if len(x) > 0]
            attr = splits[0]
            if attr == 'REFERENCE':
                if len(splits) == 1:
                    splits.append('')
                assert len(splits) == 2
                if ratj is not None:
                    dic['REFERENCES'].append(ratj)
                ratj = OrderedDict(REFERENCE=splits[1])
            elif attr == 'JOURNAL':
                j_name = ' '.join(splits[1:])
                ratj['JOURNAL'] = OrderedDict(NAME=j_name)
            elif attr in (
                    'AUTHORS',
                    'TITLE'):
                ratj[attr] = ' '.join(splits[1:])
            else:
                if attr in ID_FIRST:
                    dic[attr].append(OrderedDict(
                        {'ID': splits[1], 'NAME': ' '.join(splits[2:])}))
                else:
                    dic[attr].append(' '.join(splits[1:]))
        else:
            if attr == 'JOURNAL':
                ratj['JOURNAL']['DOI'] = line.strip()
            else:
                if attr in ID_FIRST:
                    splits = [x for x in line.split() if len(x) > 0]
                    dic[attr].append(OrderedDict(
                        {'ID': splits[0], 'NAME': ' '.join(splits[1:])}))
                else:
                    dic[attr].append(line.strip())

    if ratj is not None:
        dic['REFERENCES'].append(ratj)

    for attr in dic.keys():
        if attr in ['ENTRY', 'NAME', 'CLASS',
                    'PATHWAY_MAP', 'ORGANISM', 'KO_PATHWAY']:
            dic[attr] = dic[attr][0]

    return dic


def main():

    tsv_path = 'data/pathway/hsa.tsv'
    df = pd.read_table(tsv_path, header=None, names=['hsa', 'name'])
    Path('output/pathway/json').mkdir(parents=True, exist_ok=True)

    for hsa in tqdm(sorted(df['hsa'].values)):
        print(hsa)
        text_path = f'data/pathway/txt/{hsa}.txt'
        with open(text_path, 'r') as f:
            raw_text = f.read()

        dic = text2dict(raw_text)

        json_path = f'output/pathway/json/{hsa}.json'
        with open(json_path, 'w') as f:
            json.dump(dic, f, indent=4)


if __name__ == '__main__':
    main()
