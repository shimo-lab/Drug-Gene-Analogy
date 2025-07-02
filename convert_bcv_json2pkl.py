import json
import pickle as pkl
from pathlib import Path

import numpy as np
from tqdm import tqdm


def main():
    json_path = './data/embeddings/concept_skipgram.json'
    Path('./output/embeddings/').mkdir(parents=True, exist_ok=True)
    pkl_path = './output/embeddings/concept_skipgram.pkl'

    with open(json_path, 'r') as f:
        model = json.load(f)

    numpy_model = dict()
    for k, v in tqdm(model.items()):
        numpy_model[k] = np.array(v)

    with open(pkl_path, 'wb') as f:
        pkl.dump(numpy_model, f)


if __name__ == '__main__':
    main()
