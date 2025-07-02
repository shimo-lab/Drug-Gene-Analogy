import pickle as pkl
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from utils import Model, calc_mrr, get_logger


def main():

    logger = get_logger()

    embed = 'homemade'
    logger.info(f"Using {embed} embeddings")

    logger.info("Loading embeddings...")
    model_file = (
        "data/embeddings/years/1781-2023/min_count30/"
        "embedding_e10s1e-05l0.025w5n5m30d300"
    )
    model = np.fromfile(model_file, dtype=np.float64)
    model = model.reshape(-1, 300)
    word_to_id_path = (
        "data/embeddings/years/1781-2023/min_count30/word_to_id.pkl"
    )
    with open(word_to_id_path, "rb") as f:
        word_to_id = pkl.load(f)
    assert len(model) == len(word_to_id)
    model = Model(model, word_to_id)

    data_root_dir = Path("data/kge_datasplit")

    search_names_list = [['gene'],]
    for search_names in search_names_list:
        results = []
        for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            logger.info(f"Processing p = {p}")
            data_dir = data_root_dir / f"{p:.1f}/setting_P1-split"
            state2df = {}
            cands = []
            to_id = {}
            to_word = []
            for state in ['train', 'valid', 'test']:
                data_path = data_dir / f"{state}.txt"
                df = pd.read_csv(data_path, sep="\t", header=None)
                df.columns = ['drug_name', 'relation', 'gene_name']
                state2df[state] = df

                for _, row in df.iterrows():
                    for name in search_names:
                        k = row[f"{name}_name"]
                        if k in to_id:
                            continue
                        assert k in word_to_id
                        to_id[k] = len(to_id)
                        to_word.append(k)
                        v = model[k]
                        cands.append(v)

            cands_norm = np.linalg.norm(cands, axis=-1)
            cands_norm = np.maximum(cands_norm, 1e-8)
            cands_mean = np.mean(cands, axis=0)
            cands_c = cands - cands_mean[np.newaxis, ...]
            cands_norm_c = np.linalg.norm(cands_c, axis=-1)
            cands_norm_c = np.maximum(cands_norm_c, 1e-8)

            def nearest_words(v_pred, n=10):

                # size: vocab_num
                v_pred = v_pred - cands_mean
                cos = np.dot(cands_c, v_pred) / (
                    np.linalg.norm(v_pred) * cands_norm_c)

                # top n index
                index = np.argpartition(-cos, n)[:n]
                top = index[np.argsort(-cos[index])]
                word_list = []
                sim_list = []
                for word_id in top:
                    word = to_word[word_id]
                    sim = cos[word_id]
                    word_list.append(word)
                    sim_list.append(sim)

                return word_list, sim_list

            def calc_rr(v_pred, ans_gene_names):
                # size: vocab_num
                v_pred = v_pred - cands_mean
                cos = np.dot(cands_c, v_pred) / (
                    np.linalg.norm(v_pred) * cands_norm_c)

                # ans ids
                ans_ids = [to_id[ans_gene_name]
                           for ans_gene_name in ans_gene_names]

                # choose best ans_gene_name
                best_ans_id = ans_ids[np.argmax(cos[ans_ids])]
                best_ans_cos = cos[best_ans_id]
                rank = np.sum(cos > best_ans_cos)
                return 1 / (rank + 1)

            def calc_acc(rows, top):
                tots = []
                for row in rows:
                    ans_gene_names = set(row['ans_gene_names'])
                    c = [1]*top
                    for k in range(top):
                        pred_gene_name = row['pred_gene_names'][k]
                        if pred_gene_name in ans_gene_names:
                            break
                        else:
                            c[k] = 0
                    tots.append(c)
                accs = np.mean(np.array(tots), axis=0)
                return list(accs)

            for train_states in [['train'],]:
                train_df = pd.concat(
                    [state2df[state] for state in train_states])
                test_df = state2df['test']

                # compute relation vectors
                v_rs = []
                relation2v_rs = dict()
                for _, row in train_df.iterrows():
                    drug_name = row['drug_name']
                    gene_name = row['gene_name']
                    relation = row['relation']

                    v_d = model[drug_name]
                    v_g = model[gene_name]
                    v_r = v_g - v_d
                    v_rs.append(v_r)
                    if relation not in relation2v_rs:
                        relation2v_rs[relation] = []
                    relation2v_rs[relation].append(v_r)

                v_r = np.mean(v_rs, axis=0)
                for relation, v_rs in relation2v_rs.items():
                    relation2v_rs[relation] = np.mean(v_rs, axis=0)

                # make test set
                relation2drug2genes = defaultdict(lambda: defaultdict(set))
                for _, row in test_df.iterrows():
                    drug_name = row['drug_name']
                    gene_name = row['gene_name']
                    relation = row['relation']
                    relation2drug2genes[relation][drug_name].add(gene_name)

                rows = []
                count_use_vr = 0
                for relation in relation2drug2genes:
                    if relation not in relation2v_rs:
                        v_p = v_r
                        count_use_vr += 1
                    else:
                        v_p = relation2v_rs[relation]

                    drug2genes = relation2drug2genes[relation]
                    for drug_name, ans_gene_names in drug2genes.items():
                        v_pred = model[drug_name] + v_p
                        pred_gene_names, _ = nearest_words(v_pred, n=10)
                        rr = calc_rr(v_pred, ans_gene_names)

                        row = {
                            "drug_name": drug_name,
                            "ans_gene_names": ans_gene_names,
                            "pred_gene_names": pred_gene_names,
                            "rr": rr,
                        }
                        rows.append(row)

                accs = calc_acc(rows, 10)
                top1 = accs[0]
                top10 = accs[-1]
                mrr = calc_mrr(rows)

                result = {
                    "train_states": '_'.join(train_states),
                    "p": p,
                    "top1": top1,
                    "top10": top10,
                    "mrr": mrr,
                    "count_use_vr": count_use_vr,
                }
                logger.info(result)
                results.append(result)

        # save results
        result_df = pd.DataFrame(results)
        output_path = Path("data/kge_datasplit") /\
            f"analogy_search-{'-and-'.join(search_names)}.csv"
        result_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
