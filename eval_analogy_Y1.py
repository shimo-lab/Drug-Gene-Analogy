import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import (Y1, Model, calc_acc, calc_mrr, expectations, get_logger,
                   get_year_result_row)


def main():
    top = 10
    K = 10
    np.random.seed(0)

    output_dir = Path("output/analogy_Y1")
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "output.log"
    logger = get_logger(log_file)

    prefix_dict = {"drug": "chemical_mesh_", "gene": "gene"}
    weighted_mean = True
    centering = True

    results = []
    stats_results = []
    years = list(range(1975, 2021, 5)) + [2023]
    for year in years:
        # load model
        year_dir = Path(f"data/embeddings/years/1781-{year}/min_count30")
        model_file = year_dir / "embedding_e10s1e-05l0.025w5n5m30d300"
        model = np.fromfile(str(model_file), dtype=np.float64)
        model = model.reshape(-1, 300)
        word_to_id_path = year_dir / "word_to_id.pkl"
        with open(word_to_id_path, "rb") as f:
            word_to_id = pkl.load(f)
        assert len(word_to_id) == len(model)
        logger.info(f"vocab size={len(model)}")
        model = Model(model, word_to_id)

        # query directions
        tmp_stats_results = []
        for query_type, ans_type in (("drug", "gene"), ("gene", "drug")):
            query_direction = f"{query_type}2{ans_type}"
            stats = {"year": year, "query_direction": query_direction}

            query_prefix = prefix_dict[query_type]
            cand_prefix = prefix_dict[ans_type]

            # prepare common domains and cands
            # in the case of drug2gene,
            # domains are drugs $\mathcal{D}^y$,
            # cands are genes $\mathcal{G}^y$
            domains = []
            cands = []
            to_id = dict()
            to_word = []

            for k in word_to_id.keys():
                v = model[k]
                if k[: len(query_prefix)] == query_prefix:
                    domains.append(v)
                if k[: len(cand_prefix)] == cand_prefix:
                    cands.append(v)
                    to_id[k] = len(to_id)
                    to_word.append(k)

            domains = np.array(domains)
            cands = np.array(cands)

            stats[Y1.stats("cal_D", query_direction)] = len(domains)
            stats[Y1.stats("cal_G", query_direction)] = len(cands)

            # prepare queries
            rel_df = pd.read_csv(
                f"output/relation/homemade_{query_direction}s_relation.csv"
            )

            query_cpt_col = f"{query_type}_concept"
            ans_cpts_col = f"{ans_type}_concepts"
            assert len(rel_df) == len(rel_df[query_cpt_col].unique())

            # define query and answer set
            # in the case of drug2gene,
            # query is drug $d\in D^y$, answer is gene set $[d]^y$
            query_cpt2ans_cpts = dict()
            total_ans_list = []
            ans_cpt_freq = dict()
            for _, row in rel_df.iterrows():
                query_cpt = row[query_cpt_col]

                if query_cpt not in model:
                    continue

                ans_cpts = eval(row[ans_cpts_col])
                ans_cpts = [ans_cpt for ans_cpt in ans_cpts
                            if ans_cpt in model]

                if len(ans_cpts) == 0:
                    continue

                query_cpt2ans_cpts[query_cpt] = ans_cpts
                total_ans_list.append(ans_cpts)
                for ans_cpt in ans_cpts:
                    if ans_cpt not in ans_cpt_freq:
                        ans_cpt_freq[ans_cpt] = 0
                    ans_cpt_freq[ans_cpt] += 1
            ans_cpt_keys = np.array(list(ans_cpt_freq.keys()))
            ans_cpt_values = np.array(list(ans_cpt_freq.values()))
            ans_cpt_ps = ans_cpt_values / np.sum(ans_cpt_values)

            stats[Y1.stats("cal_R", query_direction)] = sum(
                [len(set(ans_cpts)) for ans_cpts in total_ans_list]
            )
            stats[Y1.stats("D", query_direction)] = len(query_cpt2ans_cpts)
            stats[Y1.stats("G", query_direction)] = len(
                set([ans_cpt for ans_cpts in total_ans_list
                     for ans_cpt in ans_cpts])
            )
            stats[Y1.stats("E_d", query_direction)] = np.mean(
                [len(set(ans_cpts)) for ans_cpts in total_ans_list]
            )

            # calcuate candidate norms for cosine similarity
            cands_norm = np.linalg.norm(cands, axis=-1)
            cands_norm = np.maximum(cands_norm, 1e-8)
            cands_mean = np.mean(cands, axis=0)
            cands_c = cands - cands_mean[np.newaxis, ...]
            cands_norm_c = np.linalg.norm(cands_c, axis=-1)
            cands_norm_c = np.maximum(cands_norm_c, 1e-8)

            def nearest_words(pred, n=top, centering=False):
                # size: vocab_num
                if centering:
                    pred = pred - cands_mean
                    cos = np.dot(cands_c, pred) / (
                        np.linalg.norm(pred) * cands_norm_c)
                else:
                    cos = np.dot(cands, pred) / (
                        np.linalg.norm(pred) * cands_norm)
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

            def calc_rr(pred, ans_cpts, centering=False):
                # size: vocab_num
                if centering:
                    pred = pred - cands_mean
                    cos = np.dot(cands_c, pred) / (
                        np.linalg.norm(pred) * cands_norm_c)
                else:
                    cos = np.dot(cands, pred) / (
                        np.linalg.norm(pred) * cands_norm)

                # ans ids
                ans_ids = [to_id[ans_cpt] for ans_cpt in ans_cpts]

                # choose best ans_cpt
                best_ans_id = ans_ids[np.argmax(cos[ans_ids])]
                best_ans_cos = cos[best_ans_id]
                rank = np.sum(cos > best_ans_cos)
                return 1 / (rank + 1)

            def calc_random_rr(all_cand_cpts, ans_cpts):
                for ith, cand_cpt in enumerate(all_cand_cpts):
                    if cand_cpt in ans_cpts:
                        return 1 / (ith + 1)
                return 0

            # random baseline
            logger.info(
                f"year: {year} "
                f"query_direction: {query_direction} "
                "setting: Y1 "
                f"method: random K: {K}"
            )
            rows = []
            query_num = 0
            for query_cpt, ans_cpts in tqdm(query_cpt2ans_cpts.items()):
                if query_cpt not in model:
                    continue
                query_num += 1

                for _ in range(K):
                    # sample random cands by top
                    # distribution is based on ans_cpts frequency
                    all_cand_cpts = np.random.choice(
                        ans_cpt_keys,
                        size=len(ans_cpt_keys),
                        replace=False,
                        p=ans_cpt_ps,
                    )
                    cand_cpts = list(all_cand_cpts)[:top]

                    row = {
                        "query_cpt": query_cpt,
                        "ans_cpts": ans_cpts,
                        "cand_cpts": cand_cpts,
                        "cand_sims": [0] * top,
                        "rr": calc_random_rr(all_cand_cpts, ans_cpts),
                    }
                    rows.append(row)

            result_row = get_year_result_row(
                year,
                query_direction,
                "Y1_random",
                calc_acc(rows, top),
                calc_mrr(rows)
            )
            result_row["query_num"] = query_num
            logger.info(result_row)
            results.append(result_row)

            # setting Y1
            # we can use query_cpt2ans_cpts
            v = []
            for query_cpt, ans_cpts in query_cpt2ans_cpts.items():
                if query_cpt not in model:
                    continue

                ans_cpts = [ans_cpt for ans_cpt in ans_cpts
                            if ans_cpt in model]
                if len(ans_cpts) == 0:
                    continue

                for ans_cpt in ans_cpts:
                    v.append(model[ans_cpt] - model[query_cpt])
            v = np.mean(v, axis=0)

            logger.info(
                f"year: {year} "
                f"query_direction: {query_direction} "
                "setting: Y1 "
                "method: analogy "
                f"wmean: {weighted_mean} center: {centering}"
            )
            rows = []
            query_num = 0
            for query_cpt, ans_cpts in tqdm(query_cpt2ans_cpts.items()):
                if query_cpt not in model:
                    continue

                ans_cpts = [ans_cpt for ans_cpt in ans_cpts
                            if ans_cpt in model]
                if len(ans_cpts) == 0:
                    continue
                query_num += 1

                pred = model[query_cpt] + v
                cand_cpts, cand_sims = nearest_words(pred, centering=centering)
                rr = calc_rr(pred, ans_cpts, centering=centering)
                row = {
                    "query_cpt": query_cpt,
                    "ans_cpts": ans_cpts,
                    "cand_cpts": cand_cpts,
                    "cand_sims": cand_sims,
                    "rr": rr,
                }
                rows.append(row)
            result_row = get_year_result_row(
                year,
                query_direction,
                "Y1",
                calc_acc(rows, top),
                calc_mrr(rows),
                weighted_mean,
                centering,
            )
            result_row["query_num"] = query_num
            logger.info(result_row)
            results.append(result_row)

            # save stats
            tmp_stats_results.append(stats)

        tmp_stats_results = pd.DataFrame(tmp_stats_results)
        assert len(tmp_stats_results) == 2
        first = tmp_stats_results.iloc[0]
        second = tmp_stats_results.iloc[1]
        row = dict()
        for col in ["year"] + Y1.all:
            if col == "query_direction":
                continue

            fv = first[col]
            sv = second[col]

            # na check
            if pd.isna(fv):
                assert not pd.isna(sv)
                v = sv
            elif pd.isna(sv):
                assert not pd.isna(fv)
                v = fv
            else:
                # same check
                assert fv == sv
                v = fv

            if col != "embed" and col not in expectations and v is not None:
                v = int(v)
            row[col] = v
        stats_results.append(row)

    # save results
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_dir / "result.csv", index=False)

    # save stats
    stats_result_df = pd.DataFrame(stats_results)
    stats_result_df = stats_result_df[["year"] + Y1.all]
    stats_result_df.to_csv(output_dir / "stats.csv", index=False)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
