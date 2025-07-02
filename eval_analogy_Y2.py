import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import (Y2, Model, calc_acc, calc_mrr, expectations, get_logger,
                   get_year_result_row)


def main():
    top = 10
    K = 10
    np.random.seed(0)

    output_dir = Path("output/analogy_Y2")
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "output.log"
    logger = get_logger(log_file)

    prefix_dict = {"drug": "chemical_mesh_", "gene": "gene"}
    weighted_mean = True
    centering = True

    results = []
    stats_results = []
    years = list(range(1975, 2021, 5))
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
            # cands are genes $\mathcal{G}^y\[d]^{y \mid U_y}$
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

            # prepare queries
            rel_df = pd.read_csv(
                f"output/year/{year}_{query_direction}s_relation_for_Y2.csv"
            )

            # define query and answer set
            # in the case of drug2gene,
            # query is drug $d\in D^{y \mid U_y}$,
            # answer is gene set $[d]^{y \mid U_y}$

            query_cpt2knw_cpts = dict()
            query_cpt2ans_cpts = dict()
            query_cpt2knw_ans_cpts = dict()
            ans_cpt_freq = dict()
            for _, row in rel_df.iterrows():
                query_cpt = row["query_cpt"]
                Ly_ans_cpts = row["Ly_ans_cpts"]
                if isinstance(Ly_ans_cpts, str):
                    Ly_ans_cpts = eval(Ly_ans_cpts)
                Uy_ans_cpts = row["Uy_ans_cpts"]
                if isinstance(Uy_ans_cpts, str):
                    Uy_ans_cpts = eval(Uy_ans_cpts)
                assert len(Ly_ans_cpts) + len(Uy_ans_cpts) > 0
                if len(Ly_ans_cpts):
                    query_cpt2knw_cpts[query_cpt] = Ly_ans_cpts
                if len(Uy_ans_cpts):
                    query_cpt2ans_cpts[query_cpt] = Uy_ans_cpts
                query_cpt2knw_ans_cpts[query_cpt] = Ly_ans_cpts + Uy_ans_cpts
                for ans_cpt in Uy_ans_cpts + Ly_ans_cpts:
                    if ans_cpt not in ans_cpt_freq:
                        ans_cpt_freq[ans_cpt] = 0
                    ans_cpt_freq[ans_cpt] += 1
            ans_cpt_keys = np.array(list(ans_cpt_freq.keys()))
            ans_cpt_values = np.array(list(ans_cpt_freq.values()))
            ans_cpt_ps = ans_cpt_values / np.sum(ans_cpt_values)

            stats[Y2.stats("cal_R_L", query_direction)] = sum(
                [len(v) for v in query_cpt2knw_cpts.values()]
            )
            stats[Y2.stats("cal_R_U", query_direction)] = sum(
                [len(v) for v in query_cpt2ans_cpts.values()]
            )

            stats[Y2.stats("D_U", query_direction)] = len(query_cpt2ans_cpts)
            stats[Y2.stats("G_U", query_direction)] = len(
                set([v for vs in query_cpt2ans_cpts.values() for v in vs])
            )

            stats[Y2.stats("E_d_U", query_direction)] = np.mean(
                [len(v) for v in query_cpt2ans_cpts.values()]
            )
            stats[Y2.stats("E_d_L", query_direction)] = np.mean(
                [len(v) for v in query_cpt2knw_cpts.values()]
            )

            # calcuate candidate norms for cosine similarity
            cands_norm = np.linalg.norm(cands, axis=-1)
            cands_norm = np.maximum(cands_norm, 1e-8)
            cands_mean = np.mean(cands, axis=0)
            cands_c = cands - cands_mean[np.newaxis, ...]
            cands_norm_c = np.linalg.norm(cands_c, axis=-1)
            cands_norm_c = np.maximum(cands_norm_c, 1e-8)

            def nearest_words(pred, known_cpts, n=top, centering=False):
                # size: vocab_num
                pred = pred - cands_mean
                cos = np.dot(cands_c, pred) / (
                    np.linalg.norm(pred) * cands_norm_c)

                # top n index
                res = len(known_cpts)
                index = np.argpartition(-cos, n + res)[: n + res]
                top = index[np.argsort(-cos[index])]
                word_list = []
                sim_list = []
                for word_id in top:
                    word = to_word[word_id]
                    if word in known_cpts:
                        continue
                    sim = cos[word_id]
                    word_list.append(word)
                    sim_list.append(sim)
                    if len(word_list) == n:
                        break
                return word_list, sim_list

            def calc_rr(pred, ans_cpts, known_cpts):
                # size: vocab_num
                pred = pred - cands_mean
                cos = np.dot(cands_c, pred) / (
                    np.linalg.norm(pred) * cands_norm_c)

                # ans ids
                ans_ids = [to_id[ans_cpt] for ans_cpt in ans_cpts]

                # choose best ans_cpt
                best_ans_id = ans_ids[np.argmax(cos[ans_ids])]
                best_ans_cos = cos[best_ans_id]
                rank = np.sum(cos > best_ans_cos)

                # remove known_cpts
                known_ids = [to_id[known_cpt] for known_cpt in known_cpts]
                known_cos = cos[known_ids]
                rank = rank - np.sum(known_cos > best_ans_cos)

                return 1 / (rank + 1)

            def calc_random_rr(all_cand_cpts, ans_cpts):
                for ith, cand_cpt in enumerate(all_cand_cpts):
                    if cand_cpt in ans_cpts:
                        return 1 / (ith + 1)
                return 0

            # setting Y2
            # random baseline
            logger.info(
                f"year: {year} "
                f"query_direction: {query_direction} "
                "setting: Y2 "
                f"method: random K: {K}"
            )
            rows = []
            query_num = 0
            for query_cpt, ans_cpts in tqdm(query_cpt2ans_cpts.items()):
                if len(ans_cpts) == 0:
                    continue
                if query_cpt not in query_cpt2knw_cpts:
                    known_cpts = []
                else:
                    known_cpts = query_cpt2knw_cpts[query_cpt]
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
                query_num += 1
            result_row = get_year_result_row(
                year,
                query_direction,
                "Y2_random",
                calc_acc(rows, top),
                calc_mrr(rows)
            )
            result_row["query_num"] = query_num
            logger.info(result_row)
            results.append(result_row)

            # vector analogy
            v = []
            for query_cpt, ans_cpts in query_cpt2knw_cpts.items():
                for ans_cpt in ans_cpts:
                    v.append(model[ans_cpt] - model[query_cpt])
            print(f"total_knw_num: {len(v)}")
            v = np.mean(v, axis=0)

            logger.info(
                f"year: {year} "
                f"query_direction: {query_direction} "
                "setting: Y2 "
                "method: analogy "
                f"wmean: {weighted_mean} center: {centering}"
            )
            rows = []
            query_num = 0
            for query_cpt, ans_cpts in tqdm(query_cpt2ans_cpts.items()):
                if len(ans_cpts) == 0:
                    continue
                if query_cpt not in query_cpt2knw_cpts:
                    known_cpts = []
                else:
                    known_cpts = query_cpt2knw_cpts[query_cpt]
                pred = model[query_cpt] + v
                cand_cpts, cand_sims = nearest_words(pred, known_cpts)
                rr = calc_rr(pred, ans_cpts, known_cpts)
                row = {
                    "query_cpt": query_cpt,
                    "ans_cpts": ans_cpts,
                    "cand_cpts": cand_cpts,
                    "cand_sims": cand_sims,
                    "rr": rr,
                }
                rows.append(row)
                query_num += 1
            result_row = get_year_result_row(
                year,
                query_direction,
                "Y2",
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
        for col in ["year"] + Y2.all:
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
    stats_result_df = stats_result_df[["year"] + Y2.all]
    stats_result_df.to_csv(output_dir / "stats.csv", index=False)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
