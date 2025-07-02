import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd

from utils import P1Y2_P2Y2 as YP
from utils import (Model, calc_acc, calc_mrr, expectations, get_logger,
                   get_year_result_row)


def get_csv_path(task, embed, query_direction, weighted_mean, centering):
    csv_path = f"{embed}/{query_direction}/{embed}_{query_direction}_{task}"
    if weighted_mean:
        csv_path += "_wmean"
    else:
        csv_path += "_mean"
    if centering:
        csv_path += "_center"
    else:
        csv_path += "_nocenter"
    csv_path += ".csv"
    return csv_path


def main():
    top = 10
    K = 10
    np.random.seed(0)

    output_dir = Path("output/analogy_P1Y2_and_P2Y2")
    output_dir.mkdir(exist_ok=True, parents=True)

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
        year_dir = Path(f"data/embeddings/years/1781-{year}/min_count30/")
        model_file = year_dir / "embedding_e10s1e-05l0.025w5n5m30d300"
        model = np.fromfile(str(model_file), dtype=np.float64)
        model = model.reshape(-1, 300)
        word_to_id_path = year_dir / "word_to_id.pkl"
        with open(word_to_id_path, "rb") as f:
            word_to_id = pkl.load(f)
        assert len(word_to_id) == len(model)
        model = Model(model, word_to_id)

        # query directions
        tmp_stats_results = []
        for query_type, ans_type in (("drug", "gene"), ("gene", "drug")):
            query_direction = f"{query_type}2{ans_type}"
            stats = {"year": year, "query_direction": query_direction}
            logger.info(f"query_direction: {query_direction}")

            query_prefix = prefix_dict[query_type]
            cand_prefix = prefix_dict[ans_type]

            # prepare ans_cpt_freq
            rel_df = pd.read_csv(
                f"output/year/{year}_{query_direction}s_relation_for_Y2.csv"
            )

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
                for ans_cpt in Uy_ans_cpts + Ly_ans_cpts:
                    if ans_cpt not in ans_cpt_freq:
                        ans_cpt_freq[ans_cpt] = 0
                    ans_cpt_freq[ans_cpt] += 1
            ans_cpt_keys = np.array(list(ans_cpt_freq.keys()))
            ans_cpt_values = np.array(list(ans_cpt_freq.values()))
            ans_cpt_ps = ans_cpt_values / np.sum(ans_cpt_values)

            # prepare common domains and cands
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
            p1y2_query_path = f"output/year/{year}_{query_direction}_P1Y2.pkl"
            with open(p1y2_query_path, "rb") as f:
                p1y2_query = pkl.load(f)
            p2y2_query_path = f"output/year/{year}_{query_direction}_P2Y2.pkl"
            with open(p2y2_query_path, "rb") as f:
                p2y2_query = pkl.load(f)

            # calcuate candidate norms for cosine similarity
            cands_norm = np.linalg.norm(cands, axis=-1)
            cands_norm = np.maximum(cands_norm, 1e-8)
            cands_mean = np.mean(cands, axis=0)
            cands_c = cands - cands_mean[np.newaxis, ...]
            cands_norm_c = np.linalg.norm(cands_c, axis=-1)
            cands_norm_c = np.maximum(cands_norm_c, 1e-8)

            def nearest_words(pred, known_cpts, n=top, centering=False):
                # size: vocab_num
                if centering:
                    pred = pred - cands_mean
                    cos = np.dot(cands_c, pred) / (
                        np.linalg.norm(pred) * cands_norm_c)
                else:
                    cos = np.dot(cands, pred) / (
                        np.linalg.norm(pred) * cands_norm)
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

            def calc_rr(pred, ans_cpts, known_cpts, centering=False):
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

            # prepare relation vectors
            hsa2v = dict()
            for hsa, rels in p1y2_query.items():
                v = []
                for dic in rels:
                    query_cpt = dic["query_cpt"]
                    Ly_ans_cpts = dic["Ly_ans_cpts"]
                    if isinstance(Ly_ans_cpts, str):
                        Ly_ans_cpts = eval(Ly_ans_cpts)
                    for Ly_ans_cpt in Ly_ans_cpts:
                        v.append(model[Ly_ans_cpt] - model[query_cpt])
                if len(v) == 0:
                    continue
                hsa2v[hsa] = np.mean(v, axis=0)

            if len(hsa2v) == 0:
                continue

            # year pathway setting P1Y2
            # stats
            drugs_U_list = []
            total_L_mean_list = []
            total_U_mean_list = []
            for hsa, rels in p1y2_query.items():
                if hsa not in hsa2v:
                    continue

                hsa_drugs_U_list = []
                L_num_list = []
                U_num_list = []
                for dic in rels:
                    query_cpt = dic["query_cpt"]
                    Ly_ans_cpts = dic["Ly_ans_cpts"]
                    if isinstance(Ly_ans_cpts, str):
                        Ly_ans_cpts = eval(Ly_ans_cpts)

                    Uy_ans_cpts = dic["Uy_ans_cpts"]
                    if isinstance(Uy_ans_cpts, str):
                        Uy_ans_cpts = eval(Uy_ans_cpts)
                    if len(Uy_ans_cpts) == 0:
                        continue
                    hsa_drugs_U_list.append(query_cpt)
                    assert len(Ly_ans_cpts) + len(Uy_ans_cpts) > 0
                    L_num_list.append(len(Ly_ans_cpts))
                    U_num_list.append(len(Uy_ans_cpts))

                if len(hsa_drugs_U_list):
                    drugs_U_list.append(hsa_drugs_U_list)
                    total_L_mean_list.append(np.mean(L_num_list))
                    total_U_mean_list.append(np.mean(U_num_list))

            stats[YP.stats("sum_Dp_U", query_direction)] = sum(
                [len(drugs) for drugs in drugs_U_list]
            )
            stats[YP.stats("E_dp_L", query_direction)
                  ] = np.mean(total_L_mean_list)
            stats[YP.stats("E_dp_U", query_direction)
                  ] = np.mean(total_U_mean_list)

            # random baseline
            logger.info(
                f"year: {year} "
                f"query_direction: {query_direction} "
                "setting: P1Y2 "
                f"method: random K: {K}"
            )
            rows = []
            query_num = 0
            for hsa, rels in p1y2_query.items():
                if hsa not in hsa2v:
                    continue

                for dic in rels:
                    query_cpt = dic["query_cpt"]
                    Ly_ans_cpts = dic["Ly_ans_cpts"]
                    if isinstance(Ly_ans_cpts, str):
                        Ly_ans_cpts = eval(Ly_ans_cpts)
                    Uy_ans_cpts = dic["Uy_ans_cpts"]
                    if isinstance(Uy_ans_cpts, str):
                        Uy_ans_cpts = eval(Uy_ans_cpts)

                    if len(Uy_ans_cpts) == 0:
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
                            "ans_cpts": Uy_ans_cpts,
                            "cand_cpts": cand_cpts,
                            "cand_sims": [0] * top,
                            "rr": calc_random_rr(all_cand_cpts, Uy_ans_cpts),
                        }
                        rows.append(row)

            if len(rows) > 0:
                result_row = get_year_result_row(
                    year,
                    query_direction,
                    "P1Y2_random",
                    calc_acc(rows, top),
                    calc_mrr(rows),
                )
                result_row["query_num"] = query_num
                logger.info(result_row)
                results.append(result_row)

            # vector analogy
            logger.info(
                f"year: {year} "
                f"query_direction: {query_direction} "
                "setting: P1Y2 "
                "method: analogy "
                f"wmean: {weighted_mean} center: {centering}"
            )
            rows = []
            query_num = 0
            for hsa, rels in p1y2_query.items():
                if hsa not in hsa2v:
                    continue
                v = hsa2v[hsa]
                for dic in rels:
                    query_cpt = dic["query_cpt"]
                    Ly_ans_cpts = dic["Ly_ans_cpts"]
                    if isinstance(Ly_ans_cpts, str):
                        Ly_ans_cpts = eval(Ly_ans_cpts)
                    Uy_ans_cpts = dic["Uy_ans_cpts"]
                    if isinstance(Uy_ans_cpts, str):
                        Uy_ans_cpts = eval(Uy_ans_cpts)

                    if len(Uy_ans_cpts) == 0:
                        continue
                    query_num += 1
                    pred = model[query_cpt] + v
                    cand_cpts, cand_sims = nearest_words(
                        pred, Ly_ans_cpts, centering=centering
                    )
                    rr = calc_rr(
                        pred, Uy_ans_cpts, Ly_ans_cpts, centering=centering)
                    row = {
                        "query_cpt": query_cpt,
                        "ans_cpts": Uy_ans_cpts,
                        "cand_cpts": cand_cpts,
                        "cand_sims": cand_sims,
                        "rr": rr,
                    }
                    rows.append(row)

            if len(rows) > 0:
                result_row = get_year_result_row(
                    year,
                    query_direction,
                    "P1Y2",
                    calc_acc(rows, top),
                    calc_mrr(rows),
                    weighted_mean,
                    centering,
                )
                result_row["query_num"] = query_num
                logger.info(result_row)
                results.append(result_row)

            # year pathway setting P2Y2
            # stats
            drugs_U_list = []
            total_L_mean_list = []
            total_U_mean_list = []
            for hsa, rels in p2y2_query.items():
                if hsa not in hsa2v:
                    continue

                hsa_drugs_U_list = []
                L_num_list = []
                U_num_list = []
                for dic in rels:
                    query_cpt = dic["query_cpt"]

                    Ly_ans_cpts = dic["Ly_ans_cpts"]
                    if isinstance(Ly_ans_cpts, str):
                        Ly_ans_cpts = eval(Ly_ans_cpts)

                    Uy_ans_cpts = dic["Uy_ans_cpts"]
                    if isinstance(Uy_ans_cpts, str):
                        Uy_ans_cpts = eval(Uy_ans_cpts)

                    if len(Uy_ans_cpts) == 0:
                        continue
                    hsa_drugs_U_list.append(query_cpt)
                    assert len(Ly_ans_cpts) + len(Uy_ans_cpts) > 0
                    L_num_list.append(len(Ly_ans_cpts))
                    U_num_list.append(len(Uy_ans_cpts))

                if len(hsa_drugs_U_list):
                    drugs_U_list.append(hsa_drugs_U_list)
                    total_L_mean_list.append(np.mean(L_num_list))
                    total_U_mean_list.append(np.mean(U_num_list))

            stats[YP.stats("sum_cal_cap_Dp_U", query_direction)] = sum(
                [len(drugs) for drugs in drugs_U_list]
            )
            stats[YP.stats("E_d_L", query_direction)
                  ] = np.mean(total_L_mean_list)
            stats[YP.stats("E_d_U", query_direction)
                  ] = np.mean(total_U_mean_list)

            # random baseline
            logger.info(
                f"year: {year} "
                f"query_direction: {query_direction} "
                "setting: P2Y2 "
                f"method: random K: {K}"
            )
            rows = []
            query_num = 0
            for hsa, rels in p2y2_query.items():
                if hsa not in hsa2v:
                    continue
                v = hsa2v[hsa]

                for dic in rels:
                    query_cpt = dic["query_cpt"]
                    Ly_ans_cpts = dic["Ly_ans_cpts"]
                    if isinstance(Ly_ans_cpts, str):
                        Ly_ans_cpts = eval(Ly_ans_cpts)
                    Uy_ans_cpts = dic["Uy_ans_cpts"]
                    if isinstance(Uy_ans_cpts, str):
                        Uy_ans_cpts = eval(Uy_ans_cpts)

                    if len(Uy_ans_cpts) == 0:
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
                            "ans_cpts": Uy_ans_cpts,
                            "cand_cpts": cand_cpts,
                            "cand_sims": cand_sims,
                            "rr": calc_random_rr(all_cand_cpts, Uy_ans_cpts),
                        }
                        rows.append(row)

            if len(rows) > 0:
                result_row = get_year_result_row(
                    year,
                    query_direction,
                    "P2Y2_random",
                    calc_acc(rows, top),
                    calc_mrr(rows),
                )
                result_row["query_num"] = query_num
                logger.info(result_row)
                results.append(result_row)

            # vector analogy
            logger.info(
                f"year: {year} "
                f"query_direction: {query_direction} "
                "setting: P2Y2 "
                "method: analogy "
                f"wmean: {weighted_mean} center: {centering}"
            )
            rows = []
            query_num = 0
            for hsa, rels in p2y2_query.items():
                if hsa not in hsa2v:
                    continue
                v = hsa2v[hsa]

                for dic in rels:
                    query_cpt = dic["query_cpt"]
                    Ly_ans_cpts = dic["Ly_ans_cpts"]
                    if isinstance(Ly_ans_cpts, str):
                        Ly_ans_cpts = eval(Ly_ans_cpts)
                    Uy_ans_cpts = dic["Uy_ans_cpts"]
                    if isinstance(Uy_ans_cpts, str):
                        Uy_ans_cpts = eval(Uy_ans_cpts)

                    if len(Uy_ans_cpts) == 0:
                        continue
                    query_num += 1
                    pred = model[query_cpt] + v
                    cand_cpts, cand_sims = nearest_words(
                        pred, Ly_ans_cpts, centering=centering
                    )
                    rr = calc_rr(
                        pred, Uy_ans_cpts, Ly_ans_cpts, centering=centering)
                    row = {
                        "query_cpt": query_cpt,
                        "ans_cpts": Uy_ans_cpts,
                        "cand_cpts": cand_cpts,
                        "cand_sims": cand_sims,
                        "rr": rr,
                    }
                    rows.append(row)

            if len(rows) > 0:
                result_row = get_year_result_row(
                    year,
                    query_direction,
                    "P2Y2",
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
        if len(tmp_stats_results) != 2:
            continue
        first = tmp_stats_results.iloc[0]
        second = tmp_stats_results.iloc[1]
        row = dict()
        for col in ["year"] + YP.all:
            if col == "query_direction":
                continue

            fv = first[col]
            sv = second[col]

            # na check
            if pd.isna(fv):
                v = sv
            elif pd.isna(sv):
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
    stats_result_df = stats_result_df[["year"] + YP.all]
    stats_result_df.to_csv(output_dir / "stats.csv", index=False)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
