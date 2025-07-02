import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import A, Model, P, calc_acc, calc_mrr, expectations, get_logger


def get_result_row(
    embed,
    query_direction,
    method,
    acc,
    mrr,
    weighted_mean=None,
    centering=None
):
    result_row = {
        "embed": embed,
        "query_direction": query_direction,
        "method": method,
        "weighted_mean": weighted_mean,
        "centering": centering,
    }
    for ith, acc_ in enumerate(acc):
        result_row[f"top{ith+1}_acc"] = acc_
    result_row["mrr"] = mrr
    return result_row


def get_csv_path(task, embed, query_direction, weighted_mean, centering):
    csv_path = f"{embed}_{query_direction}_{task}"
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

    output_dir = Path("output/analogy")
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "output.log"
    logger = get_logger(log_file)

    results = []
    stats_results = []
    prefix_dict = {"drug": "Chemical_MESH_", "gene": "Gene"}
    for embed in ("bcv", "homemade"):
        # load model
        if embed == "bcv":
            model_file = "output/embeddings/concept_skipgram.pkl"
            with open(model_file, "rb") as f:
                model = pkl.load(f)

        elif embed == "homemade":
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
        else:
            raise NotImplementedError

        # query directions
        for query_type, ans_type in (("drug", "gene"), ("gene", "drug")):
            query_direction = f"{query_type}2{ans_type}"

            if embed == "homemade" and query_type == "drug":
                weighted_flags = [True, False]
                centering_flags = [True, False]
            else:
                weighted_flags = [True]
                centering_flags = [True]

            stats = {"embed": embed, "query_direction": query_direction}

            query_prefix = prefix_dict[query_type]
            cand_prefix = prefix_dict[ans_type]

            # prepare common domains and cands
            # in the case of drug2gene,
            # domains are drugs $\mathcal{D}$,
            # cands are genes $\mathcal{G}$
            domains = []
            cands = []
            to_id = dict()
            to_word = []
            if embed == "bcv":
                for k, v in model.items():
                    if k[: len(query_prefix)] == query_prefix:
                        domains.append(v)
                    if k[: len(cand_prefix)] == cand_prefix:
                        cands.append(v)
                        to_id[k] = len(to_id)
                        to_word.append(k)
            else:
                for k in word_to_id.keys():
                    v = model[k]
                    if k[: len(query_prefix)] == query_prefix.lower():
                        domains.append(v)
                    if k[: len(cand_prefix)] == cand_prefix.lower():
                        cands.append(v)
                        to_id[k] = len(to_id)
                        to_word.append(k)
            domains = np.array(domains)
            cands = np.array(cands)

            stats[A.stats("cal_D", query_direction)] = len(domains)
            stats[A.stats("cal_G", query_direction)] = len(cands)

            # prepare queries
            rel_df = pd.read_csv(
                f"output/relation/{embed}_{query_direction}s_relation.csv"
            )

            query_cpt_col = f"{query_type}_concept"
            ans_cpts_col = f"{ans_type}_concepts"
            assert len(rel_df) == len(rel_df[query_cpt_col].unique())

            # define query and answer set
            # in the case of drug2gene,
            # query is drug $d\in D$, answer is gene set $[d]$
            query_cpt2ans_cpts = dict()
            total_ans_list = []
            ans_cpt_freq = dict()
            for _, row in rel_df.iterrows():
                query_cpt = row[query_cpt_col]
                ans_cpts = eval(row[ans_cpts_col])
                query_cpt2ans_cpts[query_cpt] = ans_cpts
                total_ans_list.append(ans_cpts)
                for ans_cpt in ans_cpts:
                    if ans_cpt not in ans_cpt_freq:
                        ans_cpt_freq[ans_cpt] = 0
                    ans_cpt_freq[ans_cpt] += 1
            ans_cpt_keys = np.array(list(ans_cpt_freq.keys()))
            ans_cpt_values = np.array(list(ans_cpt_freq.values()))
            ans_cpt_ps = ans_cpt_values / np.sum(ans_cpt_values)

            stats[A.stats("cal_R", query_direction)] = sum(
                [len(set(ans_cpts)) for ans_cpts in total_ans_list]
            )
            stats[A.stats("D", query_direction)] = len(query_cpt2ans_cpts)
            stats[A.stats("G", query_direction)] = len(
                set([ans_cpt for ans_cpts in total_ans_list
                     for ans_cpt in ans_cpts])
            )
            stats[A.stats("E_d", query_direction)] = np.mean(
                [len(set(ans_cpts)) for ans_cpts in total_ans_list]
            )
            # save E_d distribution
            save_path = output_dir / f"G_{query_direction}_{embed}.pkl"
            print(f"save G ans distribution to {save_path}")
            with open(save_path, "wb") as f:
                pkl.dump([len(set(ans_cpts)) for ans_cpts in total_ans_list],
                         f)

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
                f"embed: {embed} "
                f"query_direction: {query_direction} "
                "setting: G "
                f"method: random K: {K}"
            )
            rows = []
            query_num = 0
            for query_cpt, ans_cpts in tqdm(query_cpt2ans_cpts.items()):
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
                query_num += 1

            result_row = get_result_row(
                embed,
                query_direction,
                "global_random",
                calc_acc(rows, top),
                calc_mrr(rows),
            )
            result_row["query_num"] = query_num
            logger.info(result_row)
            results.append(result_row)

            # setting G
            # we can use query_cpt2ans_cpts

            # weighted mean x centering
            for weighted_mean in weighted_flags:
                # calc relation vector
                if weighted_mean:
                    v = []
                    for query_cpt, ans_cpts in query_cpt2ans_cpts.items():
                        for ans_cpt in ans_cpts:
                            v.append(model[ans_cpt] - model[query_cpt])
                    v = np.mean(v, axis=0)
                else:
                    v = np.mean(cands, axis=0) - np.mean(domains, axis=0)

                for centering in centering_flags:
                    logger.info(
                        f"embed: {embed} "
                        f"query_direction: {query_direction} "
                        "setting: G "
                        "method: analogy "
                        f"wmean: {weighted_mean} center: {centering}"
                    )
                    rows = []
                    for query_cpt, ans_cpts in tqdm(
                            query_cpt2ans_cpts.items()):
                        pred = model[query_cpt] + v
                        cand_cpts, cand_sims = nearest_words(
                            pred, centering=centering)
                        rr = calc_rr(pred, ans_cpts, centering=centering)
                        row = {
                            "query_cpt": query_cpt,
                            "ans_cpts": ans_cpts,
                            "cand_cpts": cand_cpts,
                            "cand_sims": cand_sims,
                            "rr": rr,
                        }
                        rows.append(row)
                    result_row = get_result_row(
                        embed,
                        query_direction,
                        "global",
                        calc_acc(rows, top),
                        calc_mrr(rows),
                        weighted_mean,
                        centering,
                    )
                    logger.info(result_row)
                    results.append(result_row)

            # pathway-wise setting
            # prepare queries for setting P1
            pathway_pkl = \
                f"output/pathway/{embed}_hsa_to_{query_direction}s.pkl"

            with open(pathway_pkl, "rb") as f:
                hsa2relations = pkl.load(f)
            hsas = sorted(list(hsa2relations.keys()))
            logger.info(f"hsa count: {len(hsas)}")
            stats[P.stats("cal_P", query_direction)] = len(hsas)

            # stats
            total_query_list = []
            total_ans_list = []
            total_relation_list = []
            total_ans_mean_list = []
            total_ans_num_list = []
            for hsa in tqdm(hsas):
                # query check
                query_list = []
                ans_list = []
                relation_list = []
                ans_num_list = []
                for _, query_cpt, anss in hsa2relations[hsa]:
                    query_list.append(query_cpt)
                    ans_num_list.append(len(anss))
                    for _, ans_cpt in anss:
                        ans_list.append(ans_cpt)
                        relation_list.append((query_cpt, ans_cpt))
                assert len(set(query_list)) > 1 and len(set(ans_list)) > 1
                total_query_list.append(query_list)
                total_ans_list.append(ans_list)
                total_relation_list.append(relation_list)
                total_ans_mean_list.append(np.mean(ans_num_list))
                total_ans_num_list.append(ans_num_list)

            stats[P.stats("sum_Dp", query_direction)] = sum(
                [len(set(query_list)) for query_list in total_query_list]
            )
            stats[P.stats("sum_Gp", query_direction)] = sum(
                [len(set(ans_list)) for ans_list in total_ans_list]
            )
            stats[P.stats("E_dp", query_direction)
                  ] = np.mean(total_ans_mean_list)

            # save E_dp distribution
            save_path = output_dir / f"P1_{query_direction}_{embed}.pkl"
            print(f"save P1 ans distribution to {save_path} ")
            with open(save_path, "wb") as f:
                lens = [item for sublist in total_ans_num_list
                        for item in sublist]
                pkl.dump(lens, f)

            # random
            rows = []
            query_num = 0
            logger.info(
                f"embed: {embed} "
                f"query_direction: {query_direction} "
                "setting: P1 "
                f"method: random K: {K}"
            )

            for hsa in tqdm(hsas):
                # query check
                use_query_cpts = set()
                use_ans_cpts = set()
                use_relation_set = set()
                for _, query_cpt, anss in hsa2relations[hsa]:
                    use_query_cpts.add(query_cpt)
                    for _, ans_cpt in anss:
                        use_ans_cpts.add(ans_cpt)
                        use_relation_set.add((query_cpt, ans_cpt))
                assert len(use_query_cpts) > 1 and len(use_ans_cpts) > 1

                # prediction
                hsa_rows = []
                for _, query_cpt, anss in hsa2relations[hsa]:
                    ans_cpts = [ans_cpt for _, ans_cpt in anss]
                    pred = model[query_cpt] + v
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
                        hsa_rows.append(row)
                rows += hsa_rows

            result_row = get_result_row(
                embed,
                query_direction,
                "pathway1_random",
                calc_acc(rows, top),
                calc_mrr(rows),
            )
            result_row["query_num"] = query_num
            logger.info(result_row)
            results.append(result_row)

            # weighted mean x centering
            for weighted_mean in weighted_flags:
                for centering in centering_flags:
                    logger.info(
                        f"embed: {embed} "
                        f"query_direction: {query_direction} "
                        "setting: P1 "
                        "method: analogy "
                        f"wmean: {weighted_mean} center: {centering}"
                    )

                    # hsa means pathway id
                    rows = []
                    hsa_results = []
                    query_num = 0
                    for hsa in tqdm(hsas):
                        # query check
                        use_query_cpts = set()
                        use_ans_cpts = set()
                        use_relation_set = set()
                        for _, query_cpt, anss in hsa2relations[hsa]:
                            use_query_cpts.add(query_cpt)
                            for _, ans_cpt in anss:
                                use_ans_cpts.add(ans_cpt)
                                use_relation_set.add((query_cpt, ans_cpt))
                        assert len(use_query_cpts) > 1 and \
                            len(use_ans_cpts) > 1

                        # calc relation vector
                        if weighted_mean:
                            v = []
                            for _, query_cpt, anss in hsa2relations[hsa]:
                                for _, ans_cpt in anss:
                                    v.append(model[ans_cpt] - model[query_cpt])
                            v = np.mean(v, axis=0)
                        else:
                            hsa_query_vecs = []
                            hsa_anss = set()
                            hsa_ans_vecs = []
                            for _, query_cpt, anss in hsa2relations[hsa]:
                                hsa_query_vecs.append(model[query_cpt])
                                for _, ans_cpt in anss:
                                    if ans_cpt not in hsa_anss:
                                        hsa_anss.add(ans_cpt)
                                        hsa_ans_vecs.append(model[ans_cpt])
                            hsa_query_vecs = np.array(hsa_query_vecs)
                            hsa_ans_vecs = np.array(hsa_ans_vecs)
                            v = np.mean(hsa_ans_vecs, axis=0) - np.mean(
                                hsa_query_vecs, axis=0
                            )

                        # prediction
                        hsa_rows = []
                        for _, query_cpt, anss in hsa2relations[hsa]:
                            ans_cpts = [ans_cpt for _, ans_cpt in anss]
                            pred = model[query_cpt] + v
                            query_num += 1
                            cand_cpts, cand_sims = nearest_words(
                                pred, centering=centering
                            )
                            rr = calc_rr(pred, ans_cpts, centering=centering)
                            row = {
                                "query_cpt": query_cpt,
                                "ans_cpts": ans_cpts,
                                "cand_cpts": cand_cpts,
                                "cand_sims": cand_sims,
                                "rr": rr,
                            }
                            hsa_rows.append(row)
                        rows += hsa_rows

                        # each hsa result
                        hsa_result_row = {"hsa": hsa,
                                          "query_num": len(hsa_rows)}
                        for ith, acc_ in enumerate(calc_acc(hsa_rows, top)):
                            hsa_result_row[f"top{ith+1}_acc"] = acc_
                        hsa_results.append(hsa_result_row)

                    result_row = get_result_row(
                        embed,
                        query_direction,
                        "pathway1",
                        calc_acc(rows, top),
                        calc_mrr(rows),
                        weighted_mean,
                        centering,
                    )
                    result_row["query_num"] = query_num
                    logger.info(result_row)
                    results.append(result_row)

                    # save pathway results
                    hsa_result_df = pd.DataFrame(hsa_results)
                    csv_path = get_csv_path(
                        "pathway1",
                        embed,
                        query_direction,
                        weighted_mean,
                        centering
                    )
                    hsa_result_df.to_csv(output_dir / csv_path, index=False)

            # prepare queries for setting P2
            # in the case of drug2gene,
            # all_query_cpt_set is $D$
            all_query_cpt_set = set(rel_df[query_cpt_col].values)

            # define query:
            # in the case of drug2gene,
            # regarding hsa as $p$ in $\mathcal{P}$,
            # domain_df.query(f'hsa=="{hsa}"')['concept'] is $\mathcal{D}_p$,
            # and hsa2query_cpt_set[hsa] is $\mathcal{D}_p \cap D$
            domain_df = pd.read_csv(
                f"output/pathway/{embed}_{query_type}_pathway.csv")

            hsa2query_cpt_set = dict()
            for hsa in tqdm(hsas):
                hsa2query_cpt_set[hsa] = (
                    set(
                        domain_df.query(f'hsa=="{hsa}"')[
                            f"{query_type}_concept"].values
                    )
                    & all_query_cpt_set
                )

            # define answer:
            # we can use query_cpt2ans_cpts

            # stats
            total_query_list = []
            total_ans_list = []
            for hsa in tqdm(hsas):
                total_query_list.append(list(hsa2query_cpt_set[hsa]))
                query_cpt_set = hsa2query_cpt_set[hsa]
                ans_lists = []
                for query_cpt in query_cpt_set:
                    ans_lists.append(query_cpt2ans_cpts[query_cpt])
                total_ans_list.append(ans_lists)
            stats[P.stats("sum_cal_cap_Dp", query_direction)] = sum(
                [len(set(query_list)) for query_list in total_query_list]
            )
            stats[P.stats("E_d", query_direction)] = np.mean(
                [
                    np.mean([len(set(ans_list)) for ans_list in ans_lists])
                    for ans_lists in total_ans_list
                ]
            )

            # save P2 ans distribution
            save_path = output_dir / f"P2_{query_direction}_{embed}.pkl"
            print(f"save P2 ans distribution to {save_path} ")
            with open(save_path, "wb") as f:
                lens = [
                    [len(set(ans_list)) for ans_list in ans_lists]
                    for ans_lists in total_ans_list
                ]
                lens = [item for sublist in lens for item in sublist]
                pkl.dump(lens, f)

            # random
            logger.info(
                f"embed: {embed} "
                f"query_direction: {query_direction} "
                "setting: P2 "
                f"method: random K: {K}"
            )

            rows = []
            for hsa in tqdm(hsas):
                query_cpt_set = hsa2query_cpt_set[hsa]

                # query check
                assert len(query_cpt_set) > 1
                use_ans_cpts = set()
                for query_cpt in query_cpt_set:
                    ans_cpts = query_cpt2ans_cpts[query_cpt]
                    for ans_cpt in ans_cpts:
                        use_ans_cpts.add(ans_cpt)
                assert len(use_ans_cpts) > 1

                # prediction
                hsa_rows = []
                for query_cpt in query_cpt_set:
                    ans_cpts = query_cpt2ans_cpts[query_cpt]
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
                        hsa_rows.append(row)
                rows += hsa_rows

            result_row = get_result_row(
                embed,
                query_direction,
                "pathway2_random",
                calc_acc(rows, top),
                calc_mrr(rows),
            )
            result_row["query_num"] = query_num
            logger.info(result_row)
            results.append(result_row)

            # weighted mean x centering
            for weighted_mean in weighted_flags:
                for centering in centering_flags:
                    logger.info(
                        f"embed: {embed} "
                        f"query_direction: {query_direction} "
                        "setting: P2 "
                        "method: analogy "
                        f"wmean: {weighted_mean} center: {centering}"
                    )
                    # hsa means pathway id
                    rows = []
                    hsa_results = []
                    for hsa in tqdm(hsas):
                        query_cpt_set = hsa2query_cpt_set[hsa]

                        # query check
                        assert len(query_cpt_set) > 1
                        use_ans_cpts = set()
                        for query_cpt in query_cpt_set:
                            ans_cpts = query_cpt2ans_cpts[query_cpt]
                            for ans_cpt in ans_cpts:
                                use_ans_cpts.add(ans_cpt)
                        assert len(use_ans_cpts) > 1

                        # calc relation vector
                        if weighted_mean:
                            v = []
                            for _, query_cpt, anss in hsa2relations[hsa]:
                                for _, ans_cpt in anss:
                                    v.append(model[ans_cpt] - model[query_cpt])
                            v = np.mean(v, axis=0)
                        else:
                            hsa_query_vecs = []
                            hsa_anss = set()
                            hsa_ans_vecs = []
                            for _, query_cpt, anss in hsa2relations[hsa]:
                                hsa_query_vecs.append(model[query_cpt])
                                for _, ans_cpt in anss:
                                    if ans_cpt not in hsa_anss:
                                        hsa_anss.add(ans_cpt)
                                        hsa_ans_vecs.append(model[ans_cpt])
                            hsa_query_vecs = np.array(hsa_query_vecs)
                            hsa_ans_vecs = np.array(hsa_ans_vecs)
                            v = np.mean(hsa_ans_vecs, axis=0) - np.mean(
                                hsa_query_vecs, axis=0
                            )

                        # prediction
                        hsa_rows = []
                        for query_cpt in query_cpt_set:
                            ans_cpts = query_cpt2ans_cpts[query_cpt]
                            pred = model[query_cpt] + v
                            cand_cpts, cand_sims = nearest_words(
                                pred, centering=centering
                            )
                            rr = calc_rr(pred, ans_cpts, centering=centering)
                            row = {
                                "query_cpt": query_cpt,
                                "ans_cpts": ans_cpts,
                                "cand_cpts": cand_cpts,
                                "cand_sims": cand_sims,
                                "rr": rr,
                            }
                            hsa_rows.append(row)
                        rows += hsa_rows

                        # each hsa result
                        hsa_result_row = {"hsa": hsa,
                                          "query_num": len(hsa_rows)}
                        for ith, acc_ in enumerate(calc_acc(hsa_rows, top)):
                            hsa_result_row[f"top{ith+1}_acc"] = acc_
                        hsa_results.append(hsa_result_row)
                    result_row = get_result_row(
                        embed,
                        query_direction,
                        "pathway2",
                        calc_acc(rows, top),
                        calc_mrr(rows),
                        weighted_mean,
                        centering,
                    )
                    logger.info(result_row)
                    results.append(result_row)

                    # save pathway results
                    hsa_result_df = pd.DataFrame(hsa_results)
                    csv_path = get_csv_path(
                        "pathway2",
                        embed,
                        query_direction,
                        weighted_mean,
                        centering
                    )
                    hsa_result_df.to_csv(output_dir / csv_path, index=False)

            # save stats
            stats_results.append(stats)

    # save results
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_dir / "result.csv", index=False)

    # save stats
    stats_result_df = pd.DataFrame(stats_results)
    stats_result_df = stats_result_df[
        ["embed", "query_direction"] + A.all + P.all]
    embeds = stats_result_df["embed"].unique()

    data = []
    for embed in embeds:
        embed_df = stats_result_df.query(f'embed=="{embed}"')
        assert len(embed_df) == 2
        first = embed_df.iloc[0]
        second = embed_df.iloc[1]
        row = dict()
        for col in ["embed"] + A.all + P.all:
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

            if col != "embed" and col not in expectations:
                v = int(v)
            row[col] = v

        data.append(row)

    clean_df = pd.DataFrame(data)
    clean_df.to_csv(output_dir / "stats.csv", index=False)


if __name__ == "__main__":
    main()
