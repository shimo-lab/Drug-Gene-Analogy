import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import Model, get_logger
from WeightedCorr import WeightedCorr


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


def main(embed="homemade"):
    top = 10

    logger = get_logger()

    prefix_dict = {"drug": "Chemical_MESH_", "gene": "Gene"}

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
        word_to_id_path = \
            "data/embeddings/years/1781-2023/min_count30/word_to_id.pkl"
        with open(word_to_id_path, "rb") as f:
            word_to_id = pkl.load(f)
        assert len(model) == len(word_to_id)
        model = Model(model, word_to_id)
    else:
        raise NotImplementedError

    # prepare queries
    query_type, ans_type = "drug", "gene"
    query_direction = f"{query_type}2{ans_type}"
    rel_df = pd.read_csv(
        f"output/relation/{embed}_{query_direction}s_relation.csv")

    query_cpt_col = f"{query_type}_concept"
    ans_cpts_col = f"{ans_type}_concepts"
    assert len(rel_df) == len(rel_df[query_cpt_col].unique())

    # define query and answer set
    # in the case of drug2gene,
    # query is drug $d\in D$, answer is gene set $[d]$
    query_cpt2ans_cpts = dict()
    total_ans_list = []
    word_to_freq = dict()
    for _, row in rel_df.iterrows():
        query_cpt = row[query_cpt_col]
        ans_cpts = eval(row[ans_cpts_col])
        query_cpt2ans_cpts[query_cpt] = ans_cpts
        total_ans_list.append(ans_cpts)
        word_to_freq[query_cpt] = len(ans_cpts)

    # query directions
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

    # prepare queries
    rel_df = pd.read_csv(
        f"output/relation/{embed}_{query_direction}s_relation.csv")

    query_cpt_col = f"{query_type}_concept"
    ans_cpts_col = f"{ans_type}_concepts"
    assert len(rel_df) == len(rel_df[query_cpt_col].unique())

    # define query and answer set
    # in the case of drug2gene,
    # query is drug $d\in D$, answer is gene set $[d]$
    query_cpt2ans_cpts = dict()
    total_ans_list = []
    for _, row in rel_df.iterrows():
        query_cpt = row[query_cpt_col]
        ans_cpts = eval(row[ans_cpts_col])
        query_cpt2ans_cpts[query_cpt] = ans_cpts
        total_ans_list.append(ans_cpts)

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
            cos = np.dot(cands_c, pred) / (np.linalg.norm(pred) * cands_norm_c)
        else:
            cos = np.dot(cands, pred) / (np.linalg.norm(pred) * cands_norm)
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
            cos = np.dot(cands_c, pred) / (np.linalg.norm(pred) * cands_norm_c)
        else:
            cos = np.dot(cands, pred) / (np.linalg.norm(pred) * cands_norm)

        # ans ids
        ans_ids = [to_id[ans_cpt] for ans_cpt in ans_cpts]

        # choose best ans_cpt
        best_ans_id = ans_ids[np.argmax(cos[ans_ids])]
        best_ans_cos = cos[best_ans_id]
        rank = np.sum(cos > best_ans_cos)
        return 1 / (rank + 1), rank + 1

    weighted_mean = True
    centering = True

    output_dir = Path("output/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_pkl_path = output_dir / f"{embed}_{query_direction}_relfreq.pkl"

    if rows_pkl_path.exists():
        print(f"load {rows_pkl_path}")
        with open(rows_pkl_path, "rb") as f:
            rows = pkl.load(f)
        G_rows, P1_rows, P2_rows = rows
    else:
        # setting G
        # we can use query_cpt2ans_cpts

        # weighted mean x centering
        # calc relation vector
        v = []
        for query_cpt, ans_cpts in query_cpt2ans_cpts.items():
            for ans_cpt in ans_cpts:
                v.append(model[ans_cpt] - model[query_cpt])
        v = np.mean(v, axis=0)

        logger.info(
            f"embed: {embed} "
            f"query_direction: {query_direction} "
            "setting: G "
            "method: analogy "
            f"wmean: {weighted_mean} center: {centering}"
        )
        G_rows = []
        for query_cpt, ans_cpts in tqdm(query_cpt2ans_cpts.items()):
            pred = model[query_cpt] + v
            cand_cpts, cand_sims = nearest_words(pred, centering=centering)
            rr, rank = calc_rr(pred, ans_cpts, centering=centering)
            row = {
                "query_cpt": query_cpt,
                "ans_cpts": ans_cpts,
                "cand_cpts": cand_cpts,
                "cand_sims": cand_sims,
                "rr": rr,
                "rank": rank,
                "freq": word_to_freq[query_cpt],
            }
            G_rows.append(row)

        # pathway-wise setting
        # prepare queries for setting P1
        pathway_pkl = f"output/pathway/{embed}_hsa_to_{query_direction}s.pkl"

        with open(pathway_pkl, "rb") as f:
            hsa2relations = pkl.load(f)
        hsas = sorted(list(hsa2relations.keys()))
        logger.info(f"hsa count: {len(hsas)}")

        # weighted mean x centering
        logger.info(
            f"embed: {embed} "
            f"query_direction: {query_direction} "
            "setting: P1 "
            "method: analogy "
            f"wmean: {weighted_mean} center: {centering}"
        )

        # hsa means pathway id
        P1_rows = []
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
            assert len(use_query_cpts) > 1 and len(use_ans_cpts) > 1

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
                v = np.mean(hsa_ans_vecs, axis=0) \
                    - np.mean(hsa_query_vecs, axis=0)

            # prediction
            hsa_rows = []
            for _, query_cpt, anss in hsa2relations[hsa]:
                ans_cpts = [ans_cpt for _, ans_cpt in anss]
                pred = model[query_cpt] + v
                query_num += 1
                cand_cpts, cand_sims = nearest_words(pred, centering=centering)
                rr, rank = calc_rr(pred, ans_cpts, centering=centering)
                row = {
                    "query_cpt": query_cpt,
                    "ans_cpts": ans_cpts,
                    "cand_cpts": cand_cpts,
                    "cand_sims": cand_sims,
                    "rr": rr,
                    "rank": rank,
                    "freq": word_to_freq[query_cpt],
                }
                hsa_rows.append(row)
            P1_rows += hsa_rows

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
                set(domain_df.query(f'hsa=="{hsa}"')[
                    f"{query_type}_concept"].values)
                & all_query_cpt_set
            )

        # define answer:
        # we can use query_cpt2ans_cpts

        # weighted mean x centering
        logger.info(
            f"embed: {embed} "
            f"query_direction: {query_direction} "
            "setting: P2 "
            "method: analogy "
            f"wmean: {weighted_mean} center: {centering}"
        )
        # hsa means pathway id
        P2_rows = []
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
                v = np.mean(hsa_ans_vecs, axis=0) \
                    - np.mean(hsa_query_vecs, axis=0)

            # prediction
            hsa_rows = []
            for query_cpt in query_cpt_set:
                ans_cpts = query_cpt2ans_cpts[query_cpt]
                pred = model[query_cpt] + v
                cand_cpts, cand_sims = nearest_words(pred, centering=centering)
                rr, rank = calc_rr(pred, ans_cpts, centering=centering)
                row = {
                    "query_cpt": query_cpt,
                    "ans_cpts": ans_cpts,
                    "cand_cpts": cand_cpts,
                    "cand_sims": cand_sims,
                    "rr": rr,
                    "rank": rank,
                    "freq": word_to_freq[query_cpt],
                }
                hsa_rows.append(row)
            P2_rows += hsa_rows

        # save rows
        with open(rows_pkl_path, "wb") as f:
            pkl.dump([G_rows, P1_rows, P2_rows], f)

    # draw relfreq-rank plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    label_size = 20
    tick_size = 18
    title_size = 20
    for idx, (rows, setting) in enumerate(
        zip([G_rows, P1_rows, P2_rows], ["G", "P1", "P2"])
    ):
        freqs = [row["freq"] for row in rows]
        ranks = [row["rank"] for row in rows]
        freq2count = dict()
        for freq in freqs:
            if freq not in freq2count:
                freq2count[freq] = 0
            freq2count[freq] += 1
        weights = [1 / freq2count[freq] for freq in freqs]
        axes[idx].scatter(freqs, ranks, alpha=0.2, color="darkorange")
        # xyw should be a pd.DataFrame, or x, y, w should be pd.Series
        x = pd.Series(freqs)
        y = pd.Series(ranks)
        w = pd.Series(weights)
        corr = WeightedCorr(x=x, y=y, w=w)(method="spearman")
        title = f"Setting {setting}\nWeighted Spearman's " \
            + r"$\rho$" + f": {corr:.3f}"
        axes[idx].set_title(title, fontsize=title_size, pad=10)
        axes[idx].set_yscale("log")
        if setting == "G":
            xlabel = r"$|[d]|$"
        elif setting == "P1":
            xlabel = r"$|[d]_p|$"
        else:
            xlabel = r"$|[d]|$"

        axes[idx].set_xlabel(xlabel, fontsize=label_size)
        axes[idx].set_ylabel("Search Result Rank", fontsize=label_size)
        axes[idx].tick_params(labelsize=tick_size)

    # save plot
    (output_dir / "ansfreq_rank").mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "ansfreq_rank" / f"{query_direction}_{embed}.png"
    fig.subplots_adjust(
        left=0.07, right=0.96, bottom=0.16, top=0.82, wspace=0.3)
    print(f"save {save_path}")
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main(embed="homemade")
    main(embed="bcv")
