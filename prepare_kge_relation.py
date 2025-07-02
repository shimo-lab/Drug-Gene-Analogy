import pickle as pkl
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def save_data(data, output_dir, hsa=False):
    df = pd.DataFrame(data, columns=("head", "relation", "tail"))
    tab_sep_all_path = output_dir / "all.tsv"
    df.to_csv(tab_sep_all_path, sep="\t", index=False, header=False)


def main(embed):
    query_type = "drug"
    ans_type = "gene"
    query_direction = f"{query_type}2{ans_type}"
    query_cpt_col = f"{query_type}_concept"
    ans_cpts_col = f"{ans_type}_concepts"
    rel_df = pd.read_csv(
        f"output/relation/{embed}_{query_direction}s_relation.csv")

    output_root_dir = Path(f"output/kge_files/{embed}")
    output_root_dir.mkdir(parents=True, exist_ok=True)

    # setting G
    output_dir = output_root_dir / "setting_G"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = []
    for _, row in rel_df.iterrows():
        query_cpt = row[query_cpt_col]
        ans_cpts = eval(row[ans_cpts_col])
        for ans_cpt in ans_cpts:
            head = query_cpt
            relation = query_direction
            tail = ans_cpt
            data.append((head, relation, tail))
    save_data(data, output_dir)

    # setting P1
    output_dir = output_root_dir / "setting_P1"
    output_dir.mkdir(parents=True, exist_ok=True)

    pathway_pkl = f"output/pathway/{embed}_hsa_to_{query_direction}s.pkl"
    with open(pathway_pkl, "rb") as f:
        hsa2relations = pkl.load(f)
    hsas = sorted(list(hsa2relations.keys()))

    data = []
    for hsa in tqdm(hsas):
        for _, query_cpt, anss in hsa2relations[hsa]:
            ans_cpts = [ans_cpt for _, ans_cpt in anss]
            for ans_cpt in ans_cpts:
                head = query_cpt
                relation = f"{query_direction}_in_{hsa}"
                tail = ans_cpt
                data.append((head, relation, tail))
    save_data(data, output_dir, hsa=True)

    # setting P2
    output_dir = output_root_dir / "setting_P2"
    output_dir.mkdir(parents=True, exist_ok=True)

    query_cpt2ans_cpts = dict()
    for _, row in rel_df.iterrows():
        query_cpt = row[query_cpt_col]
        ans_cpts = eval(row[ans_cpts_col])
        query_cpt2ans_cpts[query_cpt] = ans_cpts
    all_query_cpt_set = set(rel_df[query_cpt_col].values)
    domain_df = pd.read_csv(f"output/pathway/{embed}_{query_type}_pathway.csv")

    hsa2query_cpt_set = dict()
    for hsa in tqdm(hsas):
        hsa2query_cpt_set[hsa] = (
            set(domain_df.query(f'hsa=="{hsa}"')[
                f"{query_type}_concept"].values)
            & all_query_cpt_set
        )

    data = []
    for hsa in tqdm(hsas):
        query_cpt_set = hsa2query_cpt_set[hsa]
        for query_cpt in query_cpt_set:
            ans_cpts = query_cpt2ans_cpts[query_cpt]
            for ans_cpt in ans_cpts:
                head = query_cpt
                relation = f"{query_direction}_in_{hsa}"
                tail = ans_cpt
                data.append((head, relation, tail))
    save_data(data, output_dir, hsa=True)


if __name__ == "__main__":
    main("bcv")
    main("homemade")
