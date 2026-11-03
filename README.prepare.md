# Preparation for experiments

## Drug–gene relations

Create `data/relation/human_KEGG_drug_names.tsv` from [ASURATDB](https://github.com/keita-iida/ASURATDB) (GPL-3.0, built from KEGG); R is required:

```bash
$ bash scripts/asuratdb/make_human_KEGG_drug_names.sh
```

## Map gene IDs to gene names

```bash
$ python prepare_gene_cpt2name.py
```

## Convert KEGG ID to MeSH ID

Run the notebook `prepare_kegg2mesh.ipynb`.

## Relations

Create relation files for each embedding set:

```bash
$ python prepare_relation.py --embed bcv
$ python prepare_relation.py --embed homemade
```

## Pathways

KEGG pathway entries are not included. Download the 352 pathways used in the paper ([data/pathway/hsa_ids_paper.txt](data/pathway/hsa_ids_paper.txt)) via the KEGG REST API (`--all` for all current pathways):

```bash
$ python scripts/kegg/download_pathways.py
```

KEGG data are subject to the [KEGG terms of use](https://www.kegg.jp/kegg/legal.html) and are updated regularly; the pathway sets used in the paper are in [output/pathway/](output/pathway/).

Convert the saved TXT files to JSON:

```bash
$ python prepare_pathway_txt2json.py
```

Build pathway sets:

```bash
$ python prepare_pathway_set.py --embed bcv
$ python prepare_pathway_set.py --embed homemade
```

Generate pathway queries:

```bash
$ python prepare_pathway_query.py --embed bcv
$ python prepare_pathway_query.py --embed homemade
```

## Relations by year

```bash
$ python prepare_relation_for_Y2.py
```

## Pathways by year

```bash
$ python prepare_pathway_set_and_query_for_P1Y2_and_P2Y2.py
```

## Format for KGE experiments

```bash
$ python prepare_kge_relation.py
```