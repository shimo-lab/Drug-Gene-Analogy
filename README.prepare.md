# Preparation for experiments

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

KEGG pathway data were pre-crawled in two steps

1. Get the list of all human pathway IDs: https://rest.kegg.jp/list/pathway/hsa .
2. For every ID in that list, download its details (e.g., https://rest.kegg.jp/get/hsa04012 ) and save the TXT file.

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