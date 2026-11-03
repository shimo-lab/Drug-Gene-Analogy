#!/bin/bash
# Create data/relation/human_KEGG_drug_names.tsv from ASURATDB
# (https://github.com/keita-iida/ASURATDB, GPL-3.0), which is built from KEGG.
# Requires R.
set -eu
COMMIT=2e136f8a6e360a375f6785a640aa7f620e745643
OUT=data/relation/human_KEGG_drug_names.tsv
RDA=$(mktemp --suffix=.rda)
trap 'rm -f "$RDA"' EXIT

curl -sSfL -o "$RDA" \
    "https://raw.githubusercontent.com/keita-iida/ASURATDB/$COMMIT/genes2bioterm/20221102_human_KEGG_drug.rda"
mkdir -p "$(dirname "$OUT")"
Rscript --vanilla -e '
args <- commandArgs(trailingOnly = TRUE)
e <- new.env()
load(args[1], envir = e)
d <- e$human_KEGG_drug$drug[, c("ID", "Description", "Gene", "GeneID")]
write.table(d, args[2], sep = "\t", row.names = FALSE, quote = TRUE)
' "$RDA" "$OUT"
echo "saved $OUT"
