# Drug-Gene-Analogy

> [Predicting drug–gene relations via analogy tasks with word embeddings](https://www.nature.com/articles/s41598-025-01418-z)                 
> [Hiroaki Yamagiwa](https://ymgw55.github.io/), Ryoma Hashimoto, Kiwamu Arakane, Ken Murakami, Shou Soeda, [Momose Oyama](https://momoseoyama.github.io/), Yihua Zhu, Mariko Okada, [Hidetoshi Shimodaira](http://stat.sys.i.kyoto-u.ac.jp/members/shimo/)                 
> *Scientific Reports 15, 17240 (2025)* [[arXiv](https://arxiv.org/abs/2406.00984)]

![Fig. 2](.github/images/fig2b.png)

## Setup

The code is meant to run inside Docker.
If you prefer other setups, install the packages listed in [requirements.txt](requirements.txt).

### Build the Docker image

```bash
$ bash scripts/docker/build.sh
```

### Run the Docker container

```bash
$ bash scripts/docker/run.sh
```

## Embeddings

### BioConceptVec

Download the BioConceptVec skip-gram embeddings:

```bash
$ mkdir -p data/embeddings
$ wget -c -O data/embeddings/concept_skipgram.json https://ftp.ncbi.nlm.nih.gov/pub/lu/BioConceptVec/concept_skip.json
```

### Our skip-gram embeddings

Skip-gram embeddings trained on PubMed abstracts in 5-year windows from 1970 are available on ([Google Drive](https://drive.google.com/drive/folders/1G5v_G9sP0z8XDPb7MbSkWwvJVTKb-hbD?usp=sharing)).

### Generating the experimental dataset

See [README.prepare.md](README.prepare.md) for the full preprocessing pipeline.

Pre-generated data are already placed in the [data/](data/) directory.

## Code

### PCA

Fig. 2a: PCA of drugs and genes for a randomly selected relation

```bash
$ python Fig2a.py
```

Fig. 2b: PCA of drugs and genes classified to the ErbB signaling pathway

```bash
$ python Fig2b.py 
```

### Evaluation

```bash
$ python eval_analogy.py
$ python eval_analogy_Y1.py
$ python eval_analogy_Y2.py
$ python eval_analogy_P1Y1_and_P2Y1.py
$ python eval_analogy_P1Y2_and_P2Y2.py
```

Results produced with the OpenAI API are stored under [output/analogy_API/](output/analogy_API/) and are loaded by default:

```bash
$ python eval_OpenAI_API.py
```

If you wish to rerun the API experiments, adjust the scripts as needed.

### Plotting year-wise scores (Fig. 3 / Fig. S4)

```bash
$ python Fig3_FigS4.py
```

### Predictions for the ErbB signaling pathway (Table 4)

```bash
$ python Table4.py
```

### Analogy vs. TransE

Generate analogy-based predictions using 10 %, 20 %, … 60 % of the training data:

```bash
$ python eval_analogy_for_comparing_with_TransE.py
```

Compare them with TransE results:

```bash
$ python Fig4.py
```

(TransE scores are currently hard-coded inside `Fig4.py`; a dedicated script will be released later.)


## Reference

- Chen et al. Bioconceptvec: Creating and evaluating literature-based biomedical concept embeddings on a large scale. PLoS Comput Biol. (2020).

## Appendix

Distribution of answer-set sizes:

```bash
$ python FigS2_S3.py
```

Search-result rank by answer-set size:

```bash
$ python FigS5.py
```

Weighted correlations are computed with the [WeightedCorr](https://github.com/matthijsz/weightedcorr) repository.

## Citation

```bibtex
@article{Yamagiwa2025,
  title        = {Predicting drug–gene relations via analogy tasks with word embeddings},
  author       = {Yamagiwa, Hiroaki and Hashimoto, Ryoma and Arakane, Kiwamu and Murakami, Ken and Soeda, Shou and Oyama, Momose and Zhu, Yihua and Okada, Mariko and Shimodaira, Hidetoshi},
  journal      = {Scientific Reports},
  volume       = {15},
  number       = {1},
  pages        = {17240},
  year         = {2025},
  month        = {May},
  doi          = {10.1038/s41598-025-01418-z},
  url          = {https://doi.org/10.1038/s41598-025-01418-z},
  issn         = {2045-2322}
}
```

## Notes

- Embedding URLs may change; please refer to the GitHub repository rather than the raw download link.
- This directory was created by [Hiroaki Yamagiwa](https://ymgw55.github.io/).
- Embeddings were trained by Ryoma Hashimoto.
- KEGG ID to MeSH ID conversion ([prepare_kegg2mesh.ipynb](prepare_kegg2mesh.ipynb)) was implemented by Kiwamu Arakane．
- TransE prediction experiments were conducted by Yihua Zhu．