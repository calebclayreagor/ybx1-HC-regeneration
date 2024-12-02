Welcome to the repository for the manuscript "_ybx1_ acts upstream of _atoh1a_ to promote the rapid regeneration of hair cells in zebrafish lateral-line neuromasts" by Caleb C. Reagor & A. J. Hudspeth (2024). 

-**Datasets** contains the pre-processed scRNA-seq data from **baek2022_preprocessing.R** to fine-tune DELAY and infer the early GRN for neuromast regeneration, with sub-directories for hair cell (HC) and supporting cell (SC) lineages. **zfish-tfs-baek2022.csv** contains a list of known TFs in the data set.

-**DELAY_GRN_analysis.py** contains code for the analyses of the inferred GRN presented in Figure 1. **enhancer_regions.R** extracts locations of genes' enhancers for the de novo inference of Ybx1's TFBS with MEME Suite (see results files **fimo.tsv** and **streme.txt**).

-**Microscopy** contains code and quantifications for _ybx1_ mutant neuromasts presented in Figure 2.

This repository is a work in progress; please open an issue if you have additional questions.
