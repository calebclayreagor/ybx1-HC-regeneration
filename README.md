# Welcome to the repository for "_ybx1_ acts upstream of _atoh1a_ to promote the rapid regeneration of hair cells in zebrafish lateral-line neuromasts" by Caleb C. Reagor, Paloma Bravo & A. J. Hudspeth (2025). 

### [Datasets](Datasets) contains the pre-processed scRNA-seq data from [baek2022_preprocessing.R](baek2022_preprocessing.R) to fine-tune DELAY and infer the early GRN for neuromast regeneration in hair cells (HC) and central supporting cells (SC). 
[zfish-tfs-baek2022.csv](zfish-tfs-baek2022.csv) contains a list of known TFs in the data set.

## Figure 1
### [DELAY_GRN_analysis.py](DELAY_GRN_analysis.py) contains code for the analyses of the inferred GRN for early neuromast regeneration. 
[enhancer_regions.R](enhancer_regions.R) extracts locations of genes' enhancers for de novo inference of Ybx1's TFBS with MEME Suite (see [fimo.tsv](fimo.tsv) & [streme.txt](streme.txt)).

## Figure 2
### [Microscopy](Microscopy) contains quantifications and code for the analyses of _ybx1_ mutant neuromasts and Ybx1 immunostaining.
