library(data.table)
library(Matrix)
library(dplyr)
library(GenomicRanges)
library(org.Dr.eg.db)
library(TxDb.Drerio.UCSC.danRer11.refGene)

# construct GRanges for enhancer regions
n_bp_upstream <- 50000    # +/- 50kb
n_bp_downstream <- 50000  #
txdb_dr11 <- TxDb.Drerio.UCSC.danRer11.refGene
txdb_gene <- as.data.frame(genes(txdb_dr11, single.strand.genes.only = FALSE))
anno_gene <- select(org.Dr.eg.db, keys = txdb_gene$group_name, columns = c('SYMBOL', 'ENTREZID'), keytype = 'ENTREZID')
colnames(anno_gene) <- c('group_name', 'name')
txdb_gene <- left_join(txdb_gene, anno_gene, by = 'group_name', relationship = 'many-to-many')
tfs <- as(t(read.table('Datasets/BP-HC/TranscriptionFactors.csv', header = FALSE, sep = '\t')), 'character')
txdb_tfs <- subset(txdb_gene, txdb_gene$name %in% tfs)
txdb_tfs <- subset(txdb_tfs, !grepl('_alt', txdb_tfs$seqnames) | !duplicated(txdb_tfs$name))
txdb_tfs <- subset(txdb_tfs, !grepl('chrUn_', txdb_tfs$seqnames) | !duplicated(txdb_tfs$name))
txdb_tfs <- subset(txdb_tfs, !duplicated(txdb_tfs$name))
txdb_tfs_enhancers <- data.frame(txdb_tfs)
txdb_tfs_enhancers$end <- txdb_tfs_enhancers$start
txdb_tfs_enhancers$start <- txdb_tfs_enhancers$start - n_bp_upstream
txdb_tfs_enhancers$end <- txdb_tfs_enhancers$end + n_bp_downstream
txdb_tfs_enhancers$width <- txdb_tfs_enhancers$end - txdb_tfs_enhancers$start
txdb_tfs_enhancers_gr <- makeGRangesFromDataFrame(txdb_tfs_enhancers, keep.extra.columns = TRUE)
write.csv(txdb_tfs_enhancers, 'EnhancerRegions50kb.csv', row.names = FALSE)
