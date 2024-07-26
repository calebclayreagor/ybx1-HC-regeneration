library(data.table)
library(Matrix)
library(Seurat)
library(slingshot)
library(tradeSeq)

# neuromast regeneration: homeostatic, 0min, 30min, 1hr, 3hr, 5hr, 10hr
samples.list <- list("GSM5862799_homeo_filtered_gene_bc_matrices_h5.h5",
                     "GSM5862800_0min_filtered_feature_bc_matrix.h5",
                     "GSM5862801_30min_filtered_feature_bc_matrix.h5",
                     "GSM5862802_1hr_filtered_feature_bc_matrix.h5",
                     "GSM5862803_3hr_filtered_gene_bc_matrices_h5.h5",
                     "GSM5862804_5hr_filtered_feature_bc_matrix.h5",
                     "GSM5862805_10hr_filtered_feature_bc_matrix.h5")

# load datasets -> filter -> normalize -> find variable features
samples.list <- lapply(X = samples.list, FUN = function(x) {
  name <- strsplit(x, "_")[[1]][2]
  x <- Read10X_h5(paste0("GSE196211_RAW/", x))
  x <- CreateSeuratObject(x, min.features = 300, project = name)
  x[['percent.mt']] <- PercentageFeatureSet(x, pattern = "mt-")
  x <- subset(x, subset = nFeature_RNA < 6000 & percent.mt < 5)
  x <- NormalizeData(x)
  x <- FindVariableFeatures(x, nfeatures = 5000)})

samples.merge <- merge(samples.list[[1]], 
  y = c(samples.list[[2]], samples.list[[3]], samples.list[[4]], samples.list[[5]], samples.list[[6]], samples.list[[7]]), 
  add.cell.ids = c("homeo", "0min", "30min", "1hr", "3hr", "5hr", "10hr"))

# find shared variable features -> scale -> PCA
features <- SelectIntegrationFeatures(samples.list, nfeatures = 3500)
samples.list <- lapply(X = samples.list, FUN = function(x) {
  x <- ScaleData(x, features = features)
  x <- RunPCA(x, features = features) })

# integration using pairwise anchors between data sets
anchors <- FindIntegrationAnchors(samples.list, anchor.features = features, reduction = 'rpca', dims = 1:50)
samples.integrated <- IntegrateData(anchors, features = features, dims = 1:50)

# dimensional reduction: scale -> PCA -> tSNE -> plot
samples.integrated <- ScaleData(samples.integrated, features = features)
samples.integrated <- RunPCA(samples.integrated, features = features)
ElbowPlot(samples.integrated, ndims = 30)
samples.integrated <- RunTSNE(samples.integrated, dims = 1:13, seed.use = 1111, perplexity = ncol(samples.integrated) / 100)
DimPlot(samples.integrated, group.by = 'orig.ident')
FeaturePlot(samples.integrated, keep.scale = NULL, min.cutoff = 'q5', features = 
 c("atoh1a", "dld", "tekt3", "myo6b", "isl1", "sost", "mcm4", "si:ch73-261i21.5", "tspan1"))

# find neighbors -> cluster -> subset HC trajectory
samples.integrated <- FindNeighbors(samples.integrated, dims = 1:13)
set.seed(1)
samples.integrated <- FindClusters(samples.integrated, resolution = 2.)
DimPlot(samples.integrated)
sc.central.clus <- c('9', '22')
hc.prog.clus <- c('26')
hc.young.clus <- c('19')
hc.mature.clus <- c('28', '30')
DimPlot(samples.integrated, cells.highlight = WhichCells(samples.integrated, 
  idents = c(sc.central.clus, hc.prog.clus, hc.young.clus, hc.mature.clus)))
hc.traj <- subset(samples.integrated, subset = 
  ((((seurat_clusters %in% sc.central.clus) & (orig.ident %in% c('0min', '30min', '1hr', '3hr'))) |
    ((seurat_clusters %in% hc.prog.clus) & (orig.ident %in% c('1hr', '3hr', '5hr'))) | 
    ((seurat_clusters %in% hc.young.clus) & (orig.ident %in% c('3hr', '5hr', '10hr'))) | 
    ((seurat_clusters %in% hc.mature.clus) & (orig.ident %in% c('3hr', '5hr', '10hr')))) &
   (tSNE_1 > 5) & (tSNE_2 < 10)))

# subset central SC trajectory
sc.traj.clus <- c('9', '3', '15')
DimPlot(samples.integrated, cells.highlight = WhichCells(samples.integrated, idents = sc.traj.clus))
sc.traj <- subset(samples.integrated, subset = ((seurat_clusters %in% sc.traj.clus) &
                                                (orig.ident %in% c('0min', '30min', '1hr', '3hr', '5hr', '10hr')) & 
                                                (tSNE_1 > 9)))

# HC trajectory: PCA -> tSNE -> plot
hc.traj <- RunPCA(hc.traj)
ElbowPlot(hc.traj, ndims = 30)
hc.traj <- RunTSNE(hc.traj, dims = 1:7, seed.use = 1111, perplexity = 100)
DimPlot(hc.traj, group.by = 'orig.ident')
FeaturePlot(hc.traj, keep.scale = NULL, min.cutoff = 'q5', features = c("sox21a", "dld", "atoh1a", "tekt3"))
DimPlot(hc.traj)  # clus 9 -> 28

# HC trajectory: slingshot pseudotime
hc.traj.pt <- slingPseudotime(slingshot(Embeddings(hc.traj, reduction = 'tsne'), Idents(hc.traj), start.clus = 9, end.clus = 28))
colnames(hc.traj.pt) <- "PseudoTime"
hc.traj[['PseudoTime']] <- hc.traj.pt
FeaturePlot(hc.traj, keep.scale = NULL, features = "PseudoTime", cols = c('yellow', 'blue'))
#write.csv(hc.traj.pt, "Datasets/BP-HC/PseudoTime.csv", row.names = TRUE)

# HC trajectory: timepoints
hc.traj.timepoints <- as.matrix(hc.traj@meta.data$orig.ident)
rownames(hc.traj.timepoints) <- rownames(hc.traj.pt)
colnames(hc.traj.timepoints) <- "Timepoint"
#write.csv(hc.traj.timepoints, "Datasets/BP-HC/Timepoints.csv", row.names = TRUE)

# SC trajectory: PCA -> tSNE -> plot
sc.traj <- RunPCA(sc.traj)
ElbowPlot(sc.traj, ndims = 30)
sc.traj <- RunTSNE(sc.traj, dims = 1:4, seed.use = 1111, perplexity = 200)
FeaturePlot(sc.traj, keep.scale = NULL, min.cutoff = 'q5', features = c("irg1l", "atoh1a", "sox3", "sox2"))
DimPlot(sc.traj)  # clus 15 -> 3

# SC trajectory: slingshot pseudotime
sc.traj.pt <- slingPseudotime(slingshot(Embeddings(sc.traj, reduction = 'tsne'), Idents(sc.traj), start.clus = 15, end.clus = 3))
colnames(sc.traj.pt) <- "PseudoTime"
sc.traj[['PseudoTime']] <- sc.traj.pt
FeaturePlot(sc.traj, keep.scale = NULL, features = "PseudoTime", cols = c('yellow', 'blue'))
#write.csv(sc.traj.pt, "Datasets/BP-SC/PseudoTime.csv", row.names = TRUE)

# SC trajectory: timepoints
sc.traj.timepoints <- as.matrix(sc.traj@meta.data$orig.ident)
rownames(sc.traj.timepoints) <- rownames(sc.traj.pt)
colnames(sc.traj.timepoints) <- "Timepoint"
#write.csv(sc.traj.timepoints, "Datasets/BP-SC/Timepoints.csv", row.names = TRUE)

# HC/SC: load tfs -> pct expressed -> expressed tfs
tfs <- as.matrix(read.table("zfish-tfs-baek2022.csv", header = FALSE, sep = ','))
hc.traj.counts <- GetAssayData(hc.traj, assay = 'RNA', slot = 'counts')
sc.traj.counts <- GetAssayData(sc.traj, assay = 'RNA', slot = 'counts')
hc.genes.expressed <- rowMeans(hc.traj.counts > 0)
sc.genes.expressed <- rowMeans(sc.traj.counts > 0)
tfs.bool <- ((rownames(hc.traj.counts) %in% tfs) & ((hc.genes.expressed > .01) | (sc.genes.expressed > .01)))

# HC/SC: raw counts -> normalized data -> save csv files
hc.traj.counts <- hc.traj.counts[tfs.bool, ]
sc.traj.counts <- sc.traj.counts[tfs.bool, ]
hc.traj.data <- as.matrix(GetAssayData(hc.traj, assay = 'RNA', slot = 'data'))[tfs.bool, ]
sc.traj.data <- as.matrix(GetAssayData(sc.traj, assay = 'RNA', slot = 'data'))[tfs.bool, ]
#write.csv(hc.traj.counts, "Datasets/BP-HC/RawCountsData.csv", row.names = TRUE)
#write.csv(sc.traj.counts, "Datasets/BP-SC/RawCountsData.csv", row.names = TRUE)
#write.csv(hc.traj.data, "Datasets/BP-HC/NormalizedData-finetune.csv", row.names = TRUE)
#write.csv(sc.traj.data, "Datasets/BP-SC/NormalizedData-finetune.csv", row.names = TRUE)

# HC trajectory: tradeSeq -> differentially expressed tfs -> save csv
# set.seed(1)
# icMat <- evaluateK(counts = hc.traj.counts, pseudotime = hc.traj.pt, cellWeights = rep(1, ncol(hc.traj.counts)), k = 3:15, verbose = TRUE)
hc.traj.tradeseq <- fitGAM(counts = hc.traj.counts, pseudotime = hc.traj.pt, cellWeights = rep(1, ncol(hc.traj.counts)), nknots = 7, verbose = TRUE)
hc.traj.assoRes <- associationTest(hc.traj.tradeseq)
hc.traj.padj <- p.adjust(hc.traj.assoRes$pvalue, method = "bonferroni")
hc.traj.results <- cbind(hc.traj.assoRes$pvalue, hc.traj.padj, hc.traj.assoRes$meanLogFC)
rownames(hc.traj.results) <- rownames(hc.traj.counts)
colnames(hc.traj.results) <- c('pValue', 'Adjusted', 'logFC')
#write.table(hc.traj.results, "Datasets/BP-HC/tradeSeq.csv", sep = ',', col.names = NA)

# SC trajectory: tradeSeq -> differentially expressed tfs -> save csv
#set.seed(1)
#icMat <- evaluateK(counts = sc.traj.counts, pseudotime = sc.traj.pt, cellWeights = rep(1, ncol(sc.traj.counts)), k = 3:15, verbose = TRUE)
sc.traj.tradeseq <- fitGAM(counts = sc.traj.counts, pseudotime = sc.traj.pt, cellWeights = rep(1, ncol(sc.traj.counts)), nknots = 6, verbose = TRUE)
sc.traj.assoRes <- associationTest(sc.traj.tradeseq)
sc.traj.padj <- p.adjust(sc.traj.assoRes$pvalue, method = "bonferroni")
sc.traj.results <- cbind(sc.traj.assoRes$pvalue, sc.traj.padj, sc.traj.assoRes$meanLogFC)
rownames(sc.traj.results) <- rownames(sc.traj.counts)
colnames(sc.traj.results) <- c('pValue', 'Adjusted', 'logFC')
#write.table(sc.traj.results, "Datasets/BP-SC/tradeSeq.csv", sep = ',', col.names = NA)

# check known genes
key.genes <- c("atoh1a", "atoh1b", "barhl1a", "ebf3a", "emx2", "etv4", "foxn4", "gata2a", "gata2b",
               "gfi1aa", "gfi1ab", "her15.1", "her4.1", "her6", "her8a", "her9", "hes6", "hey1",
               "hey2", "insm1a", "isl1", "klf17", "klf2a", "lhx4", "myclb", "notch1a", "notch3",
               "pax2a", "pou4f1", "prdm1a", "prox1a", "six1a", "six1b", "six2a", "six2b", "sox2",
               "sox21a", "sox3", "sox4a.1", "tead1b")
#sort(sc.traj.results[rownames(sc.traj.results) %in% key.genes, 1])
FeaturePlot(sc.traj, keep.scale = NULL, min.cutoff = 'q5', features = c("sox2", "six1b", "six1a", "sox21a"))
FeaturePlot(hc.traj, keep.scale = NULL, min.cutoff = 'q5', features = c("ybx1", "fosab", "her4.1", "znf536"))
