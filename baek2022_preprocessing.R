library(data.table)
library(Matrix)
library(Seurat)
library(slingshot)
library(tradeSeq)
library(ggplot2)
library(viridis)
library(FNN)
library(SeuratDisk)

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

# find shared variable features -> scale -> PCA
features <- SelectIntegrationFeatures(samples.list, nfeatures = 3500)
samples.list <- lapply(X = samples.list, FUN = function(x) {
  x <- ScaleData(x, features = features)
  x <- RunPCA(x, features = features) })

# integration using pairwise anchors between data sets
anchors <- FindIntegrationAnchors(samples.list, anchor.features = features, 
                                  reduction = 'rpca', dims = 1:50)
samples.integrated <- IntegrateData(anchors, features = features, dims = 1:50)
samples.list.merge <- merge(x = samples.list[[1]], y = samples.list[2:7])

# dimensionality reduction: scale -> PCA -> tSNE -> plot
samples.integrated <- ScaleData(samples.integrated, features = features)
samples.integrated <- RunPCA(samples.integrated, features = features)
ElbowPlot(samples.integrated, ndims = 30)
samples.integrated <- RunTSNE(samples.integrated, dims = 1:13, seed.use = 1111, 
                        perplexity = ncol(samples.integrated) / 100, verbose = TRUE)
DimPlot(samples.integrated, group.by = 'orig.ident')

# find neighbors -> cluster -> remove skin/blood/unknown
marker.genes <- c('dld', 'atoh1a', 'atoh1b', 'myo6b', 'tekt3',
                  'isl1', 'ebf3a', 'lfng', 'si:ch73-261i21.5',
                  'sost', 'mcm4', 'tspan1', 'ovgp1', 'krt4', 
                  'krt18', 'spp1')
samples.integrated <- FindNeighbors(samples.integrated, dims = 1:13)
set.seed(1)
samples.integrated <- FindClusters(samples.integrated, resolution = .4)
DimPlot(samples.integrated)
DotPlot(samples.integrated, marker.genes)
clus.remove <- c('0', '8', '10', '12', '13')
cells.remove <- WhichCells(samples.integrated, idents = clus.remove)
DimPlot(samples.integrated, cells.highlight = cells.remove)
cells.keep.ix <- !(Cells(samples.integrated) %in% cells.remove)
samples.integrated.nm <- samples.integrated[, cells.keep.ix]

# update embedding: scale -> PCA -> tSNE -> plot
samples.integrated.nm <- ScaleData(samples.integrated.nm, features = features)
samples.integrated.nm <- RunPCA(samples.integrated.nm, features = features)
ElbowPlot(samples.integrated.nm, ndims = 30)
samples.integrated.nm <- RunTSNE(samples.integrated.nm, dims = 1:10, seed.use = 1111, 
    perplexity = ncol(samples.integrated.nm) / 15, max_iter = 5000, verbose = TRUE)
DimPlot(samples.integrated.nm)

# update clustering: find neighbors -> cluster -> assign labels -> save
samples.integrated.nm <- FindNeighbors(samples.integrated.nm, dims = 1:10)
set.seed(1)
samples.integrated.nm <- FindClusters(samples.integrated.nm, resolution = 1.7)
DimPlot(samples.integrated.nm)
DotPlot(samples.integrated.nm, marker.genes[seq(length(marker.genes) - 3)])
labels <- list('Central SCs', 'AP SCs', 'DV SCs', 'Amplifying SCs', 
               'Mantle Cells', 'HC Progenitors', 'Young HCs', 'Mature HCs')
labels.clus <- list(c('3', '4', '6', '8', '9', '10', '11', '15'),
                    c('0', '2', '12', '13', '24'), c('17', '21'),
                    c('1', '19', '22', '23'), c('5', '7', '14'),
                    c('16', '26'), c('18', '20'), c('25'))
cells.remove <- WhichCells(samples.integrated.nm, idents = '27')
cells.keep.ix <- !(Cells(samples.integrated.nm) %in% cells.remove)
samples.integrated.nm.labeled <- samples.integrated.nm[, cells.keep.ix]
labels.map <- setNames(rep(labels, sapply(labels.clus, length)), unlist(labels.clus))
samples.integrated.nm.labeled <- RenameIdents(samples.integrated.nm.labeled, labels.map)
#saveRDS(samples.integrated.nm.labeled, 'baek2022_objects/samples_integrated_nm_labeled.rds')

# plot neuromast tSNE
plt.order <- c('HC Progenitors', 'Mature HCs', 'Young HCs', 'AP SCs', 
               'Central SCs', 'Amplifying Cells', 'Mantle Cells', 'DV SCs')
color.palette <- c("#52C39E", "#009E73", "#29AF7F", "#E69F00",
                   "#00A5CF", "#D55E00", "#999999", "#CC79A7")
DimPlot(samples.integrated.nm.labeled, order = plt.order) +
  scale_color_manual(values = rev(color.palette)) + 
  scale_x_reverse() +
  theme(axis.title = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        panel.grid = element_blank(),
        panel.border = element_blank(),
        axis.line = element_blank())

# bifurcation: subset -> find variable features -> scale -> PCA -> tSNE -> plot/save
ix.csc <- (Idents(samples.integrated.nm.labeled) == 'Central SCs')
ix.prog <- (Idents(samples.integrated.nm.labeled) == 'HC Progenitors')
ix.hc <- (Idents(samples.integrated.nm.labeled) %in% c('Young HCs', 'Mature HCs'))
sample.idents <- samples.integrated.nm.labeled@meta.data$orig.ident
ix.regen <- (sample.idents != 'homeo')
ix.0.to.3 <- (sample.idents %in% c('0min', '30min', '1hr', '3hr'))
ix.1.to.5 <- (sample.idents %in% c('1hr', '3hr', '5hr'))
ix.3.to.10 <- (sample.idents %in% c('3hr', '5hr', '10hr'))
ix.sc.traj <- (ix.csc & ix.regen)
ix.hc.traj <- (ix.csc & ix.0.to.3) | (ix.prog & ix.1.to.5) | (ix.hc & ix.3.to.10)
ix.bif.traj <- ix.sc.traj | ix.hc.traj
bif.cells <- Cells(samples.integrated.nm.labeled)[ix.bif.traj]
bif.traj <- samples.list.merge[, bif.cells]
bif.traj <- FindVariableFeatures(bif.traj, nfeatures = 5000)
bif.traj <- ScaleData(bif.traj)
bif.traj <- RunPCA(bif.traj)
ElbowPlot(bif.traj, ndims = 30)
bif.traj <- RunTSNE(bif.traj, dims = 1:10, seed.use = 123, perplexity = ncol(bif.traj) / 5,
                    max_iter = 2000, verbose = TRUE)
DimPlot(bif.traj, group.by = 'orig.ident')
#saveRDS(bif.traj, 'baek2022_objects/bif_traj.rds')
#bif.traj.join <- JoinLayers(bif.traj)
#bif.traj.join.counts <- bif.traj.join@assays$RNA@layers$counts
#colnames(bif.traj.join.counts) <- colnames(bif.traj.join)
#rownames(bif.traj.join.counts) <- rownames(bif.traj.join)
#bif.traj.join[['RNA']] <- CreateAssayObject(bif.traj.join.counts)
#bif.traj.join <- NormalizeData(bif.traj.join)
#SaveH5Seurat(bif.traj.join, 'baek2022_objects/bif_traj.h5Seurat')
#Convert('baek2022_objects/bif_traj.h5Seurat', dest = 'h5ad')

# SC trajectory: subset -> slingshot pseudotime -> plot/save
sc.cells <- Cells(samples.integrated.nm.labeled)[ix.sc.traj]
sc.traj <- bif.traj[, sc.cells]
DimPlot(sc.traj, group.by = 'orig.ident')
sc.clus <- sc.traj@meta.data$orig.ident
sc.clus.uniq <- unique(sc.clus)
sc.clus.n.uniq <- length(sc.clus.uniq)
sc.adj <- matrix(0, nrow = sc.clus.n.uniq, ncol = sc.clus.n.uniq, 
                 dimnames = list(sc.clus.uniq, sc.clus.uniq))
for(i in seq_len(sc.clus.n.uniq - 1)) {sc.adj[sc.clus.uniq[i], sc.clus.uniq[i + 1]] <- 1}
sc.traj.pca <- Embeddings(sc.traj)[, 1:20]
sc.sds <- newSlingshotDataSet(
            sc.traj.pca, sc.clus, adjacency = sc.adj, lineages = list(sc.clus.uniq),
            slingParams = list(start.clus = sc.clus.uniq[1], end.clus = sc.clus.uniq[-1]))
sc.sds <- SlingshotDataSet(getCurves(sc.sds, approx_points = 30)); sc.sds
sc.traj[['PseudoTime']] <- slingPseudotime(sc.sds)
sc.traj.pt <- as.matrix(sc.traj[['PseudoTime']])
sc.curve <- as.data.frame(sc.sds@curves[[1]]$s)
sc.curve.nn <- get.knnx(sc.traj.pca, sc.curve, 1)
sc.curve$tSNE1 <- Embeddings(sc.traj, 'tsne')[sc.curve.nn$nn.index, 1]
sc.curve$tSNE2 <- Embeddings(sc.traj, 'tsne')[sc.curve.nn$nn.index, 2]
FeaturePlot(sc.traj, features = 'PseudoTime', pt.size = .01) +
  geom_path(data = sc.curve, aes(x = tSNE1, y = tSNE2), color = "#00A99D", linewidth = 1) +
  scale_color_gradientn(colors = viridis::rocket(100), labels = c('', ''),
            breaks = c(min(sc.traj[['PseudoTime']]), max(sc.traj[['PseudoTime']]))) +
  guides(color = guide_colorbar(
          barwidth = 0, barheight = 3, draw.ulim = FALSE, draw.llim = FALSE)) +
  coord_flip() + scale_x_reverse() +
  theme(axis.title = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        panel.grid = element_blank(),
        panel.border = element_blank(),
        axis.line = element_blank(),
        plot.title = element_blank())
#write.csv(sc.traj.pt, 'Datasets/BP-SC/PseudoTime.csv', row.names = TRUE)
#saveRDS(sc.traj, 'baek2022_objects/sc_traj.rds')
#sc.traj.join <- JoinLayers(sc.traj)
#sc.traj.join.counts <- sc.traj.join@assays$RNA@layers$counts
#colnames(sc.traj.join.counts) <- colnames(sc.traj.join)
#rownames(sc.traj.join.counts) <- rownames(sc.traj.join)
#sc.traj.join[['RNA']] <- CreateAssayObject(sc.traj.join.counts)
#sc.traj.join <- NormalizeData(sc.traj.join)
#SaveH5Seurat(sc.traj.join, 'baek2022_objects/sc_traj.h5Seurat')
#Convert('baek2022_objects/sc_traj.h5Seurat', dest = 'h5ad')

# HC trajectory: subset -> tSNE -> slingshot pseudotime -> plot/save
hc.traj <- samples.integrated.nm.labeled[, ix.hc.traj]
hc.traj <- RunTSNE(hc.traj, dims = 1:10, seed.use = 11, perplexity = ncol(hc.traj) / 50,
                   max_iter = 1000, verbose = TRUE)
DimPlot(hc.traj)
hc.clus <- Idents(hc.traj)
hc.traj.pca <- Embeddings(hc.traj)[, 1:15]
hc.sds <- getLineages(hc.traj.pca, hc.clus, start.clus = 'Central SCs', end.clus = 'Mature HCs')
hc.sds <- SlingshotDataSet(getCurves(hc.sds, approx_points = 50)); hc.sds
hc.traj[['PseudoTime']] <- slingPseudotime(hc.sds)
hc.traj.pt <- as.matrix(hc.traj[['PseudoTime']])
hc.traj[['PseudoTime_plt']] <- log10(hc.traj[['PseudoTime']] + 4.5)
hc.curve <- as.data.frame(hc.sds@curves[[1]]$s)
hc.curve.nn <- get.knnx(hc.traj.pca, hc.curve, 1)
hc.curve$tSNE1 <- Embeddings(hc.traj, 'tsne')[hc.curve.nn$nn.index, 1]
hc.curve$tSNE2 <- Embeddings(hc.traj, 'tsne')[hc.curve.nn$nn.index, 2]
FeaturePlot(hc.traj, features = 'PseudoTime_plt', pt.size = .01) +
  geom_path(data = hc.curve, aes(x = tSNE1, y = tSNE2), color = "#00A99D", linewidth = 1) +
  scale_color_gradientn(colors = viridis::rocket(100), labels = c('Min', 'Max'),
      breaks = c(min(hc.traj[['PseudoTime_plt']]), max(hc.traj[['PseudoTime_plt']]))) +
  guides(color = guide_colorbar(
            barwidth = .5, barheight = 3, draw.ulim = FALSE, draw.llim = FALSE)) +
  scale_y_reverse() +
  theme(axis.title = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        panel.grid = element_blank(),
        panel.border = element_blank(),
        axis.line = element_blank(),
        plot.title = element_blank())
#write.csv(hc.traj.pt, "Datasets/BP-HC/PseudoTime.csv", row.names = TRUE)
#saveRDS(hc.traj, 'baek2022_objects/hc_traj.rds')
#hc.traj.join <- JoinLayers(hc.traj, assay = 'RNA')
#hc.traj.join.counts <- hc.traj.join@assays$RNA@layers$counts
#colnames(hc.traj.join.counts) <- colnames(hc.traj.join)
#rownames(hc.traj.join.counts) <- rownames(hc.traj.join[['RNA']])
#hc.traj.join[['RNA']] <- CreateAssayObject(hc.traj.join.counts)
#hc.traj.join <- NormalizeData(hc.traj.join, assay = 'RNA')
#DefaultAssay(hc.traj.join) <- 'RNA'
#SaveH5Seurat(hc.traj.join, 'baek2022_objects/hc_traj.h5Seurat')
#Convert('baek2022_objects/hc_traj.h5Seurat', dest = 'h5ad')

# HC/SC timepoints
sc.traj.t <- as.matrix(sc.clus)
rownames(sc.traj.t) <- colnames(sc.traj)
colnames(sc.traj.t) <- "Timepoint"
hc.traj.t <- as.matrix(hc.traj@meta.data$orig.ident)
rownames(hc.traj.t) <- colnames(hc.traj)
colnames(hc.traj.t) <- "Timepoint"
#write.csv(sc.traj.t, "Datasets/BP-SC/Timepoints.csv", row.names = TRUE)
#write.csv(hc.traj.t, "Datasets/BP-HC/Timepoints.csv", row.names = TRUE)

# HC/SC: load tfs -> pct expressed -> expressed tfs
tfs <- as.matrix(read.table("zfish-tfs-baek2022.csv", header = FALSE, sep = ','))
hc.traj.counts <- GetAssayData(hc.traj, assay = 'RNA', slot = 'counts')
sc.traj.counts <- GetAssayData(sc.traj, assay = 'RNA', slot = 'counts')
hc.genes.expr <- rowMeans(hc.traj.counts > 0)
sc.genes.expr <- rowMeans(sc.traj.counts > 0)
tfs.bool <- ((rownames(hc.traj.counts) %in% tfs) & ((hc.genes.expr > .01) | (sc.genes.expr > .01)))

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
#set.seed(1)
#icMat <- evaluateK(counts = hc.traj.counts, pseudotime = hc.traj.pt, k = 3:15,
#                   cellWeights = rep(1, ncol(hc.traj.counts)), verbose = TRUE)
hc.traj.tradeseq <- fitGAM(counts = hc.traj.counts, pseudotime = hc.traj.pt, nknots = 7,
                           cellWeights = rep(1, ncol(hc.traj.counts)), verbose = TRUE)
hc.traj.assoRes <- associationTest(hc.traj.tradeseq)
hc.traj.padj <- p.adjust(hc.traj.assoRes$pvalue, method = 'bonferroni')
hc.traj.res <- cbind(hc.traj.assoRes$pvalue, hc.traj.padj, hc.traj.assoRes$meanLogFC)
rownames(hc.traj.res) <- rownames(hc.traj.counts)
colnames(hc.traj.res) <- c('pValue', 'Adjusted', 'logFC')
#write.table(hc.traj.res, "Datasets/BP-HC/tradeSeq.csv", sep = ',', col.names = NA)

# SC trajectory: tradeSeq -> differentially expressed tfs -> save csv
#set.seed(1)
#icMat <- evaluateK(counts = sc.traj.counts, pseudotime = sc.traj.pt, k = 3:15, 
#                   cellWeights = rep(1, ncol(sc.traj.counts)), verbose = TRUE)
sc.traj.tradeseq <- fitGAM(counts = sc.traj.counts, pseudotime = sc.traj.pt, nknots = 6,
                           cellWeights = rep(1, ncol(sc.traj.counts)), verbose = TRUE)
sc.traj.assoRes <- associationTest(sc.traj.tradeseq)
sc.traj.padj <- p.adjust(sc.traj.assoRes$pvalue, method = 'bonferroni')
sc.traj.res <- cbind(sc.traj.assoRes$pvalue, sc.traj.padj, sc.traj.assoRes$meanLogFC)
rownames(sc.traj.res) <- rownames(sc.traj.counts)
colnames(sc.traj.res) <- c('pValue', 'Adjusted', 'logFC')
#write.table(sc.traj.res, "Datasets/BP-SC/tradeSeq.csv", sep = ',', col.names = NA)

## check known genes
#key.genes <- c("atoh1a", "atoh1b", "barhl1a", "ebf3a", "emx2", "etv4", "foxn4", "gata2a", "gata2b",
#               "gfi1aa", "gfi1ab", "her15.1", "her4.1", "her6", "her8a", "her9", "hes6", "hey1",
#               "hey2", "insm1a", "isl1", "klf17", "klf2a", "lhx4", "myclb", "notch1a", "notch3",
#               "pax2a", "pou4f1", "prdm1a", "prox1a", "six1a", "six1b", "six2a", "six2b", "sox2",
#               "sox21a", "sox3", "sox4a.1", "tead1b")
##sort(sc.traj.results[rownames(sc.traj.results) %in% key.genes, 1])
#FeaturePlot(sc.traj, keep.scale = NULL, min.cutoff = 'q5', features = c("sox2", "six1b", "six1a", "sox21a"))
#FeaturePlot(hc.traj, keep.scale = NULL, min.cutoff = 'q5', features = c("ybx1", "fosab", "her4.1", "znf536"))
