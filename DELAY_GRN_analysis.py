#%%
import os, re
import subprocess
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib_venn import venn2
from pygam import LinearGAM, s

pred_hc = pd.read_csv('Datasets/BP-HC/regPredictions.csv', index_col = 0)
pred_sc = pd.read_csv('Datasets/BP-SC/regPredictions.csv', index_col = 0)
idx_pred = pred_hc.index.sort_values()
pred_grn_hc = pred_hc.loc[idx_pred, idx_pred]
pred_grn_sc = pred_sc.loc[idx_pred, idx_pred]
pred_grn_arr = np.maximum(pred_grn_hc.values, pred_grn_sc.values)
pred_grn = pd.DataFrame(pred_grn_arr, index = idx_pred, columns = idx_pred)
pred_grn_rank = pred_grn.rank(axis = 0, ascending = False)
pred_grn = ((pred_grn_rank <= 20) & (pred_grn > .5)).astype(int)
pred_grn_hub = pred_grn.sum(axis = 1)
pred_grn_hub = pred_grn_hub[pred_grn_hub > 0]

#%%
# Treemap of GRN hubs
width, aspect = 850, .75
fig = go.Figure(go.Treemap(
    labels = pred_grn_hub.index,
    values = pred_grn_hub.values,
    parents  = ['DELAY GRN — Hubs (Out Degree)'] * pred_grn_hub.size,
    textinfo = 'label+value',
    marker_colorscale = 'Greys'))
# fig.write_image('figures/figure-1/grn-hubs.pdf', width = width, height = width * aspect)
fig.show()

#%%
def time_to_hrs(series: pd.Series) -> np.ndarray:
    def convert(entry):
        num = float(re.findall(r'\d+', entry)[0])
        if 'min' in entry:
            return num / 60
        elif 'hr' in entry:
            return num
    return series.apply(convert).values

# load data and define functions for plotting
X_HC = pd.read_csv('Datasets/BP-HC/NormalizedData.csv', index_col = 0)
X_SC = pd.read_csv('Datasets/BP-SC/NormalizedData.csv', index_col = 0)
pt_HC = pd.read_csv('Datasets/BP-HC/PseudoTime.csv', index_col = 0)
pt_SC = pd.read_csv('Datasets/BP-SC/PseudoTime.csv', index_col = 0)
t_HC = pd.read_csv('Datasets/BP-HC/Timepoints.csv', index_col = 0)
t_SC = pd.read_csv('Datasets/BP-SC/Timepoints.csv', index_col = 0)
# _X_HC = pd.read_csv('Datasets/BP-HC/RawCountsData.csv', index_col = 0)
# _X_SC = pd.read_csv('Datasets/BP-SC/RawCountsData.csv', index_col = 0)
pt_HC = (pt_HC / pt_HC.max()).values  # [0, 1]
pt_SC = (pt_SC / pt_SC.max()).values  #
t_HC = time_to_hrs(t_HC['Timepoint'])   # [0, 10] hrs
t_SC = time_to_hrs(t_SC['Timepoint'])   #

def visualize(data: np.ndarray, pt: np.ndarray, scale: bool, 
              n_spl: int, xx: np.ndarray) -> np.ndarray:
    if scale: data = scale_expr(data)
    GAM = LinearGAM(s(0, n_splines = n_spl)).fit(pt, data.mean(0))
    data = GAM.predict(xx).reshape(1, -1)
    if scale: data = scale_expr(data)
    return data

def scale_expr(yy: np.ndarray) -> np.ndarray:
    return (yy - yy.min(1)[:, None]) / (yy.max(1) - yy.min(1))[:, None]

#%%
# Gene expression: pre-bifurcation (1-4 hrs)
ptlim, n_spl, n_pts = [.31, .665], 10, 300
X = pd.concat((X_HC, X_SC), axis = 1)
pt = np.concatenate((pt_HC, pt_SC)).flatten()
t = np.concatenate((t_HC, t_SC))
pt_ord = pt.argsort(); pt = pt[pt_ord]
pt_ix = (pt >= ptlim[0]) & (pt <= ptlim[1])
pt = pt[pt_ix]
X = X.iloc[:, pt_ord].loc[:, pt_ix]
t = t[pt_ord][pt_ix].reshape(1, -1)

# plot pseudotime model: f(pt) = t
n_spl_t, n_pts_tt = 4, 5000
xx_tt = np.linspace(ptlim[0], ptlim[1], n_pts_tt)
tt = visualize(t, pt, False, n_spl_t, xx_tt).flatten()
fig, ax = plt.subplots(1, 1, figsize = (2.5, 2.5))
ax.plot(xx_tt, tt, lw = 2, c = 'tab:red')
xticks = np.arange(ptlim[0] + .1, ptlim[1] + .1, .1).round(1)
ax.set_xticks(xticks, xticks, size = 9.5)
yticks = np.arange(2, 4.1, 1).astype(int)
ax.set_yticks(yticks, yticks, size = 9.5)
ax.set_xlabel('Pseudotime', size = 9.75)
ax.set_ylabel('Time (hrs)', size = 9.75)
plt.show()

def ix_mapped(arr1, arr2):
    ix = np.zeros_like(arr2, dtype = int)
    for i in range(arr2.size):
        ix[i] = np.argmin(np.abs(arr1 - arr2[i]))
    return ix

# inverse pseudotime model: f^-1(t) ~= pt
xx_pt = np.linspace(tt.min(), tt.max(), n_pts)
xx_ix = ix_mapped(tt, xx_pt)
xx = xx_tt[xx_ix]

# gene models: ybx1, her4, atoh1a, bhlhe40, ap1
genes = pred_grn_hub.sort_values(ascending = False).index[[0, 1, 2, 3, 4, 5, 8, 21]].values
expr_genes = np.zeros((genes.size, n_pts))
for i in range(genes.size):
    expr_genes[i, :] = visualize(X.loc[[genes[i]], :].values, pt, True, n_spl, xx)
plt_ix_genes = np.argsort(np.argmax(expr_genes, axis = 1))

# plot expression heatmap
fs1, fs2, fs3, fs4, ar, lw = 10, 9.5, 9, 8.25, 30, .65
fig, ax = plt.subplots(1, 1, figsize = (3.75, 1.95))
ax.imshow(expr_genes[plt_ix_genes], aspect = ar, interpolation = 'nearest', cmap = 'Reds')
xticks = ix_mapped(tt[xx_ix], yticks)
ax.set_xticks(xticks, yticks, size = fs3)
yticks = np.arange(genes.size)
ax.set_yticks(yticks, genes[plt_ix_genes], size = fs3, style = 'italic')
ax.set_xlabel('Time (hrs)', size = fs2)
fig.text(.51, -.195, r'Supporting Cells $\boldsymbol{\longrightarrow}$ Bifurcation', ha = 'center', size = fs4)
ax.set_title('Gene Expression', size = fs1)
for spine in ax.spines.values(): spine.set_linewidth(lw)
ax.tick_params(axis = 'both', which = 'both', width = lw)
# plt.savefig('figures/figure-1/expression-heatmap.pdf', bbox_inches = 'tight', dpi = 600)
plt.show()

# %%
# # download enhancer sequences (GRCz11) -> fragment (in silico)
# _2bit_, frg_sz = '/mnt/c/Users/../Documents/twoBitToFa', 100
# dr11_url = 'https://hgdownload.soe.ucsc.edu/goldenPath/danRer11/bigZips/danRer11.2bit'
# enh_df = pd.read_csv('motifs/EnhancerRegions50kb.csv')
# for ix in enh_df.index:
#     name, chr, start, end = enh_df.loc[ix ,['name', 'seqnames', 'start', 'end']].values
#     name_clean = re.sub(r'[-:.]', '_', name)
#     enhseq_ix_fn = f'motifs/sequences/{name_clean}.txt'
#     frg_ix_fn = f'motifs/fragments/{name_clean}.txt'
#     with open(enhseq_ix_fn, 'w') as f: pass
#     result = subprocess.run(['wsl', _2bit_, dr11_url, enhseq_ix_fn, f'-seq={chr}', f'-start={start}', f'-end={end}'], capture_output = True, text = True)
#     if result.returncode != 0: print("Error:", result.stderr)
#     with open(enhseq_ix_fn, 'r') as f1:
#         with open(frg_ix_fn, 'w') as f2:
#             seq = ''.join(''.join(f1.readlines()).split('\n')[1:])  
#             for j in range(len(seq) // frg_sz):
#                 frg_start, frg_end = (j * frg_sz), (j * frg_sz) + frg_sz 
#                 f2.write(f'>{chr}:{start + frg_start}-{start + frg_end - 1}\n')
#                 f2.write(f'{seq[frg_start : frg_end]}\n')

# # compile fragments: ybx1 targets vs. others
# ybx1_targets = list(pred_grn.index.values[pred_grn.loc['ybx1'].astype(bool)])
# d = 'motifs/ybx1_targets/streme'
# with open(f'{d}/primary_fragments.txt', 'w') as f1:
#     for i in range(len(ybx1_targets)):
#         fn = re.sub(r'[-:.]', '_', ybx1_targets[i])
#         try:
#             with open(f'motifs/fragments/{fn}.txt', 'r') as f2:
#                 f1.write(''.join(f2.readlines())) 
#         except:
#             pass

# controls = list(set(pred_grn.index) - set(ybx1_targets))
# with open(f'{d}/control_fragments.txt', 'w') as f1:
#     for i in range(len(controls)):
#         fn = re.sub(r'[-:.]', '_', controls[i])
#         try:
#             with open(f'motifs/fragments/{fn}.txt', 'r') as f2:
#                 f1.write(''.join(f2.readlines()))
#         except:
#             pass

# # compile all enhancer sequences and gene names
# with open(f'motifs/sequences-named.txt', 'w') as f1:
#     for i in range(pred_grn.index.size):
#         fn = re.sub(r'[-:.]', '_', pred_grn.index[i])
#         try:
#             with open(f'motifs/sequences/{fn}.txt', 'r') as f2:
#                 seq = f2.readlines()
#                 f1.write(re.sub(r'[-:\n]', '_', seq[0]) + f'{fn}\n')
#                 f1.write(''.join(seq[1:])) 
#         except:
#             pass

# %%
# Motif enrichment near TSS: ybx1 targets vs. others
n_bins, w_sm, xlim, w_enh = 10000, 50, [-400, 900], 50000
fimo = pd.read_csv('motifs/ybx1_targets/fimo/fimo-1e-1.tsv', sep = '\t', skipfooter = 3)
fimo[['chr', 'seq_start', 'seq_end', 'gene']] = fimo['sequence_name'].str.split('_', n = 3, expand = True)
fimo = fimo.loc[~(fimo.isnull().any(axis = 1))]  # removes alt chromosomes
fimo[['seq_start', 'seq_end']] = fimo[['seq_start', 'seq_end']].astype(int)
fimo['motif_loc'] = fimo[['start', 'stop']].mean(1) - w_enh
ybx1_targets = list(pred_grn.index.values[pred_grn.loc['ybx1'].astype(bool)])
fimo['ybx1'] = np.isin(fimo['gene'], ybx1_targets)
counts_ybx1, bins = np.histogram(fimo.loc[fimo['ybx1'], 'motif_loc'], bins = n_bins)
counts_other, _ = np.histogram(fimo.loc[~fimo['ybx1'], 'motif_loc'], bins = n_bins)
xx = (bins[:-1] + bins[1:]) / 2
ix = (xx > xlim[0]) & (xx < xlim[1])

# relative enrichment near TSS: normalized to others
yy_ybx1 = counts_ybx1 / counts_ybx1.mean()
yy_other = counts_other / counts_other.mean()
yy_ybx1 = np.convolve(yy_ybx1, np.ones(w_sm) / w_sm, 'same')                # smoothing
yy_other = np.convolve(yy_other, np.ones(w_sm) / w_sm, 'same')              #
xx = xx[ix]; yy_ybx1 = yy_ybx1[ix]; yy_other = yy_other[ix]
yy_ybx1 = (yy_ybx1 - yy_ybx1.min()) / (yy_ybx1.max() - yy_ybx1.min())       # scaling
yy_other = (yy_other - yy_other.min()) / (yy_other.max() - yy_other.min())  #
yy_ybx1 = yy_ybx1 / yy_other     # normalize

fs1, fs2, lw1, lw2 = 8.5, 8, 1.5, .65
xlim_ax = [xlim[0] - 100, xlim[1] + 100]
fig, ax = plt.subplots(1, 1, figsize = (2.5, 1.75))
ax.plot(xx, yy_ybx1, lw = lw1, c = 'b')
xticks = np.arange(xlim_ax[0], xlim_ax[1] + 1, 500)
xticklabels = ['-0.5', '0', '0.5', '1']
ax.set_xticks(xticks, xticklabels, fontsize = fs2)
ax.set_xlabel('Distance From TSS (kb)', size = fs1)
yticks = np.arange(1, 5.1, 1, dtype = int)
ax.set_yticks(yticks, yticks, fontsize = fs2)
ax.set_ylabel('Motif Enrichment', size = fs1)
for spine in ax.spines.values(): spine.set_linewidth(lw2)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis = 'both', which = 'both', width = lw2)
# plt.savefig('figures/figure-1/motif-enrichment.pdf', bbox_inches = 'tight', dpi = 600)
plt.show()

# loc = [-500, 500]
# ix = (fimo['motif_loc'] > loc[0]) & (fimo['motif_loc'] < loc[1])
# fimo_tss = fimo.loc[ix]
# # n_motifs_genes = fimo_tss['gene'].value_counts()
# # n_motifs_genes
# # list(n_motifs_genes.index)
# fimo_tss.loc[fimo_tss['gene'] == 'atoh1a', :]

#%%
# # Regulon venn diagram: ybx1 vs. her4 (= her4.1 + her4.2.1)
# ybx1_regulon = set(pred_grn.index[pred_grn.loc['ybx1'].astype(bool)])
# her4_regulon = set(pred_grn.index[pred_grn.loc[['her4.1', 'her4.2.1']].max(0).astype(bool)])
# fig, ax = plt.subplots(1, 1, figsize = (2.75, 2.75))
# venn2([ybx1_regulon, her4_regulon], ('', ''), ('magenta', 'cyan'), alpha = 1)
# # plt.savefig('figures/figure-1/ybx1-her4-venn.pdf', bbox_inches = 'tight', dpi = 600)
# plt.show()

# # Regulon similarities: ybx1, fos, her4
# genes = ['ybx1', 'fos', 'her4']   # fos = fosab, fosl1a;  her4 = her4.1, her4.2.1
# pred_grn_genes = pred_grn.loc[[genes[0], f'{genes[1]}ab', f'{genes[2]}.1']].values
# pred_grn_genes[1, :] = np.maximum(pred_grn_genes[1, :], pred_grn.loc[f'{genes[1]}l1a'].values)  # OR
# pred_grn_genes[2, :] = np.maximum(pred_grn_genes[2, :], pred_grn.loc[f'{genes[2]}.2.1'].values) #
# J = 1 - pairwise_distances(pred_grn_genes.astype(bool), metric = 'jaccard')
# np.fill_diagonal(J, 0)
# J = pd.DataFrame(J, index = genes, columns = genes)
# G = nx.from_pandas_adjacency(J)
# pos = nx.spectral_layout(G)
# elabel = {e : round(G.edges[e]['weight'], 2) for e in  nx.edges(G)}

# ns, lw, fs1, fs2, m = 800, 3, 10, 10.33, .2
# fig, ax = plt.subplots(1, 1, figsize = (2, 2))
# e = nx.draw_networkx_edges(G, pos, width = lw, edge_color = 'w')
# nx.draw_networkx_nodes(G, pos, node_shape = '$\u25AC$', node_size = ns, node_color = 'w')
# nx.draw_networkx_labels(G, pos, font_color = 'k', font_size = fs1)
# nx.draw_networkx_edge_labels(G, pos, elabel, font_color = 'w', bbox = {'fc' : 'k'})
# ax.text(0, 0, r'$J$', size = fs2, ha = 'center', c = 'w')
# ax.set_facecolor('k')
# ax.set_xmargin(m)
# ax.set_ymargin(m)
# ax.set_box_aspect(1)
# plt.savefig('figures/figure-1/ybx1-jaccard.pdf', bbox_inches = 'tight', dpi = 600)
# plt.show()   
