#%%
import os, re
import subprocess
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pygam import LinearGAM, s
from matplotlib_venn import venn2

# neuromast regeneration GRN
dir_hc = os.path.join('Datasets', 'BP-HC')
dir_sc = os.path.join('Datasets', 'BP-SC')
pred_hc = pd.read_csv(os.path.join(dir_hc, 'regPredictions.csv'), index_col = 0)
pred_sc = pd.read_csv(os.path.join(dir_sc, 'regPredictions.csv'), index_col = 0)
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
np.random.seed(65365)
class BubbleChart:
    def __init__(self, area, bubble_spacing = 0):
        """
        Setup for bubble collapse.

        Parameters
        ----------
        area : array-like
            Area of the bubbles.
        bubble_spacing : float, default: 0
            Minimal spacing between bubbles after collapsing.

        Notes
        -----
        If "area" is sorted, the results might look weird.
        """
        area = np.asarray(area)
        r = np.sqrt(area / np.pi)

        self.bubble_spacing = bubble_spacing
        self.bubbles = np.ones((len(area), 4))
        self.bubbles[:, 2] = r
        self.bubbles[:, 3] = area
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
        self.step_dist = self.maxstep / 2

        # calculate initial grid layout for bubbles
        length = np.ceil(np.sqrt(len(self.bubbles)))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[:len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[:len(self.bubbles)]
        self.com = self.center_of_mass()

    def center_of_mass(self):
        return np.average(
            self.bubbles[:, :2], axis = 0, weights = self.bubbles[:, 3]
        )

    def center_distance(self, bubble, bubbles):
        return np.hypot(bubble[0] - bubbles[:, 0],
                        bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble, bubbles):
        center_distance = self.center_distance(bubble, bubbles)
        return center_distance - bubble[2] - bubbles[:, 2] - self.bubble_spacing

    def check_collisions(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        idx_min = np.argmin(distance)
        return idx_min if type(idx_min) == np.ndarray else [idx_min]

    def collapse(self, n_iterations = 100):
        """
        Move bubbles to the center of mass.

        Parameters
        ----------
        n_iterations : int, default: 50
            Number of moves to perform.
        """
        for _i in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):
                rest_bub = np.delete(self.bubbles, i, 0)
                # try to move directly towards the center of mass
                # direction vector from bubble to the center of mass
                dir_vec = self.com - self.bubbles[i, :2]

                # shorten direction vector to have length of 1
                dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))

                # calculate new bubble position
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                # check whether new bubble collides with other bubbles
                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    # try to move around a bubble that you collide with
                    # find colliding bubble
                    for colliding in self.collides_with(new_bubble, rest_bub):
                        # calculate direction vector
                        dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                        dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))
                        # calculate orthogonal vector
                        orth = np.array([dir_vec[1], -dir_vec[0]])
                        # test which direction to go
                        new_point1 = (self.bubbles[i, :2] + orth * self.step_dist)
                        new_point2 = (self.bubbles[i, :2] - orth * self.step_dist)
                        dist1 = self.center_distance(self.com, np.array([new_point1]))
                        dist2 = self.center_distance(self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self.check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()

            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2

    def plot(self, ax, labels, c, s, fs_min):
        """
        Draw the bubble plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        labels : list
            Labels of the bubbles.
        """
        for i in range(len(self.bubbles)):
            circ = plt.Circle(self.bubbles[i, :2], self.bubbles[i, 2], color = c)
            ax.add_patch(circ)
            fs_i = np.asarray([(self.bubbles[i, 2] * s / len(labels[i])), fs_min]).max()
            ax.text(*self.bubbles[i, :2], labels[i], size = fs_i,
                    horizontalalignment = 'center', verticalalignment = 'center', fontstyle = 'italic')
            
# GRN hubs — bubble plot
min_deg, sp, fs_scale, fs1, fs2 = 20, 1.3, 7.5, 9, 7.5
fig, ax = plt.subplots(figsize = (5, 5), subplot_kw = dict(aspect = 'equal'))
pred_hub_plt = pred_grn_hub[pred_grn_hub >= min_deg].sample(frac = 1)
bubble_chart = BubbleChart(area = pred_hub_plt.values, bubble_spacing = sp)
bubble_chart.collapse()
bubble_chart.plot(ax, pred_hub_plt.index.values, 'xkcd:sky blue', fs_scale, fs2)

nn, xx, xy0 = [50, 100, 150], [0, 12, 27], [86.5, 105]
for i in range(len(nn)):
    circ = plt.Circle((xy0[0] + xx[i], xy0[1]), np.sqrt(nn[i] / np.pi), facecolor = 'w', edgecolor = 'k', linewidth = .5)
    ax.add_patch(circ)
    ax.text(xy0[0] + xx[i], xy0[1], nn[i], size = fs1, ha = 'center', va = 'center')
ax.text(xy0[0] + xx[1] + 3, xy0[1] + 11, 'Out Degree', size = fs1, ha = 'center', va = 'center')

ax.axis(False)
ax.relim()
ax.autoscale_view()
plt.tight_layout()
# plt.savefig(os.path.join('figures', 'figure-1', 'grn-hubs-bubble.pdf'), bbox_inches = 'tight', dpi = 600)
plt.show()

#%%
def time_to_hrs(series: pd.Series) -> np.ndarray:
    def convert(entry):
        num = float(re.findall(r'\d+', entry)[0])
        if 'min' in entry:
            return num / 60
        elif 'hr' in entry:
            return num
    return series.apply(convert).values

def fit_gam(Y: np.ndarray, X: np.ndarray, n_spl: int, n_pts: int) -> np.ndarray:
    Y = scale_expr(Y)
    GAM = LinearGAM(s(0, n_splines = n_spl)).fit(X, Y.mean(0))
    xx = np.linspace(X.min(), X.max(), n_pts)
    Yf = GAM.predict(xx).reshape(1, -1)
    return xx, scale_expr(Yf)

def scale_expr(Y: np.ndarray) -> np.ndarray:
    return (Y - Y.min(1)[:, None]) / (Y.max(1) - Y.min(1))[:, None]

# gene-expression GAMs: ybx1, her4, atoh1a, bhlhe40, ap1
t_max, n_spl, n_pts = 5, 4, 101
genes = np.asarray(['fosab', 'jun', 'fosl1a', 'bhlhe40', 'ybx1', 'atoh1a', 'her4.1', 'her4.2.1'])
expr = pd.read_csv(os.path.join(dir_hc, 'NormalizedData.csv'), index_col = 0)
t = time_to_hrs(pd.read_csv(os.path.join(dir_hc, 'Timepoints.csv'), index_col = 0)['Timepoint'])
ix = (t <= t_max); expr = expr.loc[:, ix]; t = t[ix]
exprf = np.zeros((genes.size, n_pts))
for i in range(genes.size):
    xx, exprf[i, :] = fit_gam(expr.loc[[genes[i]], :].values, t, n_spl, n_pts)

# plot gene-expression heatmaps
fs1, fs2, ar, lw = 10, 11, 8, .65
fig, ax = plt.subplots(1, 1, figsize = (4, 2))
im = ax.imshow(exprf, aspect = ar, interpolation = 'nearest', cmap = 'Reds')
cbax = ax.inset_axes([1.05, .3, .04, .4])
cbar = fig.colorbar(im, cax = cbax, orientation = 'vertical')
cbar.ax.set_yticks([0, 1], ['Min', 'Max'], fontsize = fs1)
cbar.ax.tick_params('y', size = 0)
cbar.outline.set_linewidth(lw)
xticks = np.arange(0, n_pts, (n_pts - 1) / t_max).astype(int)
xticklabels = xx[xticks].astype(int)
ax.set_xticks(xticks, xticklabels, size = fs1)
yticks = np.arange(genes.size)
ax.set_yticks(yticks, genes, size = fs1, style = 'italic')
ax.set_xlabel('Time (hrs)', size = fs1)
ax.set_title('Gene Expression', size = fs2)
for spine in ax.spines.values(): spine.set_linewidth(lw)
ax.tick_params(axis = 'both', which = 'both', width = lw)
# plt.savefig(os.path.join('figures', 'figure-1', 'revision', 'expression-heatmap.pdf'), bbox_inches = 'tight', dpi = 600)
plt.show()

#%%
# download enhancer sequences (GRCz11) -> fragment (in silico)
_2bit_, frg_sz = os.path.join('/mnt', 'c', 'Users', '..', 'Documents', 'twoBitToFa'), 100
dr11_url = 'https://hgdownload.soe.ucsc.edu/goldenPath/danRer11/bigZips/danRer11.2bit'
enh_df = pd.read_csv(os.path.join('motifs', 'EnhancerRegions50kb.csv'))
for ix in enh_df.index:
    name, chr, start, end = enh_df.loc[ix ,['name', 'seqnames', 'start', 'end']].values
    name_clean = re.sub(r'[-:.]', '_', name)
    enhseq_ix_fn = os.path.join('motifs', 'sequences', f'{name_clean}.txt')
    frg_ix_fn = os.path.join('motifs', 'fragments', f'{name_clean}.txt')
    with open(enhseq_ix_fn, 'w') as f: pass
    result = subprocess.run(['wsl', _2bit_, dr11_url, enhseq_ix_fn, f'-seq={chr}', f'-start={start}', f'-end={end}'], capture_output = True, text = True)
    if result.returncode != 0: print("Error:", result.stderr)
    with open(enhseq_ix_fn, 'r') as f1:
        with open(frg_ix_fn, 'w') as f2:
            seq = ''.join(''.join(f1.readlines()).split('\n')[1:])  
            for j in range(len(seq) // frg_sz):
                frg_start, frg_end = (j * frg_sz), (j * frg_sz) + frg_sz 
                f2.write(f'>{chr}:{start + frg_start}-{start + frg_end - 1}\n')
                f2.write(f'{seq[frg_start : frg_end]}\n')

# compile fragments: ybx1 targets vs. others
ybx1_targets = list(pred_grn.index.values[pred_grn.loc['ybx1'].astype(bool)])
d = os.path.join('motifs', 'ybx1_targets', 'streme')
with open(os.path.join(d, 'primary_fragments.txt'), 'w') as f1:
    for i in range(len(ybx1_targets)):
        fn = re.sub(r'[-:.]', '_', ybx1_targets[i])
        try:
            with open(os.path.join('motifs', 'fragments', f'{fn}.txt'), 'r') as f2:
                f1.write(''.join(f2.readlines())) 
        except:
            pass

controls = list(set(pred_grn.index) - set(ybx1_targets))
with open(os.path.join(d, 'control_fragments.txt'), 'w') as f1:
    for i in range(len(controls)):
        fn = re.sub(r'[-:.]', '_', controls[i])
        try:
            with open(os.path.join('motifs', 'fragments', f'{fn}.txt'), 'r') as f2:
                f1.write(''.join(f2.readlines()))
        except:
            pass

# compile all enhancer sequences and gene names
with open(os.path.join('motifs', 'sequences-named.txt'), 'w') as f1:
    for i in range(pred_grn.index.size):
        fn = re.sub(r'[-:.]', '_', pred_grn.index[i])
        try:
            with open(os.path.join('motifs', 'sequences', f'{fn}.txt'), 'r') as f2:
                seq = f2.readlines()
                f1.write(re.sub(r'[-:\n]', '_', seq[0]) + f'{fn}\n')
                f1.write(''.join(seq[1:])) 
        except:
            pass

#%%
# motif enrichment near TSS: ybx1 targets vs. others
n_bins, w_sm, xlim, w_enh = 10000, 50, [-400, 900], 50000
fimo = pd.read_csv(os.path.join('motifs', 'ybx1_targets', 'fimo', 'fimo-1e-1.tsv'), sep = '\t', skipfooter = 3)
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

# plot motif enrichment
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
# plt.savefig(os.path.join('figures', 'figure-1', 'motif-enrichment.pdf'), bbox_inches = 'tight', dpi = 600)
plt.show()

#%%
# regulon overlap — ybx1 targets vs. de novo motifs
ybx1_targets = pred_grn.index[pred_grn.loc['ybx1'].astype(bool)]
for char in ['-', ':', '.']: ybx1_targets = ybx1_targets.str.replace(char, '_')
ybx1_targets = set(ybx1_targets)
ybx1_motif = set(np.unique(fimo['gene']))

# plot venn diagram
fig, ax = plt.subplots(1, 1, figsize = (3, 3))
venn2([ybx1_targets, ybx1_motif], ('', ''), ('lightcoral', 'powderblue'), alpha = 1)
# plt.savefig(os.path.join('figures', 'figure-1', 'revision', 'ybx1-targets-venn.pdf'), bbox_inches = 'tight', dpi = 600)
plt.show()

#%%
# loc = [-500, 500]
# ix = (fimo['motif_loc'] > loc[0]) & (fimo['motif_loc'] < loc[1])
# fimo_tss = fimo.loc[ix]
# # n_motifs_genes = fimo_tss['gene'].value_counts()
# # n_motifs_genes
# # list(n_motifs_genes.index)
# fimo_tss.loc[fimo_tss['gene'] == 'atoh1a', :]

# import plotly.graph_objects as go
# # Treemap of GRN hubs
# width, aspect = 850, .75
# fig = go.Figure(go.Treemap(
#     labels = pred_grn_hub.index,
#     values = pred_grn_hub.values,
#     parents  = ['DELAY GRN — Hubs (Out Degree)'] * pred_grn_hub.size,
#     textinfo = 'label+value',
#     marker_colorscale = 'Greys'))
# # fig.write_image('figures/figure-1/grn-hubs.pdf', width = width, height = width * aspect)
# fig.show()

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
