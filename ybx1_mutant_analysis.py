#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem, mannwhitneyu, chisquare
np.random.seed(1234)

# Hair cells: WT vs. ybx1 mutant
a, ms, lw, j, g1, g2, fs1, fs2, c, ylim = .33, 30, 1.25, .2, .25, .66, 11, 10, ['green', 'orange',], [0, 20]
xx = np.array([x + ((2 + g1) * i) for i in range(2) for x in [0, 1]])
xx_plt = ['WT_Regen', 'Mut_Regen', 'WT_Dev', 'Mut_Dev']
xticklabels = ['2 dpa', 'No Ablation']
df = pd.read_csv('ybx1_mutant.csv')
ix_wt = (df['Genotype'] == 'WT')
df.loc[ix_wt, 'Genotype_Combined'] = 'WT'
df.loc[~ix_wt, 'Genotype_Combined'] = 'Mut'
df['Category'] = df['Genotype_Combined'] + '_' + df['Condition']
df['Hair_Cells_Jitter'] = df['Hair_Cells'] + np.random.uniform(-j, j, df.shape[0])

# Strip plot with mean, SEM
fig, ax = plt.subplots(1, 1, figsize = (2.85, 3.25))
for i, cat in enumerate(xx_plt):
    df_cat = df.loc[(df['Category'] == cat)]
    xx_cat = xx[i] + np.random.uniform(-j, j, df_cat.shape[0])
    mu_cat = df_cat['Hair_Cells'].mean()
    sem_cat = sem(df_cat['Hair_Cells'])
    ax.scatter(xx_cat, df_cat['Hair_Cells_Jitter'], c = c[i % 2], alpha = a, s = ms, linewidth = 0)
    ax.plot([xx[i] - j, xx[i] + j], [mu_cat] * 2, zorder = 100, linewidth = lw, c = 'k')
    ax.plot([xx[i] - j/2, xx[i] + j/2], [mu_cat + sem_cat] * 2, zorder = 100, linewidth = lw, c = 'k')
    ax.plot([xx[i] - j/2, xx[i] + j/2], [mu_cat - sem_cat] * 2, zorder = 100, linewidth = lw, c = 'k')
    ax.plot([xx[i]] * 2, [mu_cat - sem_cat, mu_cat + sem_cat], zorder = 100, linewidth = lw, c = 'k')
    print(cat, mu_cat, 'n =', df_cat.shape[0])

ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_bounds(*ylim)
ax.set_ylim(ylim[0], ylim[1] + g2)
yticks = np.arange(ylim[0], ylim[1] + 1, 4)
ax.set_yticks(yticks, yticks, fontsize = fs2)
ax.set_ylabel('Hair Cells', fontsize = fs1)
ax.set_xlim(xx[0] - g2, xx[-1] + g2)
xticks = [xx[[i, i + 1]].mean() for i in range(0, len(xx), 2)]
ax.set_xticks(xticks, xticklabels, size = fs1)
ax.tick_params('x', size = 0)
# plt.savefig('../figures/figure-2/mut-hc-num.pdf', bbox_inches = 'tight', dpi = 600)
plt.show()

# Mann-Whitney U test
for i in range(0, len(xx_plt), 2):
    cat1, cat2 = xx_plt[i], xx_plt[i + 1]
    df_cat1 = df.loc[(df['Category'] == cat1)]
    df_cat2 = df.loc[(df['Category'] == cat2)]
    print(cat1, cat2, mannwhitneyu(
        df_cat1['Hair_Cells'], 
        df_cat2['Hair_Cells'], 
        alternative = 'greater'))

#%%
# Neuromasts with new HCs (%): 12 — 24 hpa
lw1, lw2, cs, a, g, fs2, xlim, ylim = 1.66, 1, 3, .2, 2, 10.5, [12, 24], [0, 100]
xx = np.arange(xlim[0], xlim[1] + 1, 4)
df = pd.read_csv('ybx1_het_x_atoh1a_dtom.csv')
df['Category'] = df['Genotype'] + '_' + df['Timepoint'].values.astype(str)
df['Regen_init'] = (df['HCs_new'] > 0).astype(float)
df['Regen_init_percent'] = (df['Regen_init'] * ylim[1])

# Chi-squared test of proportions
def chi2_test(df_exp, df_obs):
    n_regen_exp = df_exp['Regen_init'].sum()
    n_regen_obs = df_obs['Regen_init'].sum()
    f_exp = np.array([n_regen_exp, df_exp.shape[0] - n_regen_exp])
    f_obs = np.array([n_regen_obs, df_obs.shape[0] - n_regen_obs])
    f_exp = (f_exp / f_exp.sum()) * df_obs.shape[0]
    return chisquare(f_obs, f_exp)

# Plot of means, errors
fig, ax = plt.subplots(1, 1, figsize = (2.85, 2.85))
for i, t in enumerate(xx):
    cat_wt, cat_mut = f'WT_{t}', f'Het_{t}'
    df_wt = df.loc[(df['Category'] == cat_wt)]
    df_mut = df.loc[(df['Category'] == cat_mut)]
    mu_wt = df_wt['Regen_init_percent'].mean()
    mu_mut = df_mut['Regen_init_percent'].mean()
    sem_wt = sem(df_wt['Regen_init_percent'])
    sem_mut = sem(df_mut['Regen_init_percent'])
    ax.errorbar(t, mu_wt, sem_wt, c = c[0], linewidth = lw1, capsize = cs)
    ax.errorbar(t, mu_mut, sem_mut, c = c[1], linewidth = lw1, capsize = cs)
    print(cat_wt, mu_wt.round(2), 'n =', df_wt.shape[0], '\t', 
          cat_mut, mu_mut.round(2), 'n =', df_mut.shape[0])
    print(chi2_test(df_wt, df_mut))
    if i > 0:
        t_prev = xx[i - 1]
        cat_wt_prev = f'WT_{t_prev}'
        cat_mut_prev = f'Het_{t_prev}'
        df_wt_prev = df.loc[(df['Category'] == cat_wt_prev)]
        df_mut_prev = df.loc[(df['Category'] == cat_mut_prev)]
        mu_wt_prev = df_wt_prev['Regen_init_percent'].mean()
        mu_mut_prev = df_mut_prev['Regen_init_percent'].mean()
        div_wt = chi2_test(df_wt_prev, df_wt)
        div_mut = chi2_test(df_mut_prev, df_mut)
        a_wt = 1 if div_wt.pvalue < .05 else a
        a_mut = 1 if div_mut.pvalue < .05 else a
        ax.plot([t_prev, t], [mu_wt_prev, mu_wt], c = c[0], alpha = a_wt, linewidth = lw1, zorder = 0)
        ax.plot([t_prev, t], [mu_mut_prev, mu_mut], c = c[1], alpha = a_mut, linewidth = lw1, zorder = 0)

ax.axvline(xx[2], c = 'k', linewidth = lw2, linestyle = (0, (5, 5)), zorder = 0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_bounds(*ylim)
ax.set_ylim(*ylim)
yticks = np.arange(ylim[0], ylim[1] + 1, 20)
ax.set_yticks(yticks, yticks, fontsize = fs2)
ax.set_ylabel('NMs With New HCs (%)', fontsize = fs1)
ax.set_xlim(xlim[0] - g, xlim[1] + g)
ax.set_xticks(xx, xx, size = fs2)
ax.set_xlabel('Hours Post Ablation', fontsize = fs1)
# plt.savefig('../figures/figure-2/percent-regen.pdf', bbox_inches = 'tight', dpi = 600)
plt.show()

# %%
# New hair cells: 12 — 24 hpa
fig, ax = plt.subplots(1, 1, figsize = (2.85, 1.55))
for i, t in enumerate(xx):
    cat_wt, cat_mut = f'WT_{t}', f'Het_{t}'
    df_wt = df.loc[(df['Category'] == cat_wt), 'HCs_new']
    df_mut = df.loc[(df['Category'] == cat_mut), 'HCs_new']
    mu_wt, mu_mut, sem_wt, sem_mut = \
        df_wt.mean(), df_mut.mean(), sem(df_wt), sem(df_mut)
    ax.errorbar(t, mu_wt, sem_wt, c = c[0], linewidth = lw1, capsize = cs)
    ax.errorbar(t, mu_mut, sem_mut, c = c[1], linewidth = lw1, capsize = cs)
    print(cat_wt, mu_wt.round(2), '\t', cat_mut, mu_mut.round(2))
    print(mannwhitneyu(df_wt, df_mut, alternative = 'greater'))
    if i > 0:
        t_prev = xx[i - 1]
        cat_wt_prev = f'WT_{t_prev}'
        cat_mut_prev = f'Het_{t_prev}'
        df_wt_prev = df.loc[(df['Category'] == cat_wt_prev), 'HCs_new']
        df_mut_prev = df.loc[(df['Category'] == cat_mut_prev), 'HCs_new']
        mu_wt_prev, mu_mut_prev = df_wt_prev.mean(), df_mut_prev.mean()
        stat_wt = mannwhitneyu(df_wt, df_wt_prev, alternative = 'greater')
        stat_mut = mannwhitneyu(df_mut, df_mut_prev, alternative = 'greater')
        a_wt = 1 if stat_wt.pvalue < .05 else a
        a_mut = 1 if stat_mut.pvalue < .05 else a
        ax.plot([t_prev, t], [mu_wt_prev, mu_wt], c = c[0], alpha = a_wt, linewidth = lw1, zorder = 0)
        ax.plot([t_prev, t], [mu_mut_prev, mu_mut], c = c[1], alpha = a_mut, linewidth = lw1, zorder = 0)

ylim = [0, 3.33]
ax.axvline(xx[2], c = 'k', linewidth = lw2, linestyle = (0, (5, 5)), zorder = 0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(*ylim)
yticks = np.arange(ylim[0], ylim[1], 1, dtype = int)
ax.set_yticks(yticks, yticks, fontsize = fs2)
ax.set_ylabel('New Hair Cells', fontsize = fs1)
ax.set_xlim(xlim[0] - g, xlim[1] + g)
ax.set_xticks(xx, xx, size = fs2)
ax.set_xlabel('Hours Post Ablation', fontsize = fs1)
# plt.savefig('../figures/figure-2/new-hc-regen.pdf', bbox_inches = 'tight', dpi = 600)
plt.show()

# %%
