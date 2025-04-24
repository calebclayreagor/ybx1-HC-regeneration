#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import sem, norm, ttest_ind, fisher_exact
from statsmodels.stats.power import TTestIndPower
np.random.seed(1234)

# Hair cells: WT vs. ybx1 mutant
a, ms, lw, j, g = .33, 30, 1.25, .2, [.25, .66]
xx = np.array([x + ((3 + g[0]) * i) for i in range(2) for x in range(3)])
xx_plt = ['WT_Regen', 'Het_Regen', 'Mut_Regen', 'WT_Dev', 'Het_Dev', 'Mut_Dev']
df = pd.read_csv('ybx1_mutant.csv')
df['Category'] = df['Genotype'] + '_' + df['Condition']
df['Fish_unique'] = df['Fish'] + '_' + df['Date'].astype(str)
df['Hair_Cells_Jitter'] = df['Hair_Cells'] + np.random.uniform(-j, j, df.shape[0])
N = df.groupby('Category')['Fish_unique'].nunique()[xx_plt]
n = df.groupby('Category').size()[xx_plt]

# Strip plot with mean, SEM
fig, ax = plt.subplots(1, 1, figsize = (2.85, 3.55))
c = ['green', 'orange', 'purple']
for i, cat in enumerate(xx_plt):
    ix_cat = (df['Category'] == cat)
    df_cat = df.loc[ix_cat]
    xx_cat = xx[i] + np.random.uniform(-j, j, df_cat.shape[0])
    mu_cat = df_cat['Hair_Cells'].mean()
    sem_cat = sem(df_cat['Hair_Cells'])
    ax.scatter(xx_cat, df_cat['Hair_Cells_Jitter'], c = c[i % 3], alpha = a, s = ms, linewidth = 0)
    ax.plot([xx[i] - j, xx[i] + j], [mu_cat] * 2, zorder = 100, linewidth = lw, c = 'k')
    ax.plot([xx[i] - j/2, xx[i] + j/2], [mu_cat + sem_cat] * 2, zorder = 100, linewidth = lw, c = 'k')
    ax.plot([xx[i] - j/2, xx[i] + j/2], [mu_cat - sem_cat] * 2, zorder = 100, linewidth = lw, c = 'k')
    ax.plot([xx[i]] * 2, [mu_cat - sem_cat, mu_cat + sem_cat], zorder = 100, linewidth = lw, c = 'k')

fs1, fs2, ylim = 11, 10, [0, 24]
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_bounds(*ylim)
ax.set_ylim(ylim[0] - g[1], ylim[1])
yticks = np.arange(ylim[0], ylim[1] + 1, 4)
ax.set_yticks(yticks, yticks, fontsize = fs2)
ax.set_ylabel('Hair Cells', fontsize = fs1)
ax.set_xlim(xx[0] - g[1], xx[-1] + g[1])
xticks = [xx[i : (i + 3)].mean() for i in range(0, len(xx), 3)]
xticklabels = ['2 dpa', 'No Ablation']
ax.set_xticks(xticks, xticklabels, size = fs1)
ax.tick_params('x', size = 0)
# plt.savefig(os.path.join('..', 'figures', 'figure-2', 'revision', 'mut-hc-num-updated.pdf'), bbox_inches = 'tight', dpi = 600)
plt.show()

def cohens_d_and_ci(x1: np.array,
                    x2: np.array,
                    a: float = .05
                    ) -> tuple[float]:
      n1, n2 = x1.size, x2.size
      dof = (n1 + n2 - 2)
      z = norm.ppf(1 - a / 2)
      pooled_std = np.sqrt(((n1 - 1) * np.var(x1, ddof = 1) + \
                            (n2 - 1) * np.var(x2, ddof = 1)) / dof)
      d = (x1.mean() - x2.mean()) / pooled_std
      d_conf = z * np.sqrt((n1 + n2) / (n1 * n2) + (d ** 2) / (2 * dof))
      return d, (d - d_conf), (d + d_conf)

def ttest_and_power_one_sided(x1: np.array,
                              x2: np.array,
                              a: float = .05
                              ) -> list[str, float]:
    tt_power = TTestIndPower()
    tt_res = ttest_ind(x1, x2, alternative = 'greater')
    d, d_lower, d_upper = cohens_d_and_ci(x1, x2)
    pow_res = tt_power.solve_power(effect_size = abs(d),
                                   nobs1 = x1.size,
                                   alpha = a,
                                   ratio = x2.size / x1.size,
                                   alternative = 'larger')
    return ['\np =', tt_res.pvalue,
            '\nEffect size =', d,
            '\n95% CI =', f'[{d_lower}, {d_upper}]',
            '\nPower =', pow_res, '\n']

# T-test (one-sided) and power analysis
for i in range(0, len(xx_plt), 3):
    cat1, cat2, cat3 = xx_plt[i : (i + 3)]
    ix_cat1 = (df['Category'] == cat1)
    ix_cat2 = (df['Category'] == cat2)
    ix_cat3 = (df['Category'] == cat3)
    hc_cat1 = df.loc[ix_cat1, 'Hair_Cells'].values
    hc_cat2 = df.loc[ix_cat2, 'Hair_Cells'].values
    hc_cat3 = df.loc[ix_cat3, 'Hair_Cells'].values
    hc_cat1_mu, hc_cat2_mu, hc_cat3_mu = \
      hc_cat1.mean(), hc_cat2.mean(), hc_cat3.mean()
    print(f'{cat1} = {hc_cat1_mu.round(3)} (N = {N[cat1]}; n = {n[cat1]})',
          f'\n{cat2} = {hc_cat2_mu.round(3)} (N = {N[cat2]}; n = {n[cat2]})',
          *ttest_and_power_one_sided(hc_cat1, hc_cat2))
    print(f'{cat1} = {hc_cat1_mu.round(3)} (N = {N[cat1]}; n = {n[cat1]})',
          f'\n{cat3} = {hc_cat3_mu.round(3)} (N = {N[cat3]}; n = {n[cat3]})',
          *ttest_and_power_one_sided(hc_cat1, hc_cat3))
    print(f'{cat2} = {hc_cat2_mu.round(3)} (N = {N[cat2]}; n = {n[cat2]})',
          f'\n{cat3} = {hc_cat3_mu.round(3)} (N = {N[cat3]}; n = {n[cat3]})',
          *ttest_and_power_one_sided(hc_cat2, hc_cat3))

#%%
# Nonsensory cells: WT vs. ybx1 mutant
np.random.seed(1234)
df['Nonsensory_Cells_Jitter'] = df['Nonsensory_Cells'] + np.random.uniform(-j, j, df.shape[0])
fig, ax = plt.subplots(1, 1, figsize = (2.85, 3.55))
for i, cat in enumerate(xx_plt):
    ix_nan = np.isnan(df['Nonsensory_Cells'])
    ix_cat = (df['Category'] == cat) & ~ix_nan
    df_cat = df.loc[ix_cat]
    xx_cat = xx[i] + np.random.uniform(-j, j, df_cat.shape[0])
    mu_cat = df_cat['Nonsensory_Cells'].mean()
    sem_cat = sem(df_cat['Nonsensory_Cells'])
    ax.scatter(xx_cat, df_cat['Nonsensory_Cells_Jitter'], c = c[i % 3], alpha = a, s = ms, linewidth = 0)
    ax.plot([xx[i] - j, xx[i] + j], [mu_cat] * 2, zorder = 100, linewidth = lw, c = 'k')
    ax.plot([xx[i] - j/2, xx[i] + j/2], [mu_cat + sem_cat] * 2, zorder = 100, linewidth = lw, c = 'k')
    ax.plot([xx[i] - j/2, xx[i] + j/2], [mu_cat - sem_cat] * 2, zorder = 100, linewidth = lw, c = 'k')
    ax.plot([xx[i]] * 2, [mu_cat - sem_cat, mu_cat + sem_cat], zorder = 100, linewidth = lw, c = 'k')

ylim = [0, 70]
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_bounds(*ylim)
ax.set_ylim(ylim[0] - 2, ylim[1])
yticks = np.arange(ylim[0], ylim[1] + 1, 10)
ax.set_yticks(yticks, yticks, fontsize = fs2)
ax.set_ylabel('Nonsensory Cells', fontsize = fs1)
ax.set_xlim(xx[0] - g[1], xx[-1] + g[1])
ax.set_xticks(xticks, xticklabels, size = fs1)
ax.tick_params('x', size = 0)
# plt.savefig(os.path.join('..', 'figures', 'figure-2', 'revision', 'mut-nsc-num.pdf'), bbox_inches = 'tight', dpi = 600)
plt.show()

def ttest_and_power_two_sided(x1: np.array,
                              x2: np.array,
                              _2a: float = .1
                              ) -> list[str, float]:
    tt_power = TTestIndPower()
    tt_res = ttest_ind(x1, x2, alternative = 'two-sided')
    d, d_lower, d_upper = cohens_d_and_ci(x1, x2, _2a)
    pow_res = tt_power.solve_power(effect_size = abs(d),
                                   nobs1 = x1.size,
                                   alpha = _2a / 2,
                                   ratio = x2.size / x1.size,
                                   alternative = 'two-sided')
    return ['\np =', tt_res.pvalue,
            '\nEffect size =', d,
            '\n95% CI =', f'[{d_lower}, {d_upper}]',
            '\nPower =', pow_res, '\n']

# T-test (two-sided) and power analysis
for i in range(0, len(xx_plt), 3):
    cat1, cat2, cat3 = xx_plt[i : (i + 3)]
    ix_nan = np.isnan(df['Nonsensory_Cells'])
    ix_cat1 = (df['Category'] == cat1) & ~ix_nan
    ix_cat2 = (df['Category'] == cat2) & ~ix_nan
    ix_cat3 = (df['Category'] == cat3) & ~ix_nan
    nsc_cat1 = df.loc[ix_cat1, 'Nonsensory_Cells'].values
    nsc_cat2 = df.loc[ix_cat2, 'Nonsensory_Cells'].values
    nsc_cat3 = df.loc[ix_cat3, 'Nonsensory_Cells'].values
    nsc_cat1_mu, nsc_cat2_mu, nsc_cat3_mu = \
      nsc_cat1.mean(), nsc_cat2.mean(), nsc_cat3.mean()
    print(f'{cat1} = {nsc_cat1_mu.round(3)} (N = {N[cat1]}; n = {n[cat1]})',
          f'\n{cat2} = {nsc_cat2_mu.round(3)} (N = {N[cat2]}; n = {n[cat2]})',
          *ttest_and_power_two_sided(nsc_cat1, nsc_cat2))
    print(f'{cat1} = {nsc_cat1_mu.round(3)} (N = {N[cat1]}; n = {n[cat1]})',
          f'\n{cat3} = {nsc_cat3_mu.round(3)} (N = {N[cat3]}; n = {n[cat3]})',
          *ttest_and_power_two_sided(nsc_cat1, nsc_cat3))
    print(f'{cat2} = {nsc_cat2_mu.round(3)} (N = {N[cat2]}; n = {n[cat2]})',
          f'\n{cat3} = {nsc_cat3_mu.round(3)} (N = {N[cat3]}; n = {n[cat3]})',
          *ttest_and_power_two_sided(nsc_cat2, nsc_cat3))

#%%
# % Neuromasts with new HCs: 12 — 24 hpa
lw1, lw2, cs, a, xlim = 1.66, 1, 3, .2, [12, 24]
xx = np.arange(xlim[0], xlim[1] + 1, 4)
df = pd.read_csv('ybx1_het_x_atoh1a_dtom.csv')
df['Category'] = df['Genotype'] + '_' + df['Timepoint'].astype(str)
df['Fish_unique'] = df['Fish'] + '_' + df['Date'].astype(str)
df['Regen_init'] = (df['HCs_new'] > 0).astype(int)
df['Regen_init_percent'] = (df['Regen_init'] * 100).astype(float)
N = df.groupby('Category')['Fish_unique'].nunique()
n = df.groupby('Category').size()

def fisher_exact_test_one_sided(s1: np.array, 
                                s2: np.array
                                ) -> list[str, float]:
    s1_counts = np.bincount(s1, minlength = 2).reshape(1, -1)
    s2_counts = np.bincount(s2, minlength = 2).reshape(1, -1)
    table = np.concatenate((s1_counts, s2_counts), axis = 0)
    fisher_res = fisher_exact(table, alternative = 'less')
    return ['\np =', fisher_res.pvalue, '\n']

# Plot of means, errors
fig = plt.figure(figsize = (3.65, 3.25))
gs = gridspec.GridSpec(1, 2, width_ratios = [3, .5], wspace = .2)
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
for i, t in enumerate(xx):
    cat_wt, cat_het = f'WT_{t}.0', f'Het_{t}.0'
    ix_wt = (df['Category'] == cat_wt)
    ix_het = (df['Category'] == cat_het)
    df_wt = df.loc[ix_wt]
    df_het = df.loc[ix_het]
    per_wt = df_wt['Regen_init_percent']
    per_het = df_het['Regen_init_percent']
    mu_wt, mu_het, sem_wt, sem_het = \
      per_wt.mean(), per_het.mean(), sem(per_wt), sem(per_het)
    ax0.errorbar(t, mu_wt, sem_wt, c = c[0], linewidth = lw1, capsize = cs)
    ax0.errorbar(t, mu_het, sem_het, c = c[1], linewidth = lw1, capsize = cs)
    print(f'{cat_wt} = {mu_wt.round(2)}% (N = {N[cat_wt]}; n = {n[cat_wt]})',
          f'\n{cat_het} = {mu_het.round(2)}% (N = {N[cat_het]}; n = {n[cat_het]})',
          *fisher_exact_test_one_sided(df_wt['Regen_init'], df_het['Regen_init']))
    if i > 0:
        t_prev = xx[i - 1]
        cat_wt_prev, cat_het_prev = f'WT_{t_prev}.0', f'Het_{t_prev}.0'
        ix_wt_prev = (df['Category'] == cat_wt_prev)
        ix_het_prev = (df['Category'] == cat_het_prev)
        df_wt_prev = df.loc[ix_wt_prev]
        df_het_prev = df.loc[ix_het_prev]
        per_wt_prev = df_wt_prev['Regen_init_percent']
        per_het_prev = df_het_prev['Regen_init_percent']
        mu_wt_prev, mu_het_prev = per_wt_prev.mean(), per_het_prev.mean()
        fisher_res_wt = fisher_exact_test_one_sided(df_wt['Regen_init'], df_wt_prev['Regen_init'])
        fisher_res_het = fisher_exact_test_one_sided(df_het['Regen_init'], df_het_prev['Regen_init'])
        print(f'{cat_wt} = {mu_wt.round(2)}% (N = {N[cat_wt]}; n = {n[cat_wt]})',
              f'\n{cat_wt_prev} = {mu_wt_prev.round(2)}% (N = {N[cat_wt_prev]}; n = {n[cat_wt_prev]})', 
              *fisher_res_wt)
        print(f'{cat_het} = {mu_het.round(2)}% (N = {N[cat_het]}; n = {n[cat_het]})',
              f'\n{cat_het_prev} = {mu_het_prev.round(2)}% (N = {N[cat_het_prev]}; n = {n[cat_het_prev]})',
              *fisher_res_het)
        a_wt = 1 if fisher_res_wt[1] < .05 else a
        a_het = 1 if fisher_res_het[1] < .05 else a
        ax0.plot([t_prev, t], [mu_wt_prev, mu_wt], c = c[0], alpha = a_wt, linewidth = lw1, zorder = 0)
        ax0.plot([t_prev, t], [mu_het_prev, mu_het], c = c[1], alpha = a_het, linewidth = lw1, zorder = 0)

# Non-regenerating controls
cat_wt, cat_het = 'WT_nan', 'Het_nan'
ix_wt = (df['Category'] == cat_wt)
ix_het = (df['Category'] == cat_het)
df_wt = df.loc[ix_wt]
df_het = df.loc[ix_het]
per_wt = df_wt['Regen_init_percent']
per_het = df_het['Regen_init_percent']
mu_wt, mu_het, sem_wt, sem_het = \
      per_wt.mean(), per_het.mean(), sem(per_wt), sem(per_het)
ax1.plot([-j, j], [mu_wt] * 2, c = c[0], linewidth = lw1)
ax1.plot([-j/2, j/2], [mu_wt + sem_wt] * 2, c = c[0], linewidth = lw1)
ax1.plot([-j/2, j/2], [mu_wt - sem_wt] * 2, c = c[0], linewidth = lw1)
ax1.plot([0] * 2, [mu_wt - sem_wt, mu_wt + sem_wt], c = c[0], linewidth = lw1)
ax1.plot([1 - j, 1 + j], [mu_het] * 2, c = c[1], linewidth = lw1)
ax1.plot([1 - j/2, 1 + j/2], [mu_het + sem_het] * 2, c = c[1], linewidth = lw1)
ax1.plot([1 - j/2, 1 + j/2], [mu_het - sem_het] * 2, c = c[1], linewidth = lw1)
ax1.plot([1] * 2, [mu_het - sem_het, mu_het + sem_het], c = c[1], linewidth = lw1)
print(f'{cat_wt} = {mu_wt.round(2)}% (N = {N[cat_wt]}; n = {n[cat_wt]})',
      f'\n{cat_het} = {mu_het.round(2)}% (N = {N[cat_het]}; n = {n[cat_het]})',
      *fisher_exact_test_one_sided(df_wt['Regen_init'], df_het['Regen_init']))

g, fs2, ylim = 2, 10.5, [0, 100]
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.spines['left'].set_bounds(*ylim)
ax0.set_ylim(*ylim)
yticks = np.arange(ylim[0], ylim[1] + 1, 20)
ax0.set_yticks(yticks, yticks, fontsize = fs2)
ax0.set_ylabel('NMs With New HCs (%)', fontsize = fs1)
ax0.set_xlim(xlim[0] - g, xlim[1] + g)
ax0.set_xticks(xx, xx, size = fs2)
ax0.set_xlabel('Hours Post Ablation', fontsize = fs1)

ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylim(*ylim)
ax1.set_yticks([])
ax1.set_xlim(-.5, 1.5)
ax1.set_xticks([.5], ['NA'], fontsize = fs1)

# plt.savefig(os.path.join('..', 'figures', 'figure-2', 'revision', 'percent-regen-updated.pdf'), bbox_inches = 'tight', dpi = 600)
plt.show()

#%%
# New hair cells: 12 — 24 hpa
fig = plt.figure(figsize = (3.65, 3.25))
gs = gridspec.GridSpec(1, 2, width_ratios = [3, .5], wspace = .2)
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
for i, t in enumerate(xx):
    cat_wt, cat_het = f'WT_{t}.0', f'Het_{t}.0'
    ix_wt = (df['Category'] == cat_wt)
    ix_het = (df['Category'] == cat_het)
    hc_wt = df.loc[ix_wt, 'HCs_new']
    hc_het = df.loc[ix_het, 'HCs_new']
    mu_wt, mu_het, sem_wt, sem_het = \
        hc_wt.mean(), hc_het.mean(), sem(hc_wt), sem(hc_het)
    ax0.errorbar(t, mu_wt, sem_wt, c = c[0], linewidth = lw1, capsize = cs)
    ax0.errorbar(t, mu_het, sem_het, c = c[1], linewidth = lw1, capsize = cs)
    print(f'{cat_wt} = {mu_wt.round(3)} (N = {N[cat_wt]}; n = {n[cat_wt]})',
          f'\n{cat_het} = {mu_het.round(3)} (N = {N[cat_het]}; n = {n[cat_het]})',
          *ttest_and_power_one_sided(hc_wt, hc_het))
    if i > 0:
        t_prev = xx[i - 1]
        cat_wt_prev, cat_het_prev = f'WT_{t_prev}.0', f'Het_{t_prev}.0'
        ix_wt_prev = (df['Category'] == cat_wt_prev)
        ix_het_prev = (df['Category'] == cat_het_prev)
        hc_wt_prev = df.loc[ix_wt_prev, 'HCs_new']
        hc_het_prev = df.loc[ix_het_prev, 'HCs_new']
        mu_wt_prev, mu_het_prev = hc_wt_prev.mean(), hc_het_prev.mean()
        ttest_res_wt = ttest_and_power_one_sided(hc_wt, hc_wt_prev)
        ttest_res_het = ttest_and_power_one_sided(hc_het, hc_het_prev)
        print(f'{cat_wt} = {mu_wt.round(3)} (N = {N[cat_wt]}; n = {n[cat_wt]})',
              f'\n{cat_wt_prev} = {mu_wt_prev.round(3)} (N = {N[cat_wt_prev]}; n = {n[cat_wt_prev]})',
              *ttest_res_wt)
        print(f'{cat_het} = {mu_het.round(3)} (N = {N[cat_het]}; n = {n[cat_het]})',
              f'\n{cat_het_prev} = {mu_het_prev.round(3)} (N = {N[cat_het_prev]}; n = {n[cat_het_prev]})',
              *ttest_res_het)
        a_wt = 1 if ttest_res_wt[1] < .05 else a
        a_het = 1 if ttest_res_het[1] < .05 else a
        ax0.plot([t_prev, t], [mu_wt_prev, mu_wt], c = c[0], alpha = a_wt, linewidth = lw1, zorder = 0)
        ax0.plot([t_prev, t], [mu_het_prev, mu_het], c = c[1], alpha = a_het, linewidth = lw1, zorder = 0)

# Non-regenerating controls
cat_wt, cat_het = 'WT_nan', 'Het_nan'
ix_wt = (df['Category'] == cat_wt)
ix_het = (df['Category'] == cat_het)
hc_wt = df.loc[ix_wt, 'HCs_new']
hc_het = df.loc[ix_het, 'HCs_new']
mu_wt, mu_het, sem_wt, sem_het = \
      hc_wt.mean(), hc_het.mean(), sem(hc_wt), sem(hc_het)
ax1.plot([-j, j], [mu_wt] * 2, c = c[0], linewidth = lw1)
ax1.plot([-j/2, j/2], [mu_wt + sem_wt] * 2, c = c[0], linewidth = lw1)
ax1.plot([-j/2, j/2], [mu_wt - sem_wt] * 2, c = c[0], linewidth = lw1)
ax1.plot([0] * 2, [mu_wt - sem_wt, mu_wt + sem_wt], c = c[0], linewidth = lw1)
ax1.plot([1 - j, 1 + j], [mu_het] * 2, c = c[1], linewidth = lw1)
ax1.plot([1 - j/2, 1 + j/2], [mu_het + sem_het] * 2, c = c[1], linewidth = lw1)
ax1.plot([1 - j/2, 1 + j/2], [mu_het - sem_het] * 2, c = c[1], linewidth = lw1)
ax1.plot([1] * 2, [mu_het - sem_het, mu_het + sem_het], c = c[1], linewidth = lw1)
print(f'{cat_wt} = {mu_wt.round(3)} (N = {N[cat_wt]}; n = {n[cat_wt]})',
      f'\n{cat_het} = {mu_het.round(3)} (N = {N[cat_het]}; n = {n[cat_het]})',
      *ttest_and_power_one_sided(hc_wt, hc_het))

ylim = [0, 5]
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.set_ylim(*ylim)
yticks = np.arange(ylim[0], ylim[1] + .1, 1, dtype = int)
ax0.set_yticks(yticks, yticks, fontsize = fs2)
ax0.set_ylabel('New Hair Cells', fontsize = fs1)
ax0.set_xlim(xlim[0] - g, xlim[1] + g)
ax0.set_xticks(xx, xx, size = fs2)
ax0.set_xlabel('Hours Post Ablation', fontsize = fs1)

ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylim(*ylim)
ax1.set_yticks([])
ax1.set_xlim(-.5, 1.5)
ax1.set_xticks([.5], ['NA'], fontsize = fs1)

# plt.savefig(os.path.join('..', 'figures', 'figure-2', 'revision', 'new-hc-regen-updated.pdf'), bbox_inches = 'tight', dpi = 600)
plt.show()

#%%
# Ybx1 IF: WT vs. Hets (atoh1a+)
np.random.seed(8)
a, ms, lw, j, g = .33, 30, 1.25, .2, .66
df = pd.read_csv('ybx1_het_x_atoh1a_dtom_IF_results.csv')
df['Cell_avg'] = df['Cell_avg'] / 10000
df['Fish_unique'] = df['Fish'] + '_' + df['Date'].astype(str)
N = df.groupby('Genotype')['Fish_unique'].nunique()
n = df.groupby('Genotype').size()

# Strip plot with mean, SEM
fig, ax = plt.subplots(1, 1, figsize = (1.4, 3.55))
ix_wt = (df['Genotype'] == 'WT')
ix_het = (df['Genotype'] == 'Het')
if_wt = df.loc[ix_wt, 'Cell_avg']
if_het = df.loc[ix_het, 'Cell_avg']
xx_wt = np.random.uniform(-j, j, if_wt.size)
xx_het = 1 + np.random.uniform(-j, j, if_het.size)
mu_wt, mu_het, sem_wt, sem_het = \
      if_wt.mean(), if_het.mean(), sem(if_wt), sem(if_het)
ax.scatter(xx_wt, if_wt, c = c[0], alpha = a, s = ms, linewidth = 0)
ax.plot([-j, j], [mu_wt] * 2, zorder = 100, c = 'k', linewidth = lw)
ax.plot([-j/2, j/2], [mu_wt + sem_wt] * 2, zorder = 100, c = 'k', linewidth = lw)
ax.plot([-j/2, j/2], [mu_wt - sem_wt] * 2, zorder = 100, c = 'k', linewidth = lw)
ax.plot([0] * 2, [mu_wt - sem_wt, mu_wt + sem_wt], zorder = 100, c = 'k', linewidth = lw)
ax.scatter(xx_het, if_het, c = c[1], alpha = a, s = ms, linewidth = 0)
ax.plot([1 - j, 1 + j], [mu_het] * 2, zorder = 100, c = 'k', linewidth = lw)
ax.plot([1 - j/2, 1 + j/2], [mu_het + sem_het] * 2, zorder = 100, c = 'k', linewidth = lw)
ax.plot([1 - j/2, 1 + j/2], [mu_het - sem_het] * 2, zorder = 100, c = 'k', linewidth = lw)
ax.plot([1] * 2, [mu_het - sem_het, mu_het + sem_het], zorder = 100, c = 'k', linewidth = lw)

fs, ylim = 10, [0, 8]
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_bounds(*ylim)
ax.set_ylim(*ylim)
yticks = np.arange(ylim[0], ylim[1] + 1, 1)
ax.set_yticks(yticks, yticks, fontsize = fs)
ax.set_ylabel('Ybx1 Immunofluorescence (AU)', fontsize = fs)
ax.set_xlim(-g, 1 + g)
ax.set_xticks([])
# plt.savefig(os.path.join('..', 'figures', 'figure-2', 'revision', 'ybx1-IF-atoh1a.pdf'), bbox_inches = 'tight', dpi = 600)
plt.show()

print(f'WT = {mu_wt.round(3)} (N = {N['WT']}; n = {n['WT']})',
      f'\nHet = {mu_het.round(3)} (N = {N['Het']}; n = {n['Het']})',
      *ttest_and_power_one_sided(if_wt, if_het))

#%%
