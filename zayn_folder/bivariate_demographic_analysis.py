"""
bivariate_demographic_analysis.py
==================================
Analisi bivariata delle variabili demografiche IACOFI 2023.
 
Tre assi principali di analisi:
  A) digital_skills_score  — chi ha competenze digitali più alte?
  B) income_label          — come si distribuisce il reddito nei gruppi?
  C) work_status           — chi è più vulnerabile lavorativamente?
 
Per ogni asse vengono prodotti grafici di tipo diverso (boxplot, barplot
raggruppati, heatmap) sia salvati su disco che mostrati inline (notebook).
 
USO:
    from bivariate_demographic_analysis import bivariate_demographic_analysis
    bivariate_demographic_analysis(df)          # salva + mostra inline
    bivariate_demographic_analysis(df, show=False)  # solo salva su disco
"""
 
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
 
warnings.filterwarnings('ignore')
 
# ── Costanti di stile ────────────────────────────────────────────────────────
PALETTE      = 'viridis'
PAL_CAT      = sns.color_palette('viridis', 8)
PAL_3        = sns.color_palette('viridis', 3)
FIG_W, FIG_H = 11, 6
DPI          = 150
 
AGE_ORDER    = ['18-19','20-29','30-39','40-49','50-59','60-69','70-79']
EDU_ORDER    = ['No Education','Primary','Middle School','High School','University']
REGION_ORDER = ['North-West','North-East','Center','South','Islands']
URBAN_ORDER  = ['<3k','3k-15k','15k-100k','100k-1M','>1M']
INCOME_ORDER = ['<=1750\u20ac','1751-2900\u20ac','>2900\u20ac','Unknown']
WORK_ORDER   = ['Active','Inactive','Vulnerable']
LIVING_ORDER = ['Alone','With_Partner','With_Children','Other']
 
 
def _save_and_show(fig, path: str, show: bool):
    """Salva sempre su disco; mostra inline solo se show=True."""
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    if show:
        plt.show()
    plt.close(fig)
 
 
def _filter_order(order, series):
    """Restituisce solo i valori presenti nella serie, nell'ordine dato."""
    present = set(series.dropna().unique())
    return [v for v in order if v in present]
 
 
def _add_sample_sizes(ax, data, x_col, hue_col=None):
    """Aggiunge N sotto ogni gruppo sull'asse X (uso interno)."""
    pass  # placeholder — usato direttamente nei plot dove serve
 
 
# ── SEZIONE A — digital_skills_score ────────────────────────────────────────
 
def _plot_A1_boxplot_age(df, out_dir, show):
    """Boxplot digital_skills_score per fascia d'età."""
    order = _filter_order(AGE_ORDER, df['age_group'])
    d = df.dropna(subset=['age_group', 'digital_skills_score'])
 
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    palette = sns.color_palette(PALETTE, len(order))
    sns.boxplot(data=d, x='age_group', y='digital_skills_score',
                order=order, palette=palette, ax=ax,
                linewidth=1.2, flierprops=dict(marker='o', markersize=2.5, alpha=0.3))
 
    # Annotazione mediana sopra ogni box
    for i, grp in enumerate(order):
        med = d.loc[d['age_group'] == grp, 'digital_skills_score'].median()
        n   = d.loc[d['age_group'] == grp].shape[0]
        ax.text(i, med + 0.6, f'{med:.0f}', ha='center', va='bottom',
                fontsize=8.5, fontweight='bold', color='crimson')
        ax.text(i, ax.get_ylim()[0] - 0.8, f'n={n:,}', ha='center',
                va='top', fontsize=7.5, color='gray')
 
    ax.set_title('Digital Skills Score per Fascia d\'Età\n'
                 'Le competenze digitali diminuiscono con l\'età?',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Fascia d\'età', fontsize=11)
    ax.set_ylabel('Digital Skills Score (0–28)', fontsize=11)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save_and_show(fig, os.path.join(out_dir, 'A1_digital_by_age.png'), show)
 
 
def _plot_A2_boxplot_edu(df, out_dir, show):
    """Boxplot digital_skills_score per livello di istruzione."""
    order = _filter_order(EDU_ORDER, df['edu_level_grouped'])
    d = df.dropna(subset=['edu_level_grouped', 'digital_skills_score'])
 
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    palette = sns.color_palette(PALETTE, len(order))
    sns.boxplot(data=d, x='edu_level_grouped', y='digital_skills_score',
                order=order, palette=palette, ax=ax, linewidth=1.2,
                flierprops=dict(marker='o', markersize=2.5, alpha=0.3))
 
    for i, grp in enumerate(order):
        med = d.loc[d['edu_level_grouped'] == grp, 'digital_skills_score'].median()
        n   = d.loc[d['edu_level_grouped'] == grp].shape[0]
        ax.text(i, med + 0.5, f'{med:.0f}', ha='center', va='bottom',
                fontsize=8.5, fontweight='bold', color='crimson')
        ax.text(i, ax.get_ylim()[0] - 0.8, f'n={n:,}', ha='center',
                va='top', fontsize=7.5, color='gray')
 
    ax.set_title('Digital Skills Score per Livello di Istruzione\n'
                 'Più istruzione → più competenze digitali?',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Livello di istruzione', fontsize=11)
    ax.set_ylabel('Digital Skills Score (0–28)', fontsize=11)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save_and_show(fig, os.path.join(out_dir, 'A2_digital_by_edu.png'), show)
 
 
def _plot_A3_boxplot_region(df, out_dir, show):
    """Boxplot digital_skills_score per macro-regione + gender split."""
    order  = _filter_order(REGION_ORDER, df['macro_region_label'])
    d = df.dropna(subset=['macro_region_label', 'digital_skills_score', 'gender'])
 
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    sns.boxplot(data=d, x='macro_region_label', y='digital_skills_score',
                hue='gender', order=order,
                palette={'Man': PAL_CAT[1], 'Woman': PAL_CAT[5]},
                ax=ax, linewidth=1.1,
                flierprops=dict(marker='o', markersize=2, alpha=0.25))
 
    ax.set_title('Digital Skills Score per Regione e Genere\n'
                 'Divario digitale geografico e di genere',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Macro-regione', fontsize=11)
    ax.set_ylabel('Digital Skills Score (0–28)', fontsize=11)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(title='Genere', fontsize=9)
    fig.tight_layout()
    _save_and_show(fig, os.path.join(out_dir, 'A3_digital_by_region_gender.png'), show)
 
 
def _plot_A4_boxplot_work(df, out_dir, show):
    """Boxplot digital_skills_score per work_status."""
    order = _filter_order(WORK_ORDER, df['work_status'])
    d = df.dropna(subset=['work_status', 'digital_skills_score'])
 
    fig, ax = plt.subplots(figsize=(8, FIG_H))
    palette = {'Active': PAL_CAT[2], 'Inactive': PAL_CAT[4], 'Vulnerable': PAL_CAT[6]}
    sns.boxplot(data=d, x='work_status', y='digital_skills_score',
                order=order, palette=palette, ax=ax, linewidth=1.2,
                flierprops=dict(marker='o', markersize=3, alpha=0.3))
 
    # Test ANOVA
    groups = [d.loc[d['work_status'] == g, 'digital_skills_score'].dropna() for g in order]
    if len(groups) >= 2:
        f_stat, p_val = stats.f_oneway(*groups)
        ptext = f'ANOVA: F={f_stat:.2f}, p{"<0.001" if p_val < 0.001 else f"={p_val:.3f}"}'
        ax.text(0.98, 0.97, ptext, transform=ax.transAxes,
                ha='right', va='top', fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff9c4', alpha=0.8))
 
    for i, grp in enumerate(order):
        med = d.loc[d['work_status'] == grp, 'digital_skills_score'].median()
        n   = d.loc[d['work_status'] == grp].shape[0]
        ax.text(i, med + 0.5, f'{med:.0f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='crimson')
        ax.text(i, ax.get_ylim()[0] - 0.8, f'n={n:,}', ha='center',
                va='top', fontsize=7.5, color='gray')
 
    ax.set_title('Digital Skills Score per Situazione Lavorativa\n'
                 'I lavoratori attivi hanno più competenze digitali?',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Situazione lavorativa', fontsize=11)
    ax.set_ylabel('Digital Skills Score (0–28)', fontsize=11)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save_and_show(fig, os.path.join(out_dir, 'A4_digital_by_work.png'), show)
 
 
# ── SEZIONE B — income_label ─────────────────────────────────────────────────
 
def _plot_B1_stacked_edu(df, out_dir, show):
    """Barplot 100% stackato: distribuzione income per livello di istruzione."""
    order_x   = _filter_order(EDU_ORDER, df['edu_level_grouped'])
    order_hue = _filter_order(INCOME_ORDER, df['income_label'])
    d = df.dropna(subset=['edu_level_grouped', 'income_label'])
 
    ct = pd.crosstab(d['edu_level_grouped'], d['income_label'], normalize='index') * 100
    ct = ct.reindex(index=order_x, columns=order_hue, fill_value=0)
 
    palette = sns.color_palette(PALETTE, len(order_hue))
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    bottom = np.zeros(len(order_x))
 
    for j, inc in enumerate(order_hue):
        vals = ct[inc].values
        bars = ax.bar(range(len(order_x)), vals, bottom=bottom,
                      color=palette[j], label=inc, edgecolor='white', linewidth=0.5)
        # etichette solo se la fetta è abbastanza grande
        for k, (v, b) in enumerate(zip(vals, bottom)):
            if v > 6:
                ax.text(k, b + v / 2, f'{v:.0f}%', ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold')
        bottom += vals
 
    ax.set_xticks(range(len(order_x)))
    ax.set_xticklabels(order_x, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Composizione (%)', fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_title('Distribuzione del Reddito per Livello di Istruzione\n'
                 'Istruzione più alta → reddito più alto?',
                 fontsize=13, fontweight='bold')
    ax.legend(title='Fascia reddito', bbox_to_anchor=(1.01, 1),
              loc='upper left', fontsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save_and_show(fig, os.path.join(out_dir, 'B1_income_by_edu_stacked.png'), show)
 
 
def _plot_B2_grouped_region(df, out_dir, show):
    """Barplot raggruppato: % fascia >2900€ per regione e genere."""
    d = df[df['income_label'] != 'Unknown'].dropna(
        subset=['macro_region_label', 'income_label', 'gender'])
    order_x = _filter_order(REGION_ORDER, d['macro_region_label'])
 
    # calcola % >2900€ per regione × genere
    records = []
    for reg in order_x:
        for gen in ['Man', 'Woman']:
            sub = d[(d['macro_region_label'] == reg) & (d['gender'] == gen)]
            if len(sub) > 0:
                pct = (sub['income_label'] == '>2900\u20ac').mean() * 100
                records.append({'region': reg, 'gender': gen, 'pct_high': pct, 'n': len(sub)})
 
    rdf = pd.DataFrame(records)
 
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x       = np.arange(len(order_x))
    width   = 0.35
    colors  = {'Man': PAL_CAT[1], 'Woman': PAL_CAT[5]}
 
    for i, gen in enumerate(['Man', 'Woman']):
        vals = [rdf.loc[(rdf['region'] == r) & (rdf['gender'] == gen), 'pct_high'].values[0]
                if len(rdf.loc[(rdf['region'] == r) & (rdf['gender'] == gen)]) > 0 else 0
                for r in order_x]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=gen,
                      color=colors[gen], edgecolor='white', linewidth=0.6)
        for bar, v in zip(bars, vals):
            if v > 1.5:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f'{v:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
 
    ax.set_xticks(x)
    ax.set_xticklabels(order_x, rotation=15, ha='right', fontsize=10)
    ax.set_ylabel('% rispondenti con reddito >2.900€/mese', fontsize=10)
    ax.set_title('Reddito Alto (>2.900€) per Regione e Genere\n'
                 'Gap di reddito geografico e di genere',
                 fontsize=13, fontweight='bold')
    ax.legend(title='Genere', fontsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save_and_show(fig, os.path.join(out_dir, 'B2_income_high_by_region_gender.png'), show)
 
 
def _plot_B3_heatmap_income_age(df, out_dir, show):
    """Heatmap: distribuzione income per fascia d'età (% per riga)."""
    d = df.dropna(subset=['age_group', 'income_label'])
    order_x = _filter_order(AGE_ORDER, d['age_group'])
    order_y = _filter_order(INCOME_ORDER, d['income_label'])
 
    ct = pd.crosstab(d['age_group'], d['income_label'], normalize='index') * 100
    ct = ct.reindex(index=order_x, columns=order_y, fill_value=0)
 
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(ct, annot=True, fmt='.1f', cmap='YlGn',
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': '% per fascia d\'età'},
                ax=ax, annot_kws={'size': 9})
 
    ax.set_title('Distribuzione del Reddito per Fascia d\'Età (%)\n'
                 'Come cambia il reddito nel ciclo di vita?',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Fascia di reddito', fontsize=11)
    ax.set_ylabel('Fascia d\'età', fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    _save_and_show(fig, os.path.join(out_dir, 'B3_income_by_age_heatmap.png'), show)
 
 
def _plot_B4_heatmap_income_work(df, out_dir, show):
    """Heatmap: distribuzione income per work_status (% per riga)."""
    d = df.dropna(subset=['work_status', 'income_label'])
    order_x = _filter_order(WORK_ORDER, d['work_status'])
    order_y = _filter_order(INCOME_ORDER, d['income_label'])
 
    ct = pd.crosstab(d['work_status'], d['income_label'], normalize='index') * 100
    ct = ct.reindex(index=order_x, columns=order_y, fill_value=0)
 
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(ct, annot=True, fmt='.1f', cmap='YlOrRd',
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': '% per gruppo lavorativo'},
                ax=ax, annot_kws={'size': 10})
 
    ax.set_title('Distribuzione del Reddito per Situazione Lavorativa (%)\n'
                 'I Vulnerable hanno redditi più bassi?',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Fascia di reddito', fontsize=11)
    ax.set_ylabel('Situazione lavorativa', fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    _save_and_show(fig, os.path.join(out_dir, 'B4_income_by_work_heatmap.png'), show)
 
 
# ── SEZIONE C — work_status ──────────────────────────────────────────────────
 
def _plot_C1_stacked_age(df, out_dir, show):
    """Barplot 100% stackato: composizione work_status per fascia d'età."""
    order_x   = _filter_order(AGE_ORDER, df['age_group'])
    order_hue = _filter_order(WORK_ORDER, df['work_status'])
    d = df.dropna(subset=['age_group', 'work_status'])
 
    ct = pd.crosstab(d['age_group'], d['work_status'], normalize='index') * 100
    ct = ct.reindex(index=order_x, columns=order_hue, fill_value=0)
 
    palette = {'Active': PAL_CAT[1], 'Inactive': PAL_CAT[4], 'Vulnerable': PAL_CAT[6]}
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    bottom = np.zeros(len(order_x))
 
    for ws in order_hue:
        vals = ct[ws].values
        bars = ax.bar(range(len(order_x)), vals, bottom=bottom,
                      color=palette[ws], label=ws, edgecolor='white', linewidth=0.5)
        for k, (v, b) in enumerate(zip(vals, bottom)):
            if v > 5:
                ax.text(k, b + v / 2, f'{v:.0f}%', ha='center', va='center',
                        fontsize=8.5, color='white', fontweight='bold')
        bottom += vals
 
    ns = d['age_group'].value_counts()
    for i, grp in enumerate(order_x):
        n = ns.get(grp, 0)
        ax.text(i, 101, f'n={n:,}', ha='center', va='bottom', fontsize=7.5, color='gray')
 
    ax.set_xticks(range(len(order_x)))
    ax.set_xticklabels(order_x, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Composizione (%)', fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title('Composizione Work Status per Fascia d\'Età\n'
                 'Come cambia la partecipazione al lavoro nel ciclo di vita?',
                 fontsize=13, fontweight='bold')
    ax.legend(title='Work Status', bbox_to_anchor=(1.01, 1),
              loc='upper left', fontsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save_and_show(fig, os.path.join(out_dir, 'C1_work_by_age_stacked.png'), show)
 
 
def _plot_C2_grouped_gender_region(df, out_dir, show):
    """Barplot raggruppato: % Vulnerable per regione × genere."""
    d = df.dropna(subset=['macro_region_label', 'work_status', 'gender'])
    order_x = _filter_order(REGION_ORDER, d['macro_region_label'])
 
    records = []
    for reg in order_x:
        for gen in ['Man', 'Woman']:
            sub = d[(d['macro_region_label'] == reg) & (d['gender'] == gen)]
            if len(sub) > 0:
                pct = (sub['work_status'] == 'Vulnerable').mean() * 100
                records.append({'region': reg, 'gender': gen, 'pct_vuln': pct, 'n': len(sub)})
 
    rdf = pd.DataFrame(records)
    x   = np.arange(len(order_x))
    width = 0.35
    colors = {'Man': PAL_CAT[1], 'Woman': PAL_CAT[5]}
 
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for i, gen in enumerate(['Man', 'Woman']):
        vals = [rdf.loc[(rdf['region'] == r) & (rdf['gender'] == gen), 'pct_vuln'].values[0]
                if len(rdf.loc[(rdf['region'] == r) & (rdf['gender'] == gen)]) > 0 else 0
                for r in order_x]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=gen,
                      color=colors[gen], edgecolor='white', linewidth=0.6)
        for bar, v in zip(bars, vals):
            if v > 0.5:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f'{v:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
 
    ax.set_xticks(x)
    ax.set_xticklabels(order_x, rotation=15, ha='right', fontsize=10)
    ax.set_ylabel('% rispondenti Vulnerable', fontsize=10)
    ax.set_title('% Lavoratori Vulnerabili per Regione e Genere\n'
                 'Disoccupazione e invalidità: dove è più alta?',
                 fontsize=13, fontweight='bold')
    ax.legend(title='Genere', fontsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save_and_show(fig, os.path.join(out_dir, 'C2_vulnerable_by_region_gender.png'), show)
 
 
def _plot_C3_heatmap_work_edu(df, out_dir, show):
    """Heatmap: composizione work_status per livello di istruzione (%)."""
    d = df.dropna(subset=['edu_level_grouped', 'work_status'])
    order_x = _filter_order(EDU_ORDER, d['edu_level_grouped'])
    order_y = _filter_order(WORK_ORDER, d['work_status'])
 
    ct = pd.crosstab(d['edu_level_grouped'], d['work_status'], normalize='index') * 100
    ct = ct.reindex(index=order_x, columns=order_y, fill_value=0)
 
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(ct, annot=True, fmt='.1f', cmap='Blues',
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': '% per livello istruzione'},
                ax=ax, annot_kws={'size': 10})
 
    ax.set_title('Composizione Work Status per Livello di Istruzione (%)\n'
                 'Istruzione protegge dalla vulnerabilità lavorativa?',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Situazione lavorativa', fontsize=11)
    ax.set_ylabel('Livello di istruzione', fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    _save_and_show(fig, os.path.join(out_dir, 'C3_work_by_edu_heatmap.png'), show)
 
 
def _plot_C4_living_work(df, out_dir, show):
    """Barplot raggruppato: composizione work_status per living_status."""
    d = df.dropna(subset=['living_status', 'work_status'])
    order_x   = _filter_order(LIVING_ORDER, d['living_status'])
    order_hue = _filter_order(WORK_ORDER, d['work_status'])
 
    ct = pd.crosstab(d['living_status'], d['work_status'], normalize='index') * 100
    ct = ct.reindex(index=order_x, columns=order_hue, fill_value=0)
 
    x     = np.arange(len(order_x))
    width = 0.25
    palette = {'Active': PAL_CAT[1], 'Inactive': PAL_CAT[4], 'Vulnerable': PAL_CAT[6]}
 
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for i, ws in enumerate(order_hue):
        vals   = ct[ws].values
        offset = (i - (len(order_hue) - 1) / 2) * width
        bars   = ax.bar(x + offset, vals, width, label=ws,
                        color=palette[ws], edgecolor='white', linewidth=0.6)
        for bar, v in zip(bars, vals):
            if v > 4:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{v:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
 
    ax.set_xticks(x)
    ax.set_xticklabels(order_x, fontsize=10)
    ax.set_ylabel('% all\'interno del gruppo (%)', fontsize=11)
    ax.set_title('Work Status per Situazione Abitativa\n'
                 'Chi vive solo è più vulnerabile economicamente?',
                 fontsize=13, fontweight='bold')
    ax.legend(title='Work Status', fontsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save_and_show(fig, os.path.join(out_dir, 'C4_work_by_living.png'), show)
 
 
# ── SEZIONE D — Correlazione generale ───────────────────────────────────────
 
def _plot_D1_correlation_heatmap(df, out_dir, show):
    """
    Heatmap di correlazione tra variabili numeriche e ordinali codificate.
    Permette di vedere pattern generali tra tutte le dimensioni demografiche.
    """
    # Codifiche ordinali per le categoriche
    encode = {
        'gender':            {'Man': 1, 'Woman': 0},
        'age_group':         dict(zip(AGE_ORDER, range(len(AGE_ORDER)))),
        'edu_level_grouped': dict(zip(EDU_ORDER, range(len(EDU_ORDER)))),
        'work_status':       {'Active': 2, 'Inactive': 1, 'Vulnerable': 0},
        'income_label':      {'<=1750\u20ac': 0, '1751-2900\u20ac': 1, '>2900\u20ac': 2, 'Unknown': np.nan},
        'urban_area_label':  dict(zip(URBAN_ORDER, range(len(URBAN_ORDER)))),
        'internet_access_label': {'Yes': 1, 'No': 0},
        'is_italian':        {'Italian': 1, 'Other': 0},
        'living_status':     dict(zip(LIVING_ORDER, range(len(LIVING_ORDER)))),
    }
 
    num_cols    = ['digital_skills_score', 'household_size']
    corr_labels = {
        'digital_skills_score': 'Digital Skills',
        'household_size':       'Household Size',
        'gender':               'Gender (M=1)',
        'age_group':            'Age Group',
        'edu_level_grouped':    'Education',
        'work_status':          'Work Status',
        'income_label':         'Income',
        'urban_area_label':     'Urban Level',
        'internet_access_label':'Internet Access',
        'is_italian':           'Italian',
        'living_status':        'Living Status',
    }
 
    d_enc = df[num_cols].copy()
    for col, mapping in encode.items():
        if col in df.columns:
            d_enc[col] = df[col].map(mapping)
 
    d_enc = d_enc.rename(columns=corr_labels)
    corr  = d_enc.corr(method='spearman')
 
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # mostra solo triangolo inferiore
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, vmin=-1, vmax=1,
                mask=mask,
                linewidths=0.4, linecolor='white',
                cbar_kws={'label': 'Correlazione di Spearman'},
                ax=ax, annot_kws={'size': 8.5},
                square=True)
 
    ax.set_title('Matrice di Correlazione (Spearman) — Variabili Demografiche\n'
                 'Correlazioni tra tutte le dimensioni demografiche/background',
                 fontsize=13, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    fig.tight_layout()
    _save_and_show(fig, os.path.join(out_dir, 'D1_correlation_matrix.png'), show)
 
 
# ── FUNZIONE PRINCIPALE ──────────────────────────────────────────────────────
 
def bivariate_demographic_analysis(df: pd.DataFrame,
                                    output_dir: str = 'plot/bivariate',
                                    show: bool = True) -> None:
    """
    Esegue l'analisi bivariata completa sulle variabili demografiche IACOFI 2023.
 
    Sezioni prodotte:
      A) digital_skills_score come variabile dipendente (4 grafici)
         A1 — Boxplot per fascia d'età
         A2 — Boxplot per livello di istruzione
         A3 — Boxplot per regione × genere
         A4 — Boxplot per work_status (+ test ANOVA)
 
      B) income_label come variabile dipendente (4 grafici)
         B1 — Barplot stackato 100% per istruzione
         B2 — Barplot raggruppato % reddito alto per regione × genere
         B3 — Heatmap income × fascia d'età
         B4 — Heatmap income × work_status
 
      C) work_status come variabile dipendente (4 grafici)
         C1 — Barplot stackato 100% per fascia d'età
         C2 — Barplot % Vulnerable per regione × genere
         C3 — Heatmap work_status × istruzione
         C4 — Barplot raggruppato work_status × living_status
 
      D) Correlazione generale (1 grafico)
         D1 — Matrice di correlazione Spearman tra tutte le variabili
 
    Parameters
    ----------
    df         : DataFrame dopo engineer_demographic_features()
    output_dir : cartella di output (default: 'plot/bivariate')
    show       : se True mostra i grafici inline (utile in Jupyter)
    """
    os.makedirs(output_dir, exist_ok=True)
 
    print('\n' + '='*70)
    print('📊  ANALISI BIVARIATA — VARIABILI DEMOGRAFICHE (IACOFI 2023)')
    print('='*70)
 
    sections = [
        ('A — Digital Skills Score', [
            ('A1', 'Boxplot digital skills × età',              _plot_A1_boxplot_age),
            ('A2', 'Boxplot digital skills × istruzione',       _plot_A2_boxplot_edu),
            ('A3', 'Boxplot digital skills × regione/genere',   _plot_A3_boxplot_region),
            ('A4', 'Boxplot digital skills × work_status',      _plot_A4_boxplot_work),
        ]),
        ('B — Income Label', [
            ('B1', 'Stacked bar income × istruzione',           _plot_B1_stacked_edu),
            ('B2', 'Grouped bar % reddito alto × regione/genere', _plot_B2_grouped_region),
            ('B3', 'Heatmap income × età',                      _plot_B3_heatmap_income_age),
            ('B4', 'Heatmap income × work_status',              _plot_B4_heatmap_income_work),
        ]),
        ('C — Work Status', [
            ('C1', 'Stacked bar work × età',                    _plot_C1_stacked_age),
            ('C2', 'Grouped bar % Vulnerable × regione/genere', _plot_C2_grouped_gender_region),
            ('C3', 'Heatmap work × istruzione',                 _plot_C3_heatmap_work_edu),
            ('C4', 'Grouped bar work × living_status',          _plot_C4_living_work),
        ]),
        ('D — Correlazione Generale', [
            ('D1', 'Matrice correlazione Spearman',              _plot_D1_correlation_heatmap),
        ]),
    ]
 
    total = sum(len(plots) for _, plots in sections)
    done  = 0
 
    for sec_name, plots in sections:
        print(f'\n  ── {sec_name} ──')
        for code, desc, fn in plots:
            print(f'     [{code}] {desc}...', end=' ', flush=True)
            try:
                fn(df, output_dir, show)
                print('✅')
            except Exception as e:
                print(f'⚠️  {e}')
            done += 1
 
    print(f'\n{"="*70}')
    print(f'✅  {done}/{total} grafici salvati in "{output_dir}/"')
    print(f'{"="*70}\n')