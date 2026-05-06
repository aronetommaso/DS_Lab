"""
generate_demographic_report.py
==============================
Genera un PDF di report EDA per le 12 variabili demografiche rimappate
da engineer_demographic_features() — stile Master_Report_Variabili.

USO:
    1. Incolla questo file nella stessa cartella del tuo pipeline.py
    2. Assicurati che il file_path punti al tuo CSV
    3. Lancia: python generate_demographic_report.py
       oppure importa generate_demographic_report(df) nel tuo notebook

DIPENDENZE:
    pip install reportlab matplotlib seaborn pandas numpy
"""

import io
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # backend non-interattivo: funziona anche senza display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
warnings.filterwarnings('ignore')

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, HRFlowable, PageBreak, KeepTogether
)

# ── Palette e stile coerenti con il progetto ────────────────────────────────
PALETTE      = 'viridis'
BG_DARK      = colors.HexColor('#1a1a2e')   # intestazione scura
ACCENT       = colors.HexColor('#16213e')
BAR_COLOR_1  = colors.HexColor('#2d6a4f')
TEXT_LIGHT   = colors.white
TEXT_DARK    = colors.HexColor('#1a1a2e')
GRID_ALPHA   = 0.35

# ── Metadati variabili ───────────────────────────────────────────────────────
VAR_META = {
    # col_name : (etichetta leggibile, domanda originale survey, tipo)
    'gender': (
        'Gender',
        'QD1 — Sesso del rispondente',
        'categorical'
    ),
    'age_group': (
        'Age Group',
        'QD7/QD7A — Fascia d\'età (età esatta → banda oppure banda diretta)',
        'categorical'
    ),
    'macro_region_label': (
        'Macro Region',
        'QD2 — Macro-regione di residenza (1=NW, 2=NE, 3=Centro, 4=Sud, 5=Isole)',
        'categorical'
    ),
    'urban_area_label': (
        'Urban Area',
        'QD3 — Livello di urbanizzazione del comune di residenza',
        'categorical'
    ),
    'living_status': (
        'Living Status',
        'QD5 — Con chi vivi? → Alone / With_Partner / With_Children / Other',
        'categorical'
    ),
    'household_size': (
        'Household Size',
        'QD5_AD + QD5_CH — Numero componenti del nucleo familiare (rispondente incluso)',
        'numeric'
    ),
    'digital_skills_score': (
        'Digital Skills Score',
        'QD6 (1-7) — Indice aggregato di competenze digitali (somma frequenze, range 0-28)',
        'numeric'
    ),
    'edu_level_grouped': (
        'Education Level',
        'QD9 — Titolo di studio (raggruppato in 5 macro-livelli)',
        'categorical'
    ),
    'work_status': (
        'Work Status',
        'QD10 — Situazione lavorativa → Active / Inactive / Vulnerable',
        'categorical'
    ),
    'is_italian': (
        'Nationality',
        'QD12 — Nazionalità (Italian / Other)',
        'categorical'
    ),
    'income_label': (
        'Net Monthly Income',
        'QD13 — Fascia di reddito netto mensile (4 categorie incl. Unknown)',
        'categorical'
    ),
    'internet_access_label': (
        'Internet Access',
        'QD14 — Accesso a Internet (Yes / No)',
        'categorical'
    ),
}

# Ordini logici per le categoriche
CUSTOM_ORDERS = {
    'age_group':          ['18-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79'],
    'edu_level_grouped':  ['No Education', 'Primary', 'Middle School', 'High School', 'University'],
    'income_label':       ['<=1750\u20ac', '1751-2900\u20ac', '>2900\u20ac', 'Unknown'],
    'urban_area_label':   ['<3k', '3k-15k', '15k-100k', '100k-1M', '>1M'],
    'macro_region_label': ['North-West', 'North-East', 'Center', 'South', 'Islands'],
}


# ── Funzioni di plotting ─────────────────────────────────────────────────────

def _fig_to_image_flowable(fig, width_cm=15.5):
    """Converte una figura matplotlib in un oggetto Image di ReportLab."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    # Calcola altezza proporzionale leggendo le dimensioni reali della figura
    fig_w_in, fig_h_in = fig.get_size_inches()
    aspect = fig_h_in / fig_w_in
    w = width_cm * cm
    h = w * aspect
    img = Image(buf, width=w, height=h)
    img.hAlign = 'CENTER'
    return img


def _plot_categorical(series, col):
    """Barplot frequenze relative con etichette % per variabili categoriche."""
    freq = series.value_counts(dropna=True, normalize=True) * 100

    if col in CUSTOM_ORDERS:
        order = [v for v in CUSTOM_ORDERS[col] if v in freq.index]
        # aggiungi eventuali valori non previsti in coda
        order += [v for v in freq.index if v not in order]
    else:
        order = freq.index.tolist()

    freq = freq.reindex(order).dropna()
    n_bars = len(freq)
    palette = sns.color_palette(PALETTE, n_bars)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.bar(range(n_bars), freq.values, color=palette,
                  edgecolor='white', linewidth=0.7)

    for bar, pct in zip(bars, freq.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            f'{pct:.1f}%',
            ha='center', va='bottom', fontsize=8.5, fontweight='bold',
            color='#1a1a2e'
        )

    ax.set_xticks(range(n_bars))
    ax.set_xticklabels(freq.index, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Frequenza Relativa (%)', fontsize=9)
    ax.set_ylim(0, freq.max() * 1.18)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.yaxis.grid(True, linestyle='--', alpha=GRID_ALPHA)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    fig.patch.set_facecolor('white')
    fig.tight_layout()
    return fig


def _plot_numeric(series, col):
    """Histogram + KDE + boxplot orizzontale per variabili numeriche."""
    fig, (ax_hist, ax_box) = plt.subplots(
        2, 1, figsize=(9, 4.8),
        gridspec_kw={'height_ratios': [5, 1]},
        sharex=True
    )

    n_bins = min(int(series.nunique()), 30)
    palette = sns.color_palette(PALETTE, 4)

    sns.histplot(series, kde=True, ax=ax_hist,
                 color=palette[0], bins=n_bins,
                 edgecolor='white', linewidth=0.4,
                 line_kws={'linewidth': 1.8, 'color': palette[2]})

    ax_hist.axvline(series.mean(),   color='crimson',   linestyle='--',
                    linewidth=1.4, label=f'Media: {series.mean():.2f}')
    ax_hist.axvline(series.median(), color='darkorange', linestyle=':',
                    linewidth=1.4, label=f'Mediana: {series.median():.2f}')
    ax_hist.legend(fontsize=8, framealpha=0.7)
    ax_hist.set_ylabel('Frequenza', fontsize=9)
    ax_hist.yaxis.grid(True, linestyle='--', alpha=GRID_ALPHA)
    ax_hist.set_axisbelow(True)
    ax_hist.spines[['top', 'right']].set_visible(False)

    ax_box.boxplot(
        series, vert=False, patch_artist=True,
        boxprops=dict(facecolor=palette[1], alpha=0.55),
        medianprops=dict(color='crimson', linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker='o', markersize=2.5, alpha=0.25,
                        markerfacecolor=palette[3])
    )
    ax_box.set_yticks([])
    ax_box.set_xlabel('Valore', fontsize=9)
    ax_box.spines[['top', 'right', 'left']].set_visible(False)

    fig.patch.set_facecolor('white')
    fig.tight_layout(h_pad=0.3)
    return fig


# ── Statistiche ─────────────────────────────────────────────────────────────

def _stats_categorical(series):
    """Ritorna lista di (label, valore) per la tabella statistiche."""
    n_total   = len(series)
    n_missing = series.isna().sum()
    n_valid   = n_total - n_missing
    freq = series.value_counts(dropna=True, normalize=True) * 100
    mode_val  = freq.idxmax() if len(freq) > 0 else 'N/A'
    mode_pct  = freq.max() if len(freq) > 0 else 0

    rows = [
        ('N totale',       f'{n_total:,}'),
        ('Valori validi',  f'{n_valid:,}'),
        ('Dati mancanti',  f'{n_missing:,}  ({n_missing/n_total*100:.1f}%)'),
        ('Valori unici',   f'{series.nunique(dropna=True)}'),
        ('Moda',           f'{mode_val}  ({mode_pct:.1f}%)'),
    ]

    # aggiungi le prime 5 categorie per frequenza
    for i, (val, pct) in enumerate(freq.head(5).items()):
        rows.append((f'  [{val}]', f'{pct:.1f}%'))

    return rows


def _stats_numeric(series):
    """Ritorna lista di (label, valore) per la tabella statistiche."""
    s = pd.to_numeric(series, errors='coerce').dropna()
    n_total   = len(series)
    n_missing = n_total - len(s)

    rows = [
        ('N totale',       f'{n_total:,}'),
        ('Valori validi',  f'{len(s):,}'),
        ('Dati mancanti',  f'{n_missing:,}  ({n_missing/n_total*100:.1f}%)'),
        ('Minimo',         f'{s.min():.2f}'),
        ('Q1 (25°)',       f'{s.quantile(0.25):.2f}'),
        ('Mediana',        f'{s.median():.2f}'),
        ('Media',          f'{s.mean():.2f}'),
        ('Q3 (75°)',       f'{s.quantile(0.75):.2f}'),
        ('Massimo',        f'{s.max():.2f}'),
        ('Std Dev',        f'{s.std():.2f}'),
        ('Skewness',       f'{s.skew():.3f}'),
    ]
    return rows


# ── Builder principale ───────────────────────────────────────────────────────

def generate_demographic_report(df: pd.DataFrame,
                                 output_path: str = 'Demographic_EDA_Report.pdf') -> str:
    """
    Genera il PDF di report EDA per le 12 variabili demografiche.

    Parameters
    ----------
    df          : DataFrame dopo engineer_demographic_features()
    output_path : percorso del file PDF di output

    Returns
    -------
    output_path : percorso del file generato
    """

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        topMargin=1.5 * cm,  bottomMargin=1.5 * cm,
    )

    styles = getSampleStyleSheet()

    # Stili personalizzati
    s_title = ParagraphStyle(
        'ReportTitle',
        parent=styles['Title'],
        fontSize=22, textColor=TEXT_LIGHT,
        backColor=BG_DARK,
        spaceAfter=4, spaceBefore=4,
        alignment=TA_CENTER,
        borderPad=10,
    )
    s_subtitle = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor('#adb5bd'),
        backColor=BG_DARK,
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    s_var_title = ParagraphStyle(
        'VarTitle',
        parent=styles['Heading1'],
        fontSize=13, textColor=TEXT_LIGHT,
        backColor=ACCENT,
        spaceBefore=12, spaceAfter=4,
        borderPad=7,
    )
    s_source = ParagraphStyle(
        'Source',
        parent=styles['Normal'],
        fontSize=8.5, textColor=colors.HexColor('#6c757d'),
        spaceBefore=0, spaceAfter=6,
        leftIndent=4,
    )
    s_section = ParagraphStyle(
        'Section',
        parent=styles['Heading2'],
        fontSize=10, textColor=colors.HexColor('#2d6a4f'),
        spaceBefore=6, spaceAfter=2,
    )
    s_note = ParagraphStyle(
        'Note',
        parent=styles['Normal'],
        fontSize=8, textColor=colors.HexColor('#6c757d'),
        spaceBefore=2, spaceAfter=4,
        leftIndent=6,
    )

    story = []

    # ── COPERTINA ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.2 * cm))
    story.append(Paragraph('IACOFI 2023 — DS Lab', s_title))
    story.append(Paragraph(
        'Exploratory Data Analysis · Variabili Demografiche e Background',
        s_subtitle
    ))
    story.append(Spacer(1, 0.3 * cm))

    # Mini-tabella riepilogativa in copertina
    demo_cols = [c for c in VAR_META if c in df.columns]
    n_obs     = len(df)
    n_valid_avg = int(np.mean([df[c].notna().sum() for c in demo_cols]))

    cover_data = [
        ['Osservazioni totali', f'{n_obs:,}'],
        ['Variabili analizzate', str(len(demo_cols))],
        ['Validità media per variabile', f'{n_valid_avg:,}  ({n_valid_avg/n_obs*100:.1f}%)'],
        ['Dataset', 'IACOFI 2023 — Adults Questionnaire'],
        ['Sezione codebook', 'QD1, QD5–QD7, QD9–QD14'],
    ]
    cover_table = Table(cover_data, colWidths=[8 * cm, 8 * cm])
    cover_table.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
        ('BACKGROUND',  (0, 0), (0, -1), colors.HexColor('#e9ecef')),
        ('TEXTCOLOR',   (0, 0), (-1, -1), TEXT_DARK),
        ('FONTNAME',    (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE',    (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1),
         [colors.HexColor('#f8f9fa'), colors.HexColor('#ffffff')]),
        ('GRID',        (0, 0), (-1, -1), 0.4, colors.HexColor('#dee2e6')),
        ('TOPPADDING',  (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('ALIGN',       (1, 0), (1, -1), 'RIGHT'),
    ]))
    story.append(cover_table)
    story.append(Spacer(1, 0.6 * cm))
    story.append(HRFlowable(width='100%', thickness=1.5,
                            color=colors.HexColor('#2d6a4f')))
    story.append(PageBreak())

    # ── PAGINA PER VARIABILE ─────────────────────────────────────────────────
    for col, (label, source_q, var_type) in VAR_META.items():
        if col not in df.columns:
            continue

        series = df[col]

        # ----- intestazione variabile -----
        block = []
        block.append(Paragraph(f'{label}  —  {col}', s_var_title))
        block.append(Paragraph(f'Fonte: {source_q}', s_source))

        # ----- grafico -----
        if var_type == 'categorical':
            s_plot = series.dropna().astype(str)
            if len(s_plot) == 0:
                continue
            fig  = _plot_categorical(s_plot, col)
            stat_rows = _stats_categorical(series)
        else:
            s_num = pd.to_numeric(series, errors='coerce').dropna()
            if len(s_num) == 0:
                continue
            fig  = _plot_numeric(s_num, col)
            stat_rows = _stats_numeric(series)

        block.append(Paragraph('Distribuzione', s_section))
        block.append(_fig_to_image_flowable(fig, width_cm=15.5))

        # ----- tabella statistiche -----
        block.append(Spacer(1, 0.25 * cm))
        block.append(Paragraph('Statistiche descrittive', s_section))

        # dividi in due colonne affiancate se ci sono molte righe
        mid   = (len(stat_rows) + 1) // 2
        left  = stat_rows[:mid]
        right = stat_rows[mid:]
        # padding a lunghezza uguale
        while len(right) < len(left):
            right.append(('', ''))

        tbl_data = [[l[0], l[1], r[0], r[1]] for l, r in zip(left, right)]

        col_w = [4.2 * cm, 3.0 * cm, 4.2 * cm, 3.0 * cm]
        tbl = Table(tbl_data, colWidths=col_w)
        tbl.setStyle(TableStyle([
            ('FONTNAME',  (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME',  (2, 0), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE',  (0, 0), (-1, -1), 8.5),
            ('TEXTCOLOR', (0, 0), (-1, -1), TEXT_DARK),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1),
             [colors.HexColor('#f8f9fa'), colors.white]),
            ('GRID',      (0, 0), (-1, -1), 0.3, colors.HexColor('#dee2e6')),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('ALIGN',     (1, 0), (1, -1), 'RIGHT'),
            ('ALIGN',     (3, 0), (3, -1), 'RIGHT'),
            # separatore verticale centrale
            ('LINEAFTER', (1, 0), (1, -1), 1, colors.HexColor('#adb5bd')),
        ]))
        block.append(tbl)

        # nota metodologica breve
        notes = {
            'gender':              'Valori -98 (altra risposta) e -99 (refused) → NaN.',
            'age_group':           'QD7 binned in fasce; chi ha rifiutato usa la banda diretta QD7A.',
            'living_status':       'Gerarchia: Alone > With_Children > With_Partner > Other.',
            'household_size':      'Codici -98/-99 trattati come 0 nel computo. +1 per il rispondente.',
            'digital_skills_score':'Codici negativi → 0. Score teorico 7–28; 0 se tutti item mancanti.',
            'work_status':         'Active={1,2,3}; Inactive={4,6,8,9,10}; Vulnerable={5,7}. -99→NaN.',
            'income_label':        '-97 (idk) e -99 (refused) → categoria "Unknown" (~25% campione).',
        }
        if col in notes:
            block.append(Paragraph(f'Nota: {notes[col]}', s_note))

        story.append(KeepTogether(block[:3]))   # titolo + fonte + label sezione
        story.extend(block[3:])                 # grafico + stats
        story.append(PageBreak())

    # ── BUILD ────────────────────────────────────────────────────────────────
    doc.build(story)
    print(f'✅ Report salvato in: {os.path.abspath(output_path)}')
    return output_path


# ── MAIN standalone ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Importa le tue funzioni dal file pipeline (devono essere nella stessa cartella)
    try:
        from pipeline import preprocess_complete_iacofi, engineer_demographic_features
    except ImportError:
        raise SystemExit(
            '❌  Non trovo pipeline.py. '
            'Assicurati che generate_demographic_report.py sia nella stessa cartella '
            'di pipeline.py oppure incolla le funzioni direttamente qui.'
        )

    FILE_PATH = r'c:/Users/loren/OneDrive/Desktop/Data Science/Data Science Lab/Database_ENG.csv'

    print('📥  Carico il dataset...')
    df_raw = pd.read_csv(FILE_PATH)

    print('🔧  Preprocessing...')
    df = preprocess_complete_iacofi(df_raw)
    df = engineer_demographic_features(df)

    print('📊  Genero il report PDF...')
    generate_demographic_report(df, output_path='Demographic_EDA_Report.pdf')
