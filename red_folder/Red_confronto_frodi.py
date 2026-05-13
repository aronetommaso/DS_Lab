import os
import math

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import chi2_contingency


# ==========================================
# CONFIG
# ==========================================

filepath = "C:/Users/Utente/Desktop/LUCA/università/DATA_SCIENCE/1_ANNO/cleaned_df2.csv"

target = 'cyber_fraud_victim'

cartella_output = "grafici_frodi"


# ==========================================
# CREAZIONE CARTELLA OUTPUT
# ==========================================

os.makedirs(cartella_output, exist_ok=True)


# ==========================================
# CARICAMENTO DATI
# ==========================================

def carica_e_pulisci(filepath):

    print(f"Caricamento di {filepath}...")

    df = pd.read_csv(filepath)

    return df


# ==========================================
# CATEGORIZZAZIONE
# ==========================================

def categorizzazione(df):

    df['knowledge_financial_privacy_digital_cat'] = pd.cut(
        df['knowledge_financial_privacy_digital'],
        bins=[-1, 25, 35, 45, 65],
        labels=['Basso', 'Medio', 'Alto', 'Molto Alto']
    )

    df['digital_skills_score_cat'] = pd.cut(
        df['digital_skills_score'],
        bins=[0.9, 2.5, 3.5, 4.1],
        labels=['Basso', 'Medio', 'Alto']
    )

    df['finacial_situation_cat'] = pd.cut(
        df['finacial_situation'],
        bins=[0, 14, 21, 25, 35],
        labels=['Basso', 'Medio', 'Alto', 'Molto Alto']
    )

    df['behaviour_investement-payment_cat'] = pd.cut(
        df['behaviour_investement-payment'],
        bins=[-1, 30, 40, 50, 75],
        labels=['Basso', 'Medio', 'Alto', 'Molto Alto']
    )

    return df


# ==========================================
# TROVA CATEGORICHE
# ==========================================

def trova_categoriche(df, soglia=10):

    print("Ricerca variabili categoriche in corso...")

    colonne = [
        col for col in df.columns
        if df[col].nunique() <= soglia and df[col].nunique() > 0
    ]

    return colonne


# ==========================================
# TEST CHI QUADRO
# ==========================================

def chi_quadro_test(df):

    print("\n1. Avvio Screening Statistico con Chi-Quadro...\n")

    df_stats = df.dropna(subset=[target]).copy()

    df_stats = df_stats.dropna(axis=1, how='all')

    risultati_chi2 = []

    colonne_x = df_stats.columns.tolist()

    print(f"Sto testando {len(colonne_x)} variabili...\n")

    for colonna in colonne_x:

        crosstab = pd.crosstab(
            df_stats[colonna],
            df_stats[target]
        )

        if crosstab.shape[0] > 1 and crosstab.shape[1] > 1:

            chi2_stat, p_val, dof, ex = chi2_contingency(crosstab)

            risultati_chi2.append({

                'Variabile': colonna,
                'P-Value': p_val,
                'Significativa': 'SI' if p_val < 0.05 else 'NO'
            })

    df_risultati = pd.DataFrame(risultati_chi2)

    df_risultati = df_risultati.sort_values(
        by='P-Value',
        ascending=True
    ).reset_index(drop=True)

    variabili_tenute = df_risultati[
        df_risultati['P-Value'] < 0.05
    ]['Variabile'].tolist()

    return variabili_tenute


# ==========================================
# RANDOM FOREST FEATURE IMPORTANCE
# ==========================================

def seleziona_top_features(df, target_col, top_n=15):

    print(f"\nAvvio Feature Importance Random Forest...\n")

    X = df.drop(columns=[target_col])

    y = df[target_col]

    colonne_originali = X.columns.tolist()

    X_codificata = pd.get_dummies(
        X,
        drop_first=True
    )

    X_codificata = X_codificata.fillna(
        X_codificata.median()
    )

    modello = RandomForestClassifier(

        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )

    modello.fit(X_codificata, y)

    importanze_dummy = modello.feature_importances_

    nomi_dummy = X_codificata.columns

    importanza_aggregata = {

        col: 0.0 for col in colonne_originali
    }

    for nome_dummy, importanza in zip(
        nomi_dummy,
        importanze_dummy
    ):

        for col_orig in colonne_originali:

            if (
                nome_dummy == col_orig
                or nome_dummy.startswith(col_orig + '_')
            ):

                importanza_aggregata[col_orig] += importanza

                break

    df_importanze = pd.DataFrame({

        'Variabile': list(importanza_aggregata.keys()),
        'Importanza Totale': list(importanza_aggregata.values())
    })

    df_top = df_importanze.sort_values(

        by='Importanza Totale',
        ascending=False

    ).head(top_n)

    variabili_vincenti = df_top['Variabile'].tolist()

    print("Top variabili selezionate:\n")

    print(variabili_vincenti)

    return variabili_vincenti


# ==========================================
# RADAR CHART
# ==========================================

def crea_radar_chart(
    df,
    features,
    target_col,
    titolo,
    nome_file
):

    print(f"\nGenerazione Radar Chart: {titolo}")

    features_presenti = [

        col for col in features
        if col in df.columns
    ]

    features_numeriche = df[
        features_presenti
    ].select_dtypes(include=['number']).columns.tolist()

    if target_col in features_numeriche:

        features_numeriche.remove(target_col)

    if len(features_numeriche) < 3:

        print("❌ Radar non creato: meno di 3 variabili numeriche")

        return

    print(f"Variabili numeriche usate: {features_numeriche}")

    df_lavoro = df[
        features_numeriche + [target_col]
    ].dropna().copy()

    scaler = MinMaxScaler()

    df_lavoro[features_numeriche] = scaler.fit_transform(
        df_lavoro[features_numeriche]
    )

    medie = df_lavoro.groupby(
        target_col
    )[features_numeriche].mean()

    N = len(features_numeriche)

    angoli = [
        n / float(N) * 2 * np.pi
        for n in range(N)
    ]

    angoli += angoli[:1]

    fig, ax = plt.subplots(

        figsize=(12, 12),

        subplot_kw=dict(polar=True)
    )

    if 0 in medie.index:

        valori_sicuri = medie.loc[0].values.flatten().tolist()

        valori_sicuri += valori_sicuri[:1]

        ax.plot(
            angoli,
            valori_sicuri,
            linewidth=3,
            linestyle='solid',
            label='Non Truffati',
            color='#1f77b4'
        )

        ax.fill(
            angoli,
            valori_sicuri,
            '#1f77b4',
            alpha=0.2
        )

    if 1 in medie.index:

        valori_truffati = medie.loc[1].values.flatten().tolist()

        valori_truffati += valori_truffati[:1]

        ax.plot(
            angoli,
            valori_truffati,
            linewidth=3,
            linestyle='solid',
            label='Truffati',
            color='#d62728'
        )

        ax.fill(
            angoli,
            valori_truffati,
            '#d62728',
            alpha=0.3
        )

    plt.xticks(

        angoli[:-1],
        features_numeriche,

        size=11,
        fontweight='bold'
    )

    ax.set_yticklabels([])

    plt.title(

        titolo,

        size=20,
        fontweight='bold',
        pad=40
    )

    plt.legend(

        loc='upper right',
        bbox_to_anchor=(1.25, 1.15),

        fontsize=12
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    percorso = os.path.join(
        cartella_output,
        nome_file
    )

    plt.savefig(
        percorso,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Salvato: {percorso}")

    plt.close()


# ==========================================
# BARPLOT GRID
# ==========================================

def griglia_barplot_categorie(
    df,
    features_list,
    target_col,
    titolo,
    nome_file
):

    print(f"\nGenerazione Barplot Grid: {titolo}")

    features_list = [

        col for col in features_list
        if col in df.columns
    ]

    if len(features_list) == 0:

        print("❌ Nessuna variabile disponibile")

        return

    colonne_griglia = 3

    righe_griglia = math.ceil(
        len(features_list) / colonne_griglia
    )

    fig, axes = plt.subplots(

        nrows=righe_griglia,
        ncols=colonne_griglia,

        figsize=(22, 7 * righe_griglia)
    )

    axes = np.array(axes).flatten()

    for i, colonna in enumerate(features_list):

        prop_df = (

            df.groupby(target_col)[colonna]

            .value_counts(normalize=True)

            .rename('Percentuale')

            .reset_index()
        )

        prop_df['Percentuale'] *= 100

        sns.barplot(

            data=prop_df,

            x=colonna,
            y='Percentuale',

            hue=target_col,

            palette=['#1f77b4', '#d62728'],

            ax=axes[i]
        )

        axes[i].set_title(

            colonna,

            fontsize=14,
            fontweight='bold'
        )

        axes[i].set_xlabel('')

        if i % colonne_griglia == 0:

            axes[i].set_ylabel(
                'Distribuzione (%)',
                fontsize=12
            )

        else:

            axes[i].set_ylabel('')

        axes[i].tick_params(

            axis='x',
            labelrotation=25,
            labelsize=10
        )

        if i == 0:

            axes[i].legend(

                title='Frode',

                labels=['No', 'Sì'],

                fontsize=11,
                title_fontsize=12
            )

        else:

            if axes[i].get_legend() is not None:

                axes[i].get_legend().remove()

    for j in range(i + 1, len(axes)):

        fig.delaxes(axes[j])

    fig.suptitle(

        titolo,

        fontsize=24,
        fontweight='bold',

        y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    percorso = os.path.join(
        cartella_output,
        nome_file
    )

    plt.savefig(
        percorso,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Salvato: {percorso}")

    plt.close()


# ==========================================
# MAIN
# ==========================================

def main():

    print("\n--- INIZIO PIPELINE ---\n")

    # ======================================
    # CARICAMENTO
    # ======================================

    df = carica_e_pulisci(filepath)

    print(f"Dati caricati! Shape: {df.shape}")

    # ======================================
    # CATEGORIZZAZIONE
    # ======================================

    print("\nApplico categorizzazione...")

    df = categorizzazione(df)

    # ======================================
    # SOLO CATEGORICHE
    # ======================================

    lista_cat = trova_categoriche(df)

    df = df[lista_cat]

    # ======================================
    # CHI QUADRO
    # ======================================

    variabili_tenute = chi_quadro_test(df)

    print(f"\n✅ Variabili significative: {len(variabili_tenute)}")

    print("\nVariabili tenute:\n")

    print(variabili_tenute)

    variabili_scartate = list(
        set(df.columns) - set(variabili_tenute)
    )

    if target in variabili_scartate:

        variabili_scartate.remove(target)

    print("\nVariabili scartate:\n")

    print(variabili_scartate)

    # ======================================
    # RANDOM FOREST
    # ======================================

    variabili_rf = variabili_tenute.copy()

    variabili_rf.append(target)

    features_importance_RM = seleziona_top_features(

        df[variabili_rf],

        target
    )

    # ======================================
    # GRAFICI RANDOM FOREST
    # ======================================

    crea_radar_chart(

        df,

        features_importance_RM,

        target,

        titolo='RADAR CHART — Top 15 Variabili Random Forest',

        nome_file='01_radar_random_forest.jpg'
    )

    griglia_barplot_categorie(

        df,

        features_importance_RM,

        target,

        titolo='BARPLOT GRID — Top 15 Variabili Random Forest',

        nome_file='02_barplot_random_forest.jpg'
    )

    # ======================================
    # GRAFICI VARIABILI SCARTATE
    # ======================================

    crea_radar_chart(

        df,

        variabili_scartate,

        target,

        titolo='RADAR CHART — Variabili NON significative al Chi-Quadro',

        nome_file='03_radar_variabili_scartate.jpg'
    )

    griglia_barplot_categorie(

        df,

        variabili_scartate,

        target,

        titolo='BARPLOT GRID — Variabili NON significative al Chi-Quadro',

        nome_file='04_barplot_variabili_scartate.jpg'
    )

    print("\n===================================")
    print("✅ TUTTI I GRAFICI SONO STATI SALVATI")
    print(f"📁 Cartella: {cartella_output}")
    print("===================================\n")


# ==========================================
# START
# ==========================================

if __name__ == "__main__":
    main()