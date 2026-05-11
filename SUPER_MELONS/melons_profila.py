import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# =================================================================
# 1. SETUP E PREPARAZIONE DATI
# =================================================================
OUTPUT_FOLDER = "profilazione_finanziaria"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Caricamento dataset
df = pd.read_csv("cleaned_df2.csv")

# Rinominazione tecnica (come da tuo script originale)
if "behaviour_investement-payment" in df.columns:
    df = df.rename(columns={"behaviour_investement-payment": "behaviour_investment_payment"})

# --- CREAZIONE INDICE DI CONOSCENZA SINTETICO (0-100) ---
# Include qk3, qk4, qk5, qk6, qk10 (1 punto ciascuna) + qk7_clean (batteria da 0 a 6)
qk_cols = ['qk3_clean', 'qk4_clean', 'qk5_clean', 'qk6_clean', 'qk10_clean', 'qk7_clean']
max_score = 11 
df['qk_composite_100'] = (df[qk_cols].sum(axis=1) / max_score) * 100

# Selezione variabili originali per Profilazione e PCA
features = [
    "saving_level_sophistication", "transactional_score", "saving_protection_score",
    "consumer_debt_score", "traditional_investment_score", "alternative_asset_score",
    "financial_planning_score", "digital_onboarding_score", "daily_transactional_intensity",
    "advanced_fintech_intensity", "risk_aversion_class", "qk_composite_100",
    "finacial_situation", "digital_skills_score"
]

# Pulizia per analisi numerica
df_clean = df.dropna(subset=features).copy()

# =================================================================
# 2. CLUSTERING (DEFINIZIONE PERSONA)
# =================================================================
X_clust = df_clean[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clust)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df_clean['cluster'] = kmeans.fit_predict(X_scaled)

# Mapping Persona basato sui dati (Logica Heuristica)
def label_persona(row):
    if row['qk_composite_100'] > 70 and row['traditional_investment_score'] > 0.8: return "Sophisticated Investors"
    if row['advanced_fintech_intensity'] > 2.5: return "Digitally-Engaged Planners"
    if row['financial_planning_score'] < 2.5: return "Financially Disengaged"
    if row['risk_aversion_class'] > 3.5: return "Cautious Savers"
    return "General Consumer Profile"

temp_profile = df_clean.groupby('cluster')[features].mean()
persona_map = {i: label_persona(row) for i, row in temp_profile.iterrows()}
df_clean['persona'] = df_clean['cluster'].map(persona_map)

# =================================================================
# 3. ANALISI PCA (VARIANZA DI FINANCIAL SITUATION)
# =================================================================
pca = PCA(n_components=2)
pca.fit(X_scaled)
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=features)

# =================================================================
# 4. GENERAZIONE E SALVATAGGIO OUTPUT (MULTI-PAGINA)
# =================================================================

# PAGINA 1: Driver della Varianza e Correlazioni
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
loadings['PC1'].sort_values().plot(kind='barh', color='skyblue')
plt.title("Analisi PCA: Peso delle variabili su PC1", fontweight='bold')
plt.subplot(1, 2, 2)
sns.heatmap(df_clean[features].corr()[['finacial_situation']].sort_values(by='finacial_situation', ascending=False), 
            annot=True, cmap='RdYlGn', cbar=False)
plt.title("Correlazione con 'finacial_situation'", fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "01_driver_analisi_varianza.png"))
plt.close()

# PAGINA 2: Profilazione Radar delle Persona
def plot_radar_summary(df_p, folder):
    categories = features
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    df_norm = (df_p - df_p.min()) / (df_p.max() - df_p.min() + 1e-9)
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for index, row in df_norm.iterrows():
        values = row.values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=index)
        ax.fill(angles, values, alpha=0.1)
    plt.xticks(angles[:-1], categories, size=8)
    plt.title("Confronto Profili Persona (Variabili Originali)", y=1.1, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.savefig(os.path.join(folder, "02_radar_profili.png"))
    plt.close()

plot_radar_summary(df_clean.groupby('persona')[features].mean(), OUTPUT_FOLDER)

# PAGINA 3: Deep Dive Interazione (Conoscenza vs Pianificazione)
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_clean, x='qk_composite_100', y='financial_planning_score', 
                hue='persona', size='finacial_situation', sizes=(20, 400), alpha=0.6)
plt.axvline(x=df_clean['qk_composite_100'].mean(), color='red', linestyle='--', label='Media Conoscenza')
plt.title("Mappa Strategica: Conoscenza, Pianificazione e Benessere", fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "03_mappa_strategica.png"))
plt.close()

# PAGINA 4: Boxplot Socio-Economici
fig, axes = plt.subplots(2, 1, figsize=(14, 12))
sns.boxplot(data=df_clean, x='edu_level_grouped', y='finacial_situation', palette='Set2', ax=axes[0])
axes[0].set_title("Benessere vs Livello Educativo", fontweight='bold')
sns.boxplot(data=df_clean, x='work_status', y='finacial_situation', palette='Set3', ax=axes[1])
axes[1].set_title("Benessere vs Stato Lavorativo", fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "04_analisi_socio_economica.png"))
plt.close()

# 5. ESPORTAZIONE DATI FINALI
df_clean.to_csv(os.path.join(OUTPUT_FOLDER, "dataset_finale_profilato.csv"), index=False)
loadings.to_csv(os.path.join(OUTPUT_FOLDER, "carichi_pca.csv"))

print(f"Lavoro completato. Tutti i file sono in: {OUTPUT_FOLDER}/")
