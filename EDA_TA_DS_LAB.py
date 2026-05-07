"""
IACOFI 2023 - Financial Literacy Survey: Profiling & Analysis
=============================================================
This script performs:
  1. Exploratory Data Analysis (EDA) with demographic and financial overviews
  2. Correlation heatmap across engineered features
  3. Self-Organizing Map (SOM) on investment & financial behaviour features
  4. K-Means clustering on SOM outputs for persona labelling
  5. Radar charts and profile summaries for each identified persona

Dependencies: pandas, numpy, matplotlib, seaborn, scikit-learn, minisom
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
import matplotlib.cm as cm
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ── Global style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 130,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

PALETTE = sns.color_palette("tab10", 10)
OUTPUT_DIR = "plot/som_profiling_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    """Save figure to the output directory as high-resolution PNG."""
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [saved] {path}")


# ════════════════════════════════════════════════════════════════════════════
# 0. DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_data(path: str = "cleaned_df.csv") -> pd.DataFrame:
    """Load the pre-cleaned IACOFI dataset and perform minor tidying."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower().strip() for c in df.columns]

    # Rename the behaviour column that contains a dash (causes attribute-access issues)
    df = df.rename(columns={"behaviour_investement-payment": "behaviour_investment_payment"})

    # Drop unnamed index columns if present
    df = df.loc[:, ~df.columns.str.startswith("unnamed")]

    print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 1. EDA — DEMOGRAPHIC OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

def eda_demographics(df: pd.DataFrame) -> None:
    """
    Plot distributions for the main demographic variables:
    gender, age group, education, work status, income, region, and urbanisation.
    """
    print("\n[1] Demographic EDA …")

    demo_vars = [
        ("gender",            "Gender"),
        ("age_group",         "Age Group"),
        ("edu_level_grouped", "Education Level"),
        ("work_status",       "Work Status"),
        ("income_label",      "Monthly Income Band"),
        ("macro_region_label","Macro Region"),
        ("urban_area_label",  "Urbanisation Level"),
        ("internet_access_label", "Internet Access"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle("Demographic Profile — IACOFI 2023", fontsize=15, fontweight="bold", y=1.01)

    for ax, (col, title) in zip(axes.flat, demo_vars):
        if col not in df.columns:
            ax.axis("off")
            continue
        series = df[col].dropna().astype(str)
        counts = series.value_counts().sort_values(ascending=False)
        bars = ax.bar(counts.index, counts.values,
                      color=PALETTE[:len(counts)], edgecolor="white", linewidth=0.6)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        # Add percentage labels on top of each bar
        total = counts.sum()
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + total * 0.005,
                    f"{val / total:.0%}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    save(fig, "01_demographic_overview")


# ════════════════════════════════════════════════════════════════════════════
# 2. EDA — FINANCIAL BEHAVIOUR OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

def eda_financial_behaviour(df: pd.DataFrame) -> None:
    """
    Plot key financial-behaviour engineered variables:
    planning score, saving sophistication, pension flags, debt flags, product scores, etc.
    """
    print("\n[2] Financial Behaviour EDA …")

    numeric_vars = [
        ("financial_planning_score",   "Financial Planning Score (0-6)"),
        ("saving_level_sophistication","Saving Sophistication (0-3)"),
        ("transactional_score",        "Transactional Products Score"),
        ("saving_protection_score",    "Saving & Protection Score"),
        ("consumer_debt_score",        "Consumer Debt Score"),
        ("traditional_investment_score","Traditional Investment Score"),
        ("alternative_asset_score",    "Alternative Asset Score"),
        ("digital_onboarding_score",   "Digital Onboarding Score (0-5)"),
        ("qk7_clean",                  "Financial Knowledge Score (0-6)"),
        ("digital_skills_score",       "Digital Skills Score"),
        ("finacial_situation",         "Financial Situation (composite)"),
        ("behaviour_investment_payment","Behaviour: Investment & Payment"),
        ("knowledge_financial_privacy_digital","Knowledge: Privacy & Digital"),
    ]

    fig, axes = plt.subplots(3, 5, figsize=(24, 13))
    fig.suptitle("Financial Behaviour & Scores — IACOFI 2023", fontsize=15, fontweight="bold", y=1.01)

    for ax, (col, title) in zip(axes.flat, numeric_vars):
        if col not in df.columns:
            ax.axis("off")
            continue
        series = df[col].dropna()
        if series.nunique() <= 12:
            counts = series.value_counts().sort_index()
            ax.bar(counts.index.astype(str), counts.values,
                   color="#4C72B0", edgecolor="white", linewidth=0.5)
        else:
            ax.hist(series, bins=25, color="#4C72B0", edgecolor="white", linewidth=0.5)
        ax.set_title(title, fontweight="bold", fontsize=9)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(labelsize=8)

    # Turn off any unused axes
    for ax in axes.flat[len(numeric_vars):]:
        ax.axis("off")

    plt.tight_layout()
    save(fig, "02_financial_behaviour_overview")


# ════════════════════════════════════════════════════════════════════════════
# 3. CORRELATION HEATMAP
# ════════════════════════════════════════════════════════════════════════════

def correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Compute and visualise Pearson correlations among all numeric engineered features.
    Variables with too many NaNs are excluded automatically.
    """
    print("\n[3] Correlation heatmap …")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Keep only columns with < 40 % missing
    valid = [c for c in numeric_cols if df[c].isna().mean() < 0.4]
    corr = df[valid].corr()

    fig, ax = plt.subplots(figsize=(20, 16))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=False, linewidths=0.4, linecolor="white",
        cbar_kws={"shrink": 0.7, "label": "Pearson r"},
        ax=ax
    )
    ax.set_title("Pearson Correlation Matrix — Numeric Features", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    save(fig, "03_correlation_heatmap")


# ════════════════════════════════════════════════════════════════════════════
# 4. SELF-ORGANIZING MAP (SOM)
# ════════════════════════════════════════════════════════════════════════════

# Features chosen to capture the investment & financial behaviour profile
SOM_FEATURES = [
    "saving_level_sophistication",      # 0-3: cash → investor
    "transactional_score",              # basic banking usage
    "saving_protection_score",          # pension & savings products held
    "consumer_debt_score",              # debt products held
    "traditional_investment_score",     # stocks, bonds, investment account
    "alternative_asset_score",          # mortgage, crypto, ESG
    "financial_planning_score",         # planning & budgeting behaviour
    "digital_onboarding_score",         # opened products digitally
    "daily_transactional_intensity",    # online transactional usage
    "advanced_fintech_intensity",       # manage finances / robo-advisor
    "risk_aversion_class",              # 1=risk-loving → 4=risk-averse
    "qk7_clean",                        # objective financial knowledge (0-6)
    "finacial_situation",               # composite financial wellbeing
    "behaviour_investment_payment",     # investment & payment behaviour score
]


def prepare_som_data(df: pd.DataFrame):
    """
    Select SOM features, impute missing values with the column median,
    and standardise to zero mean and unit variance.
    Returns: scaled array, scaler, imputer, and the filtered dataframe rows.
    """
    available = [f for f in SOM_FEATURES if f in df.columns]
    print(f"\n  SOM uses {len(available)}/{len(SOM_FEATURES)} features: {available}")

    sub = df[available].copy()

    # Impute medians (robust to skewed financial distributions)
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(sub)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    return X_scaled, scaler, imputer, available, sub.index


def train_som(X_scaled: np.ndarray, grid_size: int = 12, n_iter_multiplier: int = 500) -> MiniSom:
    """
    Train a 2-D Self-Organizing Map.

    Grid size: (grid_size × grid_size) neurons.
    The number of iterations is n_iter_multiplier × number of samples.
    """
    n_features = X_scaled.shape[1]
    n_iter = X_scaled.shape[0] * n_iter_multiplier // 1000  # scale iterations sensibly

    print(f"\n[4] Training SOM ({grid_size}×{grid_size}, {n_iter} iterations) …")
    som = MiniSom(
        x=grid_size, y=grid_size,
        input_len=n_features,
        sigma=2.0,                # neighbourhood radius
        learning_rate=0.5,
        neighborhood_function="gaussian",
        random_seed=42
    )
    som.random_weights_init(X_scaled)
    som.train_batch(X_scaled, n_iter, verbose=True)
    print("  SOM training complete.")
    return som


def plot_som_umatrix(som: MiniSom, X_scaled: np.ndarray, output_labels: np.ndarray = None) -> None:
    """
    Plot the U-Matrix (unified distance matrix) of the SOM.
    The U-Matrix shows distances between neighbouring neurons; darker = larger distance = cluster boundary.
    Optionally overlays cluster labels as coloured dots on top.
    """
    print("  Plotting U-Matrix …")
    umatrix = som.distance_map()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.pcolor(umatrix.T, cmap="bone_r")
    plt.colorbar(im, ax=ax, label="Mean distance to neighbours")

    # Overlay winner positions coloured by cluster if labels provided
    if output_labels is not None:
        n_clusters = len(np.unique(output_labels))
        cmap = cm.get_cmap("tab10", n_clusters)
        jitter_range = 0.3
        for i, x in enumerate(X_scaled):

            off_x = np.random.uniform(-jitter_range, jitter_range)
            off_y = np.random.uniform(-jitter_range, jitter_range)
            w = som.winner(x)
            ax.plot(w[0] + 0.5 + off_x, w[1] + 0.5 + off_y, "o",
                    color=cmap(output_labels[i]),
                    markersize=3, alpha=0.4)

        # Legend for clusters
        patches = [mpatches.Patch(color=cmap(c), label=f"Cluster {c+1}")
                   for c in range(n_clusters)]
        ax.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.8)

    ax.set_title("SOM U-Matrix — Investment & Financial Behaviour", fontsize=13, fontweight="bold")
    ax.set_xlabel("SOM x-axis")
    ax.set_ylabel("SOM y-axis")
    plt.tight_layout()
    save(fig, "04_som_umatrix")


def plot_som_feature_planes(som: MiniSom, feature_names: list) -> None:
    """
    Plot one activation plane per input feature.
    Each cell shows the weight for that feature in the neuron's prototype vector.
    High values = neurons whose prototypes are high on that feature.
    """
    print("  Plotting SOM feature planes …")
    weights = som.get_weights()          # shape: (x, y, n_features)
    n_feat = len(feature_names)
    ncols = 5
    nrows = int(np.ceil(n_feat / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3))
    fig.suptitle("SOM Feature Planes — Weight per Neuron", fontsize=13, fontweight="bold", y=1.01)

    for i, (ax, feat) in enumerate(zip(axes.flat, feature_names)):
        plane = weights[:, :, i]
        im = ax.pcolor(plane.T, cmap="RdYlGn")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(feat.replace("_", " "), fontsize=8, fontweight="bold")
        ax.axis("off")

    for ax in axes.flat[n_feat:]:
        ax.axis("off")

    plt.tight_layout()
    save(fig, "05_som_feature_planes")

    # TODO: scatterplot on the feature planes.
def plot_som_feature_planes_with_samples(som, X_scaled, feature_names, sample_labels, output_name="05b_som_feature_planes_samples"):
    """
    Sovrappone i punti del dataset (colorati per cluster) sui Feature Planes della SOM.
    """
    print("  Plotting SOM feature planes with sample overlay ...")
    weights = som.get_weights()
    n_feat = len(feature_names)
    ncols = 5
    nrows = int(np.ceil(n_feat / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.2))
    fig.suptitle("SOM Feature Planes — Personas Overlay", fontsize=15, fontweight="bold", y=1.02)

    n_clusters = len(np.unique(sample_labels))
    cmap_clusters = cm.get_cmap("tab10", n_clusters)

    # 1. Calcola le coordinate BMU per tutti i campioni
    bmus = np.array([som.winner(x) for x in X_scaled])
    
    # 2. Aggiunge un po' di "jitter" (dispersione) per evitare che i punti si sovrappongano perfettamente
    jitter_x = np.random.uniform(-0.35, 0.35, size=len(bmus))
    jitter_y = np.random.uniform(-0.35, 0.35, size=len(bmus))
    x_coords = bmus[:, 0] + 0.5 + jitter_x
    y_coords = bmus[:, 1] + 0.5 + jitter_y

    for i, (ax, feat) in enumerate(zip(axes.flat, feature_names)):
        # Plotta il piano dei pesi (Feature Plane) sbiadendolo leggermente (alpha=0.6)
        plane = weights[:, :, i]
        im = ax.pcolor(plane.T, cmap="RdYlGn", alpha=0.6) 
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Sovrappone i campioni colorati per cluster
        for c in range(n_clusters):
            mask = sample_labels == c
            ax.scatter(x_coords[mask], y_coords[mask],
                       color=cmap_clusters(c), s=4, alpha=0.7, 
                       edgecolors='none')

        ax.set_title(feat.replace("_", " "), fontsize=9, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # Rimuove gli assi vuoti in eccesso
    for ax in axes.flat[n_feat:]:
        ax.axis("off")

    # Aggiunge una legenda globale in basso
    patches = [mpatches.Patch(color=cmap_clusters(c), label=f"Cluster {c+1}") for c in range(n_clusters)]
    fig.legend(handles=patches, loc="lower center", ncol=n_clusters, bbox_to_anchor=(0.5, -0.05), fontsize=10)

    plt.tight_layout()
    save(fig, output_name)

# ════════════════════════════════════════════════════════════════════════════
# 5. K-MEANS CLUSTERING ON SOM OUTPUT
# ════════════════════════════════════════════════════════════════════════════

def find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 9)) -> int:
    """
    Use the Silhouette Score to select the optimal number of K-Means clusters.
    Returns the k with the highest average silhouette score.
    """
    print("\n[5] Searching for optimal number of clusters …")
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)
        print(f"    k={k}: silhouette={scores[k]:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"  → Best k = {best_k} (silhouette={scores[best_k]:.4f})")

    # Plot silhouette scores
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(scores.keys()), list(scores.values()), "o-", color="#2C7BB6", linewidth=2)
    ax.axvline(best_k, color="red", linestyle="--", label=f"Optimal k={best_k}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Average Silhouette Score")
    ax.set_title("Silhouette Analysis — Choosing Optimal k", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    save(fig, "06_silhouette_scores")

    return best_k


def cluster_som_neurons(som: MiniSom, X_scaled: np.ndarray, k: int):
    """
    Map each SOM neuron's prototype vector to a K-Means cluster.
    Returns per-sample cluster labels.
    """
    # Collect all BMU (best matching unit) positions
    bmu_coords = np.array([som.winner(x) for x in X_scaled])

    # Cluster on raw SOM weights (prototype vectors) rather than on BMU indices
    # to capture the actual feature values in each neuron
    weights_flat = som.get_weights().reshape(-1, X_scaled.shape[1])
    km = KMeans(n_clusters=k, random_state=42, n_init=15)
    neuron_labels = km.fit_predict(weights_flat)

    grid_x = som.get_weights().shape[0]
    # Assign each sample the cluster of its winning neuron
    sample_labels = np.array([
        neuron_labels[bmu[0] * grid_x + bmu[1]]
        for bmu in bmu_coords
    ])
    return sample_labels, bmu_coords


# ════════════════════════════════════════════════════════════════════════════
# 6. PCA PROJECTION — visualise clusters in 2-D
# ════════════════════════════════════════════════════════════════════════════

def plot_pca_clusters(X_scaled: np.ndarray, labels: np.ndarray) -> None:
    """
    Project the high-dimensional feature space onto 2 principal components
    and colour points by cluster assignment.
    """
    print("\n[6] PCA projection …")
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))
    n_clusters = len(np.unique(labels))
    cmap = cm.get_cmap("tab10", n_clusters)

    for c in range(n_clusters):
        mask = labels == c
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=cmap(c), alpha=0.45, s=18, label=f"Cluster {c+1}")

    ax.set_xlabel(f"PC1 ({var[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} variance)")
    ax.set_title("PCA — Financial Behaviour Clusters", fontsize=13, fontweight="bold")
    ax.legend(markerscale=2, fontsize=9)
    plt.tight_layout()
    save(fig, "07_pca_clusters")


# ════════════════════════════════════════════════════════════════════════════
# 7. PERSONA PROFILING — radar charts + descriptive stats
# ════════════════════════════════════════════════════════════════════════════

# Friendly labels for persona assignment (filled after inspection)
PERSONA_NAMES_PLACEHOLDER = {
    0: "Cluster 1",
    1: "Cluster 2",
    2: "Cluster 3",
    3: "Cluster 4",
    4: "Cluster 5",
    5: "Cluster 6",
}


def assign_persona_names(profile_df: pd.DataFrame) -> dict:
    """
    Heuristically assign persona labels based on cluster mean values.
    Dimensions inspected:
      - traditional_investment_score + alternative_asset_score  → investment appetite
      - consumer_debt_score                                     → debt reliance
      - digital_onboarding_score + advanced_fintech_intensity   → digital engagement
      - financial_planning_score                                → planning discipline
      - qk7_clean                                              → financial knowledge
      - risk_aversion_class (inverted: 1=high risk, 4=low risk)
    """
    names = {}
    for cluster_id, row in profile_df.iterrows():
        invest = row.get("traditional_investment_score", 0) + row.get("alternative_asset_score", 0)
        debt   = row.get("consumer_debt_score", 0)
        digital= row.get("digital_onboarding_score", 0) + row.get("advanced_fintech_intensity", 0)
        plan   = row.get("financial_planning_score", 0)
        know   = row.get("qk7_clean", 0)
        risk   = row.get("risk_aversion_class", 3)   # lower = more risk tolerant

        if invest > 1.5 and know > 3 and risk < 2.5:
            label = "📈 Sophisticated Investors"
        elif digital > 3 and plan > 3 and debt < 1:
            label = "💻 Digitally-Engaged Planners"
        elif debt > 2 and invest < 0.5:
            label = "💳 Debt-Reliant Households"
        elif plan < 2 and invest < 0.5 and digital < 1.5:
            label = "😴 Financially Disengaged"
        elif risk > 3 and plan > 2:
            label = "🛡️ Cautious Savers"
        elif invest > 0.8 and digital > 2 and risk < 3:
            label = "🌱 Emerging Investors"
        else:
            label = f"Cluster {cluster_id + 1}"

        names[cluster_id] = label
    return names


def plot_radar_profiles(profile_df: pd.DataFrame, persona_names: dict,
                        feature_names: list) -> None:
    """
    Draw a radar (spider) chart for each cluster, showing the normalised
    mean value across the SOM features.
    """
    print("\n[7] Radar profile charts …")

    # Normalise each feature to [0, 1] across clusters for comparability
    norm = (profile_df - profile_df.min()) / (profile_df.max() - profile_df.min() + 1e-9)
    cols_available = [f for f in feature_names if f in norm.columns]
    norm = norm[cols_available]

    n_clusters = len(profile_df)
    n_feats = len(cols_available)
    angles = np.linspace(0, 2 * np.pi, n_feats, endpoint=False).tolist()
    angles += angles[:1]                # close the polygon

    ncols = 3
    nrows = int(np.ceil(n_clusters / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 4.5),
                             subplot_kw=dict(polar=True))
    fig.suptitle("Persona Radar Profiles", fontsize=15, fontweight="bold", y=1.01)

    cmap = cm.get_cmap("tab10", n_clusters)
    labels_short = [c.replace("_", " ") for c in cols_available]

    for idx, (cluster_id, row) in enumerate(norm.iterrows()):
        ax = axes.flat[idx]
        values = row.tolist() + row.tolist()[:1]

        ax.plot(angles, values, "o-", linewidth=2, color=cmap(idx))
        ax.fill(angles, values, alpha=0.25, color=cmap(idx))
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels_short, size=7)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(["0.25", "0.50", "0.75"], size=6)
        ax.set_title(persona_names.get(cluster_id, f"Cluster {cluster_id+1}"),
                     size=10, fontweight="bold", pad=14, color=cmap(idx))

    for ax in axes.flat[n_clusters:]:
        ax.axis("off")

    plt.tight_layout()
    save(fig, "08_radar_persona_profiles")


def plot_cluster_demographics(df_with_labels: pd.DataFrame, persona_names: dict) -> None:
    """
    Show the demographic composition of each cluster for gender, age group,
    education, income, and work status.
    """
    print("  Plotting cluster demographic breakdown …")

    demo_vars = [
        ("gender",            "Gender"),
        ("age_group",         "Age Group"),
        ("edu_level_grouped", "Education"),
        ("income_label",      "Income Band"),
        ("work_status",       "Work Status"),
    ]

    n_clusters = df_with_labels["cluster"].nunique()
    cmap = cm.get_cmap("tab10", n_clusters)
    cluster_labels = sorted(df_with_labels["cluster"].unique())

    fig, axes = plt.subplots(1, len(demo_vars), figsize=(22, 7))
    fig.suptitle("Demographic Composition per Cluster", fontsize=14, fontweight="bold", y=1.01)

    for ax, (col, title) in zip(axes, demo_vars):
        if col not in df_with_labels.columns:
            ax.axis("off")
            continue
        ct = (df_with_labels
              .groupby(["cluster", col], observed=True)
              .size()
              .unstack(fill_value=0))
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
        ct_pct.index = [persona_names.get(i, f"C{i+1}") for i in ct_pct.index]
        ct_pct.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", legend=False)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Share (%)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    save(fig, "09_cluster_demographics")


def plot_cluster_financial_boxplots(df_with_labels: pd.DataFrame,
                                    persona_names: dict) -> None:
    """
    Box plots comparing the distribution of key financial scores across clusters.
    """
    print("  Plotting financial score distributions per cluster …")

    score_vars = [
        ("financial_planning_score",    "Planning Score"),
        ("saving_level_sophistication", "Saving Sophistication"),
        ("qk7_clean",                   "Knowledge Score"),
        ("risk_aversion_class",         "Risk Aversion"),
        ("digital_skills_score",        "Digital Skills"),
        ("finacial_situation",          "Financial Wellbeing"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Financial Scores by Persona Cluster", fontsize=14, fontweight="bold", y=1.01)

    n_clusters = df_with_labels["cluster"].nunique()
    cmap = cm.get_cmap("tab10", n_clusters)
    unique_clusters = sorted(df_with_labels["cluster"].unique())
    palette = {c: cmap(i) for i, c in enumerate(unique_clusters)}

    for ax, (col, title) in zip(axes.flat, score_vars):
        if col not in df_with_labels.columns:
            ax.axis("off")
            continue
        plot_df = df_with_labels[["cluster", col]].dropna()
        plot_df["persona"] = plot_df["cluster"].map(
            {k: v.replace("📈 ", "").replace("💻 ", "").replace("💳 ", "")
                 .replace("😴 ", "").replace("🛡️ ", "").replace("🌱 ", "")
             for k, v in persona_names.items()}
        )
        sns.boxplot(data=plot_df, x="cluster", y=col,
                    palette="tab10", ax=ax,
                    order=sorted(plot_df["cluster"].unique()))
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("")
        cluster_ticks = sorted(plot_df["cluster"].unique())
        ax.set_xticklabels(
            [persona_names.get(c, str(c))[:18] for c in cluster_ticks],
            rotation=25, ha="right", fontsize=8
        )

    plt.tight_layout()
    save(fig, "10_cluster_financial_boxplots")


def print_cluster_summary(df_with_labels: pd.DataFrame,
                          feature_names: list,
                          persona_names: dict) -> None:
    """
    Print a tabular summary of mean values for each cluster.
    """
    print("\n" + "="*70)
    print("CLUSTER PROFILE SUMMARY (means of SOM features)")
    print("="*70)
    avail = [f for f in feature_names if f in df_with_labels.columns]
    summary = (df_with_labels.groupby("cluster")[avail]
               .mean()
               .round(3))
    summary.index = [persona_names.get(i, f"Cluster {i}") for i in summary.index]
    print(summary.to_string())
    print("\nCluster sizes:")
    sizes = df_with_labels["cluster"].value_counts().sort_index()
    for c, n in sizes.items():
        print(f"  {persona_names.get(c, f'Cluster {c}'):35s}  n={n:5d} ({n/len(df_with_labels):.1%})")


# ════════════════════════════════════════════════════════════════════════════
# 8. KNOWLEDGE DEEP-DIVE
# ════════════════════════════════════════════════════════════════════════════

def plot_knowledge_deep_dive(df: pd.DataFrame) -> None:
    """
    Breakdown of individual financial knowledge questions (QK).
    Shows the share of correct answers for each objective item.
    """
    print("\n[8] Knowledge deep-dive …")

    qk_items = {
        "qk3_clean":  "Inflation (brothers)",
        "qk4_clean":  "Interest on loan",
        "qk5_clean":  "Simple interest",
        "qk6_clean":  "Compound interest",
        "qk10_clean": "Mortgage comparison",
    }
    available = {k: v for k, v in qk_items.items() if k in df.columns}
    if not available:
        print("  No QK columns found – skipping.")
        return

    correct_pct = {label: df[col].eq(1).mean() * 100
                   for col, label in available.items()}

    # Add QK7 sub-items using qk7_clean distribution as a proxy
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Financial Knowledge — Correct Answer Rate", fontsize=13, fontweight="bold")

    ax = axes[0]
    bars = ax.barh(list(correct_pct.keys()), list(correct_pct.values()),
                   color="#2C7BB6", edgecolor="white")
    ax.set_xlim(0, 105)
    ax.set_xlabel("% Correct")
    ax.set_title("Objective Knowledge Items", fontweight="bold")
    for bar, val in zip(bars, correct_pct.values()):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)

    ax2 = axes[1]
    if "qk7_clean" in df.columns:
        counts = df["qk7_clean"].dropna().value_counts().sort_index()
        ax2.bar(counts.index.astype(str), counts.values, color="#D7191C", edgecolor="white")
        ax2.set_xlabel("Score (0–6 correct T/F items)")
        ax2.set_ylabel("Count")
        ax2.set_title("True/False Battery Score Distribution", fontweight="bold")

    plt.tight_layout()
    save(fig, "11_knowledge_deepdive")


# ════════════════════════════════════════════════════════════════════════════
# 9. DIGITAL ENGAGEMENT DEEP-DIVE
# ════════════════════════════════════════════════════════════════════════════

def plot_digital_deep_dive(df: pd.DataFrame) -> None:
    """
    Analyse digital financial engagement:
    - Internet access by age group
    - Digital onboarding score distribution
    - Breakdown of online activity intensities
    """
    print("\n[9] Digital engagement deep-dive …")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Digital Financial Engagement", fontsize=13, fontweight="bold")

    # Panel 1: Internet access by age group
    ax = axes[0]
    if "internet_access_label" in df.columns and "age_group" in df.columns:
        ct = df.groupby(["age_group", "internet_access_label"], observed=True).size().unstack(fill_value=0)
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
        ct_pct.plot(kind="bar", stacked=True, ax=ax, color=["#d9534f", "#5cb85c"])
        ax.set_title("Internet Access by Age Group", fontweight="bold")
        ax.set_ylabel("Share (%)")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8)

    # Panel 2: Digital onboarding score distribution
    ax = axes[1]
    if "digital_onboarding_score" in df.columns:
        counts = df["digital_onboarding_score"].dropna().value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color="#4C72B0", edgecolor="white")
        ax.set_title("Digital Onboarding Score (0–5)", fontweight="bold")
        ax.set_xlabel("Products opened online")
        ax.set_ylabel("Count")

    # Panel 3: Online activity intensity scores
    ax = axes[2]
    intensity_cols = {
        "basic_admin_intensity": "Basic Admin",
        "daily_transactional_intensity": "Daily Transactional",
        "advanced_fintech_intensity": "Advanced Fintech",
    }
    avail = {k: v for k, v in intensity_cols.items() if k in df.columns}
    if avail:
        means = {label: df[col].mean() for col, label in avail.items()}
        stds  = {label: df[col].std()  for col, label in avail.items()}
        ax.bar(means.keys(), means.values(),
               yerr=list(stds.values()), color="#DD8452",
               edgecolor="white", capsize=5)
        ax.set_title("Mean Digital Activity Intensities", fontweight="bold")
        ax.set_ylabel("Score (mean ± std)")
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    save(fig, "12_digital_engagement")


# ════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load ────────────────────────────────────────────────────────────────
    df = load_data("cleaned_df.csv")

    # ── EDA ─────────────────────────────────────────────────────────────────
    eda_demographics(df)
    eda_financial_behaviour(df)
    correlation_heatmap(df)
    plot_knowledge_deep_dive(df)
    plot_digital_deep_dive(df)

    # ── SOM ─────────────────────────────────────────────────────────────────
    X_scaled, scaler, imputer, used_features, valid_idx = prepare_som_data(df)

    som = train_som(X_scaled, grid_size=12)

    best_k = find_optimal_k(X_scaled, k_range=range(3, 8))
    sample_labels, bmu_coords = cluster_som_neurons(som, X_scaled, best_k)
    plot_som_feature_planes(som, used_features)
    plot_som_feature_planes_with_samples(som, X_scaled, used_features, sample_labels)
    

    # ── Clustering on SOM neurons ────────────────────────────────────────────
    best_k = find_optimal_k(X_scaled, k_range=range(3, 8))
    sample_labels, bmu_coords = cluster_som_neurons(som, X_scaled, best_k)

    plot_som_umatrix(som, X_scaled, output_labels=sample_labels)
    plot_pca_clusters(X_scaled, sample_labels)

    # ── Attach cluster labels back to the dataframe ──────────────────────────
    df_clustered = df.loc[valid_idx].copy()
    df_clustered["cluster"] = sample_labels

    # ── Profile each cluster ─────────────────────────────────────────────────
    profile = (df_clustered
               .groupby("cluster")[[f for f in used_features if f in df_clustered.columns]]
               .mean())

    persona_names = assign_persona_names(profile)
    print("\nPersona names assigned:")
    for k, v in persona_names.items():
        print(f"  Cluster {k}: {v}")

    plot_radar_profiles(profile, persona_names, used_features)
    plot_cluster_demographics(df_clustered, persona_names)
    plot_cluster_financial_boxplots(df_clustered, persona_names)
    print_cluster_summary(df_clustered, used_features, persona_names)

    # ── Export labelled dataset ──────────────────────────────────────────────
    df_clustered["persona"] = df_clustered["cluster"].map(persona_names)
    out_path = "iacofi_clustered.csv"
    df_clustered.to_csv(out_path, index=False)
    print(f"\nLabelled dataset saved to: {out_path}")
    print(f"All plots saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()