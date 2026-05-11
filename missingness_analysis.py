"""
IACOFI 2023 - Missingness & Reticence Analysis
==============================================
This script profiles non-respondents (Missing Not At Random - MNAR) through:
  1. Target Definition (Income missingness or Global missingness index)
  2. Statistical Inference (Chi-Square and Mann-Whitney U)
  3. Predictive Modeling & XAI (Random Forest + SHAP)
  4. Unsupervised Sub-clustering (SOM + K-Means on 'Unknowns')
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from minisom import MiniSom

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: 'shap' library not found. Please run 'pip install shap' to enable XAI plots.")

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.dpi": 130,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUTPUT_DIR = "plot/missingness_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_plot(fig: plt.Figure, name: str) -> None:
    """Helper function to save figures."""
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [Saved] {path}")

# ════════════════════════════════════════════════════════════════════════════
# PHASE 0: OVERALL MISSINGNESS
# ════════════════════════════════════════════════════════════════════════════

def plot_column_missingness(df: pd.DataFrame, top_n: int = 25) -> None:
    """
    Calculates and plots the percentage of missing values for the top N columns.
    """
    print("\n[Phase 0] Analyzing column-wise missingness ...")

    missing_pct = df.isna().mean() * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)

    if missing_pct.empty:
        print("  No missing values found in the dataset. Skipping plot.")
        return

    top_missing = missing_pct.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = sns.barplot(x=top_missing.values, y=top_missing.index, ax=ax, palette='viridis_r', orient='h')

    ax.set_title(f'Top {top_n} Colonne con più Valori Mancanti', fontweight="bold", fontsize=14)
    ax.set_xlabel('Percentuale di Valori Mancanti (%)', fontsize=11)
    ax.set_ylabel('')
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    # Add percentage labels
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.3, p.get_y() + p.get_height() / 2, f'{width:.1f}%', va='center', ha='left', fontsize=9)

    save_plot(fig, "00_column_missingness_overview")

# ════════════════════════════════════════════════════════════════════════════
# PHASE 1: TARGET DEFINITION
# ════════════════════════════════════════════════════════════════════════════

def define_missingness_target(df: pd.DataFrame, strategy: str = 'income', threshold: float = 0.15) -> pd.DataFrame:
    """
    Defines the binary target variable 'is_unknown' based on the chosen strategy.
    - 'income': Focuses on respondents who refused to disclose their income.
    - 'global': Computes a missingness index across all columns. If the ratio of NaNs 
                exceeds the threshold, the respondent is flagged.
    """
    print(f"\n[Phase 1] Defining missingness target (Strategy: {strategy}) ...")
    df = df.copy()
    
    if strategy == 'income':
        if 'income_label' not in df.columns:
            raise ValueError("Column 'income_label' not found. Ensure demographic features are engineered.")
        # Flag individuals mapped to 'Unknown' or NaN
        df['is_unknown'] = ((df['income_label'] == 'Unknown') | (df['income_label'].isna())).astype(int)
    
    elif strategy == 'global':
        # Calculate the proportion of missing values per row
        missingness_index = df.isna().mean(axis=1)
        df['is_unknown'] = (missingness_index >= threshold).astype(int)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(missingness_index, bins=30, kde=True, color='purple', ax=ax)
        ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
        ax.set_title("Global Missingness Index Distribution", fontweight="bold")
        ax.set_xlabel("Proportion of Missing Values")
        ax.set_ylabel("Frequency")
        ax.legend()
        save_plot(fig, "01_global_missingness_distribution")
        
    else:
        raise ValueError("Strategy must be either 'income' or 'global'.")
        
    print(f"  Target defined: {df['is_unknown'].sum()} 'Unknowns' found out of {len(df)} samples ({(df['is_unknown'].mean()*100):.1f}%).")
    return df

# ════════════════════════════════════════════════════════════════════════════
# PHASE 2: STATISTICAL INFERENCE
# ════════════════════════════════════════════════════════════════════════════

def perform_bivariate_inference(df: pd.DataFrame, cat_vars: list, num_vars: list) -> None:
    """
    Performs rigorous statistical testing between the 'is_unknown' target and 
    demographic/behavioural features using Chi-Square and Mann-Whitney U tests.
    """
    print("\n[Phase 2] Performing bivariate statistical inference ...")
    
    # 1. Categorical Variables (Chi-Square Test)
    print("\n  Categorical Variables (Chi-Square Independence Test):")
    for cat in cat_vars:
        if cat in df.columns:
            contingency_table = pd.crosstab(df[cat], df['is_unknown'])
            
            # Check if there is enough data for the Chi-Square test
            if contingency_table.empty or contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                print(f"    - {cat:25s} | Skipped: not enough valid data/variance")
                continue
                
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"    - {cat:25s} | Chi2: {chi2:8.2f} | p-value: {p:.4e} {significance}")
            
            # Plotting missingness rate for significant categorical vars
            if p < 0.05:
                ct_pct = pd.crosstab(df[cat], df['is_unknown'], normalize='index') * 100
                if 1 in ct_pct.columns:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ct_pct[1].sort_values(ascending=False).plot(kind='bar', color='#E24A33', edgecolor='white', ax=ax)
                    ax.set_title(f"Missingness Rate by {cat.replace('_', ' ').title()}\n(Chi-Square p-value: {p:.4e})", fontweight="bold")
                    ax.set_ylabel("% Unknown")
                    plt.xticks(rotation=45, ha='right')
                    save_plot(fig, f"02_cat_missingness_{cat}")
            
    # 2. Numerical Variables (Mann-Whitney U Test)
    print("\n  Numerical Variables (Mann-Whitney U Test):")
    group_known = df[df['is_unknown'] == 0]
    group_unknown = df[df['is_unknown'] == 1]
    
    for num in num_vars:
        if num in df.columns:
            val_known = group_known[num].dropna()
            val_unknown = group_unknown[num].dropna()
            
            if len(val_known) > 0 and len(val_unknown) > 0:
                stat, p = mannwhitneyu(val_known, val_unknown, alternative='two-sided')
                significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                diff_medians = val_unknown.median() - val_known.median()
                print(f"    - {num:25s} | Diff Medians: {diff_medians:>5.2f} | p-value: {p:.4e} {significance}")
                
                # Intuitive Plotting (Boxplot + KDE) for significant numeric vars
                if p < 0.05:
                    fig, (ax_box, ax_kde) = plt.subplots(2, 1, figsize=(7, 6), gridspec_kw={"height_ratios": [1, 4]}, sharex=True)
                    plot_df = df[[num, 'is_unknown']].dropna()
                    plot_df['Group'] = plot_df['is_unknown'].map({0: 'Known', 1: 'Unknown (Reticent)'})
                    
                    sns.boxplot(data=plot_df, x=num, y='Group', palette=["#2C7BB6", "#D7191C"], ax=ax_box)
                    ax_box.set(xlabel='', ylabel='')
                    
                    sns.kdeplot(data=plot_df, x=num, hue='Group', fill=True, palette=["#2C7BB6", "#D7191C"], ax=ax_kde, common_norm=False)
                    ax_kde.set_title(f"Distribution Divergence: {num.replace('_', ' ').title()}\n(Mann-Whitney U p-value: {p:.4e})", fontweight="bold")
                    ax_kde.set_xlabel(num.replace('_', ' ').title())
                    save_plot(fig, f"02_dist_{num}")

# ════════════════════════════════════════════════════════════════════════════
# PHASE 3: PREDICTIVE MODELING & XAI (SHAP)
# ════════════════════════════════════════════════════════════════════════════

def train_and_explain_missingness(df: pd.DataFrame, features: list) -> None:
    """
    Trains a Random Forest classifier to predict reticence and uses SHAP
    to extract the marginal contribution of each feature (Explainable AI).
    """
    print("\n[Phase 3] Predictive Modeling and SHAP Explanations ...")
    
    if not HAS_SHAP:
        print("  Skipping Phase 3 because 'shap' is not installed.")
        return
        
    avail_feats = [f for f in features if f in df.columns]
    X = df[avail_feats].copy()
    y = df['is_unknown'].values
    
    # Impute missing features for the model
    imputer = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Train a Random Forest
    print("  Training Random Forest Ensemble ...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight='balanced')
    rf.fit(X_imp, y)
    
    print("  Calculating SHAP values ...")
    explainer = shap.TreeExplainer(rf)
    
    # We sample if the dataset is too large to speed up SHAP plotting
    sample_idx = np.random.choice(X_imp.shape[0], min(X_imp.shape[0], 2000), replace=False)
    X_sample = X_imp.iloc[sample_idx].copy()
    
    # Clean feature names for better readability in the plots
    X_sample.columns = [c.replace('_', ' ').title() for c in X_sample.columns]
    
    # For RandomForest in SHAP, shap_values is a list for each class. We take index 1 (Target=1)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values_target = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        shap_values_target = shap_values[:, :, 1]
    else:
        shap_values_target = shap_values
        
    # 1. Custom Bar Plot (Global Feature Importance)
    plt.figure(figsize=(10, 7))
    mean_abs_shap = np.abs(shap_values_target).mean(axis=0)
    shap_df = pd.DataFrame({
        'Feature': X_sample.columns,
        'Importance': mean_abs_shap
    }).sort_values(by='Importance', ascending=False).head(15)
    
    ax = sns.barplot(data=shap_df, x='Importance', y='Feature', palette='mako')
    for p in ax.patches:
        ax.annotate(f"{p.get_width():.3f}", 
                    (p.get_width(), p.get_y() + p.get_height() / 2.), 
                    va='center', xytext=(5, 0), textcoords='offset points', 
                    fontsize=9, fontweight='bold', color='#333333')
                    
    plt.title("Top 15 Drivers of Reticence (Global Feature Importance)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Mean |SHAP Value| (Average impact on predicting 'Unknown')", fontsize=11)
    plt.ylabel("")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_plot(plt.gcf(), "03a_shap_feature_importance_bar")
    
    # 2. Beeswarm Plot (Directional Impact)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_target, X_sample, max_display=15, show=False)
    fig2 = plt.gcf()
    plt.suptitle("How Features Impact Reticence (SHAP Beeswarm)", fontsize=15, fontweight="bold", y=1.02)
    plt.title("Color indicates feature value (Red=High, Blue=Low).\nDots on the RIGHT increase the probability of being a Non-Respondent.", 
              fontsize=10, style='italic', pad=15)
    plt.tight_layout()
    save_plot(fig2, "03b_shap_beeswarm_plot")

# ════════════════════════════════════════════════════════════════════════════
# PHASE 4: UNSUPERVISED SUB-CLUSTERING OF "UNKNOWNS"
# ════════════════════════════════════════════════════════════════════════════

def plot_som_feature_planes(som: MiniSom, X_scaled: np.ndarray, feature_names: list, sample_labels: np.ndarray = None) -> None:
    """
    Plot one activation plane per input feature for the 'Unknowns' subset.
    Each cell shows the weight for that feature in the neuron's prototype vector.
    High values = neurons whose prototypes are high on that feature.
    Overlay the dataset points colored by their archetype cluster.
    """
    print("  Plotting SOM feature planes for 'Unknowns' with sample overlay ...")
    weights = som.get_weights()          # shape: (x, y, n_features)
    n_feat = len(feature_names)
    ncols = min(5, n_feat)
    nrows = int(np.ceil(n_feat / ncols)) if ncols > 0 else 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.2))
    fig.suptitle("SOM Feature Planes (Unknowns) — Archetypes Overlay", fontsize=15, fontweight="bold", y=1.02)
    axes_flat = axes.flat if isinstance(axes, np.ndarray) else [axes]

    if sample_labels is not None:
        n_clusters = len(np.unique(sample_labels))
        cmap_clusters = cm.get_cmap("tab10", n_clusters)
        bmus = np.array([som.winner(x) for x in X_scaled])
        jitter_x = np.random.uniform(-0.35, 0.35, size=len(bmus))
        jitter_y = np.random.uniform(-0.35, 0.35, size=len(bmus))
        x_coords = bmus[:, 0] + 0.5 + jitter_x
        y_coords = bmus[:, 1] + 0.5 + jitter_y

    for i, (ax, feat) in enumerate(zip(axes_flat, feature_names)):
        plane = weights[:, :, i]
        im = ax.pcolor(plane.T, cmap="RdYlGn", alpha=0.6 if sample_labels is not None else 1.0)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        if sample_labels is not None:
            for c in range(n_clusters):
                mask = sample_labels == c
                ax.scatter(x_coords[mask], y_coords[mask],
                           color=cmap_clusters(c), s=6, alpha=0.8, edgecolors='none')
                           
        ax.set_title(feat.replace("_", " ").title(), fontsize=8, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_flat[n_feat:]:
        ax.axis("off")

    if sample_labels is not None:
        patches = [mpatches.Patch(color=cmap_clusters(c), label=f"Archetype {c+1}") for c in range(n_clusters)]
        fig.legend(handles=patches, loc="lower center", ncol=n_clusters, bbox_to_anchor=(0.5, -0.05), fontsize=10)

    plt.tight_layout()
    save_plot(fig, "04_som_feature_planes_unknowns")

def find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 6)) -> int:
    """
    Use the Silhouette Score to select the optimal number of K-Means clusters.
    Returns the k with the highest average silhouette score.
    """
    print("  Searching for optimal number of clusters ...")
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)
        print(f"    k={k}: silhouette={scores[k]:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"  -> Best k = {best_k} (silhouette={scores[best_k]:.4f})")

    # Plot silhouette scores
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(scores.keys()), list(scores.values()), "o-", color="#2C7BB6", linewidth=2)
    ax.axvline(best_k, color="red", linestyle="--", label=f"Optimal k={best_k}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Average Silhouette Score")
    ax.set_title("Silhouette Analysis (Unknowns) — Choosing Optimal k", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    save_plot(fig, "04a_silhouette_scores_unknowns")

    return best_k

def cluster_unknown_profiles(df: pd.DataFrame, som_features: list) -> None:
    """
    Applies SOM and K-Means strictly on the subset of respondents labeled as 'Unknown'.
    This isolates the different archetypes of reticence (e.g. Privacy-Conscious vs Insecure).
    """
    print("\n[Phase 4] Sub-Clustering the 'Unknown' Profiles (SOM + K-Means) ...")
    
    df_unk = df[df['is_unknown'] == 1].copy()
    if len(df_unk) < 50:
        print("  Not enough 'Unknown' samples to perform reliable clustering.")
        return
        
    avail_feats = [f for f in som_features if f in df_unk.columns]
    
    # Preprocessing
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(df_unk[avail_feats])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    
    # SOM Training
    grid_size = 8
    print(f"  Training SOM ({grid_size}x{grid_size}) for Sub-Clusters ...")
    som = MiniSom(grid_size, grid_size, X_scaled.shape[1], sigma=1.5, learning_rate=0.5, random_seed=42)
    som.random_weights_init(X_scaled)
    som.train_batch(X_scaled, len(X_scaled) * 10, verbose=False)
    
    # Find optimal K and apply K-Means on SOM prototypes
    n_subclusters = find_optimal_k(X_scaled, k_range=range(2, 6))
    print(f"  Applying K-Means (k={n_subclusters}) to extract Reticent Archetypes ...")
    weights_flat = som.get_weights().reshape(-1, X_scaled.shape[1])
    km = KMeans(n_clusters=n_subclusters, random_state=42)
    neuron_labels = km.fit_predict(weights_flat)
    
    # Map samples to cluster labels
    bmus = np.array([som.winner(x) for x in X_scaled])
    sample_labels = np.array([neuron_labels[bmu[0] * grid_size + bmu[1]] for bmu in bmus])
    df_unk['reticent_archetype'] = sample_labels
    
    # Plot the SOM planes using the new function with samples overlaid
    plot_som_feature_planes(som, X_scaled, avail_feats, sample_labels)
    
    # Plot Radar Chart for the sub-clusters
    profile = df_unk.groupby('reticent_archetype')[avail_feats].mean()
    norm_profile = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)
    
    angles = np.linspace(0, 2 * np.pi, len(avail_feats), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, axes = plt.subplots(1, n_subclusters, figsize=(n_subclusters * 5, 5), subplot_kw=dict(polar=True))
    fig.suptitle("Archetypes of Non-Respondents (Reticent Profiles)", fontsize=15, fontweight="bold", y=1.05)
    
    labels_short = [c.replace("_", " ")[:15] for c in avail_feats]
    colors = ['#E24A33', '#31A354', '#7B3294', '#2C7BB6']
    
    for idx, (cluster_id, row) in enumerate(norm_profile.iterrows()):
        ax = axes[idx] if n_subclusters > 1 else axes
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles, values, "o-", linewidth=2, color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[idx % len(colors)])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels_short, size=7)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_title(f"Reticent Archetype {cluster_id+1}\n(n={len(df_unk[df_unk['reticent_archetype']==cluster_id])})", 
                     size=11, fontweight="bold", pad=15)
        
    plt.tight_layout()
    save_plot(fig, "04_reticent_archetypes_radar")


def run_missingness_pipeline():
    """Main orchestrator function for the Missingness Analysis."""
    print("Loading data for missingness analysis...")
    # Assumes cleaned_df2.csv exists in the working directory
    if not os.path.exists("cleaned_df2.csv"):
        print("Dataset not found. Please run the preprocessing pipeline first.")
        return
        
    df = pd.read_csv("cleaned_df2.csv", low_memory=False)
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Phase 0: Plot overall missingness
    plot_column_missingness(df, top_n=25)

    # Target Definition: You can change strategy to 'global' and adjust the threshold
    df = define_missingness_target(df, strategy='income')
    
    # Statistical Inference
    cat_vars = [
        'gender', 'age_group', 'edu_level_grouped', 'work_status', 
        'macro_region_label', 'urban_area_label', 'living_status', 
        'is_italian', 'internet_access_label'
    ]
    num_vars = [
        'digital_skills_score', 'financial_planning_score', 'qk7_clean', 
        'risk_aversion_class', 'saving_level_sophistication', 
        'transactional_score', 'saving_protection_score', 'consumer_debt_score', 
        'traditional_investment_score', 'alternative_asset_score',
        'digital_onboarding_score', 'basic_admin_intensity', 
        'daily_transactional_intensity', 'advanced_fintech_intensity', 
        'finacial_situation', 'behaviour_investement-payment', 
        'knowledge_financial_privacy_digital', 'household_size'
    ]
    perform_bivariate_inference(df, cat_vars, num_vars)
    
    # XAI & SHAP
    train_and_explain_missingness(df, features=num_vars)
    
    # Sub-clustering
    cluster_unknown_profiles(df, som_features=num_vars)

if __name__ == "__main__":
    run_missingness_pipeline()