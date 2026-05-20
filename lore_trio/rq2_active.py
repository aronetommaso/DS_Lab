import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from matplotlib.patches import Patch
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. DATA LOADING AND PREPARATION
# ==========================================
df = pd.read_csv(r'C:\Users\HP\Desktop\data_science\primo_anno\DSLab\DS_Lab\cleaned_active_df2.csv')
target = 'cyber_fraud_victim'

# Dummification and cleaning
X_raw = df.drop(columns=[target, 'wght'], errors='ignore')
X_num = pd.get_dummies(X_raw, drop_first=True).astype(float)
X_num = X_num.fillna(X_num.median())
y = df[target].fillna(0).astype(int)

# Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X_num, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# 2. DIMENSIONAL SPACES SELECTION
# ==========================================
# The 13 "Absolute Drivers" previously validated
core_features = [
    'institutional_friction',
    'credit_excluded',
    'expenditure_shock_capacity_yes_selling_assets',
    'edu_level_grouped_Middle School',
    'private_pension_asset',
    'qk6_clean',
    'saving_level_sophistication',
    'digital_onboarding_score',
    'advanced_fintech_intensity',
    'financial_planning_score',
    'household_size',
    'digital_skills_score',
    'saving_protection_score'
]

# Safe filtering: ensure names match the dummy-encoded columns
core_cols = [c for c in core_features if c in X_num.columns]

# Dynamically isolate all columns derived from Active variables
active_cols = [
    c for c in X_num.columns
    if 'shopping_behavior' in c or 'decision_driver' in c
]

# Create the two training spaces
X_train_core = X_train[core_cols]
X_test_core = X_test[core_cols]

X_train_full = X_train[core_cols + active_cols]
X_test_full = X_test[core_cols + active_cols]

# ==========================================
# 3. MODEL TRAINING (Logistic for ROC, RF for Importance)
# ==========================================
# Logistic models for pure predictive power comparison
lr_core = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

lr_full = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

lr_core.fit(X_train_core, y_train)
lr_full.fit(X_train_full, y_train)

fpr_core, tpr_core, _ = roc_curve(
    y_test,
    lr_core.predict_proba(X_test_core)[:, 1]
)

auc_core = auc(fpr_core, tpr_core)

fpr_full, tpr_full, _ = roc_curve(
    y_test,
    lr_full.predict_proba(X_test_full)[:, 1]
)

auc_full = auc(fpr_full, tpr_full)

# Random Forest for algorithmic explainability
rf_full = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)

rf_full.fit(X_train_full, y_train)

importances = pd.DataFrame({
    'Feature': X_train_full.columns,
    'Importance': rf_full.feature_importances_,
    'Type': [
        'Active Feature' if c in active_cols else 'Core Feature'
        for c in X_train_full.columns
    ]
}).sort_values(by='Importance', ascending=True)

# ==========================================
# 4. DATA VISUALIZATION (The "Mic Drop" Plot)
# ==========================================
# Clean corporate style
plt.style.use('default')

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

fig.suptitle(
    "Impact Analysis: The Role of 'Active' Variables in the Anti-Fraud Model",
    fontsize=18,
    fontweight='bold',
    y=0.98
)

# --- PANEL 1: ROC CURVES ---
axes[0].plot(
    fpr_core,
    tpr_core,
    label=f'CORE-ONLY Model (AUC = {auc_core:.3f})',
    color='#004c6d',
    linewidth=3
)

axes[0].plot(
    fpr_full,
    tpr_full,
    label=f'CORE + ACTIVE Model (AUC = {auc_full:.3f})',
    color='#ff7c43',
    linewidth=2,
    linestyle='--'
)

axes[0].plot([0, 1], [0, 1], color='gray', linestyle=':')

axes[0].set_title(
    'Predictive Power Comparison (ROC Curve)',
    fontsize=14,
    pad=15
)

axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)

axes[0].legend(loc='lower right', fontsize=12)
axes[0].grid(alpha=0.3, linestyle='--')

# --- PANEL 2: FEATURE IMPORTANCE ---
colors = [
    '#ff7c43' if t == 'Active Feature' else '#004c6d'
    for t in importances['Type']
]

axes[1].barh(
    importances['Feature'],
    importances['Importance'],
    color=colors,
    edgecolor='none'
)

axes[1].set_title(
    'Algorithmic Decision Weight (Random Forest)',
    fontsize=14,
    pad=15
)

axes[1].set_xlabel(
    "Mean Decrease in Impurity (Gini)",
    fontsize=12
)

# Custom legend
legend_elements = [
    Patch(
        facecolor='#004c6d',
        label='Core Variables (Background)'
    ),
    Patch(
        facecolor='#ff7c43',
        label='Active Variables (Shopping/Decision)'
    )
]

axes[1].legend(
    handles=legend_elements,
    loc='lower right',
    fontsize=12
)

axes[1].grid(axis='x', alpha=0.3, linestyle='--')

# Layout optimization and saving
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig(
    r'C:\Users\HP\Desktop\data_science\primo_anno\DSLab1active_variables_impact_analysis.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print(
    "Image successfully saved as "
    "'active_variables_impact_analysis.png'"
)