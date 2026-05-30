"""
RQ2 Production Pipeline: Statistical Inference, Advanced Multi-Model Tuning,
Cost-Sensitive Evaluation, False Negative Profiling, and Explainable AI (XAI).
Optimized production script translated to English for GitHub deployment.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2_contingency, mannwhitneyu
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Force console streams to use UTF-8 encoding to prevent Windows system crashes
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

# =========================================================================
# 0. GLOBAL PATH CONFIGURATIONS & BUSINESS CONSTANTS
# =========================================================================
DATA_PATH = r'C:\Users\HP\Desktop\data_science\primo_anno\DSLab\DS_Lab\cleaned_df2.csv'
OUTPUT_PLOTS_DIR = r'C:\Users\HP\Desktop\data_science\primo_anno\DSLab\plots_rq2'

if not os.path.exists(OUTPUT_PLOTS_DIR):
    os.makedirs(OUTPUT_PLOTS_DIR)
    print(f"[INFO] Created destination folder for plots: {OUTPUT_PLOTS_DIR}")

# Asymmetric Business Weights (Ratio 5:1 based on banking industry standards)
COST_FN = 5
COST_FP = 1
cost_weights = {0: COST_FP, 1: COST_FN}
THRESHOLD = 0.35

# =========================================================================
# 1. DATA LOADING & UNIVARIATE FEATURE FILTERING
# =========================================================================
print("\n[1/10] Loading dataset and executing univariate tests...")
df = pd.read_csv(DATA_PATH)
target = 'cyber_fraud_victim'

results = []
features = [col for col in df.columns if col != target and col != 'wght']

for col in features:
    temp_df = df[[col, target]].dropna()
    if temp_df.empty or len(temp_df[col].unique()) <= 1:
        continue
        
    col_type = temp_df[col].dtype
    p_value = np.nan
    test_used = ""
    
    if pd.api.types.is_numeric_dtype(col_type):
        group0 = temp_df[temp_df[target] == 0][col]
        group1 = temp_df[temp_df[target] == 1][col]
        if len(group0) > 0 and len(group1) > 0:
            stat, p_value = mannwhitneyu(group0, group1, alternative='two-sided')
            test_used = "Mann-Whitney U"
    else:
        contingency_table = pd.crosstab(temp_df[col], temp_df[target])
        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
            stat, p_value, dof, expected = chi2_contingency(contingency_table)
            test_used = "Chi-Square"
            
    if pd.api.types.is_numeric_dtype(col_type):
        mi_score = mutual_info_classif(temp_df[[col]], temp_df[target], random_state=42)[0]
    else:
        le = LabelEncoder()
        encoded_col = le.fit_transform(temp_df[col])
        mi_score = mutual_info_classif(encoded_col.reshape(-1, 1), temp_df[target], random_state=42)[0]
        
    results.append({
        'Feature': col,
        'Test_Used': test_used,
        'P_Value': p_value,
        'Significant (p<0.05)': p_value < 0.05,
        'Mutual_Information': mi_score
    })

evaluation_df = pd.DataFrame(results).sort_values(by=['Mutual_Information', 'P_Value'], ascending=[False, True])

# =========================================================================
# 2. BLACKLIST EXCLUSIONS, MULTICOLLINEARITY (VIF) & CORE FEATURE MATRICES
# =========================================================================
print("[2/10] Running Collinearity verification and separating baseline matrices...")
vars_to_exclude = [
    target, 'wght', 'consumer_debt_score', 'personal_budget_decisions', 
    'behaviour_investement-payment', 'is_italian', 'work_status', 
    'macro_region_label', 'gap_class', 'state_employee_pension', 
    'transactional_score', 'qk5_clean', 'internet_access_label', 
    'urban_area_label', 'traditional_investment_score', 'qk4_clean',
    'subj_knowledge_label', 'use_own_resources', 'age_group'
]

# Hardcoded core features to match the exact subset of the notebook pipeline
core_features = [
    'institutional_friction', 'credit_excluded', 'expenditure_shock_capacity_yes_selling_assets',
    'edu_level_grouped_Middle School', 'private_pension_asset', 'qk6_clean',
    'saving_level_sophistication', 'digital_onboarding_score', 'advanced_fintech_intensity',
    'financial_planning_score', 'household_size', 'digital_skills_score', 'saving_protection_score'
]

X_sig = df[[col for col in df.columns if col not in vars_to_exclude]].copy()
X_numeric = pd.get_dummies(X_sig, drop_first=True).astype(float).fillna(pd.get_dummies(X_sig, drop_first=True).median())

# Train/Test Split
y = df[target].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42, stratify=y)

# Extract subsets tailored to the top 13 features
X_train_13 = X_train[core_features]
X_test_13 = X_test[core_features]

# Generate SMOTE balanced variant for Method 1
smote = SMOTE(random_state=42)
X_train_smote_13, y_train_smote = smote.fit_resample(X_train_13, y_train)

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# =========================================================================
# 3. METHOD 1: SMOTE PIPELINE MODELLING VIA HYPERPARAMETER TUNING
# =========================================================================
print("[3/10] Executing GridSearch Tuning for Method 1 (SMOTE Models)...")

# Logistic Regression
log_reg = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
log_reg.fit(X_train_13, y_train)

# Decision Tree GridSearch
dt_pipe = ImbPipeline([('smote', SMOTE(random_state=42)), ('dt', DecisionTreeClassifier(class_weight='balanced', random_state=42))])
grid_dt = GridSearchCV(dt_pipe, {'dt__max_depth': [3, 5, 7], 'dt__min_samples_split': [2, 5, 10], 'dt__min_samples_leaf': [1, 5, 10], 'dt__criterion': ['gini', 'entropy']}, cv=5, scoring='recall', n_jobs=-1)
grid_dt.fit(X_train_13, y_train)
dt_best_model = DecisionTreeClassifier(criterion=grid_dt.best_params_['dt__criterion'], max_depth=grid_dt.best_params_['dt__max_depth'], min_samples_split=grid_dt.best_params_['dt__min_samples_split'], min_samples_leaf=grid_dt.best_params_['dt__min_samples_leaf'], class_weight="balanced", random_state=42)
dt_best_model.fit(X_train_smote_13, y_train_smote)

# Random Forest GridSearch
rf_pipe = ImbPipeline([('smote', SMOTE(random_state=42)), ('rf', RandomForestClassifier(random_state=42))])
grid_rf = GridSearchCV(rf_pipe, {'rf__n_estimators': [50, 100, 200], 'rf__max_depth': [4, 6, 8, None], 'rf__min_samples_split': [2, 5, 10], 'rf__max_features': ['sqrt', 'log2']}, cv=5, scoring='recall', n_jobs=-1)
grid_rf.fit(X_train_13, y_train)
rf_best_model = RandomForestClassifier(n_estimators=grid_rf.best_params_['rf__n_estimators'], max_depth=grid_rf.best_params_['rf__max_depth'], min_samples_split=grid_rf.best_params_['rf__min_samples_split'], max_features=grid_rf.best_params_['rf__max_features'], random_state=42)
rf_best_model.fit(X_train_smote_13, y_train_smote)

# Support Vector Machine GridSearch
grid_svm_smote = GridSearchCV(estimator=SVC(probability=True, random_state=42), param_grid={'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.001], 'kernel': ['rbf', 'sigmoid']}, scoring='f1', cv=cv_strategy, n_jobs=-1)
grid_svm_smote.fit(X_train_smote_13, y_train_smote)
svm_smote = SVC(C=grid_svm_smote.best_params_['C'], gamma=grid_svm_smote.best_params_['gamma'], kernel=grid_svm_smote.best_params_['kernel'], probability=True, random_state=42)
svm_smote.fit(X_train_smote_13, y_train_smote)

# =========================================================================
# 4. METHOD 2: COST-SENSITIVE MODELLING VIA ASYMMETRIC LOSS FUNCTION WEIGHTS
# =========================================================================
print("[4/10] Executing GridSearch Tuning for Method 2 (Cost-Sensitive Models)...")

# Decision Tree Cost-Sensitive GridSearch
grid_dt_cost = GridSearchCV(estimator=DecisionTreeClassifier(class_weight=cost_weights, random_state=42), param_grid={'criterion': ['gini', 'entropy'], 'max_depth': [3, 4, 5, 6, 8], 'min_samples_split': [2, 10, 20], 'min_samples_leaf': [1, 5, 10]}, scoring='f1', cv=cv_strategy, n_jobs=-1)
grid_dt_cost.fit(X_train_13, y_train)

# Random Forest Cost-Sensitive GridSearch
grid_rf_cost = GridSearchCV(estimator=RandomForestClassifier(class_weight=cost_weights, random_state=42, n_jobs=-1), param_grid={'n_estimators': [100, 200, 300], 'max_depth': [4, 6, 8, 10], 'min_samples_split': [2, 10, 20], 'max_features': ['sqrt', 'log2']}, scoring='f1', cv=cv_strategy, n_jobs=-1)
grid_rf_cost.fit(X_train_13, y_train)

# SVM Cost-Sensitive GridSearch
grid_svm_cost = GridSearchCV(estimator=SVC(class_weight=cost_weights, probability=True, random_state=42), param_grid={'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.001], 'kernel': ['rbf', 'sigmoid']}, scoring='f1', cv=cv_strategy, n_jobs=-1)
grid_svm_cost.fit(X_train_13, y_train)

# Training Cost-Sensitive Models
log_cost = LogisticRegression(class_weight=cost_weights, max_iter=2000, random_state=42).fit(X_train_13, y_train)
dt_cost = DecisionTreeClassifier(criterion=grid_dt_cost.best_params_['criterion'], max_depth=grid_dt_cost.best_params_['max_depth'], min_samples_split=grid_dt_cost.best_params_['min_samples_split'], min_samples_leaf=grid_dt_cost.best_params_['min_samples_leaf'], class_weight=cost_weights, random_state=42).fit(X_train_13, y_train)
rf_cost = RandomForestClassifier(n_estimators=grid_rf_cost.best_params_['n_estimators'], max_depth=grid_rf_cost.best_params_['max_depth'], min_samples_split=grid_rf_cost.best_params_['min_samples_split'], max_features=grid_rf_cost.best_params_['max_features'], class_weight=cost_weights, random_state=42, n_jobs=-1).fit(X_train_13, y_train)
svm_cost = SVC(C=grid_svm_cost.best_params_['C'], gamma=grid_svm_cost.best_params_['gamma'], kernel=grid_svm_cost.best_params_['kernel'], class_weight=cost_weights, probability=True, random_state=42).fit(X_train_13, y_train)

# =========================================================================
# 5. INFERENCE COMPONENT: ODDS RATIOS (COLOR-MAPPED PLOT)
# =========================================================================
print("[5/10] Calculating Odds Ratios and exporting inference profiles...")
X_train_sm = sm.add_constant(X_train_13)
try:
    result = sm.Logit(y_train, X_train_sm).fit(disp=False)
    infer_df = pd.DataFrame({'Feature': result.params.index, 'Odds_Ratio': np.exp(result.params.values), 'P_Value': result.pvalues.values}).query("Feature != 'const'")
    infer_df['Significant'] = infer_df['P_Value'] < 0.05
    infer_df = infer_df.sort_values(by='Odds_Ratio', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_or = ['#e74c3c' if val > 1 else '#2ecc71' for val in infer_df['Odds_Ratio']]
    sns.barplot(x='Odds_Ratio', y='Feature', data=infer_df, palette=colors_or, ax=ax)
    ax.axvline(1, color='black', linestyle='--', linewidth=1.2)
    ax.set_title("Logistic Regression Inference — Odds Ratio Profile\nRed: Increases Fraud Risk | Green: Protective Factor", fontsize=11)
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, 'logistic_regression_odds_ratios.png'), bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"[WARNING] Inference engine skipped: {e}")

# =========================================================================
# 6. BUSINESS LOGISTICS EVALUATION: PERFORMANCE & ERROR COSTS METRIC MATRIX
# =========================================================================
print("[6/10] Compiling global comparative database matrices...")

def evaluate_cost(name, approach, y_true, y_prob):
    y_pred = (y_prob >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = fn * COST_FN + fp * COST_FP
    return {
        'Model': name, 'Approach': approach, 'Recall': tp/(tp+fn) if (tp+fn)>0 else 0,
        'Precision': tp/(tp+fp) if (tp+fp)>0 else 0, 'AUC': auc(*roc_curve(y_true, y_prob)[:2]),
        'FN': fn, 'Cost_FN': fn * COST_FN, 'FP': fp, 'Cost_FP': fp * COST_FP, 'Total_Cost': total_cost, 'Probabilities': y_prob
    }

probs_cost = {'LogReg': log_cost.predict_proba(X_test_13)[:, 1], 'DT': dt_cost.predict_proba(X_test_13)[:, 1], 'RF': rf_cost.predict_proba(X_test_13)[:, 1], 'SVM': svm_cost.predict_proba(X_test_13)[:, 1]}
probs_smote = {'LogReg': log_reg.predict_proba(X_test_13)[:, 1], 'DT': dt_best_model.predict_proba(X_test_13)[:, 1], 'RF': rf_best_model.predict_proba(X_test_13)[:, 1], 'SVM': svm_smote.predict_proba(X_test_13)[:, 1]}

comparison_records = []
for k in probs_smote.keys(): comparison_records.append(evaluate_cost(k, 'SMOTE', y_test, probs_smote[k]))
for k in probs_cost.keys(): comparison_records.append(evaluate_cost(k, 'Cost Matrix', y_test, probs_cost[k]))
comp_df = pd.DataFrame(comparison_records)

# Plot: SMOTE vs Cost Matrix bar chart
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Methodology Comparison: SMOTE vs Cost Matrix", fontsize=13, fontweight='bold')
models_label = ['LogReg', 'DT', 'RF', 'SVM']
metrics_list = ['Total_Cost', 'Recall', 'AUC']
titles_list = ['Total Cost (lower is better)', 'Fraud Recall (higher is better)', 'AUC-ROC (higher is better)']
colors_dict = {'SMOTE': '#3498db', 'Cost Matrix': '#e74c3c'}

for ax, metric, title in zip(axes, metrics_list, titles_list):
    x_positions = np.arange(len(models_label))
    bar_width = 0.35
    for idx, approach in enumerate(['SMOTE', 'Cost Matrix']):
        vals = comp_df[comp_df['Approach'] == approach][metric].values
        ax.bar(x_positions + (idx - 0.5) * bar_width, vals, bar_width, label=approach, color=colors_dict[approach], alpha=0.85)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models_label)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, 'smote_vs_cost_matrix_metrics.png'), bbox_inches='tight')
plt.close()

# =========================================================================
# 7. BUSINESS GRAPHICS SUBPLOTS: MULTI-GRID CONFUSION MATRICES
# =========================================================================
print("[7/10] Exporting unified multi-grid confusion matrix plots...")

def save_conf_matrix_grid(probs_dict, suffix, cmap_name):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Confusion Matrices — {suffix} Configuration (Threshold = {THRESHOLD})", fontsize=13, fontweight='bold')
    titles = {'LogReg': 'Logistic Regression', 'DT': 'Decision Tree', 'RF': 'Random Forest', 'SVM': 'Support Vector Machine'}
    cmaps = {'LogReg': 'Blues', 'DT': 'Oranges', 'RF': 'Greens', 'SVM': 'Purples'}
    
    for ax, (name, y_prob) in zip(axes.flatten(), probs_dict.items()):
        y_pred = (y_prob >= THRESHOLD).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Safe', 'Fraud']).plot(ax=ax, colorbar=False, cmap=cmaps[name])
        ax.set_title(f"{titles[name]}\nRecall: {tp/(tp+fn):.0%} | Total Cost: {fn*COST_FN+fp*COST_FP:.0f}", fontsize=11)
        ax.set_xlabel("Predicted Value")
        ax.set_ylabel("Actual Value")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, f'confusion_matrices_{suffix.lower().replace(" ", "_")}.png'), bbox_inches='tight')
    plt.close()

save_conf_matrix_grid(probs_smote, "SMOTE Models", "Blues")
save_conf_matrix_grid(probs_cost, "Cost Matrix Models", "Reds")

# =========================================================================
# 8. CORE PERFORMANCE CHARTS: GLOBAL PERFORMANCE DUAL-ROC SPECTRA
# =========================================================================
print("[8/10] Generating multi-panel ROC curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("ROC Curve Profile Matrix — Model Spectrum Evaluation", fontsize=13, fontweight='bold')
palette_colors = {'LogReg': '#e74c3c', 'DT': '#f39c12', 'RF': '#2ecc71', 'SVM': '#9b59b6'}

for ax_idx, (model_probs, title) in enumerate([(probs_smote, 'SMOTE'), (probs_cost, 'Cost Matrix')]):
    for name, prob in model_probs.items():
        fpr, tpr, thresholds = roc_curve(y_test, prob)
        axes[ax_idx].plot(fpr, tpr, linewidth=2, color=palette_colors[name], label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
        thresh_idx = np.argmin(np.abs(thresholds - THRESHOLD))
        axes[ax_idx].scatter(fpr[thresh_idx], tpr[thresh_idx], color=palette_colors[name], s=60, zorder=5)
    axes[ax_idx].plot([0, 1], [0, 1], '--', linewidth=1, color='grey', label='Baseline')
    axes[ax_idx].set_xlim([0, 1])
    axes[ax_idx].set_ylim([0, 1.05])
    axes[ax_idx].set_xlabel("False Positive Rate")
    axes[ax_idx].set_ylabel("True Positive Rate")
    axes[ax_idx].set_title(f"Framework: {title}", fontsize=11)
    axes[ax_idx].legend(loc='lower right')
    axes[ax_idx].grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, 'global_roc_comparison.png'), bbox_inches='tight')
plt.close()

# Combined individual benchmark comparisons for champions (RF and SVM)
for name, m_smote, m_cost, col in [('Random Forest', rf_best_model, rf_cost, '#2ecc71'), ('SVM', svm_smote, svm_cost, '#9b59b6')]:
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, model, ls in [(f'{name} — SMOTE', m_smote, '-'), (f'{name} — Cost Matrix', m_cost, '--')]:
        y_prob = model.predict_proba(X_test_13)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, linestyle=ls, linewidth=2, color=col, label=f"{label} (AUC = {auc(fpr, tpr):.3f})")
        thresh_idx = np.argmin(np.abs(thresholds - THRESHOLD))
        ax.scatter(fpr[thresh_idx], tpr[thresh_idx], color=col, s=80, zorder=5)
    ax.plot([0, 1], [0, 1], '--', color='grey', label='Baseline')
    ax.set_title(f"{name} Spectrum Matrix Benchmark", fontsize=11)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, f"roc_comparison_{name.lower().replace(' ', '_')}.png"), bbox_inches='tight')
    plt.close()

# =========================================================================
# 9. BUSINESS VALUE VALIDATION: LIFT CHART & CUMULATIVE GAINS CAPTURE
# =========================================================================
print("[9/10] Computing lift spectrum coordinates and generating plots...")

def compute_lift_curve(y_true, y_prob, n_bins=20):
    lift_df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob}).sort_values('y_prob', ascending=False).reset_index(drop=True)
    total_fraud = y_true.sum()
    bin_size = len(y_true) // n_bins
    records = []
    for i in range(n_bins):
        cum_chunk = lift_df.iloc[: (i + 1) * bin_size]
        pct_population = (i + 1) / n_bins
        cum_fraud_caught = cum_chunk['y_true'].sum() / total_fraud
        records.append({'pct_population': pct_population, 'cum_fraud_caught': cum_fraud_caught, 'lift': cum_fraud_caught / pct_population})
    return pd.DataFrame(records)

lift_dataset = {name: compute_lift_curve(np.array(y_test), prob) for name, prob in probs_cost.items()}
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Lift Spectrum Analysis Framework — Cost-Sensitive Systems", fontsize=13, fontweight='bold')

for name, df_l in lift_dataset.items():
    axes[0].plot(df_l['pct_population'], df_l['cum_fraud_caught'], marker='o', markersize=4, label=name, color=palette_colors[name])
    axes[1].plot(df_l['pct_population'], df_l['lift'], marker='o', markersize=4, label=name, color=palette_colors[name])

axes[0].plot([0, 1], [0, 1], '--', color='grey', label='Random Baseline')
axes[0].fill_between([0, 0.2], [0, 0.2], [0, lift_dataset['RF']['cum_fraud_caught'].iloc[3]], alpha=0.08, color='#2ecc71')
axes[0].set_xlabel("% Population Contacted (Risk Ranked)")
axes[0].set_ylabel("% Frauds Captured")
axes[0].set_title("Cumulative Gains Distribution Profile")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].axhline(1, linestyle='--', color='grey', label='Baseline (Lift = 1)')
axes[1].set_xlabel("% Population Contacted")
axes[1].set_ylabel("Lift Multiplier Factor")
axes[1].set_title("Lift Spectrum Evaluation Curve")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, 'lift_and_cumulative_gains_charts.png'), bbox_inches='tight')
plt.close()

# =========================================================================
# 10. DIAGNOSTIC INTERPRETABILITY: FALSE NEGATIVE PROFILING, BOUNDARIES & XAI
# =========================================================================
print("[10/10] Running advanced False Negative Profiling and generating diagnostic plots...")

# --- Diagnostic Sub-component 1: False Negative Profiling Matrix ---
y_prob_log_diagnostic = probs_smote['LogReg']
y_pred_diagnostic = (y_prob_log_diagnostic >= THRESHOLD).astype(int)
y_test_arr = np.array(y_test)

mask_TP = (y_test_arr == 1) & (y_pred_diagnostic == 1)
mask_FN = (y_test_arr == 1) & (y_pred_diagnostic == 0)

df_test = X_test_13.copy().reset_index(drop=True)
df_test['quadrant'] = 'Other'
df_test.loc[mask_TP, 'quadrant'] = 'TP'
df_test.loc[mask_FN, 'quadrant'] = 'FN'

df_TP = df_test[df_test['quadrant'] == 'TP']
df_FN = df_test[df_test['quadrant'] == 'FN']

num_features = ['digital_onboarding_score', 'advanced_fintech_intensity', 'financial_planning_score', 'digital_skills_score', 'saving_level_sophistication', 'saving_protection_score']
cat_features = ['institutional_friction', 'credit_excluded', 'private_pension_asset', 'expenditure_shock_capacity_yes_selling_assets']

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)
colors_profile = {'TP': '#2ecc71', 'FN': '#e74c3c'}

for idx, feat in enumerate(num_features[:6]):
    row, col = divmod(idx, 3)
    ax = fig.add_subplot(gs[row, col])
    bp = ax.boxplot([df_TP[feat].dropna(), df_FN[feat].dropna()], patch_artist=True, widths=0.5, medianprops=dict(color='white', linewidth=2))
    for patch, c in zip(bp['boxes'], [colors_profile['TP'], colors_profile['FN']]):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['TP\n(caught)', 'FN\n(missed)'])
    ax.set_title(f"Divergence: {feat}", fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

ax_bar = fig.add_subplot(gs[2, :])
x_axis_cat = np.arange(len(cat_features))
bar_w = 0.35
pct_tp = [df_TP[feat].mean() for feat in cat_features]
pct_fn = [df_FN[feat].mean() for feat in cat_features]

ax_bar.bar(x_axis_cat - bar_w/2, pct_tp, bar_w, label='TP (caught)', color=colors_profile['TP'], alpha=0.85)
ax_bar.bar(x_axis_cat + bar_w/2, pct_fn, bar_w, label='FN (missed)', color=colors_profile['FN'], alpha=0.85)
ax_bar.set_xticks(x_axis_cat)
ax_bar.set_xticklabels(cat_features, fontsize=9)
ax_bar.set_title("Categorical Distributions: True Positives vs False Negatives", fontsize=11, fontweight='bold')
ax_bar.legend()
ax_bar.grid(axis='y', alpha=0.3)
plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, 'false_negative_profiling_matrix.png'), bbox_inches='tight')
plt.close()

# --- Diagnostic Sub-component 2: SVM Decision Boundary Visualizer ---
try:
    top2_features = ['advanced_fintech_intensity', 'digital_onboarding_score']
    X_train_2d = X_train_13[top2_features].values
    X_test_2d = X_test_13[top2_features].values
    svm_2d = SVC(C=grid_svm_cost.best_params_['C'], gamma=grid_svm_cost.best_params_['gamma'], kernel=grid_svm_cost.best_params_['kernel'], class_weight=cost_weights, probability=True, random_state=42).fit(X_train_2d, y_train)
    fig, ax = plt.subplots(figsize=(8, 6))
    DecisionBoundaryDisplay.from_estimator(svm_2d, X_test_2d, response_method='predict_proba', plot_method='pcolormesh', alpha=0.3, ax=ax, cmap='RdYlGn')
    ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=np.array(y_test), cmap='RdYlGn', edgecolors='k', linewidths=0.4, s=25, alpha=0.7)
    ax.set_xlabel(top2_features[0])
    ax.set_ylabel(top2_features[1])
    ax.set_title("Support Vector Machine Decision Space Boundary Mapping", fontsize=11)
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, 'svm_decision_boundary_space.png'), bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"[WARNING] SVM Boundary plot skipped: {e}")

# --- Diagnostic Sub-component 3: Pitchable Executive Decision Tree Visualizer ---
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(dt_cost, feature_names=core_features, class_names=['Safe', 'Fraud'], filled=True, rounded=True, fontsize=9, impurity=False, proportion=False, ax=ax)
ax.set_title("Explainable Structural Decision Tree Architecture", fontsize=12, fontweight='bold')
plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, 'explainable_decision_tree_graph.png'), bbox_inches='tight')
plt.close()

# --- Diagnostic Sub-component 4: XAI MDI Feature Importance Metrics Matrix ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Explainable AI (XAI) Matrix: Random Forest Feature Importances Comparison", fontsize=13, fontweight='bold')
for ax, (model, title, col) in zip(axes, [(rf_best_model, 'SMOTE Framework', '#3498db'), (rf_cost, 'Cost Matrix Strategy', '#e74c3c')]):
    importances = pd.Series(model.feature_importances_, index=core_features).sort_values(ascending=True)
    importances.plot(kind='barh', ax=ax, color=col, alpha=0.85)
    ax.set_title(f"Framework: {title}", fontsize=11)
    ax.axvline(importances.mean(), linestyle='--', color='black', linewidth=1, label=f'Mean = {importances.mean():.3f}')
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)
plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, 'xai_random_forest_feature_importance.png'), bbox_inches='tight')
plt.close()

print("\n============================================================")
print("[SUCCESS] PIPELINE PROCESSED, METRICS ARTIFACTS EXPORTED TO .PY")
print("============================================================\n")