

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

def preprocess_complete_iacofi(df):
    """
    Comprehensive preprocessing for IACOFI 2023 dataset.
    1. Maps ALL variables from the codebook to descriptive names.
    2. Consolidates all multiple-choice "tick all that apply" variables 
       (with _1, _2, etc.) into single categorical string variables.
    """
    # Convert all columns to lowercase to ensure matching[cite: 1]
    df.columns = [col.lower() for col in df.columns]

    # --- STEP 1: CONSOLIDATE MULTI-OPTION VARIABLES ---
    # Defining groups where 1='Yes', 0='No'. 
    # These will be merged into a single categorical variable per group[cite: 1].
    multi_option_groups = {
        'qd5': { # Household composition
            'qd5_1': 'alone', 'qd5_2': 'partner', 'qd5_3': 'children_under_18',
            'qd5_4': 'children_over_18', 'qd5_5': 'adult_relatives', 'qd5_6': 'friends',
            'qd5_7': 'other_adults'
        },
        'qf2': { # Budgeting behaviors
            'qf2_1': 'plan_budget', 'qf2_2': 'note_spending', 'qf2_3': 'separate_bills_money',
            'qf2_4': 'note_upcoming_bills', 'qf2_5': 'use_banking_app', 'qf2_6': 'auto_payments'
        },
        'qf3': { # Active saving
            'qf3_1': 'cash_at_home', 'qf3_2': 'deposit_account', 'qf3_3': 'family_save',
            'qf3_4': 'informal_club', 'qf3_5': 'bonds', 'qf3_6': 'crypto', 'qf3_7': 'stocks',
            'qf3_8': 'other', 'qf3_81': 'other_financial_instruments', 'qf3_98': 'did_not_save'
        },
        'qf9': { # Retirement plans
            'qf9_1': 'gov_pension', 'qf9_2': 'occupational_pension', 'qf9_3': 'private_pension',
            'qf9_4': 'sell_financial_assets', 'qf9_5': 'sell_non_financial_assets',
            'qf9_6': 'asset_income', 'qf9_7': 'spouse_support', 'qf9_8': 'family_support',
            'qf9_9': 'savings', 'qf9_10': 'continue_work', 'qf9_11': 'business_revenue',
            'qf9_12': 'reversibility_pension'
        },
        'qf12': { # Making ends meet
            'qf12_1_1': 'draw_savings', 'qf12_1_2': 'cut_spending', 'qf12_1_3': 'sell_owned',
            'qf12_2_1': 'work_overtime', 'qf12_2_2': 'gov_support', 'qf12_2_3': 'ask_family',
            'qf12_3_1': 'borrow_family', 'qf12_3_2': 'salary_advance', 'qf12_3_3': 'pawn',
            'qf12_3_4': 'informal_loan', 'qf12_3_5': 'use_others_credit_card', 'qf12_3_6': 'flexible_mortgage',
            'qf12_3_7': 'pension_withdrawal', 'qf12_4_1': 'overdraft', 'qf12_4_2': 'credit_card_cash',
            'qf12_5_1': 'personal_loan', 'qf12_5_2': 'payday_loan', 'qf12_5_3': 'moneylender',
            'qf12_5_4': 'sms_loan', 'qf12_5_5': 'online_cash_loan', 'qf12_6_1': 'unauthorized_overdraft',
            'qf12_6_2': 'pay_late', 'qf12_7_1': 'other'
        },
        'qp7': { # Information sources
            'qp7_1': 'specialist_comparison', 'qp7_2': 'price_comparison_website',
            'qp7_3': 'independent_advisor', 'qp7_4': 'advert_brochure', 'qp7_5': 'friends_family',
            'qp7_6': 'social_media_influencers', 'qp7_7': 'provider_staff', 'qp7_81': 'tv_radio_ad',
            'qp7_82': 'other_sources'
        },
        'qp8': { # Online activities done
            'qp8_1': 'open_account_online', 'qp8_2': 'request_card_online', 'qp8_3': 'insurance_online',
            'qp8_4': 'credit_online', 'qp8_5': 'invest_online'
        },
        'qp10': { # Financial issues experienced
            'qp10_1': 'scam_investment', 'qp10_2': 'phishing_victim', 'qp10_3': 'unauthorized_card_use',
            'qp10_4': 'unrecognized_transaction', 'qp10_5': 'formal_complaint', 'qp10_8': 'denied_credit',
            'qp10_9': 'complained_remittance'
        }
    }

    # Dynamically generate mappings for Financial Products (QP1, QP2, QP3)[cite: 1]
    products_suffix = {
        '1': 'pension', '2': 'investment_account', '3': 'mortgage', '5': 'unsecured_loan',
        '7': 'credit_card', '8': 'current_account', '9': 'savings_account', '11': 'insurance',
        '12': 'stocks', '13': 'bonds', '14': 'mobile_payment', '15': 'prepaid_card',
        '16': 'crypto', '17': 'esg_products', 'add_1': 'specific_good_loan',
        'add_2': 'coop_loan', 'add_3': 'buy_now_pay_later', 'add_4': 'loan_insurance',
        'add_5': 'basic_account', '98': 'none'
    }
    
    multi_option_groups['qp1'] = {f'qp1_{k}': v for k, v in products_suffix.items()} # Heard of
    multi_option_groups['qp2'] = {f'qp2_{k}': v for k, v in products_suffix.items()} # Currently hold
    multi_option_groups['qp3'] = {f'qp3_{k}': v for k, v in products_suffix.items()} # Recently chosen

    # Function to aggregate row values into a single comma-separated string
    def aggregate_options(row, col_map):
        active = [label for col, label in col_map.items() if col in row and row[col] == 1]
        return ", ".join(active) if active else "None/Refused"

    cols_to_drop = []

    # Apply aggregation for each group
    for group_name, col_map in multi_option_groups.items():
        # Check if at least some columns from this group exist in the dataframe
        existing_cols = [c for c in col_map.keys() if c in df.columns]
        if existing_cols:
            # Create the new aggregated categorical column
            new_col_name = f"{group_name}_aggregated_summary"
            df[new_col_name] = df.apply(lambda row: aggregate_options(row, col_map), axis=1)
            # Mark original binary columns for deletion to clean up the dataset
            cols_to_drop.extend(existing_cols)

    # Drop the original _1, _2 columns that were consolidated
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # --- STEP 2: MAP ALL REMAINING SINGLE VARIABLES & LIKERT SCALES ---
    # These are not binary "tick all that apply" questions, so they remain independent columns[cite: 1].
    full_name_mapping = {
        # Personal and Household
        'qd1': 'gender', 'qd7': 'age', 'qd7_a': 'age_bands', 'qd2': 'macro_region',
        'qd3': 'urbanization_level', 'qd10': 'work_situation', 'qd14': 'internet_access',
        'qd5_ad': 'household_adults_count', 'qd5_ch': 'household_children_count',
        
        # Planning and managing finances
        'qf1_a': 'personal_budget_decisions', 'qf1': 'household_budget_decisions',
        'qf4': 'expenditure_shock_capacity', 'qf8': 'retirement_plan_confidence',
        'qf11': 'income_not_covering_costs', 'qf13': 'lost_income_survival_time',
        
        # Choosing and using products
        'qp5': 'shopping_around_behavior', 'qp7_add1': 'risk_aversion',
        
        # Financial Knowledge
        'qk1': 'self_rated_knowledge', 'qk3': 'inflation_knowledge_brothers',
        'qk4': 'interest_on_loan', 'qk5': 'simple_interest', 'qk6': 'compound_interest',
        'qk10': 'mortgage_knowledge',
        
        # True/False Knowledge Statements (0/1)
        'qk7_1': 'know_high_return_high_risk', 'qk7_2': 'know_high_inflation_cost_living',
        'qk7_3': 'know_reduce_risk_diversify', 'qk7_4': 'know_digital_contract_paper',
        'qk7_5': 'know_data_targeted_offers', 'qk7_6': 'know_crypto_legal_tender',

        # Frequency of online activities (Likert 1-4)
        'qp9_1': 'freq_check_balance_online', 'qp9_3': 'freq_pay_bills_online',
        'qp9_4': 'freq_buy_online', 'qp9_5': 'freq_transfer_money_online',
        'qp9_6': 'freq_manage_finance_online', 'qp9_7': 'freq_mobile_payment_shop',
        'qp9_10': 'freq_roboadvisor',

        # Attitudes (Likert 1-5)
        'qs1_1': 'att_spend_over_save', 'qs1_2': 'att_risk_money', 'qs1_3': 'att_money_to_spend',
        'qs1_4': 'att_satisfied_finance', 'qs1_5': 'att_watch_affairs', 'qs1_7': 'att_finance_limits_life',
        'qs1_8': 'att_set_long_term_goals', 'qs1_9': 'att_trust_bank_safety', 'qs1_10': 'att_too_much_debt',
        'qs1_13': 'att_good_time_crypto',
        
        # Behaviors (Likert 1-5)
        'qs2_1': 'beh_worry_expenses', 'qs2_2': 'beh_finances_control_life', 'qs2_3': 'beh_consider_afford',
        'qs2_4': 'beh_money_left_over', 'qs2_5': 'beh_pay_bills_on_time', 'qs2_6': 'beh_share_pins',
        'qs2_7': 'beh_check_regulated_provider', 'qs2_8': 'beh_share_finance_public',
        'qs2_9': 'beh_consider_esg',
        
        # Situation matching (Likert 1-5)
        'qs3_2': 'sit_prefer_ethical_intermediary', 'qs3_3': 'sit_feel_never_have_things',
        'qs3_9': 'sit_concern_money_wont_last', 'qs3_10': 'sit_just_getting_by',
        'qs3_11': 'sit_live_for_today', 'qs3_12': 'sit_buy_lottery', 'qs3_13': 'sit_change_passwords',
        
        # Digital Attitudes (Likert 1-5)
        'qs4_1': 'dig_safe_public_wifi', 'qs4_2': 'dig_check_website_security',
        'qs4_3': 'dig_ignore_tc', 'qs4_4': 'dig_tools_facilitate', 'qs4_5': 'dig_trust_fintech',
        'qs4_6': 'dig_ok_social_data_credit', 'qs4_7': 'dig_impulsive_online',
        'qs4_8': 'dig_read_print_paper_over_online',
        
        # ESG Attitudes (Likert 1-5)
        'qs5_4': 'esg_profit_over_env', 'qs5_5': 'esg_profit_over_social', 'qs5_6': 'esg_profit_over_gov',
        
        # Background Digital Skills (Frequency 1-4)
        'qd6_1': 'freq_write_doc', 'qd6_2': 'freq_email', 'qd6_3': 'freq_mobile_call',
        'qd6_4': 'freq_internet_call', 'qd6_5': 'freq_social_networks', 'qd6_6': 'freq_instant_messaging',
        'qd6_7': 'freq_search_online',

        # Background info
        'qd9': 'educational_level', 'qd12': 'nationality', 'qd13': 'income_band'
    }

    # Apply the final renaming
    df = df.rename(columns=full_name_mapping)

    return df

############ CLE ###############################


def clean_qk(df):
    """
    Cleans QK financial knowledge variables.

    qk1_clean:
        high = qk1 1/2
        medium-low = qk1 3/4
        not defined = missing/refused/other

    qk3, qk4, qk5, qk6, qk10:
        correct = 1
        wrong/missing = 0

    qk7_clean:
        score from 0 to 6
    """

    df = df.copy()

    df.columns = [col.lower() for col in df.columns]

    # QK1: subjective knowledge
    df["qk1_clean"] = 4 # not defined
    df.loc[df["qk1"].isin([1, 2]), "qk1_clean"] = 1 #high
    df.loc[df["qk1"].isin([4, 5]), "qk1_clean"] = 2 #medium-low
    df.loc[df["qk1"].isin([3]), "qk1_clean"] = 3 #medium-low

    # Objective QK questions: correct = 1, wrong/missing = 0
    df["qk3_clean"] = (df["qk3"] == 3).astype(int)
    df["qk4_clean"] = (df["qk4"] == 0).astype(int)
    df["qk5_clean"] = (df["qk5"] == 102).astype(int)
    df["qk6_clean"] = (df["qk6"] == 1).astype(int)
    df["qk10_clean"] = (df["qk10"] == 1).astype(int)

    # QK7: True/False battery, score 0-6
    qk7_correct = {
        "qk7_1": 1,
        "qk7_2": 1,
        "qk7_3": 1,
        "qk7_4": 0,
        "qk7_5": 1,
        "qk7_6": 0
    }

    df["qk7_clean"] = 0

    for col, correct_value in qk7_correct.items():
        if col in df.columns:
            df["qk7_clean"] += (df[col] == correct_value).astype(int)

    return df

def plot_qk_distributions(df):
    """
    Generates a panel of charts for the distribution
    of QK variables after clean_qk() preprocessing.
    """

    # --- Variable configuration ---
    binary_vars = {
        "qk3_clean": "QK3 – Inflation knowledge\n(1=correct, 0=wrong/missing)",
        "qk4_clean": "QK4 – Loan interest\n(1=correct, 0=wrong/missing)",
        "qk5_clean": "QK5 – Simple interest\n(1=correct, 0=wrong/missing)",
        "qk6_clean": "QK6 – Compound interest\n(1=correct, 0=wrong/missing)",
        "qk10_clean": "QK10 – Mortgage knowledge\n(1=correct, 0=wrong/missing)",
    }

    likert_vars = {
        "qk1_clean": {
            "title": "QK1 – Subjective knowledge\n(self-assessment)",
            "labels": {1: "High", 2: "Medium", 3: "Low", 4: "Not defined"},
            "palette": ["#3266ad", "#73726c", "#d0cec4", "#a5b8d4"],
        }
    }

    score_vars = {
        "qk7_clean": "QK7 – T/F Score (0–6)"
    }

    n_rows = 3
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 13))
    fig.suptitle("Distribution of QK Variables – Financial Knowledge",
                 fontsize=15, fontweight="bold", y=1.01)
    axes = axes.flatten()

    colors_binary = ["#3266ad", "#a5b8d4"]
    ax_idx = 0

    # --- 1. Binary variables (correct/wrong) ---
    for col, title in binary_vars.items():
        ax = axes[ax_idx]
        if col not in df.columns:
            ax.set_visible(False)
            ax_idx += 1
            continue

        counts = df[col].value_counts().reindex([1, 0], fill_value=0)
        labels = ["Correct (1)", "Wrong/Missing (0)"]
        bars = ax.bar(labels, counts.values, color=colors_binary, width=0.55,
                      edgecolor="white", linewidth=1.2)

        total = counts.sum()
        for bar, val in zip(bars, counts.values):
            pct = f"{val/total*100:.1f}%"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                    f"{val:,}\n({pct})", ha="center", va="bottom", fontsize=9.5)

        ax.set_title(title, fontsize=10, pad=8)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.set_ylim(0, counts.max() * 1.22)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=9)
        ax_idx += 1

    # --- 2. QK1 – Subjective knowledge (categorical) ---
    for col, info in likert_vars.items():
        ax = axes[ax_idx]
        if col not in df.columns:
            ax.set_visible(False)
            ax_idx += 1
            continue

        counts = df[col].value_counts().sort_index()
        labels_mapped = [info["labels"].get(k, str(k)) for k in counts.index]
        bars = ax.bar(labels_mapped, counts.values, color=info["palette"][:len(counts)],
                      width=0.55, edgecolor="white", linewidth=1.2)

        total = counts.sum()
        for bar, val in zip(bars, counts.values):
            pct = f"{val/total*100:.1f}%"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                    f"{val:,}\n({pct})", ha="center", va="bottom", fontsize=9.5)

        ax.set_title(info["title"], fontsize=10, pad=8)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.set_ylim(0, counts.max() * 1.25)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=9)
        ax_idx += 1

    # --- 3. QK7 – T/F Score distribution (0–6) ---
    for col, title in score_vars.items():
        ax = axes[ax_idx]
        if col not in df.columns:
            ax.set_visible(False)
            ax_idx += 1
            continue

        score_range = list(range(7))
        counts = df[col].value_counts().reindex(score_range, fill_value=0)

        cmap = plt.cm.Blues
        norm_colors = [cmap(0.3 + 0.7 * (i / 6)) for i in score_range]
        bars = ax.bar([str(v) for v in score_range], counts.values,
                      color=norm_colors, width=0.65, edgecolor="white", linewidth=1.2)

        total = counts.sum()
        for bar, val in zip(bars, counts.values):
            if val > 0:
                pct = f"{val/total*100:.1f}%"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.005,
                        f"{val:,}\n({pct})", ha="center", va="bottom", fontsize=8.5)

        mean_val = df[col].mean()
        ax.axvline(x=mean_val, color="#d85a30", linewidth=1.5, linestyle="--",
                   label=f"Mean: {mean_val:.2f}")
        ax.legend(fontsize=9, frameon=False)

        ax.set_title(title, fontsize=10, pad=8)
        ax.set_xlabel("Score (no. correct answers out of 6)", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.set_ylim(0, counts.max() * 1.25)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=9)
        ax_idx += 1
    # --- 4. Summary panel: % correct answers by variable (QK3–QK10 + QK7 normalized) ---
    ax = axes[ax_idx]

    # QK7 normalized to 0–1 for comparability
    df = df.copy()
    df["qk7_norm"] = df["qk7_clean"] / 6

    summary_cols   = ["qk3_clean", "qk4_clean", "qk5_clean", "qk6_clean", "qk10_clean", "qk7_norm"]
    summary_labels = ["QK3", "QK4", "QK5", "QK6", "QK10", "QK7\n(norm.)"]
    pct_correct = [df[c].mean() * 100 for c in summary_cols if c in df.columns]

    bars = ax.barh(summary_labels[:len(pct_correct)], pct_correct,
                   color="#3266ad", height=0.55, edgecolor="white", linewidth=1.2)
    ax.axvline(x=50, color="#d85a30", linewidth=1.2, linestyle="--", label="50%")
    for bar, val in zip(bars, pct_correct):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9.5)

    ax.set_title("% correct answers by\nobjective question", fontsize=10, pad=8)
    ax.set_xlabel("% correct", fontsize=9)
    ax.set_xlim(0, 110)
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
    ax_idx += 1

    # --- 5. Gap analysis: subjective perception vs objective performance ---
    ax = axes[ax_idx]

    subj_labels_map = {1: "High", 2: "Medium", 3: "Low", 4: "Not defined"}
    subj_score_map  = {1: 100, 2: 67, 3: 33, 4: 0}
    groups = ["High", "Medium", "Low", "Not defined"]

    df["qk1_label"]  = df["qk1_clean"].map(subj_labels_map)
    df["subj_score"] = df["qk1_clean"].map(subj_score_map)
    df["obj_score"]  = df[["qk3_clean", "qk4_clean", "qk5_clean",
                            "qk6_clean", "qk10_clean", "qk7_norm"]].mean(axis=1) * 100

    gap_data = df[df["qk1_label"].notna()].groupby("qk1_label")[["subj_score", "obj_score"]].mean()
    gap_data = gap_data.reindex([g for g in groups if g in gap_data.index])
    gap_data["gap"] = gap_data["subj_score"] - gap_data["obj_score"]

    bar_colors = ["#d85a30" if g > 0 else "#3266ad" for g in gap_data["gap"]]
    bars = ax.barh(gap_data.index, gap_data["gap"],
                   color=bar_colors, height=0.5,
                   edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, gap_data["gap"]):
        x_pos = val + 0.5 if val >= 0 else val - 0.5
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}", va="center", ha=ha, fontsize=10)

    ax.axvline(x=0, color="#2c2c2a", linewidth=1, linestyle="-")
    ax.set_title("Gap: subjective perception\nvs. objective performance", fontsize=10, pad=8)
    ax.set_xlabel("Gap score (subjective − objective)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d85a30", label="Overconfident (subj > obj)"),
        Patch(facecolor="#3266ad", label="Underconfident (subj < obj)")
    ]
    ax.legend(handles=legend_elements, fontsize=8.5, frameon=False)
    ax_idx += 1

    # Hide empty axes
    for i in range(ax_idx, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(h_pad=3.5, w_pad=2.5)
    plt.savefig("qk_distributions.png", dpi=150, bbox_inches="tight")
    print("saved qk_distributions.png")


    
# --- Integrazione nel main ---
def main():
    file_path = "Database_ENG.csv"
    try:
        df = pd.read_csv(file_path)

        df = clean_qk(df)
        df = preprocess_complete_iacofi(df)

        print("\nDataset preview:\n")
        print(df.head())

        print("\nQK variables:\n")
        print(df[[
            "qk1_clean", "qk3_clean", "qk4_clean",
            "qk5_clean", "qk6_clean", "qk7_clean", "qk10_clean"
        ]].head(10))

        print("\nGenerating QK distribution charts...")
        plot_qk_distributions(df)

      

        print("\nAll charts generated successfully.")
        print("\nColumn list:\n")
        print(df.columns.tolist())

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()