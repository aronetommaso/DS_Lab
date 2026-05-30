import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import traceback
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

load_dotenv()

def preprocess_complete_iacofi(df):
   
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

    # Dynamically generate mappings for Financial Products (QP1, QP2, QP3)
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
    # These are not binary "tick all that apply" questions, so they remain independent columns.
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



def preprocess_financial_variables(df):
   
    
    # 1. Map Categorical Variables & Handle Missing Values (-97, -98, -99 -> NaN)
    categorical_value_maps = {
        'qf1_a': {1: 'yes', 0: 'no'},
        'qf1': {1: 'you', 2: 'you and someone else', 3: 'someone else'},
        'qf4': {1: 'yes_immediate', 2: 'yes_selling_assets', 0: 'no'},
        'qf8': {1: 'very_confident', 2: 'confident', 3: 'middle', 4: 'not_confident', 5: 'not_confident_at_all', 6: 'no_retirement_plan'},
        'qf11': {1: 'yes', 0: 'no'},
        'qf13': {1: 'less_than_a_week', 2: '1_week_to_1_month', 3: '1_to_3_months', 4: '3_to_6_months', 5: '6_months_or_more'}
    }

    # Replace missing/refusal codes with NaN, then map to strings
    missing_codes = [-97, 97, -98, 98, -99, 99, -999]
    for col, mapping in categorical_value_maps.items():
        if col in df.columns:
            df[col] = df[col].replace(missing_codes, np.nan)
            df[col] = df[col].map(mapping)

    # 2. FEATURE ENGINEERING: Multi-option groups to Scores and Macro-Flags
    
    # helper to safely select columns that actually exist in the dataframe
    def get_existing_cols(cols):
        return [c for c in cols if c in df.columns]

    # --- QF2
    qf2_cols = get_existing_cols(['qf2_1', 'qf2_2', 'qf2_3', 'qf2_4', 'qf2_5', 'qf2_6'])
    if qf2_cols:
        # Sum only the '1' (Yes) responses. Missing values in these binary cols are treated as 0 for the score sum
        df['Financial_planning_score'] = df[qf2_cols].eq(1).sum(axis=1)

    # --- QF3
    def calc_saving_sophistication(row):
        # Level 3: Investor (Bonds, Crypto, Stocks)
        if row.get('qf3_5') == 1 or row.get('qf3_6') == 1 or row.get('qf3_7') == 1:
            return 3
        # Level 2: Basic Banking (Deposit account)
        elif row.get('qf3_2') == 1:
            return 2
        # Level 1: Informal/Physical (Cash, Family, Informal club)
        elif row.get('qf3_1') == 1 or row.get('qf3_3') == 1 or row.get('qf3_4') == 1:
            return 1
        # Level 0: Didn't save
        elif row.get('qf3_98') == 1:
            return 0
        return np.nan # If no clear answer or all NaN

    if any('qf3' in c for c in df.columns):
        df['Saving_level_sophistication'] = df.apply(calc_saving_sophistication, axis=1)

    # --- QF9
    qf9_state = get_existing_cols(['qf9_1', 'qf9_12'])
    qf9_private = get_existing_cols(['qf9_2', 'qf9_3', 'qf9_4', 'qf9_5', 'qf9_6'])
    qf9_network = get_existing_cols(['qf9_7', 'qf9_8', 'qf9_10'])

    if qf9_state: df['State_employee_pension'] = df[qf9_state].eq(1).any(axis=1).astype(int)
    if qf9_private: df['Private_pension_asset'] = df[qf9_private].eq(1).any(axis=1).astype(int)
    if qf9_network: df['Informal_network_pension'] = df[qf9_network].eq(1).any(axis=1).astype(int)

    # --- QF12: Making Ends Meet (Macro-Flags 0/1) ---
    qf12_own = get_existing_cols(['qf12_1_1', 'qf12_1_2', 'qf12_1_3'])
    qf12_help = get_existing_cols(['qf12_2_1', 'qf12_2_2', 'qf12_2_3', 'qf12_3_1'])
    qf12_debt = get_existing_cols([
        'qf12_4_1', 'qf12_4_2', 'qf12_5_1', 'qf12_5_2', 'qf12_5_3', 
        'qf12_5_4', 'qf12_5_5', 'qf12_6_1', 'qf12_6_2'
    ])

    if qf12_own: df['Use_own_resources'] = df[qf12_own].eq(1).any(axis=1).astype(int)
    if qf12_help: df['Informal_external_help'] = df[qf12_help].eq(1).any(axis=1).astype(int)
    if qf12_debt: df['Use_dangerous_debt'] = df[qf12_debt].eq(1).any(axis=1).astype(int)

    # 3. Rename base categorical columns for clarity
    base_financial_mapping = {
        'qf1_a': 'personal_budget_decisions',
        'qf1': 'household_budget_decisions',
        'qf4': 'expenditure_shock_capacity',
        'qf8': 'retirement_plan_confidence',
        'qf11': 'income_not_covering_costs',
        'qf13': 'lost_income_survival_time'
    }
    df = df.rename(columns=base_financial_mapping)

    # 4. Filter dataset to keep only the newly created and mapped features
    final_engineered_cols = list(base_financial_mapping.values()) + [
        'Financial_planning_score',
        'Saving_level_sophistication',
        'State_employee_pension',
        'Private_pension_asset',
        'Informal_network_pension',
        'Use_own_resources',
        'Informal_external_help',
        'Use_dangerous_debt'
    ]
    
    # Keep only those that were successfully created
    available_cols = [col for col in final_engineered_cols if col in df.columns]
    df_financial = df[available_cols].copy()

    # Drop original granular binary columns since we synthesized them into scores/flags
    original_binary_patterns = ['qf2_', 'qf3_', 'qf9_', 'qf12_']
    cols_to_drop = [c for c in df_financial.columns if any(p in c for p in original_binary_patterns)]
    df_financial = df_financial.drop(columns=cols_to_drop, errors='ignore')

    # Print summary to verify new cardinality
    print("Financial Dataset Overview:")
    print(df_financial.head())
    for col in df_financial.columns:
        # Using dropna() to count only valid categories
        print(f"for {col}: {len(df_financial[col].dropna().unique())}")
    
    df = df.drop(columns = ['qf2_1', 'qf2_2', 'qf2_3', 'qf2_4', 'qf2_5', 'qf2_6', 'qf3_1', 'qf3_2', 'qf3_3', 'qf3_5', 'qf3_6', 'qf3_7', 'qf3_8', 'qf3_81', 'qf3_98', 'qf9_1', 'qf9_2', 'qf9_3', 'qf9_4', 'qf9_5', 'qf9_6', 'qf9_7', 'qf9_8', 'qf9_9', 'qf9_10', 'qf9_11', 'qf9_12', 'qf12_1_1', 'qf12_1_2', 'qf12_1_3', 'qf12_2_1', 'qf12_2_2', 'qf12_3_1', 'qf12_3_2', 'qf12_3_3', 'qf12_3_7', 'qf12_4_1', 'qf12_4_2', 'qf12_5_1', 'qf12_5_3', 'qf12_6_1', 'qf12_6_2', 'qf12_7_1', 'qf12_97', 'qf12_99'])

    return df

def clean_qk(df):


    df = df.copy()
    df.columns = [col.lower() for col in df.columns]

    # QK1: subjective knowledge
    df["qk1_clean"] = 4 # not defined
    df.loc[df["qk1"].isin([1, 2]), "qk1_clean"] = 1 # high
    df.loc[df["qk1"].isin([4, 5]), "qk1_clean"] = 2 # medium-low
    df.loc[df["qk1"].isin([3]), "qk1_clean"] = 3 # medium

    # Objective QK questions: correct = 1, wrong/missing = 0
    df["qk3_clean"] = (df["qk3"] == 3).astype(int)
    df["qk4_clean"] = (df["qk4"] == 0).astype(int)
    df["qk5_clean"] = (df["qk5"] == 102).astype(int)
    df["qk6_clean"] = (df["qk6"] == 1).astype(int)
    df["qk10_clean"] = (df["qk10"] == 1).astype(int)

    # QK7: True/False battery, score 0-6
    qk7_correct = {
        "qk7_1": 1, "qk7_2": 1, "qk7_3": 1,
        "qk7_4": 0, "qk7_5": 1, "qk7_6": 0
    }

    df["qk7_clean"] = 0
    for col, correct_value in qk7_correct.items():
        if col in df.columns:
            df["qk7_clean"] += (df[col] == correct_value).astype(int)
            
    # ---  GAP CLASS ---
    obj_cols = [c for c in ['qk3_clean', 'qk4_clean', 'qk5_clean', 'qk6_clean', 'qk10_clean'] if c in df.columns]
    

    if 'qk7_clean' in df.columns:
        df['qk7_norm'] = df['qk7_clean'] / 6
        obj_cols_all = obj_cols + ['qk7_norm']
    else:
        obj_cols_all = obj_cols

    # obj score
    df['obj_score'] = df[obj_cols_all].mean(axis=1) * 100
    

    df['subj_score'] = df['qk1_clean'].map({1: 100, 2: 67, 3: 33, 4: 0})
    

    df['gap'] = df['subj_score'] - df['obj_score']
    df['subj_knowledge_label'] = df['qk1_clean'].map({1: 'High', 2: 'Medium', 3: 'Low', 4: 'Not defined'})

    def classify_gap(g):
        if pd.isna(g): return np.nan
        if g >  15:    return 'Overconfident'
        if g < -15:    return 'Underconfident'
        return 'Calibrated'

    df['gap_class'] = df['gap'].apply(classify_gap)
    df = df.drop(columns=['qk7_norm', 'subj_score', 'gap'])
    return(df)


def preprocess_products_and_digital(df):
  
    df_qp = df.copy()

    def get_existing_cols(cols):
        return [c for c in cols if c in df_qp.columns]

    qp2_cols = [c for c in df_qp.columns if c.startswith('qp2_')]
    for col in qp2_cols:
        df_qp[col] = df_qp[col].replace([-97, 97, -98, 98, -99, 99, -999], np.nan).fillna(0)

    score_transactional = ['qp2_8', 'qp2_14', 'qp2_15', 'qp2_add_5']
    score_saving = ['qp2_1', 'qp2_9', 'qp2_11', 'qp2_add_4']
    score_debt = ['qp2_5', 'qp2_7', 'qp2_add_1', 'qp2_add_2', 'qp2_add_3']
    score_invest_trad = ['qp2_2', 'qp2_12', 'qp2_13']
    score_asset_alt = ['qp2_3', 'qp2_16', 'qp2_17']

    df_qp['Transactional_Score'] = df_qp[get_existing_cols(score_transactional)].sum(axis=1)
    df_qp['Saving_Protection_Score'] = df_qp[get_existing_cols(score_saving)].sum(axis=1)
    df_qp['Consumer_Debt_Score'] = df_qp[get_existing_cols(score_debt)].sum(axis=1)
    df_qp['Traditional_Investment_Score'] = df_qp[get_existing_cols(score_invest_trad)].sum(axis=1)
    df_qp['Alternative_Asset_Score'] = df_qp[get_existing_cols(score_asset_alt)].sum(axis=1)

    if 'qp7_add1' in df_qp.columns:
        df_qp['Risk_Aversion_Class'] = df_qp['qp7_add1'].replace({-99: 5, 99: 5})
        df_qp['Risk_Aversion_Class'] = df_qp['Risk_Aversion_Class'].apply(lambda x: x if x in [1, 2, 3, 4, 5] else np.nan)

    qp8_cols = get_existing_cols(['qp8_1', 'qp8_2', 'qp8_3', 'qp8_4', 'qp8_5'])
    for col in qp8_cols:
        df_qp[col] = df_qp[col].replace({2: 0, -95: 0, -97: 0, -98: 0, -99: 0}).fillna(0)
        if 'qd14' in df_qp.columns: df_qp.loc[df_qp['qd14'] == 0, col] = 0

    df_qp['Digital_Onboarding_Score'] = df_qp[qp8_cols].sum(axis=1)

    qp9_cols = get_existing_cols(['qp9_1', 'qp9_3', 'qp9_4', 'qp9_5', 'qp9_6', 'qp9_7', 'qp9_10'])
    for col in qp9_cols:
        df_qp[col] = df_qp[col].replace({-95: 1, -97: 1, -98: 1, -99: 1}).fillna(1)
        if 'qd14' in df_qp.columns: df_qp.loc[df_qp['qd14'] == 0, col] = 1 
        df_qp[col] = df_qp[col] - 1 

    df_qp['Basic_Admin_Intensity'] = df_qp[get_existing_cols(['qp9_1', 'qp9_3'])].sum(axis=1)
    df_qp['Daily_Transactional_Intensity'] = df_qp[get_existing_cols(['qp9_4', 'qp9_5', 'qp9_7'])].sum(axis=1)
    df_qp['Advanced_Fintech_Intensity'] = df_qp[get_existing_cols(['qp9_6', 'qp9_10'])].sum(axis=1)

    qp10_cols = [c for c in df_qp.columns if c.startswith('qp10_')]
    for col in qp10_cols:
        df_qp[col] = df_qp[col].replace({-95: 0, -97: 0, -98: 0, -99: 0}).fillna(0)

    cyber = get_existing_cols(['qp10_1', 'qp10_2', 'qp10_3'])
    friction = get_existing_cols(['qp10_4', 'qp10_5', 'qp10_9'])
    
    if cyber: df_qp['Cyber_Fraud_Victim'] = df_qp[cyber].eq(1).any(axis=1).astype(int)
    if friction: df_qp['Institutional_Friction'] = df_qp[friction].eq(1).any(axis=1).astype(int)
    if 'qp10_8' in df_qp.columns: df_qp['Credit_Excluded'] = df_qp['qp10_8'].astype(int)

    if 'qp5' in df_qp.columns:
        valid_qp5 = ~df_qp['qp5'].isin([-97, -98, -99, np.nan])
        df_active = df_qp[valid_qp5].copy()
    else:
        df_active = df_qp.copy() 

    def map_qp5(val):
        if val == 1: return 'Proactive_Shopper'      
        elif val == 2: return 'Lazy_Explorer'        
        elif val in [3, 4]: return 'Passive_Inertial' 
        return np.nan
    
    if 'qp5' in df_active.columns: 
        df_active['Shopping_Behavior'] = df_active['qp5'].apply(map_qp5)

    analytical = get_existing_cols(['qp7_1', 'qp7_2', 'qp7_3'])
    institutional = get_existing_cols(['qp7_4', 'qp7_7'])
    social = get_existing_cols(['qp7_5', 'qp7_6', 'qp7_81'])

    df_active['is_analytical'] = df_active[analytical].eq(1).any(axis=1) if analytical else False
    df_active['is_institutional'] = df_active[institutional].eq(1).any(axis=1) if institutional else False
    df_active['is_social'] = df_active[social].eq(1).any(axis=1) if social else False

    def assign_driver(row):
        if row['is_analytical']: return 'Analytical_Independent'
        elif row['is_institutional']: return 'Institutional_Commercial'
        elif row['is_social']: return 'Social_Emotional'
        return 'Other_or_None'

    df_active['Decision_Driver'] = df_active.apply(assign_driver, axis=1)
    df_active = df_active.drop(columns=['is_analytical', 'is_institutional', 'is_social'], errors='ignore')

    final_qp_cols = [
        'Transactional_Score', 'Saving_Protection_Score', 'Consumer_Debt_Score', 
        'Traditional_Investment_Score', 'Alternative_Asset_Score', 'Risk_Aversion_Class', 
        'Digital_Onboarding_Score', 'Basic_Admin_Intensity', 'Daily_Transactional_Intensity', 
        'Advanced_Fintech_Intensity', 'Cyber_Fraud_Victim', 'Institutional_Friction', 'Credit_Excluded'
    ]
    
    df_qp_main = df_qp.copy()

    return df_qp_main, df_active
 
def calcola_e_sostituisci_score(df):
      
    qs_cols = [col for col in df.columns if str(col).startswith('qs')] 
    

    yellow_attributes=['qs1_1','qs1_3', 'qs1_7', 'qs1_10', 'qs1_13', 'qs2_1', 'qs2_2', 'qs2_6', 'qs2_8','qs3_3','qs3_9','qs3_10', 'qs3_11', 'qs3_12', 
                   'qs4_1', 'qs4_3', 'qs4_7', 'qs4_8', 'qs5_4', 'qs5_5', 'qs5_6']
    red_attributes = ['qs1_2', 'qs1_5', 'qs1_8', 'qs1_9', 'qs2_3', 'qs2_4', 'qs2_5', 'qs2_7', 'qs3_13', 'qs4_2', 'qs4_4', 'qs4_5', 'qs4_6']
    green_attributes = ['qs3_2', 'qs2_9', 'qs1_4']
    
    pink_cat = ['qs1_1', 'qs1_2', 'qs1_3', 'qs1_5', 'qs1_8', 'qs1_13', 'qs2_1', 'qs2_3', 'qs2_5', 'qs2_9', 'qs3_2', 'qs3_11', 'qs3_12', 'qs4_7', 'qs5_4', 'qs5_5', 'qs5_6']
    brown_cat = ['qs1_9', 'qs2_6', 'qs2_7', 'qs2_8', 'qs3_13', 'qs4_1', 'qs4_2', 'qs4_3', 'qs4_4', 'qs4_5', 'qs4_6', 'qs4_8']
    blue_cat = ['qs1_4', 'qs1_7', 'qs1_10', 'qs2_2', 'qs2_4', 'qs3_3', 'qs3_9', 'qs3_10']

    # excluding NaN
    df_calc = df[qs_cols].copy()
    df_calc = df_calc.where(df_calc.isin([1, 2, 3, 4, 5]))
    

    for col in red_attributes:
        if col in df_calc.columns:
            df_calc[col] = 6 - df_calc[col]
            
    for col in green_attributes:
        if col in df_calc.columns:
            df_calc[col] = 0
            

    df_finale = df.drop(columns=qs_cols).copy()
    

    df_finale['finacial_situation'] = df_calc[blue_cat].sum(axis=1)
    df_finale['behaviour_investement-payment'] = df_calc[pink_cat].sum(axis=1)
    df_finale['knowledge_financial_privacy_digital'] = df_calc[brown_cat].sum(axis=1)
    

    return df_finale

def plotting_financial_variables(df):
  
    output_dir = 'plot'
    os.makedirs(output_dir, exist_ok=True)

    for col in df.columns:
        plt.figure(figsize=(10, 6))
        # Remove NaN values to check data types
        series = df[col].dropna()

        if series.empty:
            plt.close()
            continue

        # Identify if the feature is categorical, binary, or an integer score
        is_categorical = pd.api.types.is_object_dtype(series)
        is_discrete = pd.api.types.is_numeric_dtype(series) and (series.nunique() <= 15 or all(series % 1 == 0))

        if is_categorical or is_discrete:
            # Countplot for classes (categorical, 0/1 flags, integer scores)
            sns.countplot(data=df, x=col, order=series.value_counts().index, hue=col, legend=False, palette='viridis')
            plt.title(f'Class count for variable: {col}')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Count')
        else:
            # Distribution histogram for continuous non-integer values
            sns.histplot(series, kde=True, color='skyblue', bins=30)
            plt.title(f'Distribution of variable: {col}')
            plt.ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}.png"))
        plt.close()
        
  

def engineer_demographic_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df.columns = [col.lower() for col in df.columns]

    # -aggregation logics
    multi_option_groups = {
        'qd5': { 'qd5_1': 'alone', 'qd5_2': 'partner', 'qd5_3': 'children_under_18', 'qd5_4': 'children_over_18', 'qd5_5': 'adult_relatives', 'qd5_6': 'friends', 'qd5_7': 'other_adults'},
        'qf2': { 'qf2_1': 'plan_budget', 'qf2_2': 'note_spending', 'qf2_3': 'separate_bills_money', 'qf2_4': 'note_upcoming_bills', 'qf2_5': 'use_banking_app', 'qf2_6': 'auto_payments'},
        'qf3': { 'qf3_1': 'cash_at_home', 'qf3_2': 'deposit_account', 'qf3_3': 'family_save', 'qf3_4': 'informal_club', 'qf3_5': 'bonds', 'qf3_6': 'crypto', 'qf3_7': 'stocks', 'qf3_8': 'other', 'qf3_81': 'other_financial_instruments', 'qf3_98': 'did_not_save'},
        'qf9': { 'qf9_1': 'gov_pension', 'qf9_2': 'occupational_pension', 'qf9_3': 'private_pension', 'qf9_4': 'sell_financial_assets', 'qf9_5': 'sell_non_financial_assets', 'qf9_6': 'asset_income', 'qf9_7': 'spouse_support', 'qf9_8': 'family_support', 'qf9_9': 'savings', 'qf9_10': 'continue_work', 'qf9_11': 'business_revenue', 'qf9_12': 'reversibility_pension'},
        'qf12': { 'qf12_1_1': 'draw_savings', 'qf12_1_2': 'cut_spending', 'qf12_1_3': 'sell_owned', 'qf12_2_1': 'work_overtime', 'qf12_2_2': 'gov_support', 'qf12_2_3': 'ask_family', 'qf12_3_1': 'borrow_family', 'qf12_3_2': 'salary_advance', 'qf12_3_3': 'pawn', 'qf12_3_4': 'informal_loan', 'qf12_3_5': 'use_others_credit_card', 'qf12_3_6': 'flexible_mortgage', 'qf12_3_7': 'pension_withdrawal', 'qf12_4_1': 'overdraft', 'qf12_4_2': 'credit_card_cash', 'qf12_5_1': 'personal_loan', 'qf12_5_2': 'payday_loan', 'qf12_5_3': 'moneylender', 'qf12_5_4': 'sms_loan', 'qf12_5_5': 'online_cash_loan', 'qf12_6_1': 'unauthorized_overdraft', 'qf12_6_2': 'pay_late', 'qf12_7_1': 'other'},
        'qp7': { 'qp7_1': 'specialist_comparison', 'qp7_2': 'price_comparison_website', 'qp7_3': 'independent_advisor', 'qp7_4': 'advert_brochure', 'qp7_5': 'friends_family', 'qp7_6': 'social_media_influencers', 'qp7_7': 'provider_staff', 'qp7_81': 'tv_radio_ad', 'qp7_82': 'other_sources'},
        'qp8': { 'qp8_1': 'open_account_online', 'qp8_2': 'request_card_online', 'qp8_3': 'insurance_online', 'qp8_4': 'credit_online', 'qp8_5': 'invest_online'},
        'qp10': { 'qp10_1': 'scam_investment', 'qp10_2': 'phishing_victim', 'qp10_3': 'unauthorized_card_use', 'qp10_4': 'unrecognized_transaction', 'qp10_5': 'formal_complaint', 'qp10_8': 'denied_credit', 'qp10_9': 'complained_remittance'}
    }

    products_suffix = { '1': 'pension', '2': 'investment_account', '3': 'mortgage', '5': 'unsecured_loan', '7': 'credit_card', '8': 'current_account', '9': 'savings_account', '11': 'insurance', '12': 'stocks', '13': 'bonds', '14': 'mobile_payment', '15': 'prepaid_card', '16': 'crypto', '17': 'esg_products', 'add_1': 'specific_good_loan', 'add_2': 'coop_loan', 'add_3': 'buy_now_pay_later', 'add_4': 'loan_insurance', 'add_5': 'basic_account', '98': 'none'}
    multi_option_groups['qp1'] = {f'qp1_{k}': v for k, v in products_suffix.items()} 
    multi_option_groups['qp2'] = {f'qp2_{k}': v for k, v in products_suffix.items()} 
    multi_option_groups['qp3'] = {f'qp3_{k}': v for k, v in products_suffix.items()} 


    def aggregate_options(row, col_map):
        active = [label for col, label in col_map.items() if col in row and row[col] == 1]
        return ", ".join(active) if active else "None/Refused"

    cols_to_drop = []
    for group_name, col_map in multi_option_groups.items():
        existing_cols = [c for c in col_map.keys() if c in df.columns]
        if existing_cols:
            new_col_name = f"{group_name}_aggregated_summary"
            df[new_col_name] = df.apply(lambda row: aggregate_options(row, col_map), axis=1)
            cols_to_drop.extend(existing_cols)

    df = df.drop(columns=cols_to_drop, errors='ignore')

    # - mapping singles variables
    full_name_mapping = {
        'qd1': 'gender', 'qd7': 'age', 'qd7_a': 'age_bands', 'qd2': 'macro_region', 'qd3': 'urbanization_level', 'qd10': 'work_situation', 'qd14': 'internet_access', 'qd5_ad': 'household_adults_count', 'qd5_ch': 'household_children_count',
        'qd6_1': 'freq_write_doc', 'qd6_2': 'freq_email', 'qd6_3': 'freq_mobile_call', 'qd6_4': 'freq_internet_call', 'qd6_5': 'freq_social_networks', 'qd6_6': 'freq_instant_messaging', 'qd6_7': 'freq_search_online',
        'qd9': 'educational_level', 'qd12': 'nationality', 'qd13': 'income_band'
    }
    df = df.rename(columns=full_name_mapping)

    # 1. GENDER
    df['gender'] = df['gender'].map({0: 'Woman', 1: 'Man'})

    # 2. AGE GROUP 
    def _map_age_to_band(val):
        try:
            val = float(val)
            if pd.isna(val) or val < 0: return pd.NA
            if 18 <= val <= 19:      return 1
            if 20 <= val <= 29:      return 2
            if 30 <= val <= 39:      return 3
            if 40 <= val <= 49:      return 4
            if 50 <= val <= 59:      return 5
            if 60 <= val <= 69:      return 6
            if 70 <= val <= 79:      return 7
            return val  
        except ValueError:
            return pd.NA

    age_band_labels = {1: '18-19', 2: '20-29', 3: '30-39', 4: '40-49', 5: '50-59', 6: '60-69', 7: '70-79'}
    age_mapped  = df['age'].apply(_map_age_to_band)
    age_unified = age_mapped.fillna(df['age_bands']).replace([-99, -97], pd.NA)
    

    df['age_group'] = age_unified.map(age_band_labels)
    

    df = df.drop(columns=['age', 'age_bands'], errors='ignore')

    # 3. MACRO-REGION
    df['macro_region_label'] = df['macro_region'].map({1: 'North-West', 2: 'North-East', 3: 'Center', 4: 'South', 5: 'Islands'})

    # 4. URBANIZATION LEVEL
    df['urban_area_label'] = df['urbanization_level'].map({1: '<3k', 2: '3k-15k', 3: '15k-100k', 4: '100k-1M', 5: '>1M'})

    # 5a. LIVING STATUS 
    def _derive_living_status(row):
        summary = str(row.get('qd5_aggregated_summary', '')).lower()
        if summary in ('', 'nan', 'none/refused'): return float('nan')
        if 'alone' in summary: return 'Alone'
        if 'children_under_18' in summary or 'children_over_18' in summary: return 'With_Children'
        if 'partner' in summary: return 'With_Partner'
        return 'Other'

    df['living_status'] = df.apply(_derive_living_status, axis=1)

    # 5b. HOUSEHOLD SIZE
    adults_count   = pd.to_numeric(df['household_adults_count'], errors='coerce').fillna(0)
    children_count = pd.to_numeric(df['household_children_count'], errors='coerce').fillna(0)
    adults_count   = adults_count.where(adults_count > 0, 0)   
    children_count = children_count.where(children_count > 0, 0)   
    df['household_size'] = adults_count + children_count + 1

    # 6. DIGITAL SKILLS SCORE 
    digital_cols = [
        'freq_write_doc', 'freq_email', 'freq_mobile_call',
        'freq_internet_call', 'freq_social_networks',
        'freq_instant_messaging', 'freq_search_online'
    ]
    
    temp_dig = df[digital_cols].apply(pd.to_numeric, errors='coerce').copy()
    temp_dig[temp_dig <= 0] = np.nan
    mean_raw = temp_dig.mean(axis=1)
    
    df['digital_skills_score'] = (5 - mean_raw)
    df['digital_skills_score'] = df['digital_skills_score'].fillna(1)
    
    # 7. EDUCATION LEVEL 
    edu_map = {10: 'No Education', 9: 'Primary', 8: 'Primary', 7: 'Middle School', 6: 'Middle School', 5: 'High School', 4: 'High School', 3: 'University', 2: 'University', 1: 'University'}
    df['edu_level_grouped'] = df['educational_level'].map(edu_map)

    # 8. EMPLOYMENT STATUS 
    work_status_map = {1: 'Active', 2: 'Active', 3: 'Active', 4: 'Inactive', 6: 'Inactive', 8: 'Inactive', 9: 'Inactive', 5: 'Vulnerable', 7: 'Vulnerable', 10: 'Inactive'}
    df['work_status'] = df['work_situation'].map(work_status_map)

    # 9. NATIONALITY
    df['is_italian'] = df['nationality'].map({1: 'Italian', 0: 'Other'})

    # 10. NET MONTHLY INCOME BAND
    income_map = {1: '<=1750€', 2: '1751-2900€', 3: '>2900€', -97: 'Unknown', -99: 'Unknown'}
    df['income_label'] = df['income_band'].map(income_map).fillna('Unknown')

    # 11. INTERNET ACCESS 
    df['internet_access_label'] = df['internet_access'].map({1: 'Yes', 0: 'No'})

        # final cleaning

    cols_to_drop_final = [
        'macro_region', 'urbanization_level', 'work_situation', 'internet_access',
        'household_adults_count', 'household_children_count', 'educational_level',
        'nationality', 'income_band', 'freq_write_doc', 'freq_email', 'freq_mobile_call',
        'freq_internet_call', 'freq_social_networks', 'freq_instant_messaging', 'freq_search_online'
    ]
    cols_to_drop_final.extend([c for c in df.columns if c.endswith('_aggregated_summary')])
    raw_q_cols = [c for c in df.columns if c.startswith(('qd', 'qf', 'qp', 'qk', 'qs')) and not c.endswith('_clean')]
    cols_to_drop_final.extend(raw_q_cols)
    
    df = df.drop(columns=cols_to_drop_final, errors='ignore')


    return df

####################
####################

def eda_research(df, output_dir='eda_research_2'):
    
    os.makedirs(output_dir, exist_ok=True)
    df = df.copy()
 
    plt.rcParams.update({
        'font.size': 12, 'axes.titlesize': 13,
        'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    })
 
    BLUE   = '#3266ad'
    ORANGE = '#d85a30'
    GREY   = '#73726c'
    LGREY  = '#d0cec4'
 
    GC_ORDER  = ['Overconfident', 'Calibrated', 'Underconfident']
    GC_COLORS = [ORANGE, BLUE, GREY]
 
    def save(name):
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}.png'),
                    dpi=130, bbox_inches='tight')
        plt.close()
 
    obj_cols = [c for c in ['qk3_clean', 'qk4_clean', 'qk5_clean',
                              'qk6_clean', 'qk10_clean'] if c in df.columns]
    if 'qk7_clean' in df.columns:
        df['qk7_norm'] = df['qk7_clean'] / 6
        obj_cols_all   = obj_cols + ['qk7_norm']
    else:
        obj_cols_all   = obj_cols
 
    df['obj_score']  = df[obj_cols_all].mean(axis=1) * 100
    df['subj_score'] = df['qk1_clean'].map({1: 100, 2: 67, 3: 33, 4: 0})
    df['gap']        = df['subj_score'] - df['obj_score']
    df['qk1_label']  = df['qk1_clean'].map(
        {1: 'High', 2: 'Medium', 3: 'Low', 4: 'Not defined'})
 
    def classify_gap(g):
        if pd.isna(g): return np.nan
        if g >  15:    return 'Overconfident'
        if g < -15:    return 'Underconfident'
        return 'Calibrated'
 
    df['gap_class'] = df['gap'].apply(classify_gap)
 
    print(f"obj_score  → mean: {df['obj_score'].mean():.1f}")
    print(f"subj_score → mean: {df['subj_score'].mean():.1f}")
    print(f"gap        → mean: {df['gap'].mean():.1f}")
    print(df['gap_class'].value_counts(dropna=False))
 
    # ==================================================================
    # HELPERS
    # ==================================================================
 
    def gap_distribution_within(col, x_labels, title, name,
                                  col_order=None, figsize=(10, 5)):
        
        cross = pd.crosstab(df[col], df['gap_class'], normalize='index') * 100
        cross = cross.reindex(columns=GC_ORDER, fill_value=0)
 
        if col_order is not None:
            cross = cross.reindex(col_order)
        cross.index = [x_labels.get(i, str(i)) for i in cross.index]
        cross = cross.dropna()
 
        x = np.arange(len(cross))
        w = 0.25
        fig, ax = plt.subplots(figsize=figsize)
 
        for i, (gc, color) in enumerate(zip(GC_ORDER, GC_COLORS)):
            vals   = cross[gc].values
            offset = (i - 1) * w
            bars   = ax.bar(x + offset, vals, w, label=gc,
                            color=color, edgecolor='white', linewidth=1.1)
            for bar, val in zip(bars, vals):
                if val > 4:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.8,
                            f'{val:.1f}%', ha='center', va='bottom',
                            fontsize=8, color=color, fontweight='bold')
 
        ax.set_xticks(x)
        ax.set_xticklabels(cross.index, rotation=15, ha='right')
        ax.set_ylabel('% within each category')
        ax.set_xlabel('')
        ax.set_title(title, fontweight='bold')
        ax.set_ylim(0, 75)
        ax.axhline(y=100/3, color='black', linewidth=0.8,
                   linestyle=':', alpha=0.4, label='Uniform baseline (33%)')
        ax.legend(title='Calibration class', frameon=False,
                  loc='upper right', fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        save(name)
 
    def simple_bar(series, title, xlabel, ylabel='N',
                   colors=None, name=None):
        fig, ax = plt.subplots(figsize=(8, 5))
        total = series.sum()
        cols  = colors if colors else [BLUE] * len(series)
        bars  = ax.bar(series.index.astype(str), series.values,
                       color=cols, edgecolor='white', linewidth=1.2)
        for bar, val in zip(bars, series.values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + total * 0.012,
                        f'{val:,}\n({val/total*100:.1f}%)',
                        ha='center', va='bottom', fontsize=9)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, series.max() * 1.25)
        ax.spines[['top', 'right']].set_visible(False)
        if name:
            save(name)
 
    def gap_by_group_hbar(groupby_col, title, name, figsize=(9, 5)):

        gap_g  = df.groupby(groupby_col)['gap'].mean().sort_values()
        colors = [ORANGE if v > 0 else BLUE for v in gap_g.values]
        fig, ax = plt.subplots(figsize=figsize)
        bars    = ax.barh(gap_g.index.astype(str), gap_g.values,
                          color=colors, edgecolor='white',
                          linewidth=1.1, height=0.6)
        for bar in bars:
            w    = bar.get_width()
            xpos = w + 0.4 if w >= 0 else w - 0.4
            ha   = 'left' if w >= 0 else 'right'
            ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                    f'{w:+.1f}', va='center', ha=ha, fontsize=10)
        ax.axvline(x=0, color='black', linewidth=1.2, linestyle='--')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Mean gap  (subjective − objective)\n'
                      'Orange = overconfident  |  Blue = underconfident')
        ax.spines[['top', 'right']].set_visible(False)
        save(name)
 
    # ==================================================================
    # subjective knowledge
    # ==================================================================
    order_qk1  = ['High', 'Medium', 'Low', 'Not defined']
    counts_qk1 = df['qk1_label'].value_counts().reindex(order_qk1)
    simple_bar(counts_qk1,
               title='How do people rate their own financial knowledge? (QK1)',
               xlabel='Self-assessed knowledge level',
               colors=[BLUE, GREY, LGREY, '#b0b0b0'],
               name='step2_subjective_knowledge')
 
    # ==================================================================
    # objective knowledge
    # ==================================================================
    if 'qk7_clean' in df.columns:
        score_range = list(range(7))
        counts_qk7  = df['qk7_clean'].value_counts().reindex(score_range, fill_value=0)
        counts_qk7.index = [str(v) for v in score_range]
        cmap   = plt.cm.Blues
        colors = [cmap(0.3 + 0.7 * i / 6) for i in score_range]
        fig, ax = plt.subplots(figsize=(9, 5))
        total   = counts_qk7.sum()
        bars    = ax.bar(counts_qk7.index, counts_qk7.values,
                         color=colors, edgecolor='white', linewidth=1.2)
        for bar, val in zip(bars, counts_qk7.values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + total * 0.006,
                        f'{val:,}\n({val/total*100:.1f}%)',
                        ha='center', va='bottom', fontsize=9)
        mean_v = df['qk7_clean'].mean()
        ax.axvline(x=mean_v, color=ORANGE, linewidth=2,
                   linestyle='--', label=f'Mean: {mean_v:.2f} / 6')
        ax.set_title('QK7 – True/False battery: how many correct out of 6?',
                     fontweight='bold')
        ax.set_xlabel('Number of correct answers (0–6)')
        ax.set_ylabel('N')
        ax.legend(frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
        save('step3_qk7_score')
 
    # ==================================================================
    # gap analysis
    # ==================================================================
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df['gap'].dropna(), bins=40, color=BLUE,
            edgecolor='white', linewidth=0.8)
    ax.axvline(x=0, color='black', linewidth=1.5, linestyle='--',
               label='Zero gap')
    ax.axvline(x=df['gap'].mean(), color=ORANGE, linewidth=2,
               linestyle='--', label=f"Mean gap: {df['gap'].mean():.1f}")
    ax.axvspan(-15, 15, alpha=0.08, color='green',
               label='Calibrated zone (±15 pts)')
    ax.set_xlabel('Gap  (subjective − objective)')
    ax.set_ylabel('N')
    ax.set_title('Distribution of the knowledge gap\n'
                 'Positive = overconfident  |  Negative = underconfident',
                 fontweight='bold')
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    save('step4a_gap_distribution')
 
    gc_counts = df['gap_class'].value_counts().reindex(GC_ORDER)
    simple_bar(gc_counts,
               title='Knowledge calibration classes\n(threshold ±15 pts)',
               xlabel='Knowledge calibration class',
               colors=GC_COLORS,
               name='step4b_gap_classes')
 
    order_qk1_m = ['High', 'Medium', 'Low', 'Not defined']
    mean_obj_g  = df.groupby('qk1_label')['obj_score'].mean().reindex(order_qk1_m)
    mean_sub_g  = df.groupby('qk1_label')['subj_score'].mean().reindex(order_qk1_m)
    x = np.arange(len(order_qk1_m))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, mean_sub_g.values, w,
           label='Subjective', color=ORANGE, edgecolor='white')
    ax.bar(x + w/2, mean_obj_g.values, w,
           label='Objective',  color=BLUE,   edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(order_qk1_m)
    ax.set_title('Do people who feel more knowledgeable actually know more?',
                 fontweight='bold')
    ax.set_ylabel('Mean score (0–100)')
    ax.set_xlabel('Self-assessed knowledge level (QK1)')
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    save('step4c_subj_vs_obj_by_qk1')
 
    # ==================================================================
    # avarage gap for demographic variables
    # ==================================================================
    demo_vars = {
        'gender':                'Gender',
        'age_group':             'Age group',
        'edu_level_grouped':     'Education level',
        'income_label':          'Monthly income band',
        'macro_region_label':    'Geographic region',
        'work_status':           'Employment status',
        'living_status':         'Living situation',
        'internet_access_label': 'Internet access',
    }
    for col, label in demo_vars.items():
        if col not in df.columns:
            continue
        gap_by_group_hbar(
            col,
            title=f'Mean knowledge gap by {label}',
            name=f'step5_gap_by_{col}',
            figsize=(9, max(4, df[col].nunique() * 0.7)))
 
    # ==================================================================
    # financial beahaviours
    # ==================================================================
 
    # 6a: saving sophistication
    if 'saving_level_sophistication' in df.columns:
        gap_distribution_within(
            'saving_level_sophistication',
            x_labels={0.0: 'Did not save', 1.0: 'Informal/Cash',
                      2.0: 'Basic banking',  3.0: 'Investor'},
            title='Knowledge calibration by saving sophistication\n'
                  'Who saves more — tends to over or underestimate?',
            name='step6_saving_sophistication',
            col_order=[0.0, 1.0, 2.0, 3.0], figsize=(10, 5))
 
    # 6b: financial planning score
    if 'financial_planning_score' in df.columns:
        gap_distribution_within(
            'financial_planning_score',
            x_labels={i: str(i) for i in range(7)},
            title='Knowledge calibration by financial planning score (0–6)\n'
                  'Who plans more — tends to over or underestimate?',
            name='step6_financial_planning',
            col_order=list(range(7)), figsize=(11, 5))
 
    # 6c: risk aversion
    if 'risk_aversion_class' in df.columns:
        gap_distribution_within(
            'risk_aversion_class',
            x_labels={1: 'Very low', 2: 'Low', 3: 'Medium',
                      4: 'High',     5: 'Very high'},
            title='Knowledge calibration by risk aversion\n'
                  'Who is more risk-averse — tends to over or underestimate?',
            name='step6_risk_aversion',
            col_order=[1, 2, 3, 4, 5], figsize=(10, 5))
 
    # 6d
    binary_behav = {
        'use_dangerous_debt': (
            'Knowledge calibration: risky debt users vs not\n'
            'Are risky debt users more often overconfident?',
            {0: 'No risky debt', 1: 'Uses risky debt'}, [0, 1]),
        'use_own_resources': (
            'Knowledge calibration: uses own resources to cope vs not',
            {0: 'No', 1: 'Uses own resources'}, [0, 1]),
        'informal_external_help': (
            'Knowledge calibration: seeks informal help vs not',
            {0: 'No', 1: 'Seeks informal help'}, [0, 1]),
        'state_employee_pension': (
            'Knowledge calibration: relies on state pension vs not\n'
            'Are state-pension reliers more overconfident?',
            {0: 'No', 1: 'State/employee pension'}, [0, 1]),
        'private_pension_asset': (
            'Knowledge calibration: private pension/assets vs not\n'
            'Are private investors more underconfident?',
            {0: 'No', 1: 'Private pension/assets'}, [0, 1]),
    }
    for col, (title, labels, order) in binary_behav.items():
        if col not in df.columns:
            continue
        gap_distribution_within(col, labels, title,
                                  f'step6_{col}',
                                  col_order=order, figsize=(8, 5))
 
    # ==================================================================
    # digital variables
    # ==================================================================
 
    # 7a: digital skills 
    if 'digital_skills_score' in df.columns:
        df['_dig_bin'] = pd.qcut(df['digital_skills_score'], q=3,
                                  labels=['Low', 'Medium', 'High'])
        gap_distribution_within(
            '_dig_bin',
            x_labels={'Low': 'Low', 'Medium': 'Medium', 'High': 'High'},
            title='Knowledge calibration by digital skills level\n'
                  'Who has higher digital skills — over or underconfident?',
            name='step7_digital_skills',
            col_order=['Low', 'Medium', 'High'], figsize=(9, 5))
 
    # 7b: digital onboarding score (0-5)
    if 'digital_onboarding_score' in df.columns:
        gap_distribution_within(
            'digital_onboarding_score',
            x_labels={i: str(i) for i in range(6)},
            title='Knowledge calibration by digital onboarding score (0–5)\n'
                  'Who onboards more digitally — over or underconfident?',
            name='step7_digital_onboarding',
            col_order=list(range(6)), figsize=(11, 5))
 
    # 7c
    binary_digital = {
        'cyber_fraud_victim': (
            'Knowledge calibration: cyber fraud victims vs not\n'
            'Are victims more often overconfident?',
            {0: 'Non victim', 1: 'Victim'}, [0, 1]),
        'credit_excluded': (
            'Knowledge calibration: denied credit vs not\n'
            'Are those denied credit more often overconfident?',
            {0: 'Not denied', 1: 'Denied credit'}, [0, 1]),
        'institutional_friction': (
            'Knowledge calibration: institutional friction vs not',
            {0: 'No friction', 1: 'Friction'}, [0, 1]),
    }
    for col, (title, labels, order) in binary_digital.items():
        if col not in df.columns:
            continue
        gap_distribution_within(col, labels, title,
                                  f'step7_{col}',
                                  col_order=order, figsize=(8, 5))
        
    # ==================================================================
    # synthesis dot plot
    # ==================================================================
    from scipy.stats import chi2_contingency
    from matplotlib.lines import Line2D

    if 'internet_access_label' in df.columns and 'internet_access' not in df.columns:
        df['internet_access'] = df['internet_access_label'].map({'Yes': 1, 'No': 0})

    baseline_oc = df['gap_class'].eq('Overconfident').mean() * 100

    variables_summary = [
        ('use_dangerous_debt',          1,   'Uses risky debt'),
        ('use_own_resources',           1,   'Uses own resources'),
        ('informal_external_help',      1,   'Seeks informal help'),
        ('saving_level_sophistication', 0.0, 'Did not save'),
        ('saving_level_sophistication', 1.0, 'Saves informally/cash'),
        ('saving_level_sophistication', 2.0, 'Basic banking saver'),
        ('saving_level_sophistication', 3.0, 'Investor'),
        ('state_employee_pension',      1,   'Relies on state pension'),
        ('private_pension_asset',       1,   'Has private pension/assets'),
        ('cyber_fraud_victim',          1,   'Cyber fraud victim'),
        ('credit_excluded',             1,   'Denied credit'),
        ('institutional_friction',      1,   'Institutional friction'),
        ('internet_access',             0,   'No internet access'),
        ('internet_access',             1,   'Has internet access'),
    ]

    summary_rows = []
    for col, pos_val, label in variables_summary:
        if col not in df.columns:
            continue
        sub = df[df[col] == pos_val]
        if len(sub) < 30:
            continue
        pct_oc = sub['gap_class'].eq('Overconfident').mean() * 100
        delta  = pct_oc - baseline_oc
        ct     = pd.crosstab(df[col], df['gap_class'])
        _, p, _, _ = chi2_contingency(ct)
        summary_rows.append({'label': label, 'delta': delta, 'p_value': p})

    res_df = pd.DataFrame(summary_rows).sort_values('delta')

    fig, ax = plt.subplots(figsize=(11, 7))
    for i, row in enumerate(res_df.itertuples()):
        color  = ORANGE if row.delta > 0 else BLUE
        alpha  = 1.0 if row.p_value < 0.05 else 0.35
        marker = 'o' if row.p_value < 0.05 else 'D'
        ax.plot([0, row.delta], [i, i], color=color, alpha=alpha,
                linewidth=1.5, zorder=1)
        ax.scatter(row.delta, i, color=color, alpha=alpha,
                   s=120, zorder=2, marker=marker)
        ha   = 'left'  if row.delta >= 0 else 'right'
        xoff = row.delta + (0.4 if row.delta >= 0 else -0.4)
        sig  = '' if row.p_value < 0.05 else ' (n.s.)'
        ax.text(xoff, i, f'{row.delta:+.1f}%{sig}',
                va='center', ha=ha, fontsize=9,
                color=color, alpha=max(alpha, 0.6))

    ax.axvline(x=0, color='black', linewidth=1.2, linestyle='--', zorder=0)
    ax.axvspan(-3, 3, alpha=0.06, color='green')
    ax.set_yticks(range(len(res_df)))
    ax.set_yticklabels(res_df['label'], fontsize=10)
    ax.set_xlabel(f'Δ % Overconfident vs baseline ({baseline_oc:.1f}%)', fontsize=11)
    ax.set_title('Who tends to overestimate their financial knowledge?\n'
                 'Delta overconfidence by group — summary view',
                 fontweight='bold', fontsize=13)
    legend_elements = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=ORANGE,
               markersize=9, label='More overconfident than baseline'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=BLUE,
               markersize=9, label='Less overconfident than baseline'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor=GREY,
               markersize=7, label='Not significant (p ≥ 0.05)'),
    ]
    ax.legend(handles=legend_elements, frameon=False, loc='lower right', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(-12, 28)
    save('step8_overconfidence_summary')
    return df

#######################################
# financial profilation
########################################
# rule based prof
vuln_str_cols = ['expenditure_shock_capacity', 'income_not_covering_costs',
                  'lost_income_survival_time', 'retirement_plan_confidence']
 
def create_dimensions(df):
    
    df = df.copy()
    for c in vuln_str_cols:
        if c in df.columns:
            df[c] = df[c].fillna('unknown')
 
    s = pd.Series(0, index=df.index)
    if 'use_dangerous_debt' in df:        s += (df['use_dangerous_debt'] == 1) * 40
    if 'credit_excluded' in df:           s += (df['credit_excluded'] == 1) * 25
    if 'income_not_covering_costs' in df:
        s += (df['income_not_covering_costs'] == 'yes')     * 25
        s += (df['income_not_covering_costs'] == 'unknown') * 20
    if 'expenditure_shock_capacity' in df:
        s += (df['expenditure_shock_capacity'] == 'no')      * 10
        s += (df['expenditure_shock_capacity'] == 'unknown') * 10
 
    df['financial_vulnerability_score'] = s
    df['financial_vulnerability_class'] = s.apply(
        lambda v: 'High' if v >= 60 else 'Medium' if v >= 30 else 'Low')
 

    if 'obj_score' in df:
        df['knowledge_level'] = pd.cut(df['obj_score'],
            bins=[-1, 50, 70, 101], labels=['Low','Medium','High'])
    if 'financial_planning_score' in df:
        df['planning_level'] = pd.cut(df['financial_planning_score'],
            bins=[-1, 2, 4, 7], labels=['Low','Medium','High'])
    if 'digital_onboarding_score' in df:
        df['digital_engagement'] = pd.cut(df['digital_onboarding_score'],
            bins=[-1, 1, 3, 6], labels=['Low','Medium','High'])
    return df
 
 
def assign_profile(df):

    def rule(r):
        v   = r.get('financial_vulnerability_class')
        g   = r.get('gap_class')
        obj = r.get('obj_score', np.nan)
        pln = r.get('financial_planning_score', 0)
        dig = r.get('digital_onboarding_score', 0)
        sav = r.get('saving_level_sophistication', np.nan)
        if v == 'High' and g == 'Overconfident':              return 'Fragile & Overconfident'
        if v == 'High':                                        return 'Fragile & At Risk'
        if obj < 50 and g == 'Overconfident':                  return 'Unskilled & Overconfident'
        if obj >= 70 and g in ('Calibrated', 'Underconfident'):return 'Conscious Expert'
        if obj >= 50 and pln >= 4:                             return 'Financial Planner'
        if obj >= 50 and dig >= 2:                             return 'Digital Adopter'
        if obj < 40 and dig <= 1 and pln <= 1:                 return 'Excluded/Passive'
        if sav >= 2 and obj >= 50 and g != 'Overconfident':    return 'Aware Saver/Investor'
        return 'Standard/Average'
    df = df.copy()
    df['financial_profile'] = df.apply(rule, axis=1)
    return df
 
 
# clustering
cluster_features = [
    'obj_score','financial_planning_score','saving_level_sophistication',
    'digital_onboarding_score','digital_skills_score','consumer_debt_score',
    'traditional_investment_score','alternative_asset_score',
    'transactional_score','saving_protection_score','risk_aversion_class',
    'basic_admin_intensity','daily_transactional_intensity',
    'advanced_fintech_intensity',
]

skewed = ['traditional_investment_score','digital_onboarding_score',
          'alternative_asset_score','advanced_fintech_intensity',
          'saving_protection_score']
 
def prepare_x(df):
    
    avail = [c for c in cluster_features if c in df.columns]
    x = df[avail].apply(pd.to_numeric, errors='coerce')
    x = x.fillna(x.median())
    for c in [s for s in skewed if s in x.columns]:
        x[c] = np.log1p(x[c])
    return pd.DataFrame(StandardScaler().fit_transform(x),
                         columns=avail, index=df.index)
 
 
def select_k_and_fit(x, k_range=range(2, 9), out='clustering_plots'):
   
    os.makedirs(out, exist_ok=True)
    inertia, sils = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(x)
        inertia.append(km.inertia_)
        sils.append(silhouette_score(x, labels))
        print(f"  k={k} | inertia={km.inertia_:.0f} | silhouette={sils[-1]:.3f}")
 
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(list(k_range), inertia, 'o-'); axes[0].set_title('Elbow')
    axes[1].plot(list(k_range), sils,    'o-'); axes[1].set_title('Silhouette')
    for a in axes: a.set_xlabel('K'); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f'{out}/k_diagnostics.png', dpi=130); plt.close()
 
    best_k = list(k_range)[int(np.argmax(sils))]
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    return km.fit_predict(x), best_k
 
 
# comparison
def compare(df, out='clustering_plots'):
    """Crosstab heatmap + ARI + NMI fra rule-based e cluster."""
    pct = pd.crosstab(df['financial_profile'], df['cluster'],
                       normalize='columns') * 100
    plt.figure(figsize=(10, 7))
    sns.heatmap(pct, annot=True, fmt='.0f', cmap='YlOrRd',
                cbar_kws={'label': '% within cluster'})
    plt.title('Rule-based × Cluster'); plt.tight_layout()
    plt.savefig(f'{out}/crosstab.png', dpi=130); plt.close()
    pct.round(1).to_csv(f'{out}/crosstab_pct.csv')
 
    ari = adjusted_rand_score(df['financial_profile'], df['cluster'])
    nmi = normalized_mutual_info_score(df['financial_profile'], df['cluster'])
    print(f"  ARI = {ari:.3f}   NMI = {nmi:.3f}")
    return ari, nmi
################
def main():
    file_path = os.getenv("DATA_PATH")
    try:
        # 1. loading
        df = pd.read_csv(file_path)
        df.columns = [col.lower() for col in df.columns]

        # 2. preprocessing pipeline
        df = preprocess_financial_variables(df)
        df_main, df_active = preprocess_products_and_digital(df)
        df_main  = clean_qk(df_main)
        df_main  = calcola_e_sostituisci_score(df_main)
        df_final = engineer_demographic_features(df_main)

        print("columns df_final:", df_final.columns.tolist())
        print("dimensions df_final:", df_final.shape)

        # 3.  df_active 
        df_active = clean_qk(df_active)
        df_active = calcola_e_sostituisci_score(df_active)
        df_active_final = engineer_demographic_features(df_active)
        df_active_final.to_csv("cleaned_active_df2.csv", index=False)

              
        #EDA


        df_final = eda_research(df_final, output_dir='eda_research2')
       
        # 4. rule-based prof

        df_final = create_dimensions(df_final)
        df_final = assign_profile(df_final)

        
        counts = df_final["financial_profile"].value_counts(dropna=False)
        pct    = df_final["financial_profile"].value_counts(
            normalize=True, dropna=False
        ) * 100
        print(pd.concat([counts.rename("n"),
                         pct.round(1).rename("%")], axis=1))

        # 5. Clustering KMeans
        
        x = prepare_x(df_final)
        print("\nVariables used for clustering:")
        print(list(x.columns))

        cluster_labels, best_k = select_k_and_fit(
            x, k_range=range(2, 10), out="clustering_plots"
        )
        df_final["cluster"] = cluster_labels

        print(f"\nBest k selected by silhouette: {best_k}")
        print(f"Cluster sizes, K={best_k}:")
        print(df_final["cluster"].value_counts().sort_index())

        # 6. comparison rule-based × cluster
        
        compare(df_final, out="clustering_plots")

        # 7. final saving
        df_final.to_csv("cleaned_df2.csv", index=False)

        

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
