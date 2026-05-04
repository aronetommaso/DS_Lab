import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

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



def preprocess_financial_variables(df):
    """
    Preprocesses financial variables from the IACOFI 2023 survey.
    Transforms multi-option questions into semantic scores/flags to reduce dimensionality,
    and maps categorical questions to readable strings while handling missing values.
    """
    
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

    # --- QF2: Score Pianificazione Finanziaria (0 to 6) ---
    qf2_cols = get_existing_cols(['qf2_1', 'qf2_2', 'qf2_3', 'qf2_4', 'qf2_5', 'qf2_6'])
    if qf2_cols:
        # Sum only the '1' (Yes) responses. Missing values in these binary cols are treated as 0 for the score sum
        df['Financial_planning_score'] = df[qf2_cols].eq(1).sum(axis=1)

    # --- QF3: Livello Sofisticazione Risparmio (0 to 3) ---
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

    # --- QF9: Pensione (Macro-Flags 0/1) ---
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
    print("\nUnique categories per column (should be much lower now!):")
    for col in df_financial.columns:
        # Using dropna() to count only valid categories
        print(f"for {col}: {len(df_financial[col].dropna().unique())}")

    return df_financial

def plotting_financial_variables(df):
    """
    Generates and saves a plot for each financial variable in the dataframe.
    Saves all plots inside the 'plot' folder.
    """
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
        
    print(f"All plots have been saved in the '{output_dir}' folder.")




def main():
    file_path = "c:/Users/tomma/Downloads/Database_ASCII_EN/Database_ENG.csv"
    try:
        df = pd.read_csv(file_path)
        df = preprocess_financial_variables(df)
        plotting_financial_variables(df)
        

            
    except Exception as e:
        print(f"An error occurred: {e}")




if __name__ == "__main__":
    main()