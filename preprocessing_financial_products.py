"""
IACOFI 2023 - Data Engineering Pipeline
Questo script esegue il preprocessing dei dati grezzi, l'ingegneria delle feature 
(Blocchi QF e QP) e salva i dataset puliti pronti per l'esplorazione e il modeling.
"""

import pandas as pd
import numpy as np
import warnings

# Ignora i warning minori per avere un output a terminale pulito
warnings.filterwarnings('ignore')

def preprocess_complete_iacofi(df):
    """
    Rinomina tutte le variabili di background dal codebook a nomi descrittivi.
    """
    full_name_mapping = {
        # Personal and Household
        'qd1': 'gender', 'qd7': 'age', 'qd7_a': 'age_bands', 'qd2': 'macro_region',
        'qd3': 'urbanization_level', 'qd10': 'work_situation', 'qd14': 'internet_access',
        'qd5_ad': 'household_adults_count', 'qd5_ch': 'household_children_count',
        
        # Financial Knowledge
        'qk1': 'self_rated_knowledge', 'qk3': 'inflation_knowledge_brothers',
        'qk4': 'interest_on_loan', 'qk5': 'simple_interest', 'qk6': 'compound_interest',
        'qk10': 'mortgage_knowledge',
        
        # True/False Knowledge Statements (0/1)
        'qk7_1': 'know_high_return_high_risk', 'qk7_2': 'know_high_inflation_cost_living',
        'qk7_3': 'know_reduce_risk_diversify', 'qk7_4': 'know_digital_contract_paper',
        'qk7_5': 'know_data_targeted_offers', 'qk7_6': 'know_crypto_legal_tender',

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
    return df.rename(columns=full_name_mapping)


def preprocess_financial_variables(df):
    """
    Preprocesses QF variables (Financial Resilience).
    """
    categorical_value_maps = {
        'qf1_a': {1: 'yes', 0: 'no'},
        'qf1': {1: 'you', 2: 'you and someone else', 3: 'someone else'},
        'qf4': {1: 'yes_immediate', 2: 'yes_selling_assets', 0: 'no'},
        'qf8': {1: 'very_confident', 2: 'confident', 3: 'middle', 4: 'not_confident', 5: 'not_confident_at_all', 6: 'no_retirement_plan'},
        'qf11': {1: 'yes', 0: 'no'},
        'qf13': {1: 'less_than_a_week', 2: '1_week_to_1_month', 3: '1_to_3_months', 4: '3_to_6_months', 5: '6_months_or_more'}
    }

    missing_codes = [-97, 97, -98, 98, -99, 99, -999]
    for col, mapping in categorical_value_maps.items():
        if col in df.columns:
            df[col] = df[col].replace(missing_codes, np.nan)
            df[col] = df[col].map(mapping)

    def get_existing_cols(cols):
        return [c for c in cols if c in df.columns]

    qf2_cols = get_existing_cols(['qf2_1', 'qf2_2', 'qf2_3', 'qf2_4', 'qf2_5', 'qf2_6'])
    if qf2_cols:
        df['Financial_planning_score'] = df[qf2_cols].eq(1).sum(axis=1)

    def calc_saving_sophistication(row):
        if row.get('qf3_5') == 1 or row.get('qf3_6') == 1 or row.get('qf3_7') == 1: return 3
        elif row.get('qf3_2') == 1: return 2
        elif row.get('qf3_1') == 1 or row.get('qf3_3') == 1 or row.get('qf3_4') == 1: return 1
        elif row.get('qf3_98') == 1: return 0
        return np.nan 

    if any('qf3' in c for c in df.columns):
        df['Saving_level_sophistication'] = df.apply(calc_saving_sophistication, axis=1)

    qf9_state = get_existing_cols(['qf9_1', 'qf9_12'])
    qf9_private = get_existing_cols(['qf9_2', 'qf9_3', 'qf9_4', 'qf9_5', 'qf9_6'])
    qf9_network = get_existing_cols(['qf9_7', 'qf9_8', 'qf9_10'])

    if qf9_state: df['State_employee_pension'] = df[qf9_state].eq(1).any(axis=1).astype(int)
    if qf9_private: df['Private_pension_asset'] = df[qf9_private].eq(1).any(axis=1).astype(int)
    if qf9_network: df['Informal_network_pension'] = df[qf9_network].eq(1).any(axis=1).astype(int)

    qf12_own = get_existing_cols(['qf12_1_1', 'qf12_1_2', 'qf12_1_3'])
    qf12_help = get_existing_cols(['qf12_2_1', 'qf12_2_2', 'qf12_2_3', 'qf12_3_1'])
    qf12_debt = get_existing_cols(['qf12_4_1', 'qf12_4_2', 'qf12_5_1', 'qf12_5_2', 'qf12_5_3', 'qf12_5_4', 'qf12_5_5', 'qf12_6_1', 'qf12_6_2'])

    if qf12_own: df['Use_own_resources'] = df[qf12_own].eq(1).any(axis=1).astype(int)
    if qf12_help: df['Informal_external_help'] = df[qf12_help].eq(1).any(axis=1).astype(int)
    if qf12_debt: df['Use_dangerous_debt'] = df[qf12_debt].eq(1).any(axis=1).astype(int)

    base_financial_mapping = {
        'qf1_a': 'personal_budget_decisions', 'qf1': 'household_budget_decisions',
        'qf4': 'expenditure_shock_capacity', 'qf8': 'retirement_plan_confidence',
        'qf11': 'income_not_covering_costs', 'qf13': 'lost_income_survival_time'
    }
    df = df.rename(columns=base_financial_mapping)

    final_engineered_cols = list(base_financial_mapping.values()) + [
        'Financial_planning_score', 'Saving_level_sophistication',
        'State_employee_pension', 'Private_pension_asset', 'Informal_network_pension',
        'Use_own_resources', 'Informal_external_help', 'Use_dangerous_debt'
    ]
    
    available_cols = [col for col in final_engineered_cols if col in df.columns]
    print("✅ QF Block (Financial Resilience) preprocessed successfully.")
    return df[available_cols].copy()


def preprocess_products_and_digital(df):
    """
    Preprocesses the QP block (Financial Products, Digital Behaviors, Issues).
    """
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
    
    df_qp_main = df_qp[[c for c in final_qp_cols if c in df_qp.columns]].copy()
    print("✅ QP Block (Products & Digital) preprocessed successfully.")
    return df_qp_main, df_active


def main():
    """
    Funzione principale: carica i dati, orchestra il preprocessing e salva i CSV.
    """
    # 1. Caricamento Dati
    file_path = "C:/Users/HP/Downloads/Database_Bancaditalia/Database_ENG.csv"
    
    try:
        df_raw = pd.read_csv(file_path, low_memory=False)
        df_raw.columns = [col.lower() for col in df_raw.columns]
        print(f"✅ Dataset RAW successfully loaded. Shape: {df_raw.shape}")
    except Exception as e:
        print(f"❌ Error loading the dataset: {e}")
        return

    print("\n🚀 Starting the Data Engineering Pipeline...\n")
    df_working_copy = df_raw.copy()
    
    # 2. Elaborazione blocchi indipendenti
    df_qf_clean = preprocess_financial_variables(df_working_copy)
    df_qp_clean, df_active_shoppers = preprocess_products_and_digital(df_working_copy)
    
    # 3. Rinominare variabili demografiche e attitudini
    df_renamed_full = preprocess_complete_iacofi(df_working_copy)
    demographic_cols = [c for c in df_renamed_full.columns if not c.startswith(('qf', 'qp', 'qd', 'qk', 'qs'))]
    df_demographics = df_renamed_full[demographic_cols].copy()

    # 4. Creazione del Master Dataset
    df_master = pd.concat([df_demographics, df_qf_clean, df_qp_clean], axis=1)
    df_master = df_master.loc[:,~df_master.columns.duplicated()]

    print("-" * 60)
    print(f"📊 MASTER DATASET CREATO - Shape: {df_master.shape}")
    print(f"🛒 ACTIVE SHOPPERS CREATO - Shape: {df_active_shoppers.shape}")
    print("-" * 60)

    # 5. Salvataggio in formato CSV per il Notebook
    print("💾 Salvataggio dei file puliti in corso...")
    df_master.to_csv("master_dataset_clean.csv", index=False)
    df_active_shoppers.to_csv("active_shoppers_clean.csv", index=False)
    print("✅ Salvataggio completato! Troverai i file CSV nella stessa cartella di questo script.")

if __name__ == "__main__":
    main()