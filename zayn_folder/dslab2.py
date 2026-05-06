import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import reportlab
import scipy

# --- OPZIONI PANDAS PER VEDERE TUTTE LE TUE COLONNE SENZA TRONCAMENTI ---
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def preprocess_complete_iacofi(df):
    """
    Comprehensive preprocessing for IACOFI 2023 dataset.
    1. Maps ALL variables from the codebook to descriptive names.
    2. Consolidates all multiple-choice "tick all that apply" variables 
       (with _1, _2, etc.) into single categorical string variables.
    """
    df.columns = [col.lower() for col in df.columns]

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

    full_name_mapping = {
        'qd1': 'gender', 'qd7': 'age', 'qd7_a': 'age_bands', 'qd2': 'macro_region', 'qd3': 'urbanization_level', 'qd10': 'work_situation', 'qd14': 'internet_access', 'qd5_ad': 'household_adults_count', 'qd5_ch': 'household_children_count',
        'qf1_a': 'personal_budget_decisions', 'qf1': 'household_budget_decisions', 'qf4': 'expenditure_shock_capacity', 'qf8': 'retirement_plan_confidence', 'qf11': 'income_not_covering_costs', 'qf13': 'lost_income_survival_time',
        'qp5': 'shopping_around_behavior', 'qp7_add1': 'risk_aversion',
        'qk1': 'self_rated_knowledge', 'qk3': 'inflation_knowledge_brothers', 'qk4': 'interest_on_loan', 'qk5': 'simple_interest', 'qk6': 'compound_interest', 'qk10': 'mortgage_knowledge',
        'qk7_1': 'know_high_return_high_risk', 'qk7_2': 'know_high_inflation_cost_living', 'qk7_3': 'know_reduce_risk_diversify', 'qk7_4': 'know_digital_contract_paper', 'qk7_5': 'know_data_targeted_offers', 'qk7_6': 'know_crypto_legal_tender',
        'qp9_1': 'freq_check_balance_online', 'qp9_3': 'freq_pay_bills_online', 'qp9_4': 'freq_buy_online', 'qp9_5': 'freq_transfer_money_online', 'qp9_6': 'freq_manage_finance_online', 'qp9_7': 'freq_mobile_payment_shop', 'qp9_10': 'freq_roboadvisor',
        'qs1_1': 'att_spend_over_save', 'qs1_2': 'att_risk_money', 'qs1_3': 'att_money_to_spend', 'qs1_4': 'att_satisfied_finance', 'qs1_5': 'att_watch_affairs', 'qs1_7': 'att_finance_limits_life', 'qs1_8': 'att_set_long_term_goals', 'qs1_9': 'att_trust_bank_safety', 'qs1_10': 'att_too_much_debt', 'qs1_13': 'att_good_time_crypto',
        'qs2_1': 'beh_worry_expenses', 'qs2_2': 'beh_finances_control_life', 'qs2_3': 'beh_consider_afford', 'qs2_4': 'beh_money_left_over', 'qs2_5': 'beh_pay_bills_on_time', 'qs2_6': 'beh_share_pins', 'qs2_7': 'beh_check_regulated_provider', 'qs2_8': 'beh_share_finance_public', 'qs2_9': 'beh_consider_esg',
        'qs3_2': 'sit_prefer_ethical_intermediary', 'qs3_3': 'sit_feel_never_have_things', 'qs3_9': 'sit_concern_money_wont_last', 'qs3_10': 'sit_just_getting_by', 'qs3_11': 'sit_live_for_today', 'qs3_12': 'sit_buy_lottery', 'qs3_13': 'sit_change_passwords',
        'qs4_1': 'dig_safe_public_wifi', 'qs4_2': 'dig_check_website_security', 'qs4_3': 'dig_ignore_tc', 'qs4_4': 'dig_tools_facilitate', 'qs4_5': 'dig_trust_fintech', 'qs4_6': 'dig_ok_social_data_credit', 'qs4_7': 'dig_impulsive_online', 'qs4_8': 'dig_read_print_paper_over_online',
        'qs5_4': 'esg_profit_over_env', 'qs5_5': 'esg_profit_over_social', 'qs5_6': 'esg_profit_over_gov',
        'qd6_1': 'freq_write_doc', 'qd6_2': 'freq_email', 'qd6_3': 'freq_mobile_call', 'qd6_4': 'freq_internet_call', 'qd6_5': 'freq_social_networks', 'qd6_6': 'freq_instant_messaging', 'qd6_7': 'freq_search_online',
        'qd9': 'educational_level', 'qd12': 'nationality', 'qd13': 'income_band'
    }
    df = df.rename(columns=full_name_mapping)
    return df


# ==============================================================================
# --- LA TUA FUNZIONE DEMOGRAFICA ---
# ==============================================================================
def engineer_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3: Full Demographic & Background Feature Engineering (IACOFI 2023).
    """
    df = df.copy()

    # 1. GENDER
    df['gender'] = df['gender'].map({0: 'Woman', 1: 'Man'})

    # 2. AGE GROUP 
    def _map_age_to_band(val):
        if val in (18, 19):      return 1
        if 20 <= val <= 29:      return 2
        if 30 <= val <= 39:      return 3
        if 40 <= val <= 49:      return 4
        if 50 <= val <= 59:      return 5
        if 60 <= val <= 69:      return 6
        if 70 <= val <= 79:      return 7
        return val  

    age_band_labels = {1: '18-19', 2: '20-29', 3: '30-39', 4: '40-49', 5: '50-59', 6: '60-69', 7: '70-79'}
    age_mapped  = df['age'].apply(_map_age_to_band)
    # Fixato il warning con replace e fillna
    age_unified = age_mapped.replace(-99, pd.NA).fillna(df['age_bands'])
    df['age_group'] = age_unified.map(age_band_labels)

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
    # 6. DIGITAL SKILLS SCORE (Inversione basata sull'evidenza dei dati grezzi)
    # 6. DIGITAL SKILLS SCORE (Scala 1-4, Inversione correttiva applicata)
    digital_cols = [
        'freq_write_doc', 'freq_email', 'freq_mobile_call',
        'freq_internet_call', 'freq_social_networks',
        'freq_instant_messaging', 'freq_search_online'
    ]
    
    # 1. Converti in numerico escludendo i codici negativi (-97, -99)
    temp_dig = df[digital_cols].apply(pd.to_numeric, errors='coerce').copy()
    temp_dig[temp_dig <= 0] = np.nan
    
    # 2. Calcola la media degli item validi (ignora i NaN)
    # Media originale: giovani ~1.5, anziani ~3.0
    mean_raw = temp_dig.mean(axis=1)
    
    # 3. TRASFORMAZIONE SPECULARE 1-4
    # Applichiamo la formula: NewValue = (Max + Min) - OldValue
    # Ovvero: 5 - Media_Grezza
    # Se media=1 (Max uso) -> Score = 4 (Molto Spesso)
    # Se media=4 (Min uso) -> Score = 1 (Mai)
    df['digital_skills_score'] = (5 - mean_raw)
    
    # 4. Gestione chi non usa internet
    # Chi ha solo risposte negative o NaN viene impostato a 1 (Mai)
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

    return df
# ==============================================================================

def eda_demographic_features(df: pd.DataFrame, output_dir: str = 'plot/demographic') -> None:
    """
    Exploratory Data Analysis per le variabili demografiche/background
    ingegnerizzate da engineer_demographic_features().
    
    Produce:
      - Barplot con frequenze relative (%) per variabili categoriche
      - Histogram + KDE per variabili numeriche continue (household_size, digital_skills_score)
      - Stampa in console: distribuzione, missing %, statistiche descrittive
    
    Parameters:
        df          : dataframe dopo engineer_demographic_features()
        output_dir  : cartella dove salvare i grafici
    """
    os.makedirs(output_dir, exist_ok=True)

    # Palette coerente con il tuo progetto
    PALETTE = 'viridis'
    FIG_W, FIG_H = 10, 6

    # Variabili create dalla tua funzione + loro tipo
    categorical_cols = [
        'gender', 'age_group', 'macro_region_label', 'urban_area_label',
        'living_status', 'edu_level_grouped', 'work_status',
        'is_italian', 'income_label', 'internet_access_label'
    ]

    # Ordini logici (non alfabetici) dove ha senso
    custom_orders = {
        'age_group':          ['18-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79'],
        'edu_level_grouped':  ['No Education', 'Primary', 'Middle School', 'High School', 'University'],
        'income_label':       ['<=1750€', '1751-2900€', '>2900€', 'Unknown'],
        'urban_area_label':   ['<3k', '3k-15k', '15k-100k', '100k-1M', '>1M'],
        'macro_region_label': ['North-West', 'North-East', 'Center', 'South', 'Islands'],
    }

    numeric_cols = ['household_size', 'digital_skills_score']

    # ------------------------------------------------------------------
    # HEADER console
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("📊  EDA — VARIABILI DEMOGRAFICHE E BACKGROUND (IACOFI 2023)")
    print("="*70)

    # ------------------------------------------------------------------
    # 1. VARIABILI CATEGORICHE — Barplot frequenze relative
    # ------------------------------------------------------------------
    for col in categorical_cols:
        if col not in df.columns:
            print(f"  ⚠️  Colonna '{col}' non trovata, skip.")
            continue

        series = df[col]
        n_total   = len(series)
        n_missing = series.isna().sum()
        pct_miss  = n_missing / n_total * 100

        # Frequenze relative (esclusi NaN per il grafico, ma segnalati)
        freq = series.value_counts(dropna=True, normalize=True) * 100

        # Applica ordine custom se disponibile, altrimenti ordina per freq
        if col in custom_orders:
            order = [v for v in custom_orders[col] if v in freq.index]
        else:
            order = freq.index.tolist()

        freq = freq.reindex(order).dropna()

        # --- Console summary ---
        print(f"\n{'─'*60}")
        print(f"VARIABILE: {col.upper()}")
        print(f"  Dati Mancanti : {n_missing} ({pct_miss:.1f}%)")
        print(f"  Valori Unici  : {series.nunique(dropna=True)}")
        print("  Frequenze Relative (su validi):")
        for val, pct in freq.items():
            print(f"    [{val}] : {pct:.1f}%")

        # --- Grafico ---
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

        bars = ax.bar(
            range(len(freq)),
            freq.values,
            color=sns.color_palette(PALETTE, len(freq)),
            edgecolor='white', linewidth=0.8
        )

        # Etichette % sopra le barre
        for bar, pct in zip(bars, freq.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold'
            )

        ax.set_xticks(range(len(freq)))
        ax.set_xticklabels(freq.index, rotation=30, ha='right', fontsize=10)
        ax.set_ylabel('Frequenza Relativa (%)', fontsize=11)
        ax.set_title(
            f'Distribuzione di {col}\n'
            f'(N={n_total - n_missing:,} validi  |  Missing: {pct_miss:.1f}%)',
            fontsize=13, fontweight='bold'
        )
        ax.set_ylim(0, freq.max() * 1.15)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{col}.png'), dpi=150)
        plt.close()

    # ------------------------------------------------------------------
    # 2. VARIABILI NUMERICHE — Histogram + KDE + boxplot laterale
    # ------------------------------------------------------------------
    num_meta = {
        'household_size': {
            'title': 'Household Size (numero componenti)',
            'xlabel': 'Numero componenti',
            'note': 'Rispondente + adulti conviventi + minori'
        },
        'digital_skills_score': {
            'title': 'Digital Skills Score (indice 0–28)',
            'xlabel': 'Score aggregato (7 items × max 4)',
            'note': 'Somma frequenze attività digitali (qd6_1–qd6_7)'
        }
    }

    for col in numeric_cols:
        if col not in df.columns:
            print(f"  ⚠️  Colonna '{col}' non trovata, skip.")
            continue

        series = pd.to_numeric(df[col], errors='coerce').dropna()
        n_missing = len(df) - len(series)
        pct_miss  = n_missing / len(df) * 100
        meta      = num_meta.get(col, {'title': col, 'xlabel': col, 'note': ''})

        # --- Console summary ---
        print(f"\n{'─'*60}")
        print(f"VARIABILE: {col.upper()}")
        print(f"  Dati Mancanti : {n_missing} ({pct_miss:.1f}%)")
        print(f"  Minimo        : {series.min():.1f}")
        print(f"  Q1            : {series.quantile(0.25):.1f}")
        print(f"  Mediana       : {series.median():.1f}")
        print(f"  Media         : {series.mean():.2f}")
        print(f"  Q3            : {series.quantile(0.75):.1f}")
        print(f"  Massimo       : {series.max():.1f}")

        # --- Grafico: histogram principale + mini boxplot sotto ---
        fig, (ax_hist, ax_box) = plt.subplots(
            2, 1, figsize=(FIG_W, FIG_H + 1),
            gridspec_kw={'height_ratios': [5, 1]},
            sharex=True
        )

        # Histogram + KDE
        sns.histplot(series, kde=True, ax=ax_hist,
                     color=sns.color_palette(PALETTE, 1)[0],
                     bins=int(series.nunique() if series.nunique() <= 30 else 30),
                     edgecolor='white', linewidth=0.5)

        # Linee verticali per media e mediana
        ax_hist.axvline(series.mean(),   color='crimson',    linestyle='--', linewidth=1.5, label=f'Media: {series.mean():.1f}')
        ax_hist.axvline(series.median(), color='darkorange',  linestyle=':',  linewidth=1.5, label=f'Mediana: {series.median():.1f}')
        ax_hist.legend(fontsize=9)
        ax_hist.set_ylabel('Frequenza', fontsize=11)
        ax_hist.set_title(
            f'{meta["title"]}\n'
            f'(N={len(series):,} validi  |  Missing: {pct_miss:.1f}%  |  {meta["note"]})',
            fontsize=12, fontweight='bold'
        )
        ax_hist.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax_hist.set_axisbelow(True)

        # Boxplot orizzontale sotto
        ax_box.boxplot(series, vert=False, patch_artist=True,
                       boxprops=dict(facecolor=sns.color_palette(PALETTE, 3)[1], alpha=0.6),
                       medianprops=dict(color='crimson', linewidth=2),
                       flierprops=dict(marker='o', markersize=3, alpha=0.3))
        ax_box.set_xlabel(meta['xlabel'], fontsize=11)
        ax_box.set_yticks([])
        ax_box.yaxis.grid(False)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{col}.png'), dpi=150)
        plt.close()

    print(f"\n{'='*70}")
    print(f"✅ EDA completata. Grafici salvati in '{output_dir}/'")
    print(f"{'='*70}\n")

# --- LA FUNZIONE DEL TUO COLLEGA (INVARIATA) ---
def preprocess_financial_variables(df):
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
        'qf1_a': 'personal_budget_decisions', 'qf1': 'household_budget_decisions', 'qf4': 'expenditure_shock_capacity', 'qf8': 'retirement_plan_confidence', 'qf11': 'income_not_covering_costs', 'qf13': 'lost_income_survival_time'
    }
    df = df.rename(columns=base_financial_mapping)

    final_engineered_cols = list(base_financial_mapping.values()) + [
        'Financial_planning_score', 'Saving_level_sophistication', 'State_employee_pension', 'Private_pension_asset', 'Informal_network_pension', 'Use_own_resources', 'Informal_external_help', 'Use_dangerous_debt'
    ]
    
    available_cols = [col for col in final_engineered_cols if col in df.columns]
    df_financial = df[available_cols].copy()

    original_binary_patterns = ['qf2_', 'qf3_', 'qf9_', 'qf12_']
    cols_to_drop = [c for c in df_financial.columns if any(p in c for p in original_binary_patterns)]
    df_financial = df_financial.drop(columns=cols_to_drop, errors='ignore')

    return df_financial


# --- LA FUNZIONE DI PLOTTING DEL TUO COLLEGA (INVARIATA) ---
def plotting_financial_variables(df):
    output_dir = 'plot'
    os.makedirs(output_dir, exist_ok=True)
    for col in df.columns:
        plt.figure(figsize=(10, 6))
        series = df[col].dropna()
        if series.empty:
            plt.close()
            continue
        is_categorical = pd.api.types.is_object_dtype(series)
        is_discrete = pd.api.types.is_numeric_dtype(series) and (series.nunique() <= 15 or all(series % 1 == 0))

        if is_categorical or is_discrete:
            sns.countplot(data=df, x=col, order=series.value_counts().index, hue=col, legend=False, palette='viridis')
            plt.title(f'Class count for variable: {col}')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Count')
        else:
            sns.histplot(series, kde=True, color='skyblue', bins=30)
            plt.title(f'Distribution of variable: {col}')
            plt.ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}.png"))
        plt.close()
        
    print(f"All plots have been saved in the '{output_dir}' folder.")


def main():
    # File path aggiornato con la "r" e formato per Windows
    file_path = r"c:/Users/loren/OneDrive/Desktop/Data Science/Data Science Lab/Database_ENG.csv"
    
    try:
        df = pd.read_csv(file_path)
        
        # 1. Mappatura base INIZIALE (fondamentale per far funzionare le altre due funzioni!)
        df = preprocess_complete_iacofi(df)
        
        # 2. LA TUA FUNZIONE: Creazione variabili Demografiche
        df = engineer_demographic_features(df)
        
        # --- STAMPA DEDICATA PER VEDERE IL TUO LAVORO ---
        print("\n" + "="*80)
        print("✅ TEST DELLE TUE VARIABILI DEMOGRAFICHE/BACKGROUND ✅")
        print("="*80)
        
        # Le colonne che la tua funzione crea
        demo_cols = ['gender', 'age_group', 'macro_region_label', 'urban_area_label',
                     'living_status', 'household_size', 'digital_skills_score',
                     'edu_level_grouped', 'work_status', 'is_italian',
                     'income_label', 'internet_access_label']
                     
        # Seleziona e mostra solo le tue colonne in console
        created_cols = [c for c in demo_cols if c in df.columns]
        print(f"-> Sono state generate con successo {len(created_cols)} colonne demografiche.\n")
        print("Ecco le prime 10 righe del tuo lavoro in espanso:\n")
        print(df[created_cols].head(10))
        print("="*80 + "\n")
        
        # 3. La funzione del tuo collega (crea df_financial isolando le variabili finanziarie)
        df_financial = preprocess_financial_variables(df)
        
        # 4. Plotting del tuo collega
        plotting_financial_variables(df_financial)

        eda_demographic_features(df)

        from generate_demographic_report import generate_demographic_report
        generate_demographic_report(df, output_path='plot/Demographic_EDA_Report.pdf')

        from bivariate_demographic_analysis import bivariate_demographic_analysis
        bivariate_demographic_analysis(df, output_dir='plot/bivariate', show=False)
            
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
