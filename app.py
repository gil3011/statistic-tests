import streamlit as st
import pandas as pd
import statisticTest
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# constant
alpha = 0.05

## Binary tests (Chi-square)
h0_binary = "אין הבדל מובהק סטטיסטית (המשתנים בלתי תלויים)"
h1_binary = "יש הבדל מובהק סטטיסטית (המשתנים תלויים)"

## Continuous, independent groups (t-test or Mann–Whitney)
h0_continuous = "הקבוצות מתפלגות באופן דומה (אין הבדל מובהק סטטיסטית)"
h1_continuous = "הקבוצות מתפלגות באופן שונה (יש הבדל מובהק סטטיסטית)"

## Continuous, paired groups (paired t-test or Wilcoxon)
h0_continuous_paired = "אין תלות מובהקת בין המדדים"
h1_continuous_paired = "יש תלות מובהקת בין המדדים"

# Load Data
@st.cache_data
def load_data():
    df = pd.read_excel("Explored_featured_data.xlsx")
    df["זכר"] = df["מין"].apply(lambda x: 1 if x == "בן" else 0)
    return df

data = load_data()
cols = data.drop(['index','מחלה נוכחית','תאריך ביקור','יישוב','מין',
                  'valid_FEV1_PCT','valid_FVC_PCT','valid_FEF25_75_PCT','valid_FEV1/FVC',
                  'FVC %','FEV1 %','FEF25-75%','FEV1/FVC'], axis=1).columns

continuous = {'FVC %':'FVC %','FEV1 %':'FEV1 %','FEF25-75%':'FEF25-75%','FEV1/FVC':'FEV1/FVC'}

binary = {'FVC %':'valid_FVC_PCT','FEV1 %':'valid_FEV1_PCT','FEF25-75%':'valid_FEF25_75_PCT','FEV1/FVC':'valid_FEV1/FVC'}

def interpret_result_binary(p_value):
    if p_value < alpha:
        return f"🔍 {h1_binary} (p = {p_value:.4f} < {alpha})."
    else:
        return f"ℹ️ {h0_binary} (p = {p_value:.4f} ≥ {alpha})."

def interpret_result_continuous(p_value):
    if p_value < alpha:
        return f"🔍 {h1_continuous} (p = {p_value:.4f} < {alpha})."
    else:
        return f"ℹ️ {h0_continuous} (p = {p_value:.4f} ≥ {alpha})."
    
def interpret_result_continuous_paired(p_value):
    if p_value < alpha:
        return f"🔍 {h1_continuous_paired} (p = {p_value:.4f} < {alpha})."
    else:
        return f"ℹ️ {h0_continuous_paired} (p = {p_value:.4f} ≥ {alpha})."

# UI
st.header("📊 Featured Statistic Analysis")
selected_feature = st.selectbox("Select a feature to analyze", cols)
st.divider()

# Distribution
st.subheader("📌 Distribution of Selected Feature")
if data[selected_feature].dropna().nunique() == 2:
    fig, score = statisticTest.plot_pie(data, selected_feature)
    col1, _, col2, _ = st.columns([0.6, 0.1, 0.3, 0.1])
    with col1:
        st.pyplot(fig)
    with col2:
        scoes_df = pd.DataFrame.from_dict(score, orient='index', columns=['Count'])
        st.table(scoes_df)
else:
    fig,stats = statisticTest.plot_continuous_distribution(data, selected_feature)
    col1,_,col2,_ = st.columns([0.5,0.1,0.3,0.1])
    with col1:
        st.pyplot(fig)
    with col2:
        stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
        st.table(stats_df)

# Tabs for 4 Targets
tab1, tab2, tab3, tab4 = st.tabs(["FVC %", "FEV1 %", "FEF25-75%", "FEV1/FVC"])

targets = {"FVC %": tab1,"FEV1 %": tab2,"FEF25-75%": tab3,"FEV1/FVC": tab4}

# Tab Logic
for target_name, tab in targets.items():
    with tab:
        st.subheader(f"📌 Analysis for {target_name}")

        # Binary feature case
        if data[selected_feature].dropna().nunique() == 2:

            # Continuous test
            st.markdown("### Continuous Target Comparison")
            colA, colB = st.columns([0.6, 0.3])  # Wider chart column
            stat, p_value, fig, group_false_mean, group_true_mean, is_normal = statisticTest.continuous_test(data, continuous[target_name], selected_feature)
            with colA:
                st.pyplot(fig)
            with colB:
                st.metric("Statistic", f"{stat:.4f}")
                st.metric("P-value", f"{p_value:.4f}")
                st.metric(f"No {selected_feature}, {target_name} means", f"{group_false_mean:.4f}")
                st.metric(f"Yes {selected_feature}, {target_name} means", f"{group_true_mean:.4f}")

            st.info(interpret_result_continuous(p_value))

            st.divider()

            # Binary test
            st.markdown("### Binary Target Comparison")
            colA, colB = st.columns([0.6, 0.4])  # Wider chart column
            stat, p_value, fig = statisticTest.binary_chi_test(data, selected_feature, binary[target_name])

            with colA:
                st.pyplot(fig)
            with colB:
                st.metric("Statistic", f"{stat:.6f}")
                st.metric("P-value", f"{p_value:.6f}")

            st.info(interpret_result_binary(p_value))
        # Continuous feature case
        else:
            st.markdown("### Continuous Target Comparison")

            corr, p_value, fig = statisticTest.paired_continuous_test(data, selected_feature, target_name)
            colA,_, colB = st.columns([0.6,0.1, 0.3])
            with colA:
                st.pyplot(fig)
            with colB:
                st.metric("Correlation", f"{corr:.4f}")
                st.metric("P-value", f"{p_value:.4f}")

            st.info(interpret_result_continuous_paired(p_value))
            
            st.divider()

            # Binary test
            st.markdown("### Binary Target Comparison")
            colA, colB = st.columns([0.6, 0.3])
            stat, p_value, fig,group_false_mean, group_true_mean, _ = statisticTest.continuous_test(data, selected_feature, binary[target_name], binary_target=True)
            with colA:
                st.pyplot(fig)
            with colB:
                st.metric("Statistic", f"{stat:.4f}")
                st.metric("P-value", f"{p_value:.4f}")
                st.metric(f"Invalid {target_name}, {selected_feature} means", f"{group_false_mean:.4f}")
                st.metric(f"Valid {target_name}, {selected_feature} means", f"{group_true_mean:.4f}")

            st.info(interpret_result_continuous(p_value))