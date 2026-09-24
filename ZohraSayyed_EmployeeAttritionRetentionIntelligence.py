"""
Employee Attrition & Retention Intelligence Dashboard
-------------------------------------------------------
Author: Zohra
Internship: AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026

WHAT THIS FILE DOES (in order):
1. Loads the raw HR dataset
2. Cleans it (drops useless columns, sanity-checks values)
3. Computes KPIs and attrition-driver analysis
4. Trains a simple, explainable Logistic Regression model
5. Displays everything as an interactive Streamlit dashboard

HOW TO RUN:
    pip install -r requirements.txt
    streamlit run ZohraSayyed_EmployeeAttritionRetentionIntelligence.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Employee Attrition Intelligence", layout="wide")

# ---------------------------------------------------------
# STEP 1 + 2: LOAD AND CLEAN DATA
# ---------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

    # Drop columns that carry zero useful information
    # (EmployeeCount, StandardHours, Over18 are constant for every row;
    #  EmployeeNumber is just an ID, not a real characteristic)
    df = df.drop(columns=["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"])

    df["Left"] = (df["Attrition"] == "Yes").astype(int)
    return df

df = load_and_clean_data()
overall_rate = df["Left"].mean()

# ---------------------------------------------------------
# STEP 3: KPI + DRIVER CALCULATIONS
# ---------------------------------------------------------
def rate_by(dataframe, col, bins=None, labels=None):
    d = dataframe.copy()
    if bins is not None:
        d[col] = pd.cut(d[col], bins=bins, labels=labels)
    return (d.groupby(col)["Left"].mean() * 100).round(1)

overtime_rates = rate_by(df, "OverTime")
jobrole_rates = rate_by(df, "JobRole").sort_values(ascending=True)
tenure_rates = rate_by(df, "YearsAtCompany", bins=[-1, 1, 3, 6, 10, 40],
                        labels=["0-1yr", "2-3yr", "4-6yr", "7-10yr", "10+yr"])

income_df = df.copy()
income_df["IncomeQ"] = pd.qcut(income_df["MonthlyIncome"], 4,
                                labels=["Q1 Lowest", "Q2", "Q3", "Q4 Highest"])
income_rates = (income_df.groupby("IncomeQ")["Left"].mean() * 100).round(1)

# High-risk compound segment: overtime + new employee + entry level
high_risk = df[(df["OverTime"] == "Yes") & (df["YearsAtCompany"] <= 3) & (df["JobLevel"] == 1)]
high_risk_rate = high_risk["Left"].mean() * 100
high_risk_count = len(high_risk)
leavers_in_segment = high_risk["Left"].sum()
total_leavers = df["Left"].sum()
pct_of_leavers = leavers_in_segment / total_leavers * 100

# ---------------------------------------------------------
# STEP 4: LOGISTIC REGRESSION MODEL
# ---------------------------------------------------------
@st.cache_data
def train_model(dataframe):
    y = dataframe["Left"]
    X = dataframe.drop(columns=["Attrition", "Left"])
    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    coefs = pd.Series(model.coef_[0], index=X_encoded.columns).sort_values()
    top_risk_factors = coefs.tail(8).sort_values(ascending=True)
    top_protective_factors = coefs.head(8)

    return metrics, top_risk_factors, top_protective_factors

metrics, top_risk_factors, top_protective_factors = train_model(df)

# ---------------------------------------------------------
# STEP 5: DASHBOARD LAYOUT
# ---------------------------------------------------------
st.title("Employee Attrition & Retention Intelligence")
st.caption("A data-driven investigation into why employees leave, and what management can do about it.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Executive Overview", "Driver Analysis", "High-Risk Segment", "Model & Recommendations"]
)

# --- TAB 1: EXECUTIVE OVERVIEW ---
with tab1:
    st.subheader("Key Performance Indicators")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Employees", len(df))
    c2.metric("Overall Attrition Rate", f"{overall_rate:.1%}")
    c3.metric("Employees Who Left", int(df["Left"].sum()))
    c4.metric("High-Risk Segment Size", high_risk_count)

    st.markdown("---")
    st.subheader("Attrition by Tenure")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(tenure_rates.index.astype(str), tenure_rates.values, color="#4C72B0")
    ax.set_ylabel("Attrition Rate (%)")
    st.pyplot(fig)
    st.caption("Attrition risk is heavily front-loaded: new employees are far more likely to leave than tenured ones.")

# --- TAB 2: DRIVER ANALYSIS ---
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Overtime vs No Overtime")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(overtime_rates.index, overtime_rates.values, color=["#4C72B0", "#C44E52"])
        ax.set_ylabel("Attrition Rate (%)")
        for i, v in enumerate(overtime_rates.values):
            ax.text(i, v + 0.5, f"{v}%", ha="center", fontweight="bold")
        st.pyplot(fig)
        st.caption("Employees working overtime leave at nearly 3x the rate of those who don't.")

    with col2:
        st.subheader("Attrition by Income Quartile")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(income_rates.index.astype(str), income_rates.values, color="#4C72B0")
        ax.set_ylabel("Attrition Rate (%)")
        st.pyplot(fig)
        st.caption("The lowest-paid quartile leaves at nearly 3x the rate of the top quartile.")

    st.subheader("Attrition by Job Role")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#C44E52" if v > 20 else "#4C72B0" for v in jobrole_rates.values]
    ax.barh(jobrole_rates.index, jobrole_rates.values, color=colors)
    ax.set_xlabel("Attrition Rate (%)")
    st.pyplot(fig)
    st.caption("Sales Representatives show by far the highest attrition rate of any role (39.8%).")

# --- TAB 3: HIGH-RISK SEGMENT ---
with tab3:
    st.subheader("The Compound High-Risk Segment")
    st.markdown(
        f"""
        A specific group of employees — those who **work overtime**, have **3 years or less
        tenure**, and are at **entry job level** — show dramatically higher attrition than
        the company average.
        """
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Segment Size", f"{high_risk_count} employees", f"{high_risk_count/len(df):.1%} of workforce")
    c2.metric("Segment Attrition Rate", f"{high_risk_rate:.1f}%", f"+{high_risk_rate - overall_rate*100:.1f} pts vs. average")
    c3.metric("Share of All Leavers", f"{pct_of_leavers:.1f}%", "from just this small group")

    fig, ax = plt.subplots(figsize=(6, 4))
    cats = ["Company Average", "High-Risk Segment"]
    vals = [overall_rate * 100, high_risk_rate]
    ax.bar(cats, vals, color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_ylabel("Attrition Rate (%)")
    st.pyplot(fig)

    st.info(
        f"**Business takeaway:** Just {high_risk_count} employees ({high_risk_count/len(df):.1%} of the "
        f"workforce) account for {pct_of_leavers:.1f}% of all attrition. This is the single highest-leverage "
        f"group for a retention intervention."
    )

# --- TAB 4: MODEL & RECOMMENDATIONS ---
with tab4:
    st.subheader("Logistic Regression Model (Confirms & Ranks Drivers)")
    st.caption(
        "A simple, explainable statistical model used to confirm which factors matter most "
        "when considered together, not to replace the analysis above."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    m2.metric("Recall", f"{metrics['recall']:.1%}", "of actual leavers caught")
    m3.metric("Precision", f"{metrics['precision']:.1%}")
    m4.metric("F1 Score", f"{metrics['f1']:.1%}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top factors that INCREASE risk**")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(top_risk_factors.index, top_risk_factors.values, color="#C44E52")
        st.pyplot(fig)
    with col2:
        st.markdown("**Top factors that DECREASE risk (protective)**")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(top_protective_factors.index, top_protective_factors.values, color="#4C72B0")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Business Recommendations")
    st.markdown(
        """
        1. **Monitor and cap overtime for new, entry-level employees** — this group drives
           disproportionate attrition; a workload review here likely has the highest ROI.
        2. **Review the Sales Representative role** — investigate whether workload, incentive
           structure, or role design is driving nearly 4 in 10 to leave.
        3. **Extend stock option eligibility earlier**, especially to entry-level staff — data
           shows a clear link between having any stock options and staying longer.
        4. **Review bottom-quartile pay bands** — being underpaid relative to peers tracks
           closely with attrition, more than low pay in isolation.
        5. **Build a formal first-3-year retention check-in program** — since risk is heavily
           front-loaded, early intervention has outsized impact.
        """
    )
    st.caption(
        "Note: this analysis shows association, not proven causation. These are "
        "data-backed hypotheses for management to investigate and act on, not certainties."
    )
