"""
Streamlit Personal Finance & Bank Transaction Categorizer Dashboard
-------------------------------------------------------------------
Interactive web interface for real-time transaction classification,
batch statement analysis, spending analytics, explainability, and PDF export.
"""

import io
import json
import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classifier import HybridTransactionClassifier
from src.data_generator import CATEGORIES
from src.explainability import explain_prediction
from src.report_generator import generate_pdf_report

# Page configuration
st.set_page_config(
    page_title="Bank Transaction Categorizer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .badge-rule {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-ml {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-fallback {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_classifier():
    """Cache the classifier in memory for fast inference."""
    return HybridTransactionClassifier()


classifier = load_classifier()

# Category emoji dictionary
CATEGORY_EMOJIS = {
    "Food & Dining": "🍕",
    "Groceries": "🛒",
    "Rent & Housing": "🏠",
    "Transport": "🚗",
    "Shopping": "🛍️",
    "Health & Medical": "💊",
    "Bills & Utilities": "📱",
    "Entertainment": "🎬",
    "Transfers": "💰",
    "Education": "📚",
    "EMI & Loans": "💳",
    "Income": "💵",
    "Other / Uncategorized": "❓"
}

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/bank.png", width=70)
    st.title("Txn Categorizer")
    st.caption("AI-Powered Financial Intelligence")

    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    if classifier.pipeline is not None:
        st.success("ML Model: **Active (TF-IDF + LR)**")
    else:
        st.warning("ML Model: **Offline (Rule-Only)**")
    st.info(f"Supported Categories: **{len(CATEGORIES)}**")

    st.markdown("---")
    st.markdown("### 💡 Quick Preset Narrations")
    demo_samples = [
        "UPI/CR/4324234/SWIGGY/ICICI/order",
        "POS 4421XXXX9102 DMART BANNERGHATTA BLR",
        "NEFT-CR-N12345678-OWNER RENT-HDFC0001",
        "ACH D- NETFLIX ENTERTAINMENT SVCS",
        "E-COM PURCHASE 5123XXXXXXXX9912 AMAZON INDIA",
        "BY TRANSFER-SALARY CREDIT INFOSYS OCT",
        "CRED CC BILL PAYMENT HDFC CARDS",
        "DELHI METRO DMRC RECHARGE TICKET"
    ]
    selected_sample = st.selectbox("Select an example to populate tester:", [""] + demo_samples)

    st.markdown("---")
    st.caption("Built with FastAPI, Scikit-Learn & Streamlit.")


# Main App Layout
st.markdown('<div class="main-header">🏦 Bank Transaction Categorizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Automatically translate cryptic bank statement narratives (UPI, NEFT, IMPS, POS, Cards) into structured spending intelligence.</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs([
    "🔍 Real-Time Classifier",
    "📊 Statement Analyzer & PDF Export",
    "📈 Model Metrics & Architecture"
])

# -------------------------------------------------------------
# TAB 1: Single Transaction Tester & Explainability
# -------------------------------------------------------------
with tab1:
    st.subheader("Test a Transaction Narration")
    st.write("Type or paste any messy bank statement narrative below to see real-time classification and explainability.")

    default_val = selected_sample if selected_sample else "UPI/CR/432423409123/SWIGGY/ICICI/payment"
    input_text = st.text_input("Transaction Narrative / Description:", value=default_val)

    col_btn, col_thresh = st.columns([1, 2])
    with col_thresh:
        conf_threshold = st.slider("Classification Confidence Threshold", 0.1, 0.9, 0.35, 0.05)

    if st.button("Categorize Transaction", type="primary"):
        if input_text.strip():
            classifier.min_confidence = conf_threshold
            res = classifier.classify_single(input_text)

            cat = res["predicted_category"]
            emoji = CATEGORY_EMOJIS.get(cat, "📁")
            conf = res["confidence"]
            src = res["engine_source"]
            needs_review = res["needs_review"]

            st.markdown("---")
            res_col1, res_col2, res_col3 = st.columns([1.5, 1, 1.5])

            with res_col1:
                st.markdown("#### Predicted Category")
                st.markdown(f"## {emoji} {cat}")
                if src == "rule_engine":
                    st.markdown('<span class="badge-rule">⚡ Rule Engine (Deterministic)</span>', unsafe_allow_html=True)
                elif src == "ml_pipeline":
                    st.markdown('<span class="badge-ml">🧠 ML Pipeline (TF-IDF + LR)</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-fallback">⚠️ Low Confidence (Needs Review)</span>', unsafe_allow_html=True)

            with res_col2:
                st.markdown("#### Confidence Score")
                st.metric(label="Model Confidence", value=f"{conf * 100:.1f}%")
                st.progress(conf)

            with res_col3:
                st.markdown("#### Preprocessed Tokens")
                st.code(res["cleaned_description"] if res["cleaned_description"] else "(None)", language="text")
                st.caption("Noise words, masked digits, and routing codes stripped.")

            # Distribution chart for top predictions
            if res.get("top_predictions"):
                st.markdown("#### Top Category Probabilities")
                top_df = pd.DataFrame(res["top_predictions"])
                fig_top = px.bar(
                    top_df,
                    x="confidence",
                    y="category",
                    orientation="h",
                    text=[f"{v * 100:.1f}%" for v in top_df["confidence"]],
                    color="confidence",
                    color_continuous_scale="Blues"
                )
                fig_top.update_layout(
                    xaxis_title="Confidence",
                    yaxis_title="",
                    yaxis=dict(autorange="reversed"),
                    height=220,
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig_top, use_container_width=True)

            # Feature Attribution Explainability
            if classifier.pipeline is not None:
                st.markdown("#### 🔬 Decision Explainability (Why was this categorized?)")
                explanation = explain_prediction(classifier.pipeline, input_text)
                features = explanation.get("top_features", [])
                if features:
                    exp_df = pd.DataFrame(features)
                    fig_exp = px.bar(
                        exp_df,
                        x="importance",
                        y="token",
                        orientation="h",
                        text="importance",
                        labels={"importance": "Attribution Weight", "token": "Extracted Token"},
                        color="importance",
                        color_continuous_scale="Viridis"
                    )
                    fig_exp.update_layout(
                        yaxis=dict(autorange="reversed"),
                        height=200,
                        margin=dict(l=10, r=10, t=10, b=10)
                    )
                    st.plotly_chart(fig_exp, use_container_width=True)
                    st.caption("Linear attribution weights show which tokens most strongly activated this category.")
                else:
                    st.info("Direct rule match triggered without requiring linear feature decomposition.")
        else:
            st.error("Please enter a non-empty transaction description.")


# -------------------------------------------------------------
# TAB 2: Statement Analyzer & Batch Processing
# -------------------------------------------------------------
with tab2:
    st.subheader("Analyze Bank Statement (CSV / Excel)")
    st.write("Upload your bank statement or load the pre-packaged sample statement to inspect spending breakdowns.")

    c1, c2 = st.columns([2, 1])
    with c1:
        uploaded_file = st.file_uploader("Upload Statement (CSV):", type=["csv"])
    with c2:
        st.write("")
        st.write("")
        load_sample = st.button("📂 Load Built-In Sample Statement", use_container_width=True)

    df_to_process = None

    if uploaded_file is not None:
        try:
            df_to_process = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(df_to_process)} rows from uploaded file.")
        except Exception as e:
            st.error(f"Error parsing file: {e}")
    elif load_sample:
        sample_path = "data/sample/sample_statement.csv"
        if os.path.exists(sample_path):
            df_to_process = pd.read_csv(sample_path)
            st.success(f"Loaded {len(df_to_process)} transactions from sample statement.")

    if df_to_process is not None:
        # Detect description column
        col_options = list(df_to_process.columns)
        desc_candidates = [c for c in col_options if any(k in c.lower() for k in ["desc", "narrat", "particular", "remark"])]
        default_desc = desc_candidates[0] if desc_candidates else col_options[0]

        amt_candidates = [c for c in col_options if any(k in c.lower() for k in ["amount", "amt", "debit", "withdrawal"])]
        default_amt = amt_candidates[0] if amt_candidates else (col_options[1] if len(col_options) > 1 else col_options[0])

        with st.expander("🛠️ Column Mapping Settings", expanded=False):
            sc1, sc2 = st.columns(2)
            with sc1:
                desc_col = st.selectbox("Description Column:", col_options, index=col_options.index(default_desc))
            with sc2:
                amt_col = st.selectbox("Amount Column:", col_options, index=col_options.index(default_amt))

        # Process Statement
        with st.spinner("Classifying transactions through hybrid engine..."):
            processed_df = classifier.process_dataframe(df_to_process, description_col=desc_col)

        # Spending Analytics Overview
        has_amt = amt_col in processed_df.columns
        has_type = "transaction_type" in processed_df.columns

        total_txns = len(processed_df)
        if has_amt and has_type:
            total_in = processed_df[processed_df["transaction_type"] == "CREDIT"][amt_col].sum()
            total_out = processed_df[processed_df["transaction_type"] == "DEBIT"][amt_col].sum()
        elif has_amt:
            total_in = 0.0
            total_out = processed_df[amt_col].sum()
        else:
            total_in = 0.0
            total_out = 0.0

        net_savings = total_in - total_out

        st.markdown("---")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Transactions", f"{total_txns:,}")
        kpi2.metric("Total Income (Credits)", f"₹{total_in:,.2f}")
        kpi3.metric("Total Spending (Debits)", f"₹{total_out:,.2f}")
        kpi4.metric("Net Balance / Savings", f"₹{net_savings:,.2f}", delta=f"{net_savings:,.2f}")

        # Charts Section
        st.markdown("### 📊 Spending Intelligence")
        ch_col1, ch_col2 = st.columns(2)

        cat_col = "predicted_category"
        if has_amt:
            spend_df = processed_df[processed_df["transaction_type"] == "DEBIT"] if has_type else processed_df
            cat_totals = spend_df.groupby(cat_col)[amt_col].sum().reset_index()
            cat_totals = cat_totals.sort_values(by=amt_col, ascending=False)

            with ch_col1:
                fig_pie = px.pie(
                    cat_totals,
                    values=amt_col,
                    names=cat_col,
                    title="Spending Breakdown by Category",
                    hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_pie, use_container_width=True)

            with ch_col2:
                fig_bar = px.bar(
                    cat_totals,
                    x=cat_col,
                    y=amt_col,
                    text_auto='.2s',
                    title="Total Spending by Category (INR)",
                    color=amt_col,
                    color_continuous_scale="Teal"
                )
                fig_bar.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_bar, use_container_width=True)

        # Tabular View with Filtering
        st.markdown("### 📋 Categorized Transactions")
        filter_col, search_col = st.columns([1, 2])
        with filter_col:
            unique_cats = ["All"] + sorted(list(processed_df[cat_col].unique()))
            selected_cat = st.selectbox("Filter by Category:", unique_cats)
        with search_col:
            search_query = st.text_input("Search description keyword:", "")

        view_df = processed_df.copy()
        if selected_cat != "All":
            view_df = view_df[view_df[cat_col] == selected_cat]
        if search_query.strip():
            view_df = view_df[view_df[desc_col].str.contains(search_query, case=False, na=False)]

        # Display table
        st.dataframe(
            view_df[[c for c in [desc_col, cat_col, "confidence", "engine_source", amt_col, "date"] if c in view_df.columns]],
            use_container_width=True,
            height=320
        )

        # Download Actions
        st.markdown("### 📥 Export Reports")
        d_col1, d_col2 = st.columns(2)

        with d_col1:
            csv_data = processed_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📄 Download Categorized CSV",
                data=csv_data,
                file_name="categorized_bank_statement.csv",
                mime="text/csv",
                use_container_width=True
            )

        with d_col2:
            try:
                pdf_bytes = generate_pdf_report(processed_df)
                st.download_button(
                    label="📕 Download Executive PDF Spending Report",
                    data=pdf_bytes,
                    file_name="financial_spending_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.warning(f"PDF export temporarily unavailable: {e}")


# -------------------------------------------------------------
# TAB 3: Model Performance & Architecture
# -------------------------------------------------------------
with tab3:
    st.subheader("Model Evaluation & System Architecture")

    # Load evaluation report
    report_path = "data/processed/evaluation_report.json"
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            eval_metrics = json.load(f)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Accuracy", f"{eval_metrics['accuracy'] * 100:.2f}%")
        m2.metric("Macro F1-Score", f"{eval_metrics['macro_f1']:.4f}")
        m3.metric("Weighted F1-Score", f"{eval_metrics['weighted_f1']:.4f}")
        m4.metric("Test Samples Evaluated", f"{eval_metrics['test_samples']:,}")

        st.markdown("---")
        st.markdown("#### Per-Category Performance Breakdown")
        cat_metrics = eval_metrics.get("per_category_metrics", {})
        cat_table_records = []
        for cat_name, metrics in cat_metrics.items():
            if isinstance(metrics, dict) and cat_name in CATEGORIES:
                cat_table_records.append({
                    "Category": cat_name,
                    "Precision": f"{metrics['precision'] * 100:.1f}%",
                    "Recall": f"{metrics['recall'] * 100:.1f}%",
                    "F1-Score": f"{metrics['f1-score']:.3f}",
                    "Support": int(metrics["support"])
                })
        if cat_table_records:
            st.dataframe(pd.DataFrame(cat_table_records), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Architecture Blueprint")
    st.markdown("""
    ```
    ┌──────────────────────┐     ┌────────────────────────┐     ┌───────────────────────────┐
    │ Raw Bank Statement   │ ──> │ PII / Regex Cleaner    │ ──> │ Hybrid Decision Arbiter   │
    │ (UPI, NEFT, POS, ACH)│     │ (Strips masks & ref IDs│     │                           │
    └──────────────────────┘     └────────────────────────┘     └─────────────┬─────────────┘
                                                                              │
                                             ┌────────────────────────────────┴───────────────────────────────┐
                                             ▼                                                                ▼
                                ┌────────────────────────┐                                       ┌────────────────────────┐
                                │ Tier 1: Rule Engine    │                                       │ Tier 2: TF-IDF + LR    │
                                │ (Exact Merchant Match) │                                       │ (Linear Confidence)    │
                                └────────────────────────┘                                       └────────────────────────┘
    ```
    """)

