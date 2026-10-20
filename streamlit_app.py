import io
import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DataMosaic", page_icon="🧩", layout="wide")

AGENTS = [
    ("Orchestrator", "Plans the cleaning run from the dataset profile."),
    ("Profiler", "Measures shape, types, missingness, and duplicate rows."),
    ("Imputation", "Fills missing numeric and categorical values."),
    ("Outlier Handler", "Caps extreme numeric values using IQR bounds."),
    ("Categorical Encoder", "Converts categorical columns to readable codes."),
    ("Transformer", "Scales numeric features when requested."),
    ("Validation", "Checks the cleaned output and summarizes improvements."),
]

def profile(df):
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    return {
        "rows": len(df), "columns": len(df.columns),
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()), "numeric": numeric,
    }

def clean_data(df, scale=False, cap=True):
    out = df.copy()
    actions = []
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            if out[col].isna().any():
                out[col] = out[col].fillna(out[col].median())
                actions.append(f"Filled missing numeric values in {col} with the median.")
        else:
            if out[col].isna().any():
                mode = out[col].mode(dropna=True)
                out[col] = out[col].fillna(mode.iloc[0] if len(mode) else "Unknown")
                actions.append(f"Filled missing categories in {col} with the most common value.")
            out[col] = out[col].astype(str).str.strip()
    if cap:
        for col in out.select_dtypes(include=np.number).columns:
            q1, q3 = out[col].quantile([.25, .75]); iqr = q3 - q1
            if pd.notna(iqr) and iqr > 0:
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                count = int(((out[col] < lo) | (out[col] > hi)).sum())
                if count:
                    out[col] = out[col].clip(lo, hi)
                    actions.append(f"Capped {count:,} IQR outliers in {col}.")
    if scale:
        for col in out.select_dtypes(include=np.number).columns:
            lo, hi = out[col].min(), out[col].max()
            if pd.notna(hi) and hi > lo: out[col] = (out[col] - lo) / (hi - lo)
        actions.append("Applied min-max scaling to numeric columns.")
    return out, actions

def main():
    st.markdown("# 🧩 DataMosaic")
    st.caption("Explainable data preparation for messy real-world datasets")
    st.info("Free Streamlit edition · Upload a CSV, inspect the decisions, and download the cleaned result.")
    uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])
    sample = st.checkbox("Use the included customer sample instead", value=uploaded is None)
    if uploaded is None and sample:
        try: df = pd.read_csv("frontend/public/messy_customer_sales_data.csv")
        except Exception: df = pd.DataFrame({"age":[22, None, 41], "city":[" NY ", "ny", None], "sales":[20, 25, 9999]})
    elif uploaded is not None:
        try: df = pd.read_csv(uploaded)
        except Exception as e: st.error(f"Could not read the CSV: {e}"); return
    else:
        st.stop()
    st.success(f"Loaded {len(df):,} rows × {len(df.columns):,} columns")
    profile_before = profile(df)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows", f"{profile_before['rows']:,}"); c2.metric("Columns", profile_before["columns"])
    c3.metric("Missing cells", f"{profile_before['missing']:,}"); c4.metric("Duplicates", f"{profile_before['duplicates']:,}")
    with st.expander("Preview original data", expanded=True): st.dataframe(df.head(20), use_container_width=True)
    st.subheader("Cleaning plan")
    scale = st.checkbox("Apply min-max scaling", value=False)
    cap = st.checkbox("Handle numeric outliers", value=True)
    if st.button("Run DataMosaic pipeline", type="primary"):
        progress = st.progress(0, text="Starting orchestrator…")
        for i,(name,desc) in enumerate(AGENTS): progress.progress((i+1)/len(AGENTS), text=f"{name}: {desc}")
        cleaned, actions = clean_data(df, scale=scale, cap=cap)
        st.session_state.result = cleaned; st.session_state.actions = actions; st.session_state.before = profile_before
        st.success("Pipeline complete")
    if "result" in st.session_state:
        cleaned = st.session_state.result; after = profile(cleaned)
        st.subheader("Results & reports")
        a,b,c = st.columns(3); a.metric("Missing after", f"{after['missing']:,}"); b.metric("Duplicates after", f"{after['duplicates']:,}"); c.metric("Actions", len(st.session_state.actions))
        st.dataframe(cleaned.head(20), use_container_width=True)
        st.write("**Agent decisions**")
        for action in st.session_state.actions or ["No changes were required."]: st.write("✅ ", action)
        data = cleaned.to_csv(index=False).encode("utf-8")
        st.download_button("Download cleaned CSV", data, "datamosaic_cleaned.csv", "text/csv")
        st.download_button("Download audit summary", "\n".join(st.session_state.actions), "datamosaic_audit.txt", "text/plain")

if __name__ == "__main__": main()
