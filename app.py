import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Steel Plant Cost Analysis", layout="wide")

# --- Helper Functions ---
def sanitize_name(name):
    sanitized = ''.join(c if c.isalnum() else '_' for c in name)
    sanitized = '_'.join(filter(None, sanitized.split('_')))
    if sanitized and sanitized[0].isdigit():
        sanitized = '_' + sanitized
    return sanitized

def get_cleaned_features_from_sheet(sheet_df, feature_col_name):
    unique_features = sheet_df[feature_col_name].dropna().unique()
    cleaned_features = set()
    for feature_raw in unique_features:
        feature_cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', str(feature_raw))
        feature_cleaned = re.sub(r'__+', '_', feature_cleaned)
        feature_cleaned = feature_cleaned.strip('_')
        cleaned_features.add(feature_cleaned)
    return cleaned_features

# --- App Title ---
st.title("Steel Plant Cost Prediction & Sensitivity Analysis")

# --- Sidebar: File Upload ---
uploaded_file = st.sidebar.file_uploader("Upload Steel Plant Data (Excel)", type=["xlsx"])

if uploaded_file:
    # Load all sheets
    all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
    st.sidebar.success("File uploaded successfully.")

    # --- Data Processing Logic (Preserved) ---
    departments_to_analyze = ['HM', 'SKIPSINTER', 'BFCOKE', 'SMS', 'SMS1', 'SMS2', 'SS']
    all_departments_results = {}

    for dept in departments_to_analyze:
        if 'Cost' not in all_sheets: continue
        
        # Preprocessing snippets preserved from original code
        df_cost_dept = all_sheets['Cost'].copy()
        df_cost_dept['Date_'] = pd.to_datetime(df_cost_dept['Date_'])
        df_cost_dept = df_cost_dept[df_cost_dept['Department'] == dept]

        df_production_dept = all_sheets['Production'].copy()
        df_production_dept['Date_'] = pd.to_datetime(df_production_dept['Date_'])
        df_production_dept = df_production_dept[df_production_dept['Department'] == dept]

        df_techno_dept = all_sheets['Techno'].copy()
        df_techno_dept['Date_'] = pd.to_datetime(df_techno_dept['Date_'])
        df_techno_dept['Value'] = df_techno_dept['Value'].abs()
        df_techno_dept = df_techno_dept[df_techno_dept['Department'] == dept]

        # ... (Pivot logic from original code)
        df_techno_pivot_dept = pd.DataFrame(columns=['Date_', 'Department'])
        if not df_techno_dept.empty:
            df_techno_pivot_dept = df_techno_dept.pivot_table(index=['Date_', 'Department'], columns='Deptt_Item', values='Value', aggfunc='mean').reset_index()
            df_techno_pivot_dept.columns.name = None

        # Merge
        df_merged_dept = pd.merge(df_cost_dept, df_production_dept, on=['Date_', 'Department'], how='left')
        if len(df_techno_pivot_dept.columns) > 2:
            df_merged_dept = pd.merge(df_merged_dept, df_techno_pivot_dept, on=['Date_', 'Department'], how='left')

        # Cleaning & PCA
        numerical_cols_dept = df_merged_dept.select_dtypes(include=np.number).columns
        df_merged_dept[numerical_cols_dept] = df_merged_dept[numerical_cols_dept].ffill().bfill()
        df_merged_dept.dropna(inplace=True)

        if not df_merged_dept.empty:
            y_dept = df_merged_dept['SumOfTotalCost']
            X_dept = df_merged_dept.drop(columns=['Date_', 'SumOfTotalCost', 'Department', 'InputFor'], errors='ignore')
            
            # Sanitize Column Names
            X_dept.columns = X_dept.columns.str.strip().str.replace(r'[\W]', '_', regex=True).str.replace(r'__+', '_', regex=True).str.strip('_')

            if not X_dept.empty:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_dept)
                
                pca_full = PCA().fit(X_scaled)
                cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
                n_comp = np.where(cumulative_variance >= 0.95)[0][0] + 1 if any(cumulative_variance >= 0.95) else 1
                
                pca = PCA(n_components=n_comp)
                X_pca = pca.fit_transform(X_scaled)

                X_train, X_test, y_train, y_test = train_test_split(X_pca, y_dept, test_size=0.2, random_state=42)
                rf = RandomForestRegressor(random_state=42).fit(X_train, y_train)

                all_departments_results[dept] = {
                    'status': 'Processed',
                    'n_components_95': n_comp,
                    'mae_pca': mean_absolute_error(y_test, rf.predict(X_test)),
                    'r2_pca': r2_score(y_test, rf.predict(X_test)),
                    'X_dataframe': X_dept,
                    'scaler': scaler,
                    'pca_model': pca,
                    'rf_model': rf,
                    'median_features': X_dept.median()
                }

    # --- Feature Mapping (Preserved) ---
    important_independent_features_map = {}
    sheets_to_check = {'Production': 'Production', 'Techno': 'Deptt_Item', 'InputRate': 'Particulars_', 'InterDepartmentTransfer': 'CostOf'}
    
    for sheet_name, feat_col in sheets_to_check.items():
        if sheet_name in all_sheets and 'Sensitivity' in all_sheets[sheet_name].columns:
            df_s = all_sheets[sheet_name]
            sensitive_rows = df_s[df_s['Sensitivity'] == 1]
            dept_col = 'Department_' if sheet_name == 'InputRate' else 'Department'
            
            for _, row in sensitive_rows.iterrows():
                d, f = str(row[dept_col]), str(row[feat_col] if sheet_name != 'Production' else 'Production')
                f_clean = re.sub(r'[^a-zA-Z0-9_]', '_', f).strip('_')
                important_independent_features_map.setdefault(d, []).append(f_clean)

    # --- UI Layout ---
    selected_dept = st.sidebar.selectbox("Select Department", options=list(all_departments_results.keys()))

    if selected_dept:
        res = all_departments_results[selected_dept]
        st.header(f"Analysis for Department: {selected_dept}")
        
        # Display Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("PCA Components", res['n_components_95'])
        m2.metric("MAE", f"{res['mae_pca']:,.2f}")
        m3.metric("R2 Score", f"{res['r2_pca']:.2f}")

        # Controls
        features = list(set(important_independent_features_map.get(selected_dept, [])))
        X_df = res['X_dataframe']
        
        st.subheader("Sensitivity Controls")
        slider_vals = {}
        
        # Layout sliders in columns
        cols = st.columns(3)
        for i, feat in enumerate(features):
            if feat in X_df.columns:
                col = cols[i % 3]
                min_v, max_v = float(X_df[feat].min()), float(X_df[feat].max())
                curr_v = float(X_df[feat].iloc[-1])
                slider_vals[feat] = col.slider(feat, min_v, max_v, curr_v)

        # Prediction Logic
        if slider_vals:
            input_data = res['median_features'].copy()
            for f, v in slider_vals.items():
                input_data[f] = v
            
            # Reconstruct and Predict
            input_row = pd.DataFrame([input_data.reindex(X_df.columns).values], columns=X_df.columns)
            input_scaled = res['scaler'].transform(input_row)
            input_pca = res['pca_model'].transform(input_scaled)
            prediction = res['rf_model'].predict(input_pca)[0]

            st.markdown(f"### Predicted SumOfTotalCost: :green[{prediction:,.0f}]")

            # Visualization
            reconstructed_scaled = res['pca_model'].inverse_transform(input_pca)
            reconstructed_orig = res['scaler'].inverse_transform(reconstructed_scaled)
            recon_df = pd.DataFrame(reconstructed_orig, columns=X_df.columns).T
            recon_df.columns = ['Value']
            recon_df = recon_df.sort_values('Value', ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=recon_df.index, y='Value', data=recon_df, ax=ax, palette='viridis')
            plt.xticks(rotation=45)
            st.pyplot(fig)
else:
    st.info("Please upload an Excel file to begin.")
