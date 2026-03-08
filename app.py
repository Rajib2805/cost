import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# --- PAGE CONFIG ---
st.set_page_config(page_title="Steel Plant Cost Analysis", layout="wide")

def sanitize_name(name):
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
    sanitized = re.sub(r'__+', '_', sanitized)
    return sanitized.strip('_')

def get_cleaned_features(df, col):
    if col not in df.columns:
        return set()
    unique_f = df[col].dropna().unique()
    return {sanitize_name(f) for f in unique_f}

# --- HEADER ---
st.title("🏭 Steel Plant Operations: PCA & Cost Prediction")
st.markdown("Upload your Excel file to perform sensitivity analysis across departments.")

# --- FILE UPLOADER ---
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    # Load all sheets
    all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
    
    departments_to_analyze = ['HM', 'SKIPSINTER', 'BFCOKE', 'SMS', 'SMS1', 'SMS2', 'SS']
    all_departments_results = {}
    
    # 1. --- DATA PROCESSING & MODEL BUILDING ---
    with st.spinner("Processing departments and training models..."):
        for dept in departments_to_analyze:
            try:
                # Basic Filtering
                df_cost = all_sheets['Cost'].copy()
                df_cost = df_cost[df_cost['Department'] == dept]
                
                df_prod = all_sheets['Production'].copy()
                df_prod = df_prod[df_prod['Department'] == dept]
                
                df_techno = all_sheets['Techno'].copy()
                df_techno = df_techno[df_techno['Department'] == dept]
                df_techno['Value'] = df_techno['Value'].abs()
                
                # Merging & Pivoting
                df_merged = pd.merge(df_cost, df_prod, on=['Date_', 'Department'], how='left')
                
                if not df_techno.empty:
                    tp = df_techno.pivot_table(index=['Date_', 'Department'], columns='Deptt_Item', values='Value', aggfunc='mean').reset_index()
                    df_merged = pd.merge(df_merged, tp, on=['Date_', 'Department'], how='left')

                # Clean NaNs
                num_cols = df_merged.select_dtypes(include=np.number).columns
                df_merged[num_cols] = df_merged[num_cols].ffill().bfill()
                df_merged.dropna(subset=['SumOfTotalCost'], inplace=True)

                if df_merged.empty or len(df_merged) < 5:
                    continue

                # Prepare X, y
                y = df_merged['SumOfTotalCost']
                X = df_merged.drop(columns=['Date_', 'SumOfTotalCost', 'Department', 'InputFor'], errors='ignore')
                X.columns = [sanitize_name(c) for c in X.columns]

                # PCA
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                pca_full = PCA().fit(X_scaled)
                cum_var = np.cumsum(pca_full.explained_variance_ratio_)
                n_comp = np.where(cum_var >= 0.95)[0][0] + 1
                
                pca = PCA(n_components=n_comp)
                X_pca = pca.fit_transform(X_scaled)

                # Train Model
                X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
                rf = RandomForestRegressor(random_state=42).fit(X_train, y_train)

                all_departments_results[dept] = {
                    'rf': rf, 'pca': pca, 'scaler': scaler, 'X': X, 
                    'median': X.median(), 'r2': r2_score(y_test, rf.predict(X_test))
                }
            except Exception as e:
                st.sidebar.warning(f"Skipped {dept}: {e}")

    # 2. --- INDEPENDENT FEATURE MAPPING ---
    important_map = {}
    sheets_info = {
        'Production': ('Production', 'Department'),
        'Techno': ('Deptt_Item', 'Department'),
        'InputRate': ('Particulars_', 'Department_')
    }
    
    for s_name, (feat_col, dept_col) in sheets_info.items():
        if s_name in all_sheets and 'Sensitivity' in all_sheets[s_name].columns:
            df_s = all_sheets[s_name]
            sens = df_s[df_s['Sensitivity'] == 1]
            for _, row in sens.iterrows():
                d, f = str(row[dept_col]), sanitize_name(row[feat_col])
                important_map.setdefault(d, []).append(f)

    # 3. --- UI LAYOUT ---
    selected_dept = st.sidebar.selectbox("Select Department", options=list(all_departments_results.keys()))

    if selected_dept:
        res = all_departments_results[selected_dept]
        st.subheader(f"Analysis for Department: {selected_dept} (Model R²: {res['r2']:.2f})")
        
        ctrl_features = list(set(important_map.get(selected_dept, [])) & set(res['X'].columns))
        
        if not ctrl_features:
            st.info("No sensitivity features found for this department.")
        else:
            # Create Sliders
            st.markdown("### Adjust Input Parameters")
            cols = st.columns(3)
            user_inputs = res['median'].copy()
            
            for i, feat in enumerate(ctrl_features):
                col_idx = i % 3
                min_v = float(res['X'][feat].min())
                max_v = float(res['X'][feat].max())
                step = (max_v - min_v) / 100 if max_v != min_v else 0.1
                
                user_inputs[feat] = cols[col_idx].slider(
                    feat, min_v, max_v, float(res['median'][feat]), step=step
                )

            # Prediction Logic
            input_df = pd.DataFrame([user_inputs.reindex(res['X'].columns).values], columns=res['X'].columns)
            input_pca = res['pca'].transform(res['scaler'].transform(input_df))
            prediction = res['rf'].predict(input_pca)[0]

            st.metric("Predicted Total Cost", f"₹ {prediction:,.2f}")

            # Reconstruction Visualization
            st.markdown("### Feature Influence (Reconstructed from PCA)")
            recon_scaled = res['pca'].inverse_transform(input_pca)
            recon_orig = res['scaler'].inverse_transform(recon_scaled)
            recon_df = pd.Series(recon_orig[0], index=res['X'].columns).sort_values(ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(x=recon_df.index, y=recon_df.values, ax=ax, palette="mako")
            plt.xticks(rotation=45)
            st.pyplot(fig)

else:
    st.info("Please upload an Excel file in the sidebar to begin.")