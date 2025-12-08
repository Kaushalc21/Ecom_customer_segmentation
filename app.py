# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
import io
import base64
import warnings

warnings.filterwarnings("ignore")

# Try importing hdbscan optionally
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except Exception:
    HDBSCAN_AVAILABLE = False

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

# --------------------------
# Helper functions
# --------------------------
def compute_rfm_from_transactions(df, customer_col="CustomerID", invoice_col="InvoiceDate",
                                  invoice_id_col="InvoiceNo", qty_col="Quantity", price_col=None, ref_date=None):
    """
    Compute RFM table from raw invoices-level dataframe.
    price_col can be 'UnitPrice' or 'TotalPrice'. If only UnitPrice present, TotalPrice = Quantity * UnitPrice.
    """
    if price_col is None:
        if "TotalPrice" in df.columns:
            price_col = "TotalPrice"
        elif "UnitPrice" in df.columns:
            price_col = None
        else:
            price_col = None

    # ensure invoice date parsed
    df[invoice_col] = pd.to_datetime(df[invoice_col], errors='coerce')
    if ref_date is None:
        ref_date = df[invoice_col].max()

    # Ensure a revenue column
    if "TotalPrice" not in df.columns:
        if ("UnitPrice" in df.columns) and (qty_col in df.columns):
            df["TotalPrice"] = df[qty_col] * df["UnitPrice"]
        else:
            df["TotalPrice"] = 0.0

    rfm = df.groupby(customer_col).agg(
        Recency = (invoice_col, lambda x: (ref_date - x.max()).days if pd.notnull(x.max()) else np.nan),
        Frequency = (invoice_id_col, "nunique"),
        Monetary = ("TotalPrice", "sum"),
        Quantity = (qty_col, "sum")
    ).reset_index()

    # If Quantity or Recency have NaN due to missing data, fill with 0
    rfm['Quantity'] = rfm['Quantity'].fillna(0)
    rfm['Recency'] = rfm['Recency'].fillna(rfm['Recency'].max())
    return rfm

def scale_features(df, features):
    scaler = StandardScaler()
    return scaler.fit_transform(df[features].fillna(0))

def safe_silhouette(X, labels):
    try:
        return silhouette_score(X, labels)
    except Exception:
        return None

def df_to_csv_download(df, filename="rfm_segmented.csv"):
    towrite = io.BytesIO()
    df.to_csv(towrite, index=False)
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">⬇️ Download processed CSV</a>'
    return href

# --------------------------
# Sidebar - Inputs
# --------------------------
with st.sidebar:
    st.title("Configuration")
    st.markdown("Upload raw transaction CSV (Invoice-level). Required columns: `CustomerID`, `InvoiceNo`, `InvoiceDate`, `Quantity`, and `UnitPrice` or `TotalPrice`.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    st.markdown("---")
    algo_option = st.selectbox("Select algorithm (switch anytime)", ["Agglomerative", "GMM", "KMeans", "HDBSCAN (if available)"])
    # default clusters for algorithms that need n_clusters
    n_clusters = st.number_input("Number of clusters (for Agg/GMM/KMeans)", min_value=2, max_value=12, value=4, step=1)

    # HDBSCAN params if chosen
    if algo_option == "HDBSCAN (if available)":
        min_cluster_size = st.number_input("HDBSCAN min_cluster_size", min_value=5, max_value=500, value=50, step=5)
        # selection method
        st.info("HDBSCAN will label noise as -1. If hdbscan package is not installed this option will not run.")
    st.markdown("---")
    st.checkbox("Remove outliers (1%-99%) before clustering", value=True, key="outliers")
    st.markdown("---")
    st.markdown("RFM Reference (optional)")
    ref_date_input = st.date_input("Reference date (optional, defaults to max InvoiceDate)")

    st.markdown("---")
    st.markdown("Display options")
    show_3d = st.checkbox("Show 3D scatter", value=True)
    show_treemap = st.checkbox("Show treemap", value=True)

# --------------------------
# Main - App
# --------------------------
st.title("🔬   Ecommerce Customer Segmentation  —  Dashboard (RFM)")

if uploaded_file is None:
    st.info("Please upload a transaction CSV to start. Use sample dataset or your own file.")
    st.stop()

# Read file
# Read CSV with smart encoding fallback
# ---- CSV LOADING ----
try:
    df = pd.read_csv(uploaded_file, encoding="latin1", low_memory=False)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

# 🔍 Debug: Show actual column names
st.write("Detected columns:")
st.write([repr(col) for col in df.columns])


# Basic column checks and normalization
expected_cols = ["CustomerID", "InvoiceNo", "InvoiceDate", "Quantity"]
if not all(col in df.columns for col in expected_cols):
    st.warning(f"Your file is missing some expected columns. Found columns: {list(df.columns)}. The app will attempt to proceed if possible.")
df_columns_display = st.expander("Show detected columns")
with df_columns_display:
    st.write(list(df.columns))

# Determine ref_date
try:
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
except Exception:
    pass

ref_date = None
if ref_date_input is not None:
    # if user selected a date, use it; else fallback to max date in dataset
    try:
        ref_date = pd.to_datetime(ref_date_input)
    except Exception:
        ref_date = None

# Compute RFM
rfm = compute_rfm_from_transactions(df, ref_date=ref_date)

# Optionally remove outliers
if st.session_state.get("outliers", True):
    for col in ['Recency', 'Frequency', 'Monetary', 'Quantity']:
        low = rfm[col].quantile(0.01)
        high = rfm[col].quantile(0.99)
        rfm = rfm[(rfm[col] >= low) & (rfm[col] <= high)]
    rfm = rfm.reset_index(drop=True)

# Scale RFM features used for clustering
features = ['Recency', 'Frequency', 'Monetary']
X_scaled = scale_features(rfm, features)

# Run clustering based on selection
cluster_col = None
cluster_labels = None
sil_score = None

if algo_option == "Agglomerative":
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = model.fit_predict(X_scaled)
    cluster_labels = labels
    cluster_col = "AGG_Cluster"
    rfm[cluster_col] = labels
    sil_score = safe_silhouette(X_scaled, labels)

elif algo_option == "GMM":
    model = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm_labels = model.fit_predict(X_scaled)
    rfm['GMM_Cluster'] = gmm_labels
    cluster_col = "GMM_Cluster"
    cluster_labels = gmm_labels
    sil_score = safe_silhouette(X_scaled, gmm_labels)

elif algo_option == "KMeans":
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km_labels = model.fit_predict(X_scaled)
    rfm['KMEANS_Cluster'] = km_labels
    cluster_col = "KMEANS_Cluster"
    cluster_labels = km_labels
    sil_score = safe_silhouette(X_scaled, km_labels)

elif algo_option == "HDBSCAN (if available)":
    if not HDBSCAN_AVAILABLE:
        st.error("HDBSCAN package is not installed in this environment. Install `hdbscan` to use this algorithm.")
        st.stop()
    else:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom')
        h_labels = clusterer.fit_predict(X_scaled)
        rfm['HDBSCAN_Cluster'] = h_labels
        cluster_col = "HDBSCAN_Cluster"
        cluster_labels = h_labels
        sil_score = safe_silhouette(X_scaled, h_labels)

# Basic segment naming (optional) - simple ordering by mean Monetary
if cluster_col:
    # create an ordered label for color consistency
    rfm[cluster_col] = rfm[cluster_col].astype(int)
    cluster_order = rfm.groupby(cluster_col)['Monetary'].sum().sort_values(ascending=False).index.tolist()
    seg_name_map = {c: f"Segment {c}" for c in rfm[cluster_col].unique()}
    # Assign a human readable name optionally (user can change mapping later)
    rfm['Segment'] = rfm[cluster_col].map(lambda x: seg_name_map.get(x, f"Segment {x}"))

# --------------------------
# Top KPIs - Overview Tab
# --------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Segment Explorer", "Trend Analysis", "Customer View"])

with tab1:
    st.subheader("Overview")
    total_customers = rfm['CustomerID'].nunique()
    total_revenue = rfm['Monetary'].sum()
    avg_monetary = rfm['Monetary'].mean()

    c1, c2, c3, c4 = st.columns([1.5,1.5,1.5,1.2])
    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Total Revenue", f"${total_revenue:,.0f}")
    c3.metric("Average Monetary", f"${avg_monetary:,.0f}")
    c4.metric("Silhouette", f"{sil_score:.3f}" if sil_score is not None else "N/A")

    st.markdown("### Revenue by Segment (Donut)")
    rev_summary = rfm.groupby('Segment')['Monetary'].sum().reset_index().sort_values('Monetary', ascending=False)
    if not rev_summary.empty:
        donut = px.pie(rev_summary, names='Segment', values='Monetary', hole=0.45,
                       title="Revenue share by Segment")
        st.plotly_chart(donut, use_container_width=True)

    if show_treemap:
        st.markdown("### Treemap: Revenue by Segment")
        treemap_df = rfm.groupby('Segment').agg(Customer_Count=('CustomerID','nunique'), Total_Revenue=('Monetary','sum')).reset_index()
        fig_tm = px.treemap(treemap_df, path=['Segment'], values='Total_Revenue', color='Customer_Count',
                            color_continuous_scale='Blues', title="Treemap: Revenue")
        st.plotly_chart(fig_tm, use_container_width=True)

with tab2:
    st.subheader("Segment Explorer")
    st.markdown("Profile each segment in RFM space and inspect raw customers.")

    # Distribution charts
    seg = st.selectbox("Choose segment to inspect", options=sorted(rfm['Segment'].unique()))
    seg_df = rfm[rfm['Segment'] == seg]
    st.markdown(f"**Segment:** {seg} — Customers: {seg_df.shape[0]}, Revenue: ${seg_df['Monetary'].sum():,.0f}")

    # Radar / Parallel coordinates (use plotly parallel coordinates)
    # Normalize features for parallel coordinates
    pc_df = seg_df.copy()
    if not pc_df.empty:
        for f in features:
            pc_df[f+"_norm"] = (pc_df[f] - pc_df[f].min()) / (pc_df[f].max() - pc_df[f].min() + 1e-9)
        try:
            pc_fig = px.parallel_coordinates(pc_df, dimensions=[f+"_norm" for f in features], color=pc_df['Monetary'],
                                             labels={f+"_norm": f for f in features}, title="Parallel Coordinates (normalized)")
            st.plotly_chart(pc_fig, use_container_width=True)
        except Exception:
            st.write("Parallel coordinates not available for this segment.")

    # Show head of actual customers in this segment
    st.markdown("Sample customers in this segment")
    cols_to_show = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Quantity']
    st.dataframe(seg_df[cols_to_show].sort_values('Monetary', ascending=False).head(50))

with tab3:
    st.subheader("Trend Analysis")
    st.markdown("Monthly revenue trends per segment (using invoice-level data).")

    # Merge clusters back to original df to compute monthly revenue over time
    df_for_merge = df.copy()
    # Ensure customer id column types match
    if 'CustomerID' in df_for_merge.columns and 'CustomerID' in rfm.columns:
        # if types mismatch try converting
        try:
            df_for_merge['CustomerID'] = df_for_merge['CustomerID'].astype(rfm['CustomerID'].dtype)
        except Exception:
            pass

    df_for_merge = df_for_merge.merge(rfm[['CustomerID', 'Segment', cluster_col]], on='CustomerID', how='left')

    # create month index relative to first invoice in dataset (1..n)
    df_for_merge['InvoiceDate'] = pd.to_datetime(df_for_merge['InvoiceDate'], errors='coerce')
    first_date = df_for_merge['InvoiceDate'].min()
    df_for_merge['MonthIndex'] = ((df_for_merge['InvoiceDate'].dt.year - first_date.year) * 12 +
                              (df_for_merge['InvoiceDate'].dt.month - first_date.month))
    # Clip to first 24 months for display or full range if less
    max_months = int(st.slider("Show months range (1..N)", min_value=6, max_value=36, value=24))
    df_for_merge = df_for_merge[df_for_merge['MonthIndex'].between(1, max_months)]

    monthly_rev = df_for_merge.groupby(['MonthIndex', 'Segment'])['TotalPrice'].sum().reset_index()
    if monthly_rev.empty:
        st.info("No invoice-level revenue available for monthly trend. Ensure TotalPrice or UnitPrice+Quantity exist.")
    else:
        line_fig = px.line(monthly_rev, x='MonthIndex', y='TotalPrice', color='Segment',
                           title="Monthly Revenue per Segment", markers=True)
        line_fig.update_layout(xaxis_title="Month Index", yaxis_title="Revenue")
        st.plotly_chart(line_fig, use_container_width=True)

with tab4:
    st.subheader("Customer View")
    st.markdown("Search and inspect individual customers and download results.")
    cust_id = st.text_input("Enter CustomerID to filter (exact match)", value="")
    if cust_id != "":
        try:
            cust_filtered = rfm[rfm['CustomerID'].astype(str) == str(cust_id)]
            if cust_filtered.empty:
                st.warning("Customer not found in RFM (maybe filtered by outlier removal).")
            else:
                st.dataframe(cust_filtered)
        except Exception as e:
            st.error(f"Filter error: {e}")

    st.markdown("Download processed RFM and cluster assignments")
    st.markdown(df_to_csv_download(rfm, filename="rfm_segmented.csv"), unsafe_allow_html=True)

# --------------------------
# Optional: 3D scatter / PCA view shown in Overview or separate expander
# --------------------------
if show_3d:
    st.markdown("### 3D PCA Scatter (explore clusters in reduced space)")
    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(X_scaled)
    rfm['PCA1'] = pca_data[:,0]; rfm['PCA2'] = pca_data[:,1]; rfm['PCA3'] = pca_data[:,2]
    fig3d = px.scatter_3d(rfm, x='PCA1', y='PCA2', z='PCA3', color='Segment', hover_data=['CustomerID', 'Monetary'])
    st.plotly_chart(fig3d, use_container_width=True)

st.markdown("---")
st.caption("Built with ❤️ — upload your dataset and switch algorithms freely. Export RFM + cluster labels with the download link above.")
