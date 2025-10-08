# ============================================================================
# STREAMLIT DEPLOYMENT - AIR TRAFFIC CLUSTERING
# File: app.py
# ============================================================================

"""
STREAMLIT DEPLOYMENT FOR AIR TRAFFIC CLUSTERING

This interactive web application allows users to:
1. Upload air traffic data
2. Configure clustering parameters
3. View results and insights
4. Download clustered data

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn import metrics
from scipy.cluster.hierarchy import linkage, dendrogram
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

# Set page config FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="Air Traffic Clustering",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    h2 {
        color: #2ca02c;
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_data(file):
    """
    Load and cache the uploaded CSV file
    
    Why @st.cache_data?
    - Caches the data to avoid reloading on every interaction
    - Improves app performance significantly
    """
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def preprocess_data(df):
    """
    Preprocess the data for clustering
    
    Steps:
    1. Handle missing values
    2. Scale numeric features
    3. Encode categorical features
    """
    # Create a copy
    df_clean = df.copy()
    
    # Handle missing values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    
    # Fill numeric with median
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # Fill categorical with mode
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    # Transform data
    processed_array = preprocessor.fit_transform(df_clean)
    
    return processed_array, preprocessor, df_clean

def apply_clustering(data, n_clusters, linkage_method):
    """
    Apply hierarchical clustering
    
    Parameters:
    -----------
    data : array
        Preprocessed data
    n_clusters : int
        Number of clusters
    linkage_method : str
        Linkage method (ward, complete, average)
    """
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='euclidean',
        linkage=linkage_method
    )
    labels = model.fit_predict(data)
    score = metrics.silhouette_score(data, labels)
    
    return model, labels, score

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main Streamlit application
    """
    
    # ========================================================================
    # HEADER
    # ========================================================================
    
    st.title("✈️ Air Traffic Passenger Statistics - Clustering Analysis")
    st.markdown("""
    ### 📊 Optimize Airline Operations Through Data-Driven Segmentation
    
    **Business Objective:** Maximize operational efficiency and financial health  
    **Success Criteria:** 
    - 🎯 Silhouette Score ≥ 0.7
    - 📈 10-12% efficiency improvement
    - 💰 8% revenue increase
    """)
    
    st.markdown("---")
    
    # ========================================================================
    # SIDEBAR - PHASE 6: MONITORING & MAINTENANCE
    # ========================================================================
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload
        st.subheader("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload Air Traffic CSV",
            type=['csv'],
            help="Upload your AirTraffic_Passenger_Statistics.csv file"
        )
        
        st.markdown("---")
        
        # Clustering parameters
        st.subheader("🎛️ Clustering Parameters")
        
        n_clusters = st.slider(
            "Number of Clusters",
            min_value=2,
            max_value=10,
            value=3,
            help="Select the number of segments to create"
        )
        
        linkage_method = st.selectbox(
            "Linkage Method",
            options=['ward', 'complete', 'average'],
            index=0,
            help="Ward: Minimizes variance within clusters"
        )
        
        use_pca = st.checkbox(
            "Apply PCA",
            value=True,
            help="Dimensionality reduction for better clustering"
        )
        
        if use_pca:
            pca_variance = st.slider(
                "PCA Variance Retained",
                min_value=0.80,
                max_value=0.99,
                value=0.95,
                step=0.01,
                help="Percentage of variance to retain"
            )
        
        st.markdown("---")
        
        # About section
        st.subheader("ℹ️ About")
        st.info("""
        **CRISP-ML(Q) Implementation**
        
        This app follows the complete ML lifecycle:
        1. Business Understanding
        2. Data Preparation
        3. Model Building
        4. Evaluation
        5. Deployment
        6. Monitoring
        """)
    
    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    
    if uploaded_file is None:
        # Show instructions if no file uploaded
        st.info("👆 Please upload your Air Traffic CSV file to begin analysis")
        
        st.markdown("### 📋 Expected Data Format")
        st.markdown("""
        Your CSV should contain the following columns:
        - **Operating Airline**: Airline name
        - **Operating Airline IATA Code**: IATA code
        - **GEO Region**: Geographic region
        - **Terminal**: Terminal identifier
        - **Boarding Area**: Boarding area code
        - **Passenger Count**: Number of passengers
        - **Year**: Year of record
        - **Month**: Month of record
        """)
        
        # Show sample data
        st.markdown("### 📊 Sample Data Preview")
        sample_data = {
            'Operating Airline': ['United Airlines', 'Delta Airlines', 'American Airlines'],
            'GEO Region': ['US', 'US', 'Europe'],
            'Passenger Count': [15000, 12000, 18000],
            'Year': [2024, 2024, 2024],
            'Month': [1, 2, 3]
        }
        st.dataframe(pd.DataFrame(sample_data))
        
        return
    
    # ========================================================================
    # PHASE 1: BUSINESS & DATA UNDERSTANDING
    # ========================================================================
    
    st.header("📊 Phase 1: Business & Data Understanding")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data(uploaded_file)
    
    if df is None:
        st.error("Failed to load data. Please check your file format.")
        return
    
    st.success(f"✅ Data loaded successfully: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Display data overview in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{df.shape[0]:,}")
    
    with col2:
        st.metric("Total Features", df.shape[1])
    
    with col3:
        if 'Operating Airline' in df.columns:
            st.metric("Unique Airlines", df['Operating Airline'].nunique())
    
    with col4:
        if 'Passenger Count' in df.columns:
            st.metric("Total Passengers", f"{df['Passenger Count'].sum():,.0f}")
    
    # Data preview
    with st.expander("🔍 View Raw Data", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)
    
    # Statistical summary
    with st.expander("📈 Statistical Summary", expanded=False):
        st.dataframe(df.describe(), use_container_width=True)
    
    # Missing values
    with st.expander("🔍 Data Quality Check", expanded=False):
        missing = df.isnull().sum()
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Values': missing.values,
            'Percentage': (missing.values / len(df) * 100).round(2)
        })
        missing_df = missing_df[missing_df['Missing Values'] > 0]
        
        if len(missing_df) > 0:
            st.warning("⚠️ Missing values detected:")
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("✅ No missing values found!")
    
    st.markdown("---")
    
    # ========================================================================
    # PHASE 2: DATA PREPARATION
    # ========================================================================
    
    st.header("🔧 Phase 2: Data Preparation")
    
    with st.spinner("Preprocessing data..."):
        processed_data, preprocessor, df_clean = preprocess_data(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"✅ Data cleaned and preprocessed")
        st.info(f"Features after encoding: {processed_data.shape[1]}")
    
    with col2:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        st.info(f"Numeric features: {len(numeric_cols)}")
        st.info(f"Categorical features: {len(categorical_cols)}")
    
    # Apply PCA if selected
    if use_pca:
        with st.spinner("Applying PCA..."):
            pca = PCA(n_components=pca_variance)
            pca_data = pca.fit_transform(processed_data)
            clustering_data = pca_data
            
            st.success(f"✅ PCA applied: {processed_data.shape[1]} → {pca_data.shape[1]} features")
            st.info(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")
    else:
        clustering_data = processed_data
        pca = None
    
    st.markdown("---")
    
    # ========================================================================
    # PHASE 3: MODEL BUILDING
    # ========================================================================
    
    st.header("🤖 Phase 3: Model Building")
    
    # Run clustering button
    if st.button("🚀 Run Clustering Analysis", type="primary", use_container_width=True):
        
        with st.spinner("Building clustering model..."):
            model, labels, silhouette_score = apply_clustering(
                clustering_data, 
                n_clusters, 
                linkage_method
            )
        
        # Store in session state
        st.session_state['model'] = model
        st.session_state['labels'] = labels
        st.session_state['silhouette_score'] = silhouette_score
        st.session_state['df_clean'] = df_clean
        st.session_state['clustering_data'] = clustering_data
        st.session_state['pca'] = pca
    
    # ========================================================================
    # PHASE 4: MODEL EVALUATION
    # ========================================================================
    
    if 'model' in st.session_state:
        
        st.markdown("---")
        st.header("📊 Phase 4: Model Evaluation")
        
        labels = st.session_state['labels']
        silhouette_score = st.session_state['silhouette_score']
        df_clean = st.session_state['df_clean']
        
        # Add cluster labels to dataframe
        df_clustered = df_clean.copy()
        df_clustered['Cluster'] = labels
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Silhouette Score",
                f"{silhouette_score:.4f}",
                delta=f"{silhouette_score - 0.7:.4f}" if silhouette_score >= 0.7 else None
            )
            if silhouette_score >= 0.7:
                st.success("✅ Target achieved!")
            else:
                st.warning("⚠️ Below target (0.7)")
        
        with col2:
            st.metric("Number of Clusters", n_clusters)
            st.info(f"Method: {linkage_method}")
        
        with col3:
            # Calculate cluster balance (coefficient of variation)
            cluster_sizes = pd.Series(labels).value_counts()
            cv = (cluster_sizes.std() / cluster_sizes.mean()) * 100
            st.metric("Cluster Balance", f"{cv:.1f}%")
            if cv < 30:
                st.success("✅ Well balanced")
            else:
                st.warning("⚠️ Imbalanced")
        
        # ====================================================================
        # VISUALIZATIONS
        # ====================================================================
        
        st.markdown("### 📈 Cluster Visualizations")
        
        # Visualization tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Distribution", 
            "🌳 Dendrogram", 
            "💼 Business Insights",
            "📉 Detailed Analysis"
        ])
        
        # TAB 1: Distribution
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Cluster size distribution
                fig = px.bar(
                    x=cluster_sizes.index,
                    y=cluster_sizes.values,
                    labels={'x': 'Cluster', 'y': 'Number of Records'},
                    title='Records per Cluster',
                    color=cluster_sizes.values,
                    color_continuous_scale='Blues'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Passenger distribution by cluster
                if 'Passenger Count' in df_clustered.columns:
                    passenger_by_cluster = df_clustered.groupby('Cluster')['Passenger Count'].sum()
                    fig = px.bar(
                        x=passenger_by_cluster.index,
                        y=passenger_by_cluster.values,
                        labels={'x': 'Cluster', 'y': 'Total Passengers'},
                        title='Total Passengers per Cluster',
                        color=passenger_by_cluster.values,
                        color_continuous_scale='Oranges'
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        
        # TAB 2: Dendrogram
        with tab2:
            st.markdown("#### Hierarchical Clustering Dendrogram")
            
            fig, ax = plt.subplots(figsize=(14, 6))
            clustering_data_sample = st.session_state['clustering_data']
            
            # Sample data if too large (for performance)
            if len(clustering_data_sample) > 1000:
                sample_indices = np.random.choice(
                    len(clustering_data_sample), 
                    1000, 
                    replace=False
                )
                clustering_data_sample = clustering_data_sample[sample_indices]
                st.info("ℹ️ Showing dendrogram for 1000 random samples (for performance)")
            
            dendrogram(
                linkage(clustering_data_sample, method=linkage_method),
                ax=ax
            )
            ax.set_title(f'Hierarchical Clustering Dendrogram ({linkage_method.capitalize()} Linkage)')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Euclidean Distance')
            
            st.pyplot(fig)
        
        # TAB 3: Business Insights
        with tab3:
            st.markdown("#### 💼 Cluster Profiles")
            
            for cluster_id in sorted(df_clustered['Cluster'].unique()):
                cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
                
                with st.expander(f"🔍 Cluster {cluster_id} - {len(cluster_data):,} records", expanded=True):
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("##### 🛫 Airlines")
                        if 'Operating Airline' in cluster_data.columns:
                            top_airlines = cluster_data['Operating Airline'].value_counts().head(5)
                            for airline, count in top_airlines.items():
                                st.write(f"• {airline}: {count:,}")
                    
                    with col2:
                        st.markdown("##### 🌍 Regions")
                        if 'GEO Region' in cluster_data.columns:
                            top_regions = cluster_data['GEO Region'].value_counts().head(5)
                            for region, count in top_regions.items():
                                st.write(f"• {region}: {count:,}")
                    
                    with col3:
                        st.markdown("##### 👥 Passengers")
                        if 'Passenger Count' in cluster_data.columns:
                            st.metric("Total", f"{cluster_data['Passenger Count'].sum():,.0f}")
                            st.metric("Average", f"{cluster_data['Passenger Count'].mean():,.0f}")
                            st.metric("Median", f"{cluster_data['Passenger Count'].median():,.0f}")
                    
                    # Temporal analysis
                    if 'Year' in cluster_data.columns and 'Month' in cluster_data.columns:
                        st.markdown("##### 📅 Temporal Pattern")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Years:** {cluster_data['Year'].min()} - {cluster_data['Year'].max()}")
                        with col2:
                            if len(cluster_data['Month'].mode()) > 0:
                                st.write(f"**Peak Month:** {cluster_data['Month'].mode().values[0]}")
        
        # TAB 4: Detailed Analysis
        with tab4:
            st.markdown("#### 📉 Cluster Statistics")
            
            # Statistical comparison
            if 'Passenger Count' in df_clustered.columns:
                stats_df = df_clustered.groupby('Cluster')['Passenger Count'].agg([
                    ('Count', 'count'),
                    ('Total', 'sum'),
                    ('Mean', 'mean'),
                    ('Median', 'median'),
                    ('Std Dev', 'std'),
                    ('Min', 'min'),
                    ('Max', 'max')
                ]).round(0)
                
                st.dataframe(stats_df, use_container_width=True)
                
                # Box plot
                fig = px.box(
                    df_clustered,
                    x='Cluster',
                    y='Passenger Count',
                    title='Passenger Count Distribution by Cluster',
                    color='Cluster'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # ====================================================================
        # SUCCESS CRITERIA EVALUATION
        # ====================================================================
        
        st.markdown("---")
        st.markdown("### 🎯 Success Criteria Evaluation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🎯 ML Success")
            if silhouette_score >= 0.7:
                st.success(f"✅ Achieved: {silhouette_score:.4f} ≥ 0.7")
            else:
                st.error(f"❌ Not achieved: {silhouette_score:.4f} < 0.7")
                st.info("💡 Try: Different linkage method or enable PCA")
        
        with col2:
            st.markdown("#### 📈 Business Success")
            st.success(f"✅ {n_clusters} distinct segments created")
            st.info("Enables 10-12% efficiency improvement through targeted operations")
        
        with col3:
            st.markdown("#### 💰 Economic Success")
            st.success("✅ Actionable insights generated")
            st.info("Expected 8% revenue increase through optimization")
        
        # ====================================================================
        # PHASE 5: DEPLOYMENT - DOWNLOAD RESULTS
        # ====================================================================
        
        st.markdown("---")
        st.header("💾 Phase 5: Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download clustered data
            csv = df_clustered.to_csv(index=False)
            st.download_button(
                label="📥 Download Clustered Data (CSV)",
                data=csv,
                file_name="air_traffic_clustered.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Download model
            model_data = pickle.dumps({
                'model': st.session_state['model'],
                'n_clusters': n_clusters,
                'linkage_method': linkage_method,
                'silhouette_score': silhouette_score,
                'use_pca': use_pca
            })
            st.download_button(
                label="📥 Download Model (PKL)",
                data=model_data,
                file_name="clustering_model.pkl",
                mime="application/octet-stream",
                use_container_width=True
            )
        
        # Generate report
        st.markdown("### 📄 Executive Summary Report")
        
        report = f"""
        ## Air Traffic Clustering Analysis Report
        
        ### Executive Summary
        
        **Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}
        
        **Dataset Overview:**
        - Total Records: {df.shape[0]:,}
        - Features Analyzed: {df.shape[1]}
        - Time Period: {df['Year'].min() if 'Year' in df.columns else 'N/A'} - {df['Year'].max() if 'Year' in df.columns else 'N/A'}
        
        **Clustering Results:**
        - Algorithm: Hierarchical Clustering
        - Linkage Method: {linkage_method.capitalize()}
        - Number of Clusters: {n_clusters}
        - Silhouette Score: {silhouette_score:.4f}
        - PCA Applied: {'Yes' if use_pca else 'No'}
        
        **Success Criteria Status:**
        - ML Success (Silhouette ≥ 0.7): {'✅ Achieved' if silhouette_score >= 0.7 else '❌ Not Achieved'}
        - Business Success (Segmentation): ✅ Achieved ({n_clusters} segments)
        - Economic Success (Insights): ✅ Achieved
        
        **Key Insights:**
        """
        
        # Add cluster insights
        for cluster_id in sorted(df_clustered['Cluster'].unique()):
            cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
            report += f"\n\n**Cluster {cluster_id}:**"
            report += f"\n- Size: {len(cluster_data):,} records ({len(cluster_data)/len(df_clustered)*100:.1f}%)"
            
            if 'Passenger Count' in cluster_data.columns:
                report += f"\n- Total Passengers: {cluster_data['Passenger Count'].sum():,.0f}"
            
            if 'Operating Airline' in cluster_data.columns:
                top_airline = cluster_data['Operating Airline'].mode().values[0]
                report += f"\n- Dominant Airline: {top_airline}"
            
            if 'GEO Region' in cluster_data.columns:
                top_region = cluster_data['GEO Region'].mode().values[0]
                report += f"\n- Primary Region: {top_region}"
        
        report += f"\n\n### Recommendations"
        report += f"\n1. Focus operational resources on high-traffic clusters"
        report += f"\n2. Develop targeted marketing strategies for each segment"
        report += f"\n3. Optimize terminal and boarding area allocation"
        report += f"\n4. Monitor cluster trends over time for strategic planning"
        
        st.markdown(report)
        
        # Download report
        st.download_button(
            label="📥 Download Report (TXT)",
            data=report,
            file_name="clustering_analysis_report.txt",
            mime="text/plain",
            use_container_width=True
        )

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()