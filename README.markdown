[![Project Banner](image.png)](https://srikanthsk1-clustering-and-segmentation-app-yjtslr.streamlit.app/)

# ✈️ CLICK THE IMAGE TO VIEW THE LIVE PROJECT

# ✈️ Air Traffic Passenger Clustering Project



## 📖 Overview

This project implements a **data-driven clustering solution** for air traffic passenger statistics, following the **CRISP-ML(Q)** methodology. The goal is to optimize airline and terminal operations, enhance passenger satisfaction, and maximize profitability through actionable segmentation insights. The solution includes a backend clustering pipeline and an interactive **Streamlit** web application for user-friendly analysis and visualization.

### 🎯 Business Objectives

- **Business Success**: Achieve 10-12% improvement in operational efficiency through targeted resource allocation.
- **ML Success**: Attain a Silhouette Score ≥ 0.7 for high-quality clustering.
- **Economic Success**: Drive an 8% revenue increase via optimized operations and targeted strategies.

## 📂 Project Structure

```
air_traffic_clustering/
│
├── clustering_pipeline.py       # Backend ML pipeline
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── AirTraffic_Passenger_Statistics.csv  # Dataset
├── README.md                   # Project documentation
├── silhouette_comparison.png   # Visualization output
├── cluster_distribution.png    # Visualization output
├── dendrogram.png              # Visualization output
└── clustered_data.csv          # Clustered output data
```

## 🛠️ Setup Instructions

### Prerequisites

- **Python**: Version 3.9+
- **Dependencies**: Listed in `requirements.txt`

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-repo/air_traffic_clustering.git
   cd air_traffic_clustering
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App**:

   ```bash
   streamlit run app.py
   ```

4. **Access the App**:
   Open your browser and navigate to `http://localhost:8501`.

## 📋 Dependencies

The project relies on the following Python libraries (see `requirements.txt` for versions):

- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` & `seaborn` - Static visualizations
- `scikit-learn` - Clustering and preprocessing
- `scipy` - Hierarchical clustering
- `streamlit` - Web application framework
- `plotly` - Interactive visualizations
- `sqlalchemy` & `pymysql` - Database connectivity

## 📊 Data Requirements

The input dataset (`AirTraffic_Passenger_Statistics.csv`) should include:

- **Operating Airline**: Name of the airline
- **Operating Airline IATA Code**: Airline IATA code
- **GEO Region**: Geographic region
- **Terminal**: Terminal identifier
- **Boarding Area**: Boarding area code
- **Passenger Count**: Number of passengers
- **Year**: Year of record
- **Month**: Month of record

### Sample Data Preview

| Operating Airline | GEO Region | Passenger Count | Year | Month |
| ----------------- | ---------- | --------------- | ---- | ----- |
| United Airlines   | US         | 15,000          | 2024 | 1     |
| Delta Airlines    | US         | 12,000          | 2024 | 2     |
| American Airlines | Europe     | 18,000          | 2024 | 3     |

## 🔄 CRISP-ML(Q) Methodology

The project follows the **CRISP-ML(Q)** framework for a structured ML workflow:

1. **Business & Data Understanding**

   - Load and explore data to understand airline operations and identify key variables.
   - Key metrics: Total airlines, geographic regions, passenger counts.

2. **Data Preparation**

   - Clean data: Remove duplicates, handle missing values, and cap outliers (winsorization).
   - Preprocess: Scale numeric features (MinMaxScaler) and encode categorical features (OneHotEncoder).
   - Dimensionality reduction: Apply PCA to retain 95% variance.

3. **Model Building**

   - Use hierarchical clustering (AgglomerativeClustering) with multiple linkage methods (`ward`, `complete`, `average`).
   - Test 2 to 10 clusters to find the optimal configuration based on Silhouette Score.

4. **Model Evaluation**

   - Metrics: Silhouette Score, cluster size distribution, within-cluster variance, and business insights (e.g., passenger counts per cluster).
   - Visualizations: Silhouette comparison, cluster distribution, dendrogram.

5. **Deployment**

   - Deploy via Streamlit for interactive data upload, parameter tuning, and result visualization.
   - Features: File upload, interactive clustering parameters, downloadable results, and executive report.

6. **Monitoring & Maintenance**
   - Track Silhouette Score and cluster balance over time.
   - Retrain model with new data or if performance degrades.

## 🚀 Using the Streamlit App

1. **Upload Data**: Upload your `AirTraffic_Passenger_Statistics.csv` file.
2. **Configure Parameters**:
   - Select number of clusters (2-10).
   - Choose linkage method (`ward`, `complete`, `average`).
   - Enable/disable PCA and set variance retention (0.80-0.99).
3. **Run Analysis**: Click "Run Clustering Analysis" to generate clusters.
4. **Explore Results**:
   - View interactive visualizations (cluster distribution, dendrogram, business insights).
   - Analyze cluster profiles (top airlines, regions, passenger stats).
   - Download clustered data, model, and executive report.

![Streamlit Interface](https://via.placeholder.com/800x400.png?text=Streamlit+App+Interface)

## 📈 Expected Results

### Clustering Quality

| Scenario             | Silhouette Score | Status               |
| -------------------- | ---------------- | -------------------- |
| Poor Clustering      | 0.1 - 0.3        | ❌ Needs improvement |
| Fair Clustering      | 0.3 - 0.5        | ⚠️ Acceptable        |
| Good Clustering      | 0.5 - 0.7        | ✅ Good              |
| Excellent Clustering | 0.7 - 1.0        | ✅ Excellent         |

### Business Impact

- **Operational Efficiency (10-12%)**:
  - High-volume clusters: Allocate more staff and larger aircraft.
  - Medium clusters: Optimize standard operations.
  - Low-volume clusters: Use smaller aircraft, reduce frequency.
- **Revenue Increase (8%)**:
  - High-volume: Premium services, higher pricing.
  - Medium: Balanced pricing, loyalty programs.
  - Low-volume: Dynamic pricing, cost reduction.

## 📸 Visualizations

The pipeline generates the following visualizations:

1. **Silhouette Comparison**: Compares clustering performance across configurations.
   ![Silhouette Comparison](silhouette_comparison.png)
2. **Cluster Distribution**: Shows record and passenger distribution across clusters.
   ![Cluster Distribution](cluster_distribution.png)
3. **Dendrogram**: Visualizes hierarchical clustering structure.
   ![Dendrogram](dendrogram.png)

## 🛡️ Common Issues & Solutions

| Issue                        | Solution                                                                   |
| ---------------------------- | -------------------------------------------------------------------------- |
| Low Silhouette Score (< 0.3) | Enable PCA, try different linkage methods (`ward`, `complete`, `average`). |
| Imbalanced Clusters          | Adjust number of clusters (try 4-5), use `ward` linkage.                   |
| Too Many Features            | Apply feature selection or increase PCA variance threshold (e.g., 0.90).   |

## 📦 Deployment Options

1. **Streamlit Cloud** (Free):
   - Push code to GitHub.
   - Connect repository to Streamlit Cloud and deploy `app.py`.
2. **Local Server**:
   ```bash
   streamlit run app.py --server.port 8501
   ```
3. **Docker**:
   ```dockerfile
   FROM python:3.9
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["streamlit", "run", "app.py"]
   ```

## 🎓 Key Takeaways

### Technical Skills

- End-to-end ML pipeline using CRISP-ML(Q).
- Hierarchical clustering with scikit-learn.
- Data preprocessing (scaling, encoding, PCA).
- Interactive web deployment with Streamlit.
- Comprehensive model evaluation (Silhouette Score, visualizations).

### Business Skills

- Translating business problems to ML solutions.
- Defining and measuring success criteria (ML, business, economic).
- Generating actionable insights from clustering results.
- Communicating findings to stakeholders via reports and visualizations.

## 🔗 Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [CRISP-ML(Q)](https://ml-ops.org/content/crisp-ml)

## 📝 Notes

- Ensure data quality for optimal clustering results.
- Test multiple clustering configurations to find the best setup.
- Validate business insights with domain experts.
- Regularly monitor and retrain the model to maintain performance.

## 🎉 Acknowledgments

Developed as a production-ready solution for airline operation optimization, combining robust ML techniques with an intuitive user interface.

---

_Happy Clustering!_ ✈️
