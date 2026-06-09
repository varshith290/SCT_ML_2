## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/customer-segmentation.git
cd customer-segmentation
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Download the dataset
- Go to [kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- Click **Download** → unzip
- Place `Mall_Customers.csv` in the project folder

### 4. Run the script
```bash
python customer_segmentation.py
```

## What the Script Does

| Step | Description |
|------|-------------|
| 1 | Load data and inspect shape, dtypes, missing values |
| 2 | Data cleaning — handle nulls, remove duplicates, encode gender |
| 3 | Feature engineering — TotalBath, Age Group, Income Group, Ratio |
| 4 | EDA — distributions, correlations, key statistics |
| 5 | Find optimal K — Elbow method + Silhouette score |
| 6 | Train K-Means — 2D, 3D, and full feature models |
| 7 | Label clusters — auto-label based on income & spending |
| 8 | Visualize — 8 plots saved as PNG |

## Features Used

| Feature | Description |
|---------|-------------|
| Age | Customer age in years |
| Annual_Income | Annual income in thousands of dollars |
| Spending_Score | Score assigned by mall (1–100) based on behaviour |
| Gender_Enc | Gender encoded as 0 (Female) / 1 (Male) |
| Income_Spending_Ratio | Annual income divided by spending score |
| Age_Group | Binned age — Young / Adult / Middle / Senior |
| Income_Group | Binned income — Low / Medium / High / Very High |

## How K-Means Works
## Finding Optimal K

Two methods used together:

| Method | How it works |
|--------|-------------|
| Elbow Method | Plot inertia vs K — pick the "elbow" point where curve flattens |
| Silhouette Score | Measures how well each point fits its cluster — higher is better |

## Customer Segments Identified

| Cluster | Type | Description |
|---------|------|-------------|
| 0 | High Income, High Spender | Premium customers — top priority for loyalty rewards |
| 1 | High Income, Low Spender | Careful spenders — target with exclusive offers |
| 2 | Low Income, High Spender | Impulsive buyers — respond well to discounts |
| 3 | Low Income, Low Spender | Budget conscious — target with value deals |
| 4 | Average Income, Average Spender | Standard customers — general promotions |

> Note: Cluster labels may vary slightly depending on K and random seed.

## Plots Generated

| File | Description |
|------|-------------|
| plot1_eda_overview.png | Age, income, spending distributions + gender pie |
| plot2_optimal_k.png | Elbow curve + silhouette scores to find best K |
| plot3_clusters_2d.png | Main 2D cluster plot — income vs spending |
| plot4_clusters_3d.png | 3D cluster plot — age + income + spending |
| plot5_cluster_profiles.png | Mean age, income, spending per cluster |
| plot6_cluster_size_gender.png | Cluster sizes + gender breakdown per cluster |
| plot7_correlation.png | Feature correlation heatmap |
| plot8_pca_clusters.png | PCA dimensionality reduction cluster view |

## Key Findings

- **5 distinct customer segments** identified using K-Means
- **High Income + High Spending** group is the most valuable segment for targeted marketing
- **Low Income + High Spending** group is impulsive — responds well to flash sales
- **Gender** has minimal correlation with spending score
- **Age** has a slight negative correlation with spending score — younger customers spend more
- **PCA** confirms clusters are well separated in reduced dimensions

## Libraries Used

- `pandas` — data loading and cleaning
- `numpy` — numerical operations
- `matplotlib` — base plotting and 3D scatter
- `seaborn` — heatmaps and styled charts
- `scikit-learn` — KMeans, StandardScaler, PCA, silhouette score
- 
