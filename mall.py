# ================================================================
#  Customer Segmentation — K-Means Clustering
#  Dataset: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
# ================================================================

# ── 1. IMPORTS ──────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")


# ── 2. LOAD DATA ────────────────────────────────────────────────
# Download Mall_Customers.csv from Kaggle and place it
# in the same folder as this script.

df = pd.read_csv("Mall_Customers.csv")

print("=" * 60)
print("  CUSTOMER SEGMENTATION — K-MEANS CLUSTERING")
print("=" * 60)
print(f"\nDataset shape  : {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nColumn names   : {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nBasic statistics:\n{df.describe()}")


# ── 3. DATA CLEANING ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 1 — DATA CLEANING")
print("=" * 60)

# Missing value audit
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({
    "Missing Count": missing,
    "Missing %"    : missing_pct
}).query("`Missing Count` > 0")

if len(missing_report) == 0:
    print("\n✔ No missing values found")
else:
    print(f"\nMissing value report:\n{missing_report}")
    df.fillna(df.median(numeric_only=True), inplace=True)
    print("✔ Filled missing values with median")

# Check duplicates
dupes = df.duplicated().sum()
print(f"✔ Duplicate rows: {dupes}")
if dupes > 0:
    df.drop_duplicates(inplace=True)
    print(f"✔ Removed {dupes} duplicates")

# Rename columns for clarity
df.rename(columns={
    "CustomerID"              : "CustomerID",
    "Genre"                   : "Gender",
    "Age"                     : "Age",
    "Annual Income (k$)"      : "Annual_Income",
    "Spending Score (1-100)"  : "Spending_Score"
}, inplace=True)

# Encode gender
df["Gender_Enc"] = (df["Gender"] == "Male").astype(int)

print(f"\nFinal shape    : {df.shape}")
print(f"Null remaining : {df.isnull().sum().sum()}")
print(f"\nGender distribution:\n{df['Gender'].value_counts()}")


# ── 4. FEATURE ENGINEERING ───────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 2 — FEATURE ENGINEERING")
print("=" * 60)

# Income-to-spending ratio
df["Income_Spending_Ratio"] = (
    df["Annual_Income"] / df["Spending_Score"]
).round(2)

# Age group bins
df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[0, 25, 35, 50, 100],
    labels=["Young (≤25)", "Adult (26-35)",
            "Middle (36-50)", "Senior (50+)"]
)

# Income group bins
df["Income_Group"] = pd.cut(
    df["Annual_Income"],
    bins=[0, 40, 70, 100, 200],
    labels=["Low (<40k)", "Medium (40-70k)",
            "High (70-100k)", "Very High (100k+)"]
)

print("New features created:")
for f in ["Income_Spending_Ratio", "Age_Group", "Income_Group"]:
    print(f"  • {f}")

print(f"\nAge group distribution:\n{df['Age_Group'].value_counts()}")
print(f"\nIncome group distribution:\n{df['Income_Group'].value_counts()}")


# ── 5. EDA ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 3 — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print(f"\nAge stats:\n{df['Age'].describe()}")
print(f"\nAnnual Income stats:\n{df['Annual_Income'].describe()}")
print(f"\nSpending Score stats:\n{df['Spending_Score'].describe()}")

print(f"\nCorrelation matrix:")
num_cols = ["Age", "Annual_Income", "Spending_Score",
            "Income_Spending_Ratio"]
print(df[num_cols].corr().round(3))


# ── 6. FIND OPTIMAL K — ELBOW + SILHOUETTE ──────────────────────
print("\n" + "=" * 60)
print("  STEP 4 — FINDING OPTIMAL NUMBER OF CLUSTERS")
print("=" * 60)

# Features for clustering
features_2d  = ["Annual_Income", "Spending_Score"]
features_3d  = ["Age", "Annual_Income", "Spending_Score"]
features_full = ["Age", "Annual_Income", "Spending_Score",
                 "Gender_Enc"]

scaler    = StandardScaler()
X_2d      = scaler.fit_transform(df[features_2d])
X_3d      = scaler.fit_transform(df[features_3d])
X_full    = scaler.fit_transform(df[features_full])

inertias       = []
silhouette_scores = []
k_range        = range(2, 12)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_2d)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_2d, km.labels_))
    print(f"  K={k}  Inertia={km.inertia_:,.1f}  "
          f"Silhouette={silhouette_score(X_2d, km.labels_):.4f}")

optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\n✔ Optimal K = {optimal_k} "
      f"(highest silhouette score = "
      f"{max(silhouette_scores):.4f})")


# ── 7. TRAIN FINAL K-MEANS MODEL ────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 5 — TRAINING K-MEANS MODEL")
print("=" * 60)

# 2D model (Income + Spending) — most interpretable
km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster_2D"] = km_final.fit_predict(X_2d)

# 3D model (Age + Income + Spending)
km_3d = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster_3D"] = km_3d.fit_predict(X_3d)

# Full model (all features)
km_full = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster_Full"] = km_full.fit_predict(X_full)

sil_2d   = silhouette_score(X_2d,   df["Cluster_2D"])
sil_3d   = silhouette_score(X_3d,   df["Cluster_3D"])
sil_full = silhouette_score(X_full, df["Cluster_Full"])

print(f"\nSilhouette Scores:")
print(f"  2D  (Income + Spending)         : {sil_2d:.4f}")
print(f"  3D  (Age + Income + Spending)   : {sil_3d:.4f}")
print(f"  Full (all features)             : {sil_full:.4f}")

# Cluster profiles
print(f"\nCluster profiles (2D model):")
profile = df.groupby("Cluster_2D")[
    ["Age","Annual_Income","Spending_Score"]
].mean().round(2)
print(profile)


# ── 8. LABEL CLUSTERS ────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 6 — CLUSTER LABELLING")
print("=" * 60)

# Auto-label based on income & spending means
cluster_means = df.groupby("Cluster_2D")[
    ["Annual_Income","Spending_Score"]
].mean()

labels_map = {}
for cluster_id, row in cluster_means.iterrows():
    inc = row["Annual_Income"]
    spd = row["Spending_Score"]
    inc_med = cluster_means["Annual_Income"].median()
    spd_med = cluster_means["Spending_Score"].median()

    if inc >= inc_med and spd >= spd_med:
        labels_map[cluster_id] = "High Income\nHigh Spender"
    elif inc >= inc_med and spd < spd_med:
        labels_map[cluster_id] = "High Income\nLow Spender"
    elif inc < inc_med and spd >= spd_med:
        labels_map[cluster_id] = "Low Income\nHigh Spender"
    else:
        labels_map[cluster_id] = "Low Income\nLow Spender"

df["Cluster_Label"] = df["Cluster_2D"].map(labels_map)

print("\nCluster labels assigned:")
for k, v in labels_map.items():
    count = (df["Cluster_2D"] == k).sum()
    print(f"  Cluster {k} → {v.replace(chr(10),' | ')} "
          f"({count} customers)")


# ── 9. VISUALIZATIONS ────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 7 — SAVING PLOTS")
print("=" * 60)

cluster_colors = ["#3266ad","#E24B4A","#1D9E75",
                  "#BA7517","#533AB7","#63991e"]


# ── Plot 1: EDA overview
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Customer Data — EDA Overview",
             fontsize=14, fontweight="bold")

# Age distribution
axes[0,0].hist(df["Age"], bins=20, color="#3266ad",
               edgecolor="white", linewidth=0.4)
axes[0,0].set_title("Age Distribution")
axes[0,0].set_xlabel("Age")
axes[0,0].set_ylabel("Count")

# Annual income distribution
axes[0,1].hist(df["Annual_Income"], bins=20, color="#1D9E75",
               edgecolor="white", linewidth=0.4)
axes[0,1].set_title("Annual Income Distribution")
axes[0,1].set_xlabel("Annual Income (k$)")
axes[0,1].set_ylabel("Count")

# Spending score distribution
axes[0,2].hist(df["Spending_Score"], bins=20, color="#BA7517",
               edgecolor="white", linewidth=0.4)
axes[0,2].set_title("Spending Score Distribution")
axes[0,2].set_xlabel("Spending Score (1-100)")
axes[0,2].set_ylabel("Count")

# Gender pie
gender_counts = df["Gender"].value_counts()
axes[1,0].pie(gender_counts.values,
              labels=gender_counts.index,
              colors=["#3266ad","#E24B4A"],
              autopct="%1.1f%%", startangle=90)
axes[1,0].set_title("Gender Distribution")

# Age group bar
age_counts = df["Age_Group"].value_counts()
axes[1,1].bar(age_counts.index, age_counts.values,
              color=["#3266ad","#1D9E75","#BA7517","#E24B4A"])
axes[1,1].set_title("Age Group Distribution")
axes[1,1].set_ylabel("Count")
axes[1,1].tick_params(axis="x", rotation=20)

# Income vs Spending scatter
axes[1,2].scatter(df["Annual_Income"], df["Spending_Score"],
                  alpha=0.5, color="#533AB7", s=30)
axes[1,2].set_title("Income vs Spending Score")
axes[1,2].set_xlabel("Annual Income (k$)")
axes[1,2].set_ylabel("Spending Score")

plt.tight_layout()
plt.savefig("plot1_eda_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: plot1_eda_overview.png")


# ── Plot 2: Elbow + Silhouette
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Finding Optimal Number of Clusters",
             fontsize=13, fontweight="bold")

# Elbow curve
axes[0].plot(list(k_range), inertias, "o-",
             color="#3266ad", linewidth=2, markersize=8)
axes[0].axvline(optimal_k, color="#E24B4A", linestyle="--",
                linewidth=1.5, label=f"Optimal K = {optimal_k}")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Silhouette scores
axes[1].plot(list(k_range), silhouette_scores, "o-",
             color="#1D9E75", linewidth=2, markersize=8)
axes[1].axvline(optimal_k, color="#E24B4A", linestyle="--",
                linewidth=1.5, label=f"Optimal K = {optimal_k}")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plot2_optimal_k.png", dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: plot2_optimal_k.png")


# ── Plot 3: Main 2D cluster plot
plt.figure(figsize=(10, 7))
for i in range(optimal_k):
    mask = df["Cluster_2D"] == i
    plt.scatter(
        df[mask]["Annual_Income"],
        df[mask]["Spending_Score"],
        s=60, alpha=0.7,
        color=cluster_colors[i % len(cluster_colors)],
        label=f"Cluster {i} — {labels_map[i].replace(chr(10),' | ')}"
    )

# Plot centroids (inverse transform back to original scale)
centroids_orig = scaler.inverse_transform(km_final.cluster_centers_)
plt.scatter(centroids_orig[:, 0], centroids_orig[:, 1],
            s=250, c="black", marker="X",
            zorder=5, label="Centroids")

plt.title(f"Customer Segments — K-Means (K={optimal_k})",
          fontsize=14, fontweight="bold")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(fontsize=9, loc="upper left")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("plot3_clusters_2d.png", dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: plot3_clusters_2d.png")


# ── Plot 4: 3D scatter plot
fig = plt.figure(figsize=(12, 8))
ax  = fig.add_subplot(111, projection="3d")

for i in range(optimal_k):
    mask = df["Cluster_3D"] == i
    ax.scatter(
        df[mask]["Age"],
        df[mask]["Annual_Income"],
        df[mask]["Spending_Score"],
        s=40, alpha=0.7,
        color=cluster_colors[i % len(cluster_colors)],
        label=f"Cluster {i}"
    )

ax.set_xlabel("Age")
ax.set_ylabel("Annual Income (k$)")
ax.set_zlabel("Spending Score")
ax.set_title(f"3D Customer Clusters (Age + Income + Spending)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("plot4_clusters_3d.png", dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: plot4_clusters_3d.png")


# ── Plot 5: Cluster profiles
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Cluster Profiles — Mean Feature Values",
             fontsize=13, fontweight="bold")

profile_cols = ["Age", "Annual_Income", "Spending_Score"]
ylabels      = ["Age (years)", "Annual Income (k$)",
                "Spending Score (1-100)"]

for ax, col, ylabel in zip(axes, profile_cols, ylabels):
    means = df.groupby("Cluster_2D")[col].mean()
    bars  = ax.bar(
        [f"C{i}" for i in means.index],
        means.values,
        color=cluster_colors[:len(means)]
    )
    ax.set_title(f"Mean {col}")
    ax.set_xlabel("Cluster")
    ax.set_ylabel(ylabel)
    for bar, val in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("plot5_cluster_profiles.png", dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: plot5_cluster_profiles.png")


# ── Plot 6: Cluster size + gender breakdown
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Cluster Size & Gender Breakdown",
             fontsize=13, fontweight="bold")

# Cluster sizes
cluster_sizes = df["Cluster_2D"].value_counts().sort_index()
axes[0].bar(
    [f"Cluster {i}" for i in cluster_sizes.index],
    cluster_sizes.values,
    color=cluster_colors[:len(cluster_sizes)]
)
axes[0].set_title("Number of Customers per Cluster")
axes[0].set_ylabel("Count")
for i, v in enumerate(cluster_sizes.values):
    axes[0].text(i, v + 0.5, str(v),
                 ha="center", fontweight="bold")

# Gender per cluster
gender_cluster = df.groupby(
    ["Cluster_2D","Gender"]
).size().unstack(fill_value=0)
gender_cluster.plot(
    kind="bar", ax=axes[1],
    color=["#E24B4A","#3266ad"], rot=0
)
axes[1].set_title("Gender Distribution per Cluster")
axes[1].set_xlabel("Cluster")
axes[1].set_ylabel("Count")
axes[1].legend(title="Gender", fontsize=9)

plt.tight_layout()
plt.savefig("plot6_cluster_size_gender.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: plot6_cluster_size_gender.png")


# ── Plot 7: Correlation heatmap
plt.figure(figsize=(8, 6))
corr = df[["Age","Annual_Income","Spending_Score",
           "Income_Spending_Ratio","Gender_Enc"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8})
plt.title("Feature Correlation Matrix",
          fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot7_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: plot7_correlation.png")


# ── Plot 8: PCA visualization
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_full)

plt.figure(figsize=(9, 6))
for i in range(optimal_k):
    mask = df["Cluster_Full"] == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                s=50, alpha=0.7,
                color=cluster_colors[i % len(cluster_colors)],
                label=f"Cluster {i}")
plt.title("PCA — Customer Clusters (All Features)",
          fontsize=13, fontweight="bold")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.legend(fontsize=9)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("plot8_pca_clusters.png", dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: plot8_pca_clusters.png")


# ── 10. FINAL SUMMARY ────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
print(f"\n  Total customers     : {len(df)}")
print(f"  Optimal clusters    : {optimal_k}")
print(f"  Silhouette score    : {sil_2d:.4f}")
print(f"\n  Cluster breakdown:")

for i in range(optimal_k):
    subset = df[df["Cluster_2D"] == i]
    print(f"\n  Cluster {i} — "
          f"{labels_map[i].replace(chr(10),' | ')}")
    print(f"    Customers     : {len(subset)}")
    print(f"    Avg Age       : {subset['Age'].mean():.1f}")
    print(f"    Avg Income    : ${subset['Annual_Income'].mean():.1f}k")
    print(f"    Avg Spending  : {subset['Spending_Score'].mean():.1f}/100")

print(f"\n✔ All 8 plots saved. Script complete.")
print("=" * 60)