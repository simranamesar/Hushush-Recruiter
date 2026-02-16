# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# %%
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
DATA_DIR = CODE_DIR / "dataset"
csv_path = DATA_DIR / "devto_github_final_dataset.csv"
df = pd.read_csv(csv_path)



# %%
# ===============================
# 3. DROP UNWANTED COLUMNS
# ===============================
cols_to_drop = [
    "slug",
    "user_profile_image",
    "published_at",
    "social_image",
    "published_timestamp",
    "positive_reactions_count"
]

df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

print("After Dropping Columns:", df.shape)

# %%
# ===============================
# 4. SELECT TECHNICAL NUMERIC FEATURES
# ===============================
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# Remove obvious IDs if present
numeric_cols = [c for c in numeric_cols if "id" not in c.lower()]

print("Numeric Columns Used:", numeric_cols)

# %%
# ===============================
# 5. HANDLE MISSING VALUES
# ===============================
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(df[numeric_cols])

# %%
# ===============================
# 6. LOG TRANSFORM (for skewed distributions)
# ===============================
X_log = np.log1p(X)

# %%
# ===============================
# 7. STANDARD SCALING
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# %%
# ===============================
# 8. ELBOW METHOD
# ===============================
inertia = []
K_range = range(1, 8)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure()
plt.plot(list(K_range), inertia)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# %%
# ===============================
# 9. SILHOUETTE SCORE FOR k=2
# ===============================
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

sil_score = silhouette_score(X_scaled, clusters)
print("Silhouette Score (k=2):", sil_score)

# %%
# ===============================
# 10. PCA VISUALIZATION
# ===============================
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("KMeans Clusters (k=2)")
plt.show()

# %%
# ===============================
# 11. ADD CLUSTER LABELS
# ===============================
df["cluster"] = clusters

# %%
# ===============================
# 12. RANDOM FOREST (SUPERVISED MODEL)
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, clusters, test_size=0.3, random_state=42
)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred))

# %%
# ===============================
# 13. FEATURE IMPORTANCE
# ===============================
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

plt.figure()
plt.plot(range(len(importances)), importances[sorted_idx])
plt.xlabel("Feature Rank")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.show()

# %%
# ===============================
# 11. IDENTIFY BEST TECHNICAL CLUSTER
# ===============================
cluster_means = df.groupby("cluster")[numeric_cols].mean()

# cluster with highest stars + followers
best_cluster = cluster_means.sum(axis=1).idxmax()

print("Best Technical Cluster:", best_cluster)

# %%
# ===============================
# 12. CREATE TECHNICAL SCORE
# ===============================
df_best = df[df["cluster"] == best_cluster].copy()

df_best["technical_score"] = df_best[numeric_cols].sum(axis=1)

df_best = df_best.sort_values("technical_score", ascending=False)

# %%
# ===============================
# 13. TOP 10 GITHUB TECHNICAL USERS
# ===============================
top_10 = df_best.head(5)


print(top_10[["user_name", "github_username"] + numeric_cols + ["technical_score"]])

def get_randomforest_github_table():
    df_table = top_10[["user_name", "github_username"] + numeric_cols + ["technical_score"]]
    return df_table


