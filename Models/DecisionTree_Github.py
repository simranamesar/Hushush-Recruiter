# Import required libraries
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

# %%
# Load dataset
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
DATA_DIR = CODE_DIR / "dataset"
csv_path = DATA_DIR / "devto_github_final_dataset.csv"
df = pd.read_csv(csv_path)

print(df.shape)

# %%
# -----------------------------
# 1. Drop unnecessary columns
# -----------------------------
columns_to_drop = [
    "slug", "user_profile_image", "published_at",
    "social_image", "published_timestamp",
    "positive_reactions_count"
]

df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# -----------------------------
# 2. Keep only GitHub-related columns
# -----------------------------
github_columns = [col for col in df.columns if "github" in col.lower()]
df_github = df[github_columns].copy()

print("GitHub Columns Used:")
print(df_github.columns.tolist())
print("\nDataset Shape:", df_github.shape)

# -----------------------------
# 3. Handle Missing Values
# -----------------------------
numeric_cols = df_github.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df_github.select_dtypes(exclude=np.number).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Preprocess data
X_processed = preprocessor.fit_transform(df_github)

# %%
# -----------------------------
# 4. Elbow Method
# -----------------------------
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_processed)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(K_range, inertia)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method For Optimal k")
plt.show()

# %%
# -----------------------------
# 5. KMeans Clustering (k=2)
# -----------------------------
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_processed)

df_github["Cluster"] = clusters

# %%
# Visualize cluster distribution
plt.figure()
df_github["Cluster"].value_counts().sort_index().plot(kind="bar")
plt.xlabel("Cluster")
plt.ylabel("Number of Users")
plt.title("Cluster Distribution")
plt.show()

# %%
# -----------------------------
# 6. Decision Tree Supervised Model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, clusters, test_size=0.2, random_state=42
)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)

print("\nDecision Tree Classification Report:\n")
print(classification_report(y_test, y_pred))

# Remove email column
if "github_email" in df_github.columns:
    df_github = df_github.drop(columns=["github_email"])

print("Columns after removing email:")
print(df_github.columns.tolist())

# %%
# -----------------------------
# 7. Identify Top Technical Users
# -----------------------------
# Assuming higher numeric GitHub metrics = stronger technical skills
if len(numeric_cols) > 0:
    df_scores = df_github.copy()
    df_scores["technical_score"] = df_scores[numeric_cols].sum(axis=1)
    top_users = df_scores.sort_values(by="technical_score", ascending=False).head(10)

    print("\nTop 10 Technical Users Based on GitHub Metrics:\n")
    print(top_users.head(10))
else:
    print("\nNo numeric GitHub metrics found to compute technical score.")

def get_decisiontree_github_table():
    df_table = top_users.head(5)
    return df_table