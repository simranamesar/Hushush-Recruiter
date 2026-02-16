#import libraries
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

np.random.seed(42)

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
DATA_DIR = CODE_DIR / "dataset"
csv_path = DATA_DIR / "codeforces_technical_users.csv"
df = pd.read_csv(csv_path)


# check for missing values
missing_values = df.isnull().sum()
print("\nmissing values :\n",missing_values)

# check for columns with missing values
columns_with_missing_values = missing_values[missing_values>0].index.tolist()
print("\nColumns with missing values:",columns_with_missing_values)

df_cleaned = df.copy()

# impute numerical columns with median
numerical_cols =  df.select_dtypes(include=['int64', 'float64']).columns
num_medians = df_cleaned[numerical_cols].median()
df_cleaned[numerical_cols] = df_cleaned[numerical_cols].fillna(num_medians)

# impute categorical columns with mode
categorical_cols = ['rank']
categorical_modes = df_cleaned[categorical_cols].mode().iloc[0]
df_cleaned[categorical_cols] = df_cleaned[categorical_cols].fillna(categorical_modes)

# convert top_tags and top_languages to string
df_cleaned['top_tags'] = df_cleaned['top_tags'].fillna("").astype(str)
df_cleaned['top_languages'] = df_cleaned['top_languages'].fillna("").astype(str)

# vectorize top_tags
tag_vectorizer = CountVectorizer(
    tokenizer=lambda x: [i.strip() for i in x.split(',') if i.strip() != ""],
    token_pattern=None
)
tag_matrix = tag_vectorizer.fit_transform(df_cleaned['top_tags'])

tag_df = pd.DataFrame(
    tag_matrix.toarray(),
    columns=tag_vectorizer.get_feature_names_out(),
    index=df_cleaned.index
)

# vectorize top_languages
lang_vectorizer = CountVectorizer(
    tokenizer=lambda x: [i.strip() for i in x.split(',') if i.strip() != ""],
    token_pattern=None
)
lang_matrix = lang_vectorizer.fit_transform(df_cleaned['top_languages'])

lang_df = pd.DataFrame(
    lang_matrix.toarray(),
    columns=lang_vectorizer.get_feature_names_out(),
    index=df_cleaned.index
)

numeric_df = df_cleaned[['rating','max_rating','total_solved','hard_solved','contests']]

# combine numeric features with tag and language features
X = pd.concat(
    [numeric_df, tag_df, lang_df],
    axis=1
)
print("\nFeature Matrix Shape:", X.shape)

X_scaled = X.copy()

num_cols = ['rating','max_rating','total_solved','hard_solved','contests']

# Scale numeric features using StandardScaler
scaler = StandardScaler()
X_scaled[num_cols] = scaler.fit_transform(X[num_cols])
print("\nScaled data shape:", X_scaled.shape)

# Elbow method to find optimal K
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, n_init=10, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Total Distance from Centers (WCSS)')
plt.xticks(range(1, 11))
plt.grid(True)
# plt.show()

features = ['rating','max_rating','total_solved','hard_solved','contests']

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)
cluster_labels = kmeans.fit_predict(X_scaled)
df_cleaned['cluster'] = cluster_labels


# Clusters using PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10,6))
plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df_cleaned['cluster'],
    s=60,
    alpha=0.7
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clusters (PCA Projection)")
plt.colorbar(label="Cluster")
plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()


# cluster analysis and candidate selection using centroid method
cluster_centroids = df_cleaned.groupby('cluster')[features].mean()

print("\nCluster Centroid Metrics:\n")
print(cluster_centroids)


cluster_strength = cluster_centroids.mean(axis=1)
best_cluster = cluster_strength.idxmax()
print("Strongest cluster:", best_cluster)


print("\nCluster Strength Score:\n")
print(cluster_strength)

best_cluster = cluster_strength.idxmax()
print("\nStrongest Cluster Selected:", best_cluster)

cluster_summary = df_cleaned.groupby('cluster')[features].mean()
print(cluster_summary)

# prepare data for modeling
y = df_cleaned['cluster']

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# train Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb_model.fit(X_train, y_train)

y_pred_gb = gb_model.predict(X_test)

print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_gb))

# feature importance
feature_importance_gb = pd.DataFrame({
    'Feature': X.columns,
    'Importance': gb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Important Features (GB):")
print(feature_importance_gb.head(10))


# gradient boosting predictions
all_predictions = gb_model.predict(X_scaled)
df_cleaned['gb_predicted_cluster'] = all_predictions

prediction_probabilities = gb_model.predict_proba(X_scaled)
df_cleaned['gb_confidence'] = prediction_probabilities.max(axis=1)

df_strength_gb = df_cleaned.copy()

# calculate average strength for each cluster based on important features
df_strength_gb[features] = X_scaled[features]
cluster_strength_gb = df_strength_gb.groupby(
    'gb_predicted_cluster'
)[features].mean().mean(axis=1)

strongest_cluster_gb = cluster_strength_gb.idxmax()

top_candidates_gb = df_strength_gb[
    (df_strength_gb['gb_predicted_cluster'] == strongest_cluster_gb) &
    (df_strength_gb['gb_confidence'] > 0.8)
]

top_5_gb = top_candidates_gb.sort_values(
    by=features,
    ascending=False
).head(5)


def get_gradientboosting_codeforces_table():
    df_table = df_cleaned.loc[top_5_gb.index][
        ['username', 'rating', 'max_rating', 'total_solved', 'hard_solved', 'contests']
    ]
    return df_table





