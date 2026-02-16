import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



SCRIPT_DIR = Path(__file__).resolve().parent
# Go up one level to code/
CODE_DIR = SCRIPT_DIR.parent
DATA_DIR = CODE_DIR / "dataset"
csv_path = DATA_DIR / "codeforces_technical_users.csv"

df = pd.read_csv(csv_path)
print(df.head())

# %%
features = [
    'rating',
    'max_rating',
    'total_solved',
    'hard_solved',
    'contests'
]

# Handle missing values
df[features] = df[features].fillna(df[features].mean())

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# %%
kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Identify GOOD cluster (higher average rating)
cluster_means = df.groupby('Cluster')['rating'].mean()
good_cluster = cluster_means.idxmax()

df['Good_Label'] = df['Cluster'].apply(
    lambda x: 1 if x == good_cluster else 0
)

df[['username', 'rating', 'Good_Label']].head()
print(df[['username', 'Good_Label']].head())



X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    df['Good_Label'],
    test_size=0.2,
    random_state=42,
    stratify=df['Good_Label']
)


rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight='balanced'
)

rf.fit(X_train, y_train)


df['Good_Probability'] = rf.predict_proba(X_scaled)[:, 1]

df[['username', 'Good_Probability']].head()
print(df[['username', 'Good_Probability']].head())


ranked_candidates = df.sort_values(
    by=['Good_Probability', 'rating'],
    ascending=False
)

ranked_candidates.head(10)


top_5 = ranked_candidates.head(5)

top_5[['username', 'rating', 'total_solved', 'hard_solved', 'contests']]

print(ranked_candidates.head(10))
print(top_5[['username', 'rating', 'total_solved', 'hard_solved', 'contests']])


def get_randomforest_codeforces_table():
    df_table = top_5[['username', 'rating', 'total_solved', 'hard_solved', 'contests']]
    return df_table