import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.stats import entropy
import seaborn as sns
import matplotlib.pyplot as plt

#Seed for reproducibility
np.random.seed(42)

#Load and Clean data
df = pd.read_csv("data/pbp_export.csv")

#Drop rows with no play_type (not a real play)
df = df.dropna(subset=["play_type"])

#Focus on core offensive plays
main_types = ["run", "pass"]
df = df[df["play_type"].isin(main_types)]

#Handle NA values
numeric_fill = {
    "down": 0,
    "ydstogo": -1,
    "yardline_100": -1,
    "score_differential": 0,
    "game_seconds_remaining": -1,
}


for col, val in numeric_fill.items():
    df[col] = df[col].fillna(val)

cat_cols = [
    "offense_personnel",
    "defense_personnel",
    "posteam",
    "season",
    "game_id"
]

num_cols = [
    "down",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "game_seconds_remaining"
]

target_col = "play_type"

X = df[num_cols + cat_cols]
y = df[target_col]

#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)

pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", model)
])

#Train the model
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))

#Predict probabilities
probs = pipe.predict_proba(X_test)
class_labels = pipe.named_steps["model"].classes_

probs_df = pd.DataFrame(
    probs,
    index=X_test.index,
    columns=[f"prob_{c}" for c in class_labels]
)

results = pd.concat(
    [
        df.loc[X_test.index, ["posteam", "play_type"]],
        probs_df
    ],
    axis=1
)

#Entropy & Accuracy Metrics
results["entropy"] = entropy(probs.T)
results["predicted_play_type"] = pipe.predict(X_test)
results["correct"] = (results["play_type"] == results["predicted_play_type"])

team_stats = results.groupby("posteam").agg(
    accuracy=("correct", "mean"),
    avg_entropy=("entropy", "mean"),
    n_plays=("correct", "size")
).reset_index()

team_stats = team_stats.sort_values(
    ["accuracy", "avg_entropy"],
    ascending=[False, True]
)

print("\nMost predictable teams:")
print(team_stats.head(10))

#Visualize team predictability
plt.figure(figsize=(10,6))
sns.scatterplot(
    data=team_stats,
    x="avg_entropy",
    y="accuracy",
    hue="posteam"
)
plt.xlabel("Average prediction entropy (lower = more predictable)")
plt.ylabel("Model Accuracy per team")
plt.title("Team-level play predictability")
plt.savefig("reports/team_play_predictability.png", dpi=300)