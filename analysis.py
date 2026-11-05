"""Wine Data Analysis
Exploratory data analysis of the UCI Wine dataset (178 wines, 3 cultivars, 13 chemical features).
Run: python analysis.py   -> writes data/wine.csv, charts in images/, metrics in results.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
IMG = ROOT / "images"
DATA = ROOT / "data"
IMG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
PALETTE = {"Cultivar 1": "#2a6fdb", "Cultivar 2": "#f28e2b", "Cultivar 3": "#59a14f"}

# 1. Load and save the data
raw = load_wine(as_frame=True)
df = raw.frame.rename(columns={"od280/od315_of_diluted_wines": "od280_od315"})
df["cultivar"] = df.pop("target").map({0: "Cultivar 1", 1: "Cultivar 2", 2: "Cultivar 3"})
df.to_csv(DATA / "wine.csv", index=False)
features = [c for c in df.columns if c != "cultivar"]

# 2. Data quality
quality = {"rows": len(df), "columns": len(features), "missing_values": int(df.isna().sum().sum()),
           "duplicates": int(df.duplicated().sum())}

# 3. Class balance
counts = df["cultivar"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(counts.index, counts.values, color=[PALETTE[c] for c in counts.index])
ax.bar_label(bars, padding=4)
ax.set_title("Number of wines per cultivar")
ax.set_ylabel("Wines")
ax.set_ylim(0, counts.max() * 1.15)
fig.tight_layout()
fig.savefig(IMG / "01_class_balance.png", dpi=120)
plt.close(fig)

# 4. Correlation heatmap
corr = df[features].corr()
fig, ax = plt.subplots(figsize=(13, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=True, fmt=".2f",
            annot_kws={"size": 10}, linewidths=0.5, cbar_kws={"shrink": 0.7}, ax=ax)
ax.set_title("Correlation between chemical features")
fig.tight_layout()
fig.savefig(IMG / "02_correlation_heatmap.png", dpi=110)
plt.close(fig)
pairs = (corr.where(~mask).stack().rename("r").reset_index()
         .assign(abs_r=lambda d: d["r"].abs()).sort_values("abs_r", ascending=False))
top_pairs = pairs.head(3)

# 5. Which features separate the cultivars best? (ratio of between-group to within-group variance)
def f_ratio(col):
    groups = [g[col].values for _, g in df.groupby("cultivar")]
    grand = df[col].mean()
    between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (len(groups) - 1)
    within = sum(((g - g.mean()) ** 2).sum() for g in groups) / (len(df) - len(groups))
    return between / within

sep = pd.Series({c: f_ratio(c) for c in features}).sort_values(ascending=False)
top4 = list(sep.index[:4])
fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
for ax, col in zip(axes, top4):
    sns.boxplot(data=df, x="cultivar", y=col, hue="cultivar", palette=PALETTE, legend=False, ax=ax)
    ax.set_title(col.replace("_", " ").title())
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=12)
fig.suptitle("Top 4 features that best separate the cultivars", y=1.02)
fig.tight_layout()
fig.savefig(IMG / "03_top_features_boxplot.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# 6. Mean profile per cultivar for the top features
profile = df.groupby("cultivar")[top4].mean().round(2)

# 7. PCA 2D view
X = StandardScaler().fit_transform(df[features])
pca = PCA(n_components=2).fit(X)
pc = pca.transform(X)
fig, ax = plt.subplots(figsize=(9, 7))
for name, color in PALETTE.items():
    m = (df["cultivar"] == name).values
    ax.scatter(pc[m, 0], pc[m, 1], s=55, alpha=0.8, color=color, label=name, edgecolor="white", linewidth=0.6)
ev = pca.explained_variance_ratio_
ax.set_xlabel(f"PC1 ({ev[0]:.0%} of variance)")
ax.set_ylabel(f"PC2 ({ev[1]:.0%} of variance)")
ax.set_title("Wines projected onto 2 principal components")
ax.legend()
fig.tight_layout()
fig.savefig(IMG / "04_pca_scatter.png", dpi=120)
plt.close(fig)

# 8. Simple classification check
model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=300, random_state=42))
scores = cross_val_score(model, df[features], df["cultivar"], cv=5)
model.fit(df[features], df["cultivar"])
imp = pd.Series(model[-1].feature_importances_, index=features).sort_values()
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(imp.index, imp.values, color="#2a6fdb")
ax.set_title("Random Forest feature importance")
ax.set_xlabel("Importance")
fig.tight_layout()
fig.savefig(IMG / "05_feature_importance.png", dpi=120)
plt.close(fig)

results = {
    "data_quality": quality,
    "class_counts": counts.to_dict(),
    "top_correlations": [{"a": r.level_0, "b": r.level_1, "r": round(r.r, 2)} for r in top_pairs.itertuples()],
    "best_separating_features": sep.round(1).head(4).to_dict(),
    "mean_profile_top_features": profile.to_dict(orient="index"),
    "pca_explained_variance_2d": round(float(ev.sum()), 3),
    "pca_explained_variance": [round(float(v), 3) for v in ev],
    "rf_cv_accuracy_mean": round(float(scores.mean()), 3),
    "rf_cv_accuracy_std": round(float(scores.std()), 3),
    "rf_top3_importance": imp.sort_values(ascending=False).head(3).round(3).to_dict(),
}
(ROOT / "results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
