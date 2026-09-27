import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=0.95)

df = pd.read_csv("respiratory_disease_dataset.csv")

# Class distribution
plt.figure(figsize=(6, 4))
order = df["diagnosis"].value_counts().index
sns.countplot(data=df, y="diagnosis", order=order, hue="diagnosis",
              palette="viridis", legend=False)
plt.title(f"Class distribution across the {len(df)} synthetic records")
plt.xlabel("Number of patients")
plt.ylabel("")
plt.tight_layout()
plt.savefig("fig_class_distribution.png", dpi=150)
plt.close()

# Feature distributions by class (respiratory rate, SpO2)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
sns.boxplot(data=df, x="diagnosis", y="respiratory_rate", hue="diagnosis",
            order=order, palette="viridis", ax=axes[0], legend=False)
axes[0].set_title("Respiratory rate by diagnosis")
axes[0].set_xlabel("")
axes[0].tick_params(axis='x', rotation=30)

sns.boxplot(data=df, x="diagnosis", y="oxygen_saturation", hue="diagnosis",
            order=order, palette="viridis", ax=axes[1], legend=False)
axes[1].set_title("Oxygen saturation (SpO2) by diagnosis")
axes[1].set_xlabel("")
axes[1].tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig("fig_rr_spo2_boxplots.png", dpi=150)
plt.close()

# Correlation matrix (numeric features)
numeric_cols = ["age", "respiratory_rate", "oxygen_saturation", "temperature",
                 "coughing_severity", "fatigue", "chest_pain"]
corr = df[numeric_cols].corr()
plt.figure(figsize=(6.5, 5.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
            square=True, cbar_kws={"shrink": 0.8})
plt.title("Correlation matrix of numeric medical indicators")
plt.tight_layout()
plt.savefig("fig_correlation_matrix.png", dpi=150)
plt.close()

# Scatter: RR vs SpO2 coloured by diagnosis
plt.figure(figsize=(6.5, 5))
sns.scatterplot(data=df, x="respiratory_rate", y="oxygen_saturation",
                 hue="diagnosis", palette="viridis", alpha=0.75, s=40)
plt.title("Respiratory rate vs oxygen saturation by diagnosis")
plt.xlabel("Respiratory rate (breaths/min)")
plt.ylabel("Oxygen saturation (%)")
plt.legend(title="Diagnosis", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("fig_rr_vs_spo2_scatter.png", dpi=150)
plt.close()

# Smoking history vs diagnosis
plt.figure(figsize=(7, 4.5))
ct = pd.crosstab(df["diagnosis"], df["smoking_history"], normalize="index")
ct = ct.loc[order]
ct.plot(kind="bar", stacked=True, colormap="viridis", ax=plt.gca())
plt.title("Smoking history proportion by diagnosis")
plt.ylabel("Proportion of patients")
plt.xlabel("")
plt.xticks(rotation=30)
plt.legend(title="Smoking history", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("fig_smoking_by_diagnosis.png", dpi=150)
plt.close()

print("Descriptive statistics:\n", df[numeric_cols].describe().T)
print("\nEDA figures saved.")
