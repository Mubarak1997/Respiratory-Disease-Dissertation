# This trains and compares all 5 models and saves the best one.
#
# I tried XGBoost like I said I would in the proposal but couldn't get it
# working in my dev environment at the time, so I used GradientBoostingClassifier
# instead since it's basically the same idea (sklearn already has it built in).
#
# Also saves the plots I used in the results chapter and the final model
# as a pickle file so the GUI can load it later.

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix)

RANDOM_STATE = 42
sns.set_theme(style="whitegrid", font_scale=0.9)

# load the data
df = pd.read_csv("respiratory_disease_dataset.csv")
df = df.drop(columns=["patient_id"])  # don't need this for modelling

numeric_features = ["age", "respiratory_rate", "oxygen_saturation", "temperature",
                     "coughing_severity", "fatigue", "chest_pain"]
categorical_features = ["smoking_history"]

X = df.drop(columns=["diagnosis"])
y_raw = df["diagnosis"]

le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_
print("Classes:", list(class_names))

# preprocessing: fill in missing values, scale the numbers, one-hot encode smoking history
preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ]), categorical_features),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain set: {X_train.shape[0]} records | Test set: {X_test.shape[0]} records")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# quick test: what happens if I don't scale the features first?
# (logistic regression is supposed to need this, wanted to check)
X_train_raw = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
X_test_raw = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)
X_test_raw = X_test_raw.reindex(columns=X_train_raw.columns, fill_value=0)
X_train_raw = X_train_raw.fillna(X_train_raw.median(numeric_only=True))
X_test_raw = X_test_raw.fillna(X_train_raw.median(numeric_only=True))

lr_unscaled = LogisticRegression(max_iter=200, random_state=RANDOM_STATE)
lr_unscaled.fit(X_train_raw, y_train)
pred_unscaled = lr_unscaled.predict(X_test_raw)
acc_unscaled = accuracy_score(y_test, pred_unscaled)
print(f"\nLogistic Regression with NO scaling: accuracy = {acc_unscaled:.3f}")
print("did it converge properly?", lr_unscaled.n_iter_ < 200)
# -> it barely worked and threw a convergence warning, confirms I need StandardScaler


# quick test: decision tree with no max_depth set
# wanted to see how badly it overfits if I don't tweak it
dt_pipe_untuned = Pipeline([
    ("prep", preprocessor),
    ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))
])
dt_pipe_untuned.fit(X_train, y_train)
train_acc = accuracy_score(y_train, dt_pipe_untuned.predict(X_train))
test_acc = accuracy_score(y_test, dt_pipe_untuned.predict(X_test))
print(f"\nUn-tuned Decision Tree: train accuracy = {train_acc:.3f}, test accuracy = {test_acc:.3f}")
print("(big gap between train and test = overfitting, need GridSearchCV to fix this)")


# now the real models, each tuned with GridSearchCV
# doing these one at a time instead of in a loop so it's easier to read/debug

results = []
fitted_models = {}

# Logistic Regression
pipe = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])
param_grid = {"clf__C": [0.01, 0.1, 1, 10], "clf__class_weight": [None, "balanced"]}
search = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
search.fit(X_train, y_train)
fitted_models["Logistic Regression"] = search.best_estimator_
pred = search.best_estimator_.predict(X_test)
results.append({
    "Model": "Logistic Regression", "Best Params": search.best_params_,
    "CV F1 (macro)": search.best_score_,
    "Test Accuracy": accuracy_score(y_test, pred),
    "Test Precision (macro)": precision_score(y_test, pred, average="macro"),
    "Test Recall (macro)": recall_score(y_test, pred, average="macro"),
    "Test F1 (macro)": f1_score(y_test, pred, average="macro"),
})
print(f"\nLogistic Regression best params: {search.best_params_}")
print(f"  CV F1 = {search.best_score_:.3f} | Test acc = {results[-1]['Test Accuracy']:.3f}")

# Naive Bayes
pipe = Pipeline([("prep", preprocessor), ("clf", GaussianNB())])
param_grid = {"clf__var_smoothing": [1e-9, 1e-8, 1e-7]}
search = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
search.fit(X_train, y_train)
fitted_models["Naive Bayes"] = search.best_estimator_
pred = search.best_estimator_.predict(X_test)
results.append({
    "Model": "Naive Bayes", "Best Params": search.best_params_,
    "CV F1 (macro)": search.best_score_,
    "Test Accuracy": accuracy_score(y_test, pred),
    "Test Precision (macro)": precision_score(y_test, pred, average="macro"),
    "Test Recall (macro)": recall_score(y_test, pred, average="macro"),
    "Test F1 (macro)": f1_score(y_test, pred, average="macro"),
})
print(f"\nNaive Bayes best params: {search.best_params_}")
print(f"  CV F1 = {search.best_score_:.3f} | Test acc = {results[-1]['Test Accuracy']:.3f}")

# Decision Tree
pipe = Pipeline([("prep", preprocessor), ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))])
param_grid = {
    "clf__max_depth": [3, 4, 5, 6, 8, None],
    "clf__min_samples_leaf": [1, 3, 5],
    "clf__class_weight": [None, "balanced"],
}
search = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
search.fit(X_train, y_train)
fitted_models["Decision Tree"] = search.best_estimator_
pred = search.best_estimator_.predict(X_test)
results.append({
    "Model": "Decision Tree", "Best Params": search.best_params_,
    "CV F1 (macro)": search.best_score_,
    "Test Accuracy": accuracy_score(y_test, pred),
    "Test Precision (macro)": precision_score(y_test, pred, average="macro"),
    "Test Recall (macro)": recall_score(y_test, pred, average="macro"),
    "Test F1 (macro)": f1_score(y_test, pred, average="macro"),
})
print(f"\nDecision Tree best params: {search.best_params_}")
print(f"  CV F1 = {search.best_score_:.3f} | Test acc = {results[-1]['Test Accuracy']:.3f}")

# Random Forest
pipe = Pipeline([("prep", preprocessor), ("clf", RandomForestClassifier(random_state=RANDOM_STATE))])
param_grid = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [None, 6, 10],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__class_weight": [None, "balanced"],
}
search = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
search.fit(X_train, y_train)
fitted_models["Random Forest"] = search.best_estimator_
pred = search.best_estimator_.predict(X_test)
results.append({
    "Model": "Random Forest", "Best Params": search.best_params_,
    "CV F1 (macro)": search.best_score_,
    "Test Accuracy": accuracy_score(y_test, pred),
    "Test Precision (macro)": precision_score(y_test, pred, average="macro"),
    "Test Recall (macro)": recall_score(y_test, pred, average="macro"),
    "Test F1 (macro)": f1_score(y_test, pred, average="macro"),
})
print(f"\nRandom Forest best params: {search.best_params_}")
print(f"  CV F1 = {search.best_score_:.3f} | Test acc = {results[-1]['Test Accuracy']:.3f}")

# Gradient Boosting (this is my XGBoost substitute)
pipe = Pipeline([("prep", preprocessor), ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))])
param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__learning_rate": [0.05, 0.1],
    "clf__max_depth": [2, 3, 4],
}
search = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
search.fit(X_train, y_train)
fitted_models["Gradient Boosting (XGBoost substitute)"] = search.best_estimator_
pred = search.best_estimator_.predict(X_test)
results.append({
    "Model": "Gradient Boosting (XGBoost substitute)", "Best Params": search.best_params_,
    "CV F1 (macro)": search.best_score_,
    "Test Accuracy": accuracy_score(y_test, pred),
    "Test Precision (macro)": precision_score(y_test, pred, average="macro"),
    "Test Recall (macro)": recall_score(y_test, pred, average="macro"),
    "Test F1 (macro)": f1_score(y_test, pred, average="macro"),
})
print(f"\nGradient Boosting best params: {search.best_params_}")
print(f"  CV F1 = {search.best_score_:.3f} | Test acc = {results[-1]['Test Accuracy']:.3f}")


# put it all together and see which model won
results_df = pd.DataFrame(results).sort_values("Test F1 (macro)", ascending=False)
results_df.to_csv("model_results_summary.csv", index=False)
print("\n=== FINAL MODEL COMPARISON (sorted by Test F1 macro) ===")
print(results_df[["Model", "Test Accuracy", "Test Precision (macro)",
                   "Test Recall (macro)", "Test F1 (macro)"]].to_string(index=False))

best_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_name]
print(f"\nBest performing model: {best_name}")
print("\nClassification report for best model:\n",
      classification_report(y_test, best_model.predict(X_test), target_names=class_names))


# plot 1: bar chart comparing all 5 models
plt.figure(figsize=(8, 4.5))
plot_df = results_df.sort_values("Test F1 (macro)")
sns.barplot(data=plot_df, y="Model", x="Test F1 (macro)", hue="Model", palette="viridis", legend=False)
plt.title("Model comparison - macro F1-score on held-out test set")
plt.xlabel("Macro F1-score")
plt.ylabel("")
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig("fig_model_comparison.png", dpi=150)
plt.close()

# plot 2: confusion matrix for every model
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for i, (name, model) in enumerate(fitted_models.items()):
    cm = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=axes[i])
    axes[i].set_title(name, fontsize=10)
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")
    axes[i].tick_params(axis='x', rotation=30)
    axes[i].tick_params(axis='y', rotation=0)
for j in range(len(fitted_models), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig("fig_confusion_matrices.png", dpi=150)
plt.close()

# plot 3: feature importance (use Random Forest if best model doesn't have this)
if hasattr(best_model.named_steps["clf"], "feature_importances_"):
    importance_source = best_model
    importance_model_name = best_name
else:
    importance_source = fitted_models["Random Forest"]
    importance_model_name = "Random Forest"

clf_step = importance_source.named_steps["clf"]
cat_names = list(importance_source.named_steps["prep"]
                  .named_transformers_["cat"].named_steps["onehot"]
                  .get_feature_names_out(categorical_features))
feature_names = numeric_features + cat_names
importances = pd.Series(clf_step.feature_importances_, index=feature_names).sort_values()

plt.figure(figsize=(7, 5))
importances.plot(kind="barh", color=sns.color_palette("viridis", len(importances)))
plt.title(f"Feature importance ({importance_model_name})")
plt.xlabel("Relative importance")
plt.tight_layout()
plt.savefig("fig_feature_importance.png", dpi=150)
plt.close()

# save the winning model so the GUI (app.py) can load it later
artifact = {
    "model": best_model,
    "label_encoder": le,
    "class_names": list(class_names),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "model_name": best_name,
}
with open("respiratory_disease_model.pkl", "wb") as f:
    pickle.dump(artifact, f)

print("\nSaved final model artifact -> respiratory_disease_model.pkl")
print("Saved comparison table -> model_results_summary.csv")