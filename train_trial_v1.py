import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("respiratory_dataset_v1_trial.csv")
df = pd.get_dummies(df, columns=["smoking_history"], drop_first=True)

X = df.drop(columns=["diagnosis"])
y = LabelEncoder().fit_transform(df["diagnosis"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)

print("Trial v1 Random Forest accuracy:", accuracy_score(y_test, pred))
print("Trial v1 Random Forest macro F1:", f1_score(y_test, pred, average="macro"))
