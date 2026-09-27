# first attempt at the dataset - used simple ranges that don't overlap
# between classes. turned out to be way too easy (100% accuracy when I
# tested it in train_trial_v1.py), so I rebuilt it properly in
# generate_dataset_v2_final.py with normal distributions instead

import numpy as np
import pandas as pd

np.random.seed(42)

N_PER_CLASS = 100
CLASSES = ["Healthy", "Asthma", "Pneumonia", "Bronchitis", "COPD"]

def make_class(label, rr_range, spo2_range, temp_range, cough_range,
               fatigue_range, chest_pain_range, age_range, n=N_PER_CLASS):
    return pd.DataFrame({
        "age": np.random.randint(age_range[0], age_range[1], n),
        "respiratory_rate": np.random.randint(rr_range[0], rr_range[1], n),
        "oxygen_saturation": np.random.randint(spo2_range[0], spo2_range[1], n),
        "temperature": np.round(np.random.uniform(temp_range[0], temp_range[1], n), 1),
        "smoking_history": np.random.choice(["Never", "Former", "Current"], n),
        "coughing_severity": np.random.randint(cough_range[0], cough_range[1], n),
        "fatigue": np.random.randint(fatigue_range[0], fatigue_range[1], n),
        "chest_pain": np.random.randint(chest_pain_range[0], chest_pain_range[1], n),
        "diagnosis": label
    })

# Deliberately non-overlapping ranges per class (naive first attempt)
healthy    = make_class("Healthy",    (12, 18), (97, 100), (36.4, 37.0), (0, 1), (0, 1), (0, 1), (18, 80))
asthma     = make_class("Asthma",     (20, 24), (93, 96),  (36.5, 37.2), (2, 3), (1, 2), (1, 2), (10, 60))
pneumonia  = make_class("Pneumonia",  (26, 30), (88, 91),  (38.2, 39.5), (3, 4), (2, 3), (2, 3), (40, 90))
bronchitis = make_class("Bronchitis", (18, 21), (94, 96),  (37.3, 38.0), (3, 4), (1, 2), (0, 1), (30, 75))
copd       = make_class("COPD",       (22, 26), (86, 90),  (36.6, 37.3), (2, 3), (2, 3), (0, 1), (55, 90))

df = pd.concat([healthy, asthma, pneumonia, bronchitis, copd], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("respiratory_dataset_v1_trial.csv", index=False)
print(df["diagnosis"].value_counts())
print(df.head())