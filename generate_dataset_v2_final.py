# Generates the synthetic respiratory disease dataset.
# 1500 fake patients split into 5 groups: Healthy, Asthma, Bronchitis, COPD, Pneumonia.
#
# The numbers I used for each group (average respiratory rate, oxygen level etc.)
# are based on NHS NEWS2 ranges and NICE guidance for these conditions, not
# just made up randomly.
#
# This is the second version of this script. My first attempt (generate_dataset_v1_trial.py)
# used fixed ranges that didn't overlap between classes at all, which made the models
# get 100% accuracy - way too easy and not realistic. So here I use normal distributions
# instead, which means classes can overlap a bit like real symptoms would.
#
# Originally this dataset was only 500 rows. I wasn't sure that was enough for the
# models to learn from properly, so I bumped it up to 1500 - just multiplied all the
# class sizes by 3 and kept everything else the same.

import numpy as np
import pandas as pd

np.random.seed(7)

TOTAL = 1500

# same proportions as the original 500-row version, just x3
CLASS_SIZES = {
    "Healthy": 390,
    "Asthma": 270,
    "Bronchitis": 285,
    "COPD": 285,
    "Pneumonia": 270,
}
assert sum(CLASS_SIZES.values()) == TOTAL


def clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def gen_smoking_history(n, p_never, p_former, p_current):
    return np.random.choice(["Never", "Former", "Current"], size=n, p=[p_never, p_former, p_current])


# makes number of patients for one class - normal distributions for the vitals,
# fixed probabilities for the symptom scores
def make_class(label, n, age_mu, age_sd, age_bounds,
               rr_mu, rr_sd, spo2_mu, spo2_sd, temp_mu, temp_sd,
               cough_probs, fatigue_probs, chestpain_probs,
               smoking_probs):

    age = clip(np.random.normal(age_mu, age_sd, n), *age_bounds).round().astype(int)
    rr = clip(np.random.normal(rr_mu, rr_sd, n), 10, 40).round().astype(int)
    spo2 = clip(np.random.normal(spo2_mu, spo2_sd, n), 80, 100).round().astype(int)
    temp = clip(np.random.normal(temp_mu, temp_sd, n), 35.0, 40.5).round(1)
    cough = np.random.choice([0, 1, 2, 3], size=n, p=cough_probs)
    fatigue = np.random.choice([0, 1, 2, 3], size=n, p=fatigue_probs)
    chest_pain = np.random.choice([0, 1, 2, 3], size=n, p=chestpain_probs)
    smoking = gen_smoking_history(n, *smoking_probs)

    return pd.DataFrame({
        "age": age,
        "respiratory_rate": rr,
        "oxygen_saturation": spo2,
        "temperature": temp,
        "smoking_history": smoking,
        "coughing_severity": cough,
        "fatigue": fatigue,
        "chest_pain": chest_pain,
        "diagnosis": label,
    })


# Healthy - normal vitals, barely any symptoms
healthy = make_class(
    "Healthy", CLASS_SIZES["Healthy"],
    age_mu=42, age_sd=16, age_bounds=(18, 90),
    rr_mu=15, rr_sd=2.0, spo2_mu=97.5, spo2_sd=1.2, temp_mu=36.6, temp_sd=0.3,
    cough_probs=[0.75, 0.20, 0.04, 0.01], fatigue_probs=[0.70, 0.22, 0.06, 0.02],
    chestpain_probs=[0.85, 0.12, 0.02, 0.01], smoking_probs=(0.65, 0.25, 0.10),
)

# Asthma - breathless, slightly low O2, wheezy cough, no fever, tends to be younger
asthma = make_class(
    "Asthma", CLASS_SIZES["Asthma"],
    age_mu=34, age_sd=15, age_bounds=(10, 75),
    rr_mu=22, rr_sd=3.2, spo2_mu=94.3, spo2_sd=2.2, temp_mu=36.8, temp_sd=0.4,
    cough_probs=[0.10, 0.30, 0.40, 0.20], fatigue_probs=[0.15, 0.35, 0.35, 0.15],
    chestpain_probs=[0.25, 0.40, 0.25, 0.10], smoking_probs=(0.55, 0.30, 0.15),
)

# Bronchitis - the main thing is a really bad productive cough, mild fever, smokers
bronchitis = make_class(
    "Bronchitis", CLASS_SIZES["Bronchitis"],
    age_mu=48, age_sd=17, age_bounds=(20, 85),
    rr_mu=19.5, rr_sd=2.5, spo2_mu=95.2, spo2_sd=1.8, temp_mu=37.4, temp_sd=0.5,
    cough_probs=[0.03, 0.12, 0.35, 0.50], fatigue_probs=[0.20, 0.35, 0.30, 0.15],
    chestpain_probs=[0.55, 0.30, 0.12, 0.03], smoking_probs=(0.30, 0.35, 0.35),
)

# COPD - older patients, low O2 (BTS says target 88-92% for these patients), chronic cough, mostly smokers or ex-smokers
copd = make_class(
    "COPD", CLASS_SIZES["COPD"],
    age_mu=66, age_sd=10, age_bounds=(45, 90),
    rr_mu=23, rr_sd=3.0, spo2_mu=90.0, spo2_sd=2.6, temp_mu=36.9, temp_sd=0.5,
    cough_probs=[0.05, 0.15, 0.40, 0.40], fatigue_probs=[0.05, 0.20, 0.40, 0.35],
    chestpain_probs=[0.55, 0.30, 0.12, 0.03], smoking_probs=(0.10, 0.45, 0.45),
)

# Pneumonia - acute and severe: high fever, fast breathing, low O2, chest pain
pneumonia = make_class(
    "Pneumonia", CLASS_SIZES["Pneumonia"],
    age_mu=58, age_sd=20, age_bounds=(18, 95),
    rr_mu=27, rr_sd=3.5, spo2_mu=91.0, spo2_sd=2.8, temp_mu=38.6, temp_sd=0.7,
    cough_probs=[0.05, 0.15, 0.35, 0.45], fatigue_probs=[0.05, 0.15, 0.35, 0.45],
    chestpain_probs=[0.10, 0.25, 0.35, 0.30], smoking_probs=(0.40, 0.30, 0.30),
)

# stick all 5 groups together and shuffle so they're not in class order
df = pd.concat([healthy, asthma, bronchitis, copd, pneumonia], ignore_index=True)
df = df.sample(frac=1, random_state=7).reset_index(drop=True)
df.insert(0, "patient_id", [f"P{str(i+1).zfill(4)}" for i in range(len(df))])

# add a bit of missing data (~3%) in SpO2 and temperature so the
# missing-value handling step actually has something to do
rng = np.random.default_rng(7)
for col in ["temperature", "oxygen_saturation"]:
    missing_idx = rng.choice(df.index, size=int(0.03 * len(df)), replace=False)
    df.loc[missing_idx, col] = np.nan

df.to_csv("respiratory_disease_dataset.csv", index=False)

print("Final dataset shape:", df.shape)
print(df["diagnosis"].value_counts())
print("\nMissing values per column:\n", df.isna().sum())
print("\nSample rows:\n", df.head())
