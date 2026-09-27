# GUI for the respiratory disease model, built with Streamlit.
# Loads the model I saved in train_models_final.py and lets you fill in
# a patient's vitals/symptoms to get a predicted diagnosis.
#
# obviously this is just a demo for my dissertation, trained on made up
# data - not something anyone should actually use to diagnose someone.
#
# to run: streamlit run app.py

import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Respiratory Disease Detection Tool", page_icon="🫁", layout="centered")


@st.cache_resource
def load_model():
    with open("respiratory_disease_model.pkl", "rb") as f:
        return pickle.load(f)


artifact = load_model()
model = artifact["model"]
class_names = artifact["class_names"]
model_name = artifact["model_name"]

st.title("Respiratory Disease Detection Tool")
st.caption(f"Dissertation prototype | Best-performing model: {model_name}")

st.warning(
    "**This is an academic decision support prototype trained using synthetic data.** "
    "It is not a validated diagnostic tool and must not replace professional clinical judgement."
)

# all the patient info goes in the sidebar
with st.sidebar:
    st.header("Patient Observations")
    age = st.slider("Age (years)", 10, 95, 50)
    respiratory_rate = st.slider("Respiratory rate (breaths/min)", 10, 40, 18)
    oxygen_saturation = st.slider("Oxygen saturation - SpO2 (%)", 80, 100, 97)
    temperature = st.slider("Body temperature (C)", 35.0, 40.5, 36.8, step=0.1)
    smoking_history = st.selectbox("Smoking history", ["Never", "Former", "Current"])
    coughing_severity = st.select_slider("Coughing severity", options=[0, 1, 2, 3],
                                          format_func=lambda x: ["None", "Mild", "Moderate", "Severe"][x])
    fatigue = st.select_slider("Fatigue", options=[0, 1, 2, 3],
                                format_func=lambda x: ["None", "Mild", "Moderate", "Severe"][x])
    chest_pain = st.select_slider("Chest pain", options=[0, 1, 2, 3],
                                   format_func=lambda x: ["None", "Mild", "Moderate", "Severe"][x])
    predict_clicked = st.button("Predict diagnosis", type="primary", use_container_width=True)

if predict_clicked:
    # put the inputs into a dataframe with the same column names the model was trained on
    input_df = pd.DataFrame([{
        "age": age,
        "respiratory_rate": respiratory_rate,
        "oxygen_saturation": oxygen_saturation,
        "temperature": temperature,
        "smoking_history": smoking_history,
        "coughing_severity": coughing_severity,
        "fatigue": fatigue,
        "chest_pain": chest_pain,
    }])

    proba = model.predict_proba(input_df)[0]
    pred_idx = proba.argmax()
    pred_label = class_names[pred_idx]
    confidence = proba[pred_idx]

    st.subheader("Prediction Result")
    st.metric("Most likely classification", pred_label, f"{confidence:.1%} confidence")

    results_df = pd.DataFrame({
        "Diagnosis": class_names,
        "Predicted probability": proba,
    }).sort_values("Predicted probability", ascending=False)

    st.bar_chart(results_df.set_index("Diagnosis"))
    st.dataframe(
        results_df.style.format({"Predicted probability": "{:.1%}"}),
        hide_index=True, use_container_width=True,
    )

    st.info(
        "This result is based on patterns learned from a synthetic medically-informed "
        "dataset. It is for decision support only and should always be considered "
        "alongside professional clinical assessment."
    )
else:
    st.info("Set the patient's observations in the sidebar, then click **Predict diagnosis**.")

st.divider()
st.caption(
    "Source: Machine Learning Respiratory Disease Detection Using Synthetic Data, 2026."
)
