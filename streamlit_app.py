import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Churn Predictor UI", layout="wide")

st.title("📞 Churn Prediction Dashboard")
st.write("Interface Streamlit pour interagir avec votre API FastAPI.")

# ==========================================================
#                1️⃣  PREDICTION SECTION
# ==========================================================
st.header("🔮 Faire une prédiction")

with st.form("prediction_form"):
    st.subheader("Données d'entrée")

    total_charge = st.number_input("Total charge", min_value=0.0, value=110.0)
    area_415 = st.number_input("Area code 415", min_value=0, max_value=1, value=0)
    area_408 = st.number_input("Area code 408", min_value=0, max_value=1, value=1)
    area_510 = st.number_input("Area code 510", min_value=0, max_value=1, value=0)

    cust_calls = st.number_input("Customer service calls", min_value=0, value=2)
    intl_calls = st.number_input("Total intl calls", min_value=0, value=3)

    international_plan = st.selectbox("International plan", [0, 1], index=0)

    vmail = st.number_input("Number vmail messages", min_value=0, value=5)
    
    state_sc = st.number_input("State_SC", min_value=0, max_value=1, value=0)
    state_tx = st.number_input("State_TX", min_value=0, max_value=1, value=1)
    state_mt = st.number_input("State_MT", min_value=0, max_value=1, value=0)
    state_il = st.number_input("State_IL", min_value=0, max_value=1, value=0)

    intl_charge = st.number_input("Total intl charge", min_value=0.0, value=2.7)
    cs_rate = st.number_input("CScalls Rate", min_value=0.0, value=0.01)

    submitted = st.form_submit_button("Predict")

    if submitted:
        features = {
            "Total charge": total_charge,
            "Area code_415": area_415,
            "Area code_408": area_408,
            "Customer service calls": cust_calls,
            "Area code_510": area_510,
            "Total intl calls": intl_calls,
            "International plan": international_plan,
            "Number vmail messages": vmail,
            "State_SC": state_sc,
            "State_TX": state_tx,
            "State_MT": state_mt,
            "Total intl charge": intl_charge,
            "State_IL": state_il,
            "CScalls Rate": cs_rate,
        }

        resp = requests.post(f"{API_URL}/predict", json=features)

        if resp.status_code == 200:
            result = resp.json()
            st.info(f"Churn probability: {result['churn_probability']:.4f}")
        else:
            st.error("Erreur lors de la requête à l'API.")


# ==========================================================
#                2️⃣  RETRAIN SECTION
# ==========================================================
st.header("🔧 Réentraîner le modèle")

with st.form("retrain_form"):
    st.subheader("Choisir les hyperparamètres")

    n_estimators = st.slider("n_estimators", 50, 500, 150)
    max_depth = st.slider("max_depth", 2, 12, 4)
    learning_rate = st.number_input("learning_rate", min_value=0.001, max_value=1.0, value=0.05)

    submitted_retrain = st.form_submit_button("Retrain Model")

    if submitted_retrain:
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate
        }

        resp = requests.post(f"{API_URL}/retrain", json=params)

        if resp.status_code == 200:
            st.success("🎉 Modèle réentraîné avec succès !")
            st.json(resp.json())
        else:
            st.error("Erreur pendant le réentraînement du modèle.")
