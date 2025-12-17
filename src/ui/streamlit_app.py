import streamlit as st
import requests
import os

# ==========================================================
# CONFIG
# ==========================================================
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="📞 Churn Prediction Dashboard",
    layout="wide",
    page_icon="📊",
)

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("Interface de prédiction et de réentraînement du modèle.")

mode = st.sidebar.radio(
    "Choisissez une action :",
    ["🔮 Prédiction", "🔧 Réentraînement"],
)

st.sidebar.markdown("---")
st.sidebar.caption("MLOps Project – Churn Prediction")

# ==========================================================
# MAIN TITLE
# ==========================================================
st.title("📞 Customer Churn Prediction")
st.markdown(
    """
    Cette application permet :
    - 🔮 **Prédire le churn d’un client**
    - 🔧 **Réentraîner le modèle via l’API**
    """
)

# ==========================================================
# 1️⃣ PREDICTION MODE
# ==========================================================
if mode == "🔮 Prédiction":

    st.header("🔮 Prédiction du churn client")

    with st.form("prediction_form"):

        st.subheader("📋 Informations client")

        col1, col2, col3 = st.columns(3)

        with col1:
            total_charge = st.number_input("💰 Total charge ($)", min_value=0.0, value=110.0)
            intl_charge = st.number_input("🌍 International charge ($)", min_value=0.0, value=2.7)
            intl_calls = st.number_input("📞 International calls", min_value=0, value=3)

        with col2:
            cust_calls = st.number_input("☎️ Customer service calls", min_value=0, value=2)
            cs_rate = st.number_input("📊 CS calls rate", min_value=0.0, value=0.01)
            vmail = st.number_input("📨 Voice mail messages", min_value=0, value=5)

        with col3:
            international_plan = st.selectbox(
                "🌐 International plan",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )

            area_code = st.selectbox(
                "📍 Area code",
                ["408", "415", "510"],
            )

            state = st.selectbox(
                "🏙️ State",
                ["TX", "SC", "MT", "IL"],
            )

        submitted = st.form_submit_button("🚀 Predict churn")

    if submitted:

        # Encode categorical fields
        features = {
            "Total charge": total_charge,
            "Customer service calls": cust_calls,
            "Total intl calls": intl_calls,
            "Total intl charge": intl_charge,
            "International plan": international_plan,
            "Number vmail messages": vmail,
            "CScalls Rate": cs_rate,
            "Area code_408": int(area_code == "408"),
            "Area code_415": int(area_code == "415"),
            "Area code_510": int(area_code == "510"),
            "State_TX": int(state == "TX"),
            "State_SC": int(state == "SC"),
            "State_MT": int(state == "MT"),
            "State_IL": int(state == "IL"),
        }

        with st.spinner("🔄 Calling FastAPI..."):
            try:
                resp = requests.post(f"{API_URL}/predict", json=features, timeout=10)
                resp.raise_for_status()
                result = resp.json()

                churn_prob = result["churn_probability"]

                st.success("✔ Prediction successful")

                st.metric(
                    label="📊 Churn probability",
                    value=f"{churn_prob:.2%}",
                )

                st.progress(min(churn_prob, 1.0))

                if churn_prob > 0.5:
                    st.error("⚠️ High risk of churn")
                else:
                    st.success("✅ Low churn risk")

            except Exception as e:
                st.error("❌ API error")
                st.exception(e)

# ==========================================================
# 2️⃣ RETRAIN MODE
# ==========================================================
if mode == "🔧 Réentraînement":

    st.header("🔧 Réentraîner le modèle")

    st.warning(
        "⚠️ Cette action déclenche un entraînement complet "
        "et enregistre une nouvelle version du modèle."
    )

    with st.form("retrain_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            n_estimators = st.slider("n_estimators", 50, 500, 150)

        with col2:
            max_depth = st.slider("max_depth", 2, 12, 6)

        with col3:
            learning_rate = st.number_input(
                "learning_rate", min_value=0.001, max_value=1.0, value=0.05
            )

        submitted_retrain = st.form_submit_button("🔁 Retrain model")

    if submitted_retrain:
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
        }

        with st.spinner("🧠 Training model..."):
            try:
                resp = requests.post(f"{API_URL}/retrain", json=params, timeout=30)
                resp.raise_for_status()

                st.success("🎉 Model retrained successfully")
                st.json(resp.json())

            except Exception as e:
                st.error("❌ Retraining failed")
                st.exception(e)
