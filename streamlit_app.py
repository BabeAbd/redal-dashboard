import streamlit as st
st.set_page_config(page_title="Tableau de bord Redal AI", layout="wide")

import pandas as pd
import numpy as np
import plotly.express as px

# Chargement des données prédictives
@st.cache_data
def load_data():
    df = pd.read_csv("df_prediction.csv")
    df["mois"] = df["mois"].astype(str)
    df["consommation_reelle"] = np.exp(df["consommation_log"])
    return df

df = load_data()

st.title("💧 Dashboard IA – Prédiction de la consommation d'eau")
st.markdown("Ce tableau de bord montre les prédictions basées sur un modèle panel IA appliqué à la consommation d’eau.")

# Section performance du modèle
st.header("📈 Performance du modèle IA")
df["erreur"] = df["predicted_conso"] - df["consommation_reelle"]
df["erreur_abs"] = abs(df["erreur"])
df["erreur_sq"] = df["erreur"]**2
df["erreur_pct"] = df["erreur_abs"] / (df["consommation_reelle"] + 1e-6)

mae = df["erreur_abs"].mean()
mse = df["erreur_sq"].mean()
rmse = np.sqrt(mse)
mape = df["erreur_pct"].mean() * 100
sst = ((df["consommation_reelle"] - df["consommation_reelle"].mean())**2).sum()
sse = (df["erreur"]**2).sum()
r2 = 1 - sse / sst

st.metric("MAE (Erreur absolue moyenne)", f"{mae:.2f} m³")
st.metric("RMSE (Erreur quadratique moyenne)", f"{rmse:.2f} m³")
st.metric("MAPE (Erreur relative moyenne)", f"{mape:.2f}%")
st.metric("R² (Variance expliquée)", f"{r2:.4f}")

# Visualisation globale
st.header("📊 Consommation totale réelle vs prédite")
df_mois = df.groupby("mois").agg({
    "consommation_reelle": "sum",
    "predicted_conso": "sum"
}).reset_index()

df_melted = df_mois.melt(id_vars="mois", value_vars=["consommation_reelle", "predicted_conso"],
                         var_name="Type", value_name="Consommation")

fig = px.line(df_melted, x="mois", y="Consommation", color="Type",
              title="Consommation totale par mois – Réel vs Prédit")
st.plotly_chart(fig, use_container_width=True)

# Prédiction individuelle
st.header("🔍 Consommation par client")
selected_client = st.selectbox("Choisir un client :", sorted(df["CIL"].unique()))
df_client = df[df["CIL"] == selected_client].sort_values("mois")

fig2 = px.line(df_client.melt(id_vars="mois", value_vars=["consommation_reelle", "predicted_conso"],
                              var_name="Type", value_name="Consommation"),
               x="mois", y="Consommation", color="Type",
               title=f"Client {selected_client} – Consommation réelle vs prédite")
st.plotly_chart(fig2, use_container_width=True)

# Seuil d'alerte
st.header("⚠️ Alerte de seuil")
seuil = st.slider("Fixer un seuil (m³/mois)", 5, 100, 20)
alertes = df[df["predicted_conso"] > seuil]
nb_alertes = alertes["CIL"].nunique()

st.warning(f"{nb_alertes} client(s) prédisent dépasser le seuil de {seuil} m³.")
st.dataframe(alertes[["CIL", "mois", "predicted_conso"]])

# Simulation personnalisée de prédiction (scénarios multiples)
st.header("🧠 Simulation de consommation totale – Tester plusieurs scénarios")

n = st.number_input("Nombre de scénarios à tester", min_value=1, max_value=10, value=1)

# Coefficients du modèle
coefs = {
    "z_lag_conso_log": 0.2942309,
    "z_log_temperature": 0.0125661,
    "z_log_precipitation": 0.0037697,
    "z_log_humidity": 0.0027898,
    "intercept": 0
}

nb_clients = df["CIL"].nunique()
scenarios = []

for i in range(int(n)):
    st.subheader(f"Scénario {i+1}")
    lag_input = st.slider(f"Lag conso (log) - scénario {i+1}", 0.0, 5.0, 2.0, 0.1, key=f"lag_{i}")
    temp_input = st.slider(f"Température (°C) - scénario {i+1}", 1.0, 40.0, 20.0, 0.5, key=f"temp_{i}")
    prec_input = st.slider(f"Précipitation (mm) - scénario {i+1}", 0.0, 50.0, 5.0, 0.5, key=f"prec_{i}")
    hum_input = st.slider(f"Humidité (%) - scénario {i+1}", 10.0, 100.0, 60.0, 1.0, key=f"hum_{i}")

    # Normalisation simple (à adapter selon dataset réel)
    lag_std = (lag_input - 2.5) / 0.5
    temp_std = (np.log(temp_input + 1) - 3.0) / 0.3
    prec_std = (np.log(prec_input + 1) - 2.0) / 0.5
    hum_std = (np.log(hum_input + 1) - 4.0) / 0.4

    prediction_log = coefs["z_lag_conso_log"] * lag_std + \
                     coefs["z_log_temperature"] * temp_std + \
                     coefs["z_log_precipitation"] * prec_std + \
                     coefs["z_log_humidity"] * hum_std

    prediction_indiv = np.exp(prediction_log)
    prediction_total = prediction_indiv * nb_clients

    st.success(f"🔮 Consommation totale prédite (scénario {i+1}) : {prediction_total:.2f} m³")
    scenarios.append({
        "Scénario": f"Scénario {i+1}",
        "Consommation totale prédite (m³)": round(prediction_total, 2)
    })

if scenarios:
    st.dataframe(pd.DataFrame(scenarios))
