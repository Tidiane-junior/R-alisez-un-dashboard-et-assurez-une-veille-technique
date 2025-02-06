import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib  # Pour charger le modèle
import plotly.graph_objects as go
import plotly.express as px
import shap

# Configuration de la page
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Accueil scoring - crédit",
)

# --- Initialisation de SessionState ---
if "load_state" not in st.session_state:
    st.session_state.load_state = False

# --- Layout de la page d'accueil ---
st.title("Dashbord - Scoring Crédit")
st.subheader("Outils d'aide à la décision d'octroi de crédit")
st.write("""
    Cette application assiste l'agent de prêt dans sa décision d'accorder un prêt à un client.
    Pour ce faire, un algorithme d'apprentissage automatique est utilisé pour prédire les risques d'un client à rembourser le prêt.
    Pour plus de transparence, cette application fournit également des informations pour expliquer l'algorithme et les prédictions.
         """)

st.write(" ")
st.image("image/logo.png")  # Logo de l'application
st.write(" ")
st.subheader("**Prédiction des résultats du client :**")

# --- Sidebar pour informations supplémentaires ---
st.sidebar.title("À propos du tableau de bord")
st.sidebar.write(
    """
    **Auteur :** [Ch. Tidiane THIAM](https://www.linkedin.com/in/cheikh-tid-thiam)
    """
)
st.sidebar.write(
    """
    Ce tableau de bord de scoring permet :
    - De prédire le risque de remboursement en entrant l'ID du client.
    - Une barre bleue dynamique en fonction du **crédit score**.
    - D'avoir la probabilité, le seuil optimal ainsi que l'écart.
    - D'afficher les caractéristiques qui ont pesé sur le modèle.
    - D'avoir les caractéristiques ayant joué sur la décision.
    - Visualiser la tranche d'âge du client.
    """
)

# --- Chargement des données et modèle ---
@st.cache_data
def load_data_and_model():
    df = pd.read_pickle("train_red_format.pkl")  # Chemin vers vos données
    model = joblib.load("model.pkl")  # Chargement du modèle
    return df, model

df, model = load_data_and_model()

# --- Saisie de l'utilisateur pour SK_ID_CURR ---
sk_id_curr = st.number_input(
    "Entrez l'ID du client :", 
    min_value=int(df["SK_ID_CURR"].min()), 
    max_value=int(df["SK_ID_CURR"].max()), 
    step=1
)

if st.button("Afficher les résultats du client"):
    if sk_id_curr in df["SK_ID_CURR"].values:
        client_data = df[df["SK_ID_CURR"] == sk_id_curr]

        selected_columns = [
            "EXT_SOURCE_2", "DAYS_BIRTH", "EXT_SOURCE_3", "SK_ID_CURR", 
            "bureau_DAYS_CREDIT_max", "bureau_DAYS_CREDIT_min", 
            "bureau_DAYS_CREDIT_UPDATE_mean", "bureau_DAYS_CREDIT_mean", 
            "bureau_CREDIT_ACTIVE_Closed_mean", "bureau_CREDIT_ACTIVE_Active_mean"
        ]
        
        X_client = client_data[selected_columns].values
        
        # Prédire la probabilité pour le client
        proba = model.predict_proba(X_client)[0, 1]
        seuil = 0.49  # Seuil optimal
        ecart = round((proba - seuil) * 100, 0)

        # Message basé sur le risque
        if proba <= seuil:
            message = f"Crédit accordé avec un risque de défaut de {100 * proba:.0f}%."
        else:
            message = f"Crédit refusé avec un risque de défaut de {100 * proba:.0f}%."

        # --- Jauge de risque ---
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(proba, 2),
            title={"text": "Risque de défaut de remboursement"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "darkblue"},
                'steps': [
                    {"range": [0, 0.4], "color": "darkgreen"},
                    {"range": [0.4, seuil], "color": "lightgreen"},
                    {"range": [seuil, 0.6], "color": "yellow"},
                    {"range": [0.6, 1], "color": "red"}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": seuil,
                },
            },
        ))
        fig_gauge.add_annotation(
            x=0.49, y=0.5, text="Seuil 0.49", showarrow=True, arrowhead=2, ax=0, ay=-40
        )
        st.plotly_chart(fig_gauge)

        # --- Affichage des résultats ---
        st.subheader("Décision")
        st.markdown(f"<h3 style='color: {'green' if proba <= seuil else 'red'};'>{message}</h3>", unsafe_allow_html=True)

        if ecart <= 0:
            ecart_message = "L'écart est en dessous du seuil, le risque est acceptable."
        else:
            ecart_message = "L'écart est au-dessus du seuil, le risque est élevé."

        st.write(f"Écart par rapport au seuil : {ecart} points")
        st.markdown(f"<h5 style='color: {'green' if ecart <= 0 else 'red'};'>{ecart_message}</h5>", unsafe_allow_html=True)

        # Affichage des données client
        st.subheader("Données du client")
        st.dataframe(client_data)

        # # Calculer les features importantes globalement
        feature_importances = model.feature_importances_
        
        global_importance = pd.DataFrame({
            "Feature": selected_columns,
            "Importance Globale": feature_importances
        }).sort_values(by="Importance Globale", ascending=True)

        # Importance locale via SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_client)
        local_importance = pd.DataFrame({
            "Feature": selected_columns,
            "Importance Locale": np.abs(shap_values[0])
        }).sort_values(by="Importance Locale", ascending=True)
        # Configuration des colonnes
        col1, col2 = st.columns(2)

        with col1:
            # Caractéristiques ayant pesé sur le modèle
            fig_global = px.bar(global_importance, x="Importance Globale"
                            , y="Feature", orientation='h'
                            , title="Caractéristiques ayant pesé sur le modèle"
                            )
            st.plotly_chart(fig_global)

        # --- Description dans la deuxième colonne ---
        with col2:
            # Caractéristiques qui ont joué sur le client
            fig_local = px.bar(local_importance, x="Importance Locale"
                        , y="Feature", orientation='h'
                        , title="Caractéristiques qui ont joué sur la décision"
                        )
            st.plotly_chart(fig_local)

    else:
        st.error("L'ID client saisi n'existe pas dans les données.")

# --- Analyse exploratoire sur les tranches d'âge ---
if st.checkbox("Distibution de clients par tranches d'âge"):
    df["AGE"] = (-df["DAYS_BIRTH"] / 365).astype(int)  # Conversion en années
    age_bins = [20, 30, 40, 50, 60, 70, 80]
    age_labels = ["20-30 ans", "30-40 ans", "40-50 ans", "50-60 ans", "60-70 ans", "70-80 ans"]
    df["AGE_GROUP"] = pd.cut(df["AGE"], bins=age_bins, labels=age_labels, right=False)

    client_age = df.loc[df["SK_ID_CURR"] == sk_id_curr, "AGE"].iloc[0]
    client_age_group = df.loc[df["SK_ID_CURR"] == sk_id_curr, "AGE_GROUP"].iloc[0]

    # --- Répartition des scores de crédit par tranches d'âge ---
    st.title("Le risque de crédit par tranche d'âge")
    
    # Calculer la moyenne des scores de crédit par tranche d'âge
    age_group_scores = df.groupby("AGE_GROUP")["TARGET"].mean().reset_index()
    
    # Créer un graphique en barres
    fig_age_scores = px.bar(
        age_group_scores, 
        x="AGE_GROUP", 
        y="TARGET", 
        title="Moyenne des scores de crédit par tranche d'âge",
        labels={"AGE_GROUP": "Tranche d'âge", "TARGET": "Score de crédit moyen"},
        color_discrete_sequence=["#00CC96"]
    )
    
    # Colorer en rouge la tranche d'âge du client sélectionné
    fig_age_scores.update_traces(marker_color=[
        "red" if age_group == client_age_group else "#00CC96" for age_group in age_group_scores["AGE_GROUP"]
    ])
    
    # Afficher le graphique
    st.plotly_chart(fig_age_scores)

    # --- Histogramme de la tranche d'âge du client ---
    fig_age = px.histogram(
        df, x="AGE_GROUP", title="Répartition des tranches d'âge",
        labels={"AGE_GROUP": "Tranche d'âge", "count": "Nombre de clients"},
        color_discrete_sequence=["#636EFA"]
    )
    fig_age.add_annotation(
        x=str(client_age_group), 
        y=df["AGE_GROUP"].value_counts()[client_age_group], 
        text=f"Client {sk_id_curr} : {client_age} ans", 
        showarrow=True, arrowhead=2, ax=0, ay=-40
    )
    st.plotly_chart(fig_age)

# --- Nouvelle section : Comparaison des distributions de variables ---
@st.cache_data
def load_data():
    data = pd.read_pickle("train_red_format.pkl")
    return data.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")  # Suppression des colonnes non pertinentes

data = load_data()

# Interface utilisateur
st.title("Situation par rapport autres clients")

# Définition des groupes de variables
group1 = [ "bureau_DAYS_CREDIT_max", "bureau_DAYS_CREDIT_min", "bureau_DAYS_CREDIT_mean"]
group2 = ["EXT_SOURCE_2", "EXT_SOURCE_3", "bureau_CREDIT_ACTIVE_Closed_mean"
          , "bureau_CREDIT_ACTIVE_Active_mean"]

# Interface avec deux sélecteurs côte à côte
col3, col4 = st.columns(2)

with col3:
    selected_vars1 = st.multiselect("Sélectionnez des variables du groupe 1"
                                    , group1, default=group1[:0])

with col4:
    selected_vars2 = st.multiselect("Sélectionnez des variables du groupe 2"
                                    , group2, default=group2[:0])

# Affichage des distributions en deux colonnes
col3, col4 = st.columns(2)

# Affichage des histogrammes du premier groupe
with col3:
    if selected_vars1:
        for var in selected_vars1:
            fig_hist, ax = plt.subplots(figsize=(5, 4))
            n, bins, patches = ax.hist(data[var], bins=15, edgecolor='black', alpha=0.7)
            ax.set_title(f"Distribution de {var}")
            ax.set_xlabel(var)
            ax.set_ylabel("Fréquence")

            # Colorer en rouge la barre du client sélectionné
            if sk_id_curr in df["SK_ID_CURR"].values:
                client_data = df[df["SK_ID_CURR"] == sk_id_curr]
                client_value = client_data[var].values[0]  # Extraire la valeur scalaire
                bin_index = np.digitize(client_value, bins) - 1
                patches[bin_index].set_facecolor('red')

            st.pyplot(fig_hist)

# Affichage des histogrammes du second groupe
with col4:
    for var in selected_vars2:
        fig_hist, ax = plt.subplots(figsize=(5, 4))
        n, bins, patches = ax.hist(data[var], bins=15, edgecolor='black', alpha=0.7)
        ax.set_title(f"Distribution de {var}")
        ax.set_xlabel(var)
        ax.set_ylabel("Fréquence")

        # Colorer en rouge la barre du client sélectionné
        if sk_id_curr in df["SK_ID_CURR"].values:
            client_data = df[df["SK_ID_CURR"] == sk_id_curr]
            client_value = client_data[var].values[0]  # Extraire la valeur scalaire
            bin_index = np.digitize(client_value, bins) - 1
            patches[bin_index].set_facecolor('red')

        st.pyplot(fig_hist)

# Message si aucun choix
if not selected_vars1 and not selected_vars2:
    st.warning("Sélectionnez au moins une variable dans l'un des groupes pour afficher les distributions.")


# --- Évaluer un nouveau dossier client ---
st.title("Évaluer à nouveau le dossier du client")

# Dictionnaire de traductions en français (sans underscores)
traductions = {
    "EXT_SOURCE_2": "Source Externe 2",
    "DAYS_BIRTH": "Âge du client (en jours depuis la naissance, valeur négative)",
    "EXT_SOURCE_3": "Source Externe 3",
    "SK_ID_CURR": "ID Client",
    "bureau_DAYS_CREDIT_max": "Durée maximale des crédits précédents (en jours)",
    "bureau_DAYS_CREDIT_min": "Durée minimale des crédits précédents (en jours)",
    "bureau_DAYS_CREDIT_UPDATE_mean": "Moyenne des mises à jour des crédits (en jours)",
    "bureau_DAYS_CREDIT_mean": "Durée moyenne des crédits précédents (en jours)",
    "bureau_CREDIT_ACTIVE_Closed_mean": "Proportion moyenne des crédits fermés",
    "bureau_CREDIT_ACTIVE_Active_mean": "Proportion moyenne des crédits actifs"
}

# Saisie des valeurs pour le nouveau dossier client
new_client_data = {}
selected_columns = list(traductions.keys())

for col in selected_columns:
    new_client_data[col] = st.number_input(
        # f"Entrez la valeur pour {traductions[col]} :",  # Utilisation de la traduction en français
        f"{traductions[col]} :", value=float(df[col].mean())
    )

# Bouton pour évaluer le nouveau dossier
if st.button("Évaluer", key="eval_button"):
    new_client_df = pd.DataFrame(new_client_data, index=[0])
    new_proba = model.predict_proba(new_client_df[selected_columns])[0, 1]
    seuil = 0.49  # Seuil optimal

    # Message basé sur le risque
    if new_proba <= seuil:
        new_message = f"Crédit accordé avec une probabilité de remboursement de {100 * new_proba:.0f}%."
    else:
        new_message = f"Crédit refusé avec un risque de défaut de {100 * new_proba:.0f}%."

    # Affichage du résultat
    st.markdown(f"<h3 style='color: {'green' if new_proba <= seuil else 'red'};'>{new_message}</h3>", unsafe_allow_html=True)