import streamlit as st
import requests
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler

# 1. Configuration du Logger pour envoyer les logs à Application Insights
def get_logger():
    logger = logging.getLogger("azure_logger")
    if not logger.handlers:
        logger.setLevel(logging.WARNING)
        azure_handler = AzureLogHandler(connection_string="InstrumentationKey=3d18c767-dc27-49d5-b8f0-ab528fc6142b")
        azure_handler.addFilter(lambda record: "Tweet non validé" in record.msg)
        logger.addHandler(azure_handler)
    return logger

logger = get_logger()

# 2. URL de l'API FastAPI (déployée sur Azure)
API_URL = "https://tweetsocp7-gjb7g2cwehe5c7ef.uksouth-01.azurewebsites.net/predict"
FEEDBACK_URL = "https://tweetsocp7-gjb7g2cwehe5c7ef.uksouth-01.azurewebsites.net/feedback"

# 3. Interface Streamlit
st.title("Test API - Prédiction de Sentiment")
st.write("Entrez un tweet et obtenez une prédiction via l'API déployée.")

# 4. Initialisation des variables de session
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "validation" not in st.session_state:
    st.session_state.validation = None
if "log_sent" not in st.session_state:
    st.session_state.log_sent = False

# 5. Saisie de texte par l'utilisateur
user_input = st.text_area("Entrez votre tweet :")

# 6. Envoi de la requête API pour prédire le sentiment
if st.button("Prédire"):
    if user_input.strip():
        payload = {"text": user_input}
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            sentiment = result["sentiment"]
            probability = result["probability_positive"]

            st.session_state.prediction_result = (user_input, sentiment, probability)
            st.session_state.validation = None
            st.session_state.log_sent = False
        else:
            st.error("Erreur lors de la récupération de la prédiction.")

# 7. Affichage des résultats
if st.session_state.prediction_result:
    texte, sentiment, prob = st.session_state.prediction_result

    st.write(f"**Texte analysé** : {texte}")
    st.write(f"**Sentiment prédit** : {sentiment}")
    st.write(f"**Probabilité calculée** : {prob:.2f}")

    # 8. Validation par l’utilisateur
    if st.session_state.validation is None:
        st.write("Validez-vous cette prédiction ?")
        col_yes, col_no = st.columns(2)

        with col_yes:
            if st.button("Oui"):
                st.session_state.validation = "validée"
                requests.post(FEEDBACK_URL, json={"correct": True})
                st.success("Merci de votre retour !")

        with col_no:
            if st.button("Non"):
                st.session_state.validation = "rejetée"
                requests.post(FEEDBACK_URL, json={"correct": False})

                if not st.session_state.log_sent:
                    message = f"Tweet non validé : {texte}, Prédiction : {sentiment}"
                    logger.warning(message)
                    st.session_state.log_sent = True
                st.error("Merci pour votre retour, la prédiction sera analysée.")

# 9. Ajout d'informations dans la barre latérale
st.sidebar.header("Informations")
st.sidebar.write("Cette application utilise une API FastAPI hébergée sur Azure pour prédire les sentiments des tweets.")
st.sidebar.write("Le modèle exploite une architecture GRU avec des embeddings GloVe.")
st.sidebar.write("Si la probabilité est inférieure à 0.5, le sentiment est considéré comme négatif ; sinon, il est positif.")