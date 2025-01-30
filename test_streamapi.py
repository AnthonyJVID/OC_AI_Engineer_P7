import streamlit as st
import requests
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler

# Initialiser le logger pour Application Insights
def get_logger():
    logger = logging.getLogger("azure_logger")
    if not logger.handlers:
        logger.setLevel(logging.WARNING)
        azure_handler = AzureLogHandler(
            connection_string="InstrumentationKey=9d9b8b78-bac1-4b34-ad55-6a1e85354a11;IngestionEndpoint=https://westeurope-5.in.applicationinsights.azure.com/;LiveEndpoint=https://westeurope.livediagnostics.monitor.azure.com/;ApplicationId=502c9ce7-c180-47a4-ab7b-3e29bab8837e"
        )
        logger.addHandler(azure_handler)
    return logger

logger = get_logger()

# URL de votre API locale
API_URL = "http://127.0.0.1:8000/predict"

# Fonction pour appeler l'API
def call_api(tweet):
    try:
        response = requests.post(API_URL, json={"text": tweet})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API : {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API : {str(e)}")
        return None

# Fonction pour enregistrer les traces en cas de non-validation
def log_non_validation(tweet, sentiment, probability):
    message = f"Tweet non validé : '{tweet}', Prédiction : '{sentiment}', Probabilité : {probability:.2f}"
    logger.warning(message)

# Interface Streamlit
st.title("Interface de Test : Analyse de Sentiment")
st.write("Testez l'API d'analyse de sentiment en saisissant un tweet ci-dessous.")

# Zone de saisie du tweet
user_tweet = st.text_area("Entrez un tweet à analyser :")

if st.button("Analyser le tweet"):
    if user_tweet.strip():
        # Appeler l'API
        prediction = call_api(user_tweet)
        if prediction:
            text = prediction.get("text")
            sentiment = prediction.get("sentiment")
            probability = prediction.get("probability_positive")

            # Afficher la prédiction
            st.write(f"**Tweet analysé :** {text}")
            st.write(f"**Sentiment prédit :** {sentiment}")
            st.write(f"**Probabilité (positif) :** {probability:.2f}")

            # Demander la validation à l'utilisateur
            st.write("Est-ce que cette prédiction est correcte ?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Oui, valide"):
                    st.success("Merci pour votre validation.")
            with col2:
                if st.button("Non, invalide"):
                    log_non_validation(user_tweet, sentiment, probability)
                    st.error("La prédiction a été enregistrée comme non valide.")
    else:
        st.error("Veuillez saisir un tweet avant de lancer l'analyse.")