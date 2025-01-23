import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler

# Fonction qui retourne un logger configuré une seule fois
def get_logger():
    # On choisit un nom spécifique pour éviter les conflits
    logger = logging.getLogger("azure_logger")
    if not logger.handlers:  # Vérifie s'il n'y a pas déjà de handlers
        logger.setLevel(logging.WARNING)
        azure_handler = AzureLogHandler(
            connection_string="InstrumentationKey=9d9b8b78-bac1-4b34-ad55-6a1e85354a11;"
                              "IngestionEndpoint=https://westeurope-5.in.applicationinsights.azure.com/;"
                              "LiveEndpoint=https://westeurope.livediagnostics.monitor.azure.com/;"
                              "ApplicationId=502c9ce7-c180-47a4-ab7b-3e29bab8837e"
        )
        # Filtrer pour n'envoyer que les messages contenant "Tweet non validé"
        azure_handler.addFilter(lambda record: "Tweet non validé" in record.msg)
        logger.addHandler(azure_handler)
    return logger

logger = get_logger()

@st.cache_resource
def load_resources():
    glove_model = load_model("Glove_gru_model.keras")
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    with open("maxlen.pickle", "rb") as handle:
        max_len = pickle.load(handle)
    return glove_model, tokenizer, max_len

def log_non_validation(tweet, prediction):
    message = f"Tweet non validé : {tweet}, Prédiction : {prediction}"
    print(f"[INFO] Trace locale : {message}")
    logger.warning(message)

def predict_sentiment(model, tokenizer, max_len, text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    prediction = model.predict(pad)
    sentiment = "positif" if prediction[0] > 0.5 else "négatif"
    return sentiment, float(prediction[0][0])

glove_model, tokenizer, max_len = load_resources()

st.title("Prédiction de Sentiment de Tweets")
st.write("Cette application utilise un modèle GloVe pour analyser les sentiments des tweets.")

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "validation" not in st.session_state:
    st.session_state.validation = None
if "log_sent" not in st.session_state:
    st.session_state.log_sent = False

user_input = st.text_area("Entrez votre message :")

if st.button("Prédire"):
    if user_input.strip():
        sentiment, prob = predict_sentiment(glove_model, tokenizer, max_len, user_input)
        st.session_state.prediction_result = (user_input, sentiment, prob)
        st.session_state.validation = None
        st.session_state.log_sent = False

if st.session_state.prediction_result:
    texte, sentiment, prob = st.session_state.prediction_result
    st.write(f"**Texte analysé** : {texte}")
    st.write(f"**Sentiment prédit** : {sentiment}")
    st.write(f"**Probabilité calculée** : {prob:.2f}")

    if st.session_state.validation is None:
        st.write("Validez-vous cette prédiction ?")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("Oui"):
                st.session_state.validation = "validée"
        with col_no:
            if st.button("Non"):
                st.session_state.validation = "rejetée"
                if not st.session_state.log_sent:
                    log_non_validation(texte, sentiment)
                    st.session_state.log_sent = True

if st.session_state.validation == "validée":
    st.success("Merci de votre réponse !")
elif st.session_state.validation == "rejetée":
    st.error("Merci pour votre retour, la prédiction sera analysée.")

st.sidebar.header("Informations")
st.sidebar.write("Cette application utilise des embeddings GloVe pour prédire les sentiments des tweets.")
st.sidebar.write("Le modèle exploite une architecture GRU bidirectionnelle pour une analyse avancée.")
st.sidebar.write("Si la probabilité est inférieure à 0.5, le sentiment est considéré comme négatif ; sinon, il est positif.")
