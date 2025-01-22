import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Chargement des ressources
@st.cache_resource
def load_resources():
    # Charger le modèle GloVe
    glove_model = load_model("Glove_gru_model.keras")

    # Charger le tokenizer
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)

    # Charger la longueur maximale
    with open("maxlen.pickle", "rb") as handle:
        max_len = pickle.load(handle)

    return glove_model, tokenizer, max_len

# Charger les ressources
glove_model, tokenizer, max_len = load_resources()

# Fonction pour prédire le sentiment
def predict_sentiment(model, tokenizer, max_len, text):
    # Prétraitement du texte
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

    # Prédiction
    prediction = model.predict(pad)
    sentiment = "positif" if prediction[0] > 0.5 else "négatif"
    prob = prediction[0][0]
    return sentiment, prob

# Interface utilisateur Streamlit
st.title("Prédiction de Sentiment de Tweets")
st.write("Cette application utilise un modèle basé sur GloVe pour analyser les sentiments des tweets.")

# Initialisation des variables d'état
if "validation" not in st.session_state:
    st.session_state.validation = None

# Saisie utilisateur
user_input = st.text_area("Entrez votre message :")

if st.button("Prédire"):
    if user_input.strip():
        # Prédire le sentiment
        sentiment, prob = predict_sentiment(glove_model, tokenizer, max_len, user_input)

        # Afficher le résultat
        st.write(f"**Texte analysé** : {user_input}")
        st.write(f"**Sentiment prédit** : {sentiment}")
        st.write(f"**Probabilité calculée** : {prob:.2f}")

        # Validation de l'utilisateur
        st.write("Validez-vous cette prédiction ?")
        if st.button("Oui", key="yes_button"):
            st.session_state.validation = "validée"
        if st.button("Non", key="no_button"):
            st.session_state.validation = "rejetée"

# Afficher le statut de validation
if st.session_state.validation == "validée":
    st.success("Merci pour votre validation !")
elif st.session_state.validation == "rejetée":
    st.error("Merci pour votre retour, la prédiction sera analysée.")

# Informations sur l'application
st.sidebar.header("Informations")
st.sidebar.write("Cette application utilise des embeddings GloVe pour prédire les sentiments des tweets.")
st.sidebar.write("Le modèle exploite une architecture GRU bidirectionnelle pour une analyse avancée.")
st.sidebar.write("Si la probabilité est inférieure à 0.5, le sentiment est considéré comme négatif ; sinon, il sera positif.")