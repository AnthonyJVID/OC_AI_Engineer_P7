import pytest
from fastapi.testclient import TestClient
from sentiment_api.app import app, predict_sentiment

client = TestClient(app)

def test_predict_endpoint():
    # Préparation de la charge utile pour l'endpoint /predict
    payload = {"text": "Votre texte à analyser"}
    response = client.post("/predict", json=payload)

    # Vérification du status code HTTP
    assert response.status_code == 200

    # Récupération et vérification des clés de la réponse
    data = response.json()
    assert "text" in data
    assert "probability_positive" in data
    assert "sentiment" in data

    # Vérification que la probabilité est dans l'intervalle [0, 1]
    probability = data["probability_positive"]
    assert 0 <= probability <= 1

    # Vérification que le label correspond à la valeur de la probabilité
    if probability >= 0.5:
        assert data["sentiment"] == "positif"
    else:
        assert data["sentiment"] == "negatif"

def test_predict_sentiment_function():
    # Appel direct de la fonction predict_sentiment
    text = "Votre texte à analyser"
    probability, sentiment = predict_sentiment(text)

    # Vérification de la validité de la probabilité
    assert 0 <= probability <= 1

    # Vérification de la cohérence entre la probabilité et le label
    if probability >= 0.5:
        assert sentiment == "positif"
    else:
        assert sentiment == "negatif"